"""Pure policy for Jev-style structured reads on the autoregressive Qwen service.

vLLM PR 57250 (merged as 1b3b88e) reads DiffusionGemma's canvas: the answer template sits
in the canvas with noise at each answer slot, one denoise step returns a distribution per
slot, and an example server turns a question schema into those reads behind the Jev
``POST /v1/systemone`` contract. This module implements that contract for Qwen3.8, which
is autoregressive:

- The answer template follows the prompt as ordinary tokens. The distribution over a slot
  is the next-token distribution at the row before it, so every slot of a template comes
  from one prefill that returns several logit rows.
- A slot's row sees only the template before it. A slot's fill therefore reaches later
  slots only, and tokens after the last slot's row never matter; a read stops there.
- The first draw refines to a fixed point by default: each step fills every slot with the
  previous step's most likely label and reads again, until the fills stop changing. The
  fixed point is the greedy label chain, the conditioning an autoregressive model was
  trained on. The first step fills slots with a placeholder (``?``), or with the PR's
  random tokens when ``first_fill`` is ``noise``.
- A later draw reads once, with each earlier slot filled by a label sampled from the
  refined first draw. The spread across draws measures how much a later answer depends
  on the plausible earlier ones.
- Draws with the same visible fills are the same prefill. Reads are memoized per
  decision before they reach the device, so a one-question read costs one prefill for any
  sample count or step count, and a confident earlier answer adds no device reads.
- The first slot's row sees no earlier slot, so the auto policy adds draws only when a
  later slot's first-read entropy passes the threshold.

The shell supplies the tokenizer, the chat prompt, the device read, and the thought
generation as ports, so every decision here is testable without a device.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

MAXIMUM_QUESTIONS = 64
MAXIMUM_SAMPLES = 32
MAXIMUM_ALTERNATIVES = 26
MINIMUM_ALTERNATIVES = 2
MAXIMUM_STEPS = 8
# The first draw refines to its fixed point by default: the greedy label chain, which an
# autoregressive model reads far more accurately on long schemas than one noisy read.
DEFAULT_STEPS = MAXIMUM_STEPS
MINIMUM_CHUNK_ROWS = 8
DEFAULT_AUTO_MAXIMUM = 4
DEFAULT_AUTO_THRESHOLD = 0.1
DEFAULT_SEED = 42
TOP_K = 20
# One read returns at most this many slot rows: the device's logit tile holds 32 rows and
# keeps row 0 for the generation prefill's own next-token logit.
MAXIMUM_SLOTS_PER_READ = 31
# A schema with more questions switches to the compact "indexed" answer format.
LINES_FORMAT_QUESTION_LIMIT = 10
# Seed offsets between noise draws and between the reads of one decision.
DRAW_SEED_STRIDE = 7919
GROUP_SEED_STRIDE = 104729
SCORE_DIGIT_LABEL_LIMIT = 9
# The first draw's fill for an answer slot; a slot where it is not one token takes noise.
PLACEHOLDER_LABEL = "?"
FILL_PLACEHOLDER = "placeholder"
FILL_NOISE = "noise"
ENGINE_NAME = "qwen38-p150x2"
FORMAT_LINES = "lines"
FORMAT_INDEXED = "indexed"
QUESTION_NOUL = "noul"
QUESTION_CHOICE = "choice"
QUESTION_SCORE = "score"
POLICY_AUTO = "auto"
POLICY_FIXED = "fixed"
CHUNK_PROMPT_OWN = "own"
CHUNK_PROMPT_SHARED = "shared"
SCHEMA_EXTENSIONS = (
    "instructions",
    "samples",
    "auto_max",
    "auto_threshold",
    "steps",
    "think",
    "ask",
    "chunk_rows",
    "chunk_prompt",
    "sequential",
    "first_fill",
)
PREAMBLE = (
    "Answer a fixed set of questions about the state the user provides. Each question lists "
    "its allowed answers; reply with exactly one label per question.\n"
)
CHUNKED_NOTE = (
    " A reply may cover only some of the questions; answer every line that is present."
)


@dataclass(frozen=True)
class AnswerFormat:
    join: str
    lead: str
    instruction: str


FORMATS = {
    FORMAT_LINES: AnswerFormat(
        join="\n",
        lead="{id}: ",
        instruction='Reply with one line per question, in this order, formatted as "id: label".',
    ),
    FORMAT_INDEXED: AnswerFormat(
        join=" ",
        lead="{id}",
        instruction=(
            "Reply on one line with each question's id immediately followed by its label, "
            "separated by single spaces."
        ),
    ),
}


class SchemaError(ValueError):
    """The request does not describe a readable question schema."""


@dataclass(frozen=True)
class Question:
    id: str
    kind: str
    instructions: str
    choices: tuple[tuple[str, str | None], ...]
    labels: tuple[str, ...]
    depends_on: tuple[str, ...]
    ask_if: tuple[tuple[str, tuple[str, ...]], ...]
    alone: bool


@dataclass(frozen=True)
class SamplePolicy:
    mode: str
    count: int
    threshold: float = DEFAULT_AUTO_THRESHOLD

    def as_dict(self) -> dict[str, Any]:
        if self.mode == POLICY_AUTO:
            return {"mode": POLICY_AUTO, "max": self.count, "threshold": self.threshold}
        return {"mode": POLICY_FIXED, "n": self.count}


@dataclass(frozen=True)
class Schema:
    questions: tuple[Question, ...]
    instructions: str | None
    policy: SamplePolicy
    steps: int
    think: int
    ask: tuple[str, ...] | None
    chunk_rows: int | None
    chunk_prompt: str
    sequential: bool
    answer_format: str
    first_fill: str = FILL_PLACEHOLDER


@dataclass(frozen=True)
class Slot:
    position: int
    label_ids: tuple[int, ...]
    # The token of PLACEHOLDER_LABEL at this slot, or None when it is not one token there.
    placeholder: int | None = None


@dataclass(frozen=True)
class Template:
    tokens: tuple[int, ...]
    slots: tuple[Slot, ...]


@dataclass(frozen=True)
class ReadRequest:
    """One prefill: the next-token logits after each of ``positions`` are read."""

    token_ids: tuple[int, ...]
    positions: tuple[int, ...]


@dataclass(frozen=True)
class SlotRead:
    probabilities: tuple[float, ...]
    label_mass: float
    entropy: float
    argmax_is_label: bool


@dataclass(frozen=True)
class DeviceRead:
    """Host logits [len(positions), vocab] for one ReadRequest and its device time."""

    logits: torch.Tensor
    seconds: float
    route: str


@dataclass(frozen=True)
class ReadPorts:
    """The shell's tokenizer, prompt, device, and generation operations."""

    encode: Callable[[str], list[int]]
    decode: Callable[[Sequence[int]], str]
    chat_prompt: Callable[[str, str, bool], list[int]]
    read: Callable[[tuple[ReadRequest, ...]], tuple[DeviceRead, ...]]
    think: Callable[[list[int], int], list[int]]
    # Noise draws come from the ordinary vocabulary, below the first special token.
    noise_upper: int
    thought_close_id: int
    thought_close_ids: tuple[int, ...]
    maximum_sequence_length: int


@dataclass
class _ReadCounter:
    """Device reads of one decision, memoized by prefill."""

    device_reads: int = 0
    device_seconds: float = 0.0
    routes: set[str] = field(default_factory=set)
    cache: dict[ReadRequest, DeviceRead] = field(default_factory=dict)


def _require_str_id(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    if not text or ":" in text or "\n" in text:
        raise SchemaError(
            f"question id {text!r} must be non-empty, without ':' or a newline"
        )
    return text


def _question(entry: dict[str, Any]) -> Question:
    qid = _require_str_id(entry.get("id"))
    kind = entry.get("type")
    if kind in (QUESTION_NOUL, "bool", "boolean"):
        kind = QUESTION_NOUL
        criteria = entry.get("criteria") or {}
        if not isinstance(criteria, dict):
            raise SchemaError(
                f"question {qid!r}: noul criteria must be an object with true and false"
            )
        choices = (("yes", criteria.get("true")), ("no", criteria.get("false")))
        labels = ("yes", "no")
    elif kind == QUESTION_CHOICE:
        options = entry.get("options") or []
        if not isinstance(options, list):
            raise SchemaError(f"question {qid!r}: options must be a list")
        choices = tuple(
            (str(option["name"]), option.get("description"))
            if isinstance(option, dict) and "name" in option
            else (str(option), None)
            for option in options
        )
        labels = tuple(chr(ord("A") + index) for index in range(len(choices)))
    elif kind == QUESTION_SCORE:
        levels = entry.get("levels") or []
        if not isinstance(levels, list):
            raise SchemaError(f"question {qid!r}: score levels must be an ordered list")
        choices = tuple((str(level), None) for level in levels)
        labels = tuple(
            str(index + 1)
            if len(choices) <= SCORE_DIGIT_LABEL_LIMIT
            else chr(ord("A") + index)
            for index in range(len(choices))
        )
    else:
        raise SchemaError(f"question {qid!r}: unknown type {kind!r}")
    if len(choices) < MINIMUM_ALTERNATIVES:
        raise SchemaError(f"question {qid!r}: needs at least two alternatives")
    if len(choices) > MAXIMUM_ALTERNATIVES:
        raise SchemaError(
            f"question {qid!r}: at most {MAXIMUM_ALTERNATIVES} alternatives"
        )
    names = [name for name, _description in choices]
    if len(set(names)) != len(names):
        raise SchemaError(f"question {qid!r}: alternatives must have distinct names")
    depends_on = entry.get("depends_on") or []
    ask_if = entry.get("ask_if") or {}
    if not isinstance(depends_on, list) or not all(
        isinstance(dep, str) for dep in depends_on
    ):
        raise SchemaError(
            f"question {qid!r}: depends_on must be a list of question ids"
        )
    if not isinstance(ask_if, dict) or not all(
        isinstance(values, list) and values and all(isinstance(v, str) for v in values)
        for values in ask_if.values()
    ):
        raise SchemaError(
            f"question {qid!r}: ask_if must map a question id to a non-empty list of its answers"
        )
    instructions = entry.get("instructions", "")
    return Question(
        id=qid,
        kind=kind,
        instructions=instructions
        if isinstance(instructions, str)
        else json.dumps(instructions),
        choices=choices,
        labels=labels,
        depends_on=tuple(dict.fromkeys([*depends_on, *ask_if])),
        ask_if=tuple((dep, tuple(values)) for dep, values in ask_if.items()),
        alone=bool(entry.get("alone", False)),
    )


def schedule(questions: Sequence[Question]) -> tuple[tuple[Question, ...], ...]:
    """Questions in stages; a question's stage follows every stage it depends on.

    Declaration order holds within a stage. A dependency outside ``questions`` counts as
    answered. Raises SchemaError on a cycle."""
    ids = {question.id for question in questions}
    pending = list(questions)
    done: set[str] = set()
    levels = []
    while pending:
        level = [
            question
            for question in pending
            if all(dep in done or dep not in ids for dep in question.depends_on)
        ]
        if not level:
            raise SchemaError(
                "schema: dependency cycle among "
                + ", ".join(question.id for question in pending)
            )
        levels.append(tuple(level))
        done.update(question.id for question in level)
        pending = [question for question in pending if question.id not in done]
    return tuple(levels)


def _first_fill(value: Any) -> str:
    if value not in (FILL_PLACEHOLDER, FILL_NOISE):
        raise SchemaError(
            f'schema: first_fill must be "{FILL_PLACEHOLDER}" or "{FILL_NOISE}"'
        )
    return value


def _int_field(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError(f"schema: {name} must be an integer")
    return value


def parse_schema(value: Any, *, maximum_think_tokens: int) -> Schema:
    """Validate the internal schema form: a questions array plus the extensions."""
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("questions"), list)
        or not value["questions"]
    ):
        raise SchemaError("schema: needs a non-empty questions array")
    if len(value["questions"]) > MAXIMUM_QUESTIONS:
        raise SchemaError(f"schema: at most {MAXIMUM_QUESTIONS} questions")
    questions = []
    seen: set[str] = set()
    for entry in value["questions"]:
        if not isinstance(entry, dict):
            raise SchemaError("schema: every question must be an object")
        question = _question(entry)
        if question.id in seen:
            raise SchemaError(f"duplicate question id {question.id!r}")
        seen.add(question.id)
        questions.append(question)
    by_id = {question.id: question for question in questions}
    for question in questions:
        for dep in question.depends_on:
            if dep not in by_id or dep == question.id:
                raise SchemaError(
                    f"question {question.id!r}: depends on unknown question {dep!r}"
                )
        for dep, values in question.ask_if:
            names = [name for name, _description in by_id[dep].choices]
            if any(v not in names for v in values):
                raise SchemaError(
                    f"question {question.id!r}: ask_if values for {dep!r} must be among {names}"
                )
    schedule(questions)
    samples = value.get("samples", POLICY_AUTO)
    if samples == POLICY_AUTO:
        auto_max = _int_field(value.get("auto_max", DEFAULT_AUTO_MAXIMUM), "auto_max")
        threshold = value.get("auto_threshold", DEFAULT_AUTO_THRESHOLD)
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
        ):
            raise SchemaError("schema: auto_threshold must be a finite number")
        policy = SamplePolicy(
            POLICY_AUTO, max(1, min(auto_max, MAXIMUM_SAMPLES)), float(threshold)
        )
    elif isinstance(samples, int) and not isinstance(samples, bool) and samples >= 1:
        policy = SamplePolicy(POLICY_FIXED, min(samples, MAXIMUM_SAMPLES))
    else:
        raise SchemaError('schema: samples must be a positive count or "auto"')
    ask = value.get("ask")
    if ask is not None:
        if not isinstance(ask, list) or not ask or any(a not in seen for a in ask):
            raise SchemaError("schema: ask must list question ids from this schema")
        for question in questions:
            if question.id in ask and any(
                dep not in ask for dep in question.depends_on
            ):
                raise SchemaError(
                    f"schema: ask names {question.id!r} but not everything it depends on"
                )
    chunk_rows = value.get("chunk_rows")
    if (
        chunk_rows is not None
        and _int_field(chunk_rows, "chunk_rows") < MINIMUM_CHUNK_ROWS
    ):
        raise SchemaError(
            f"schema: chunk_rows must be an integer of at least {MINIMUM_CHUNK_ROWS}"
        )
    chunk_prompt = value.get("chunk_prompt", CHUNK_PROMPT_OWN)
    if chunk_prompt not in (CHUNK_PROMPT_OWN, CHUNK_PROMPT_SHARED):
        raise SchemaError('schema: chunk_prompt must be "shared" or "own"')
    think = _int_field(value.get("think", 0), "think")
    if not 0 <= think <= maximum_think_tokens:
        raise SchemaError(
            f"schema: think must be a thought budget from 0 to {maximum_think_tokens} tokens "
            "on this service"
        )
    steps = _int_field(value.get("steps", DEFAULT_STEPS), "steps")
    instructions = value.get("instructions")
    return Schema(
        questions=tuple(questions),
        instructions=None if instructions is None else str(instructions),
        policy=policy,
        steps=max(1, min(steps, MAXIMUM_STEPS)),
        think=think,
        ask=None if ask is None else tuple(ask),
        chunk_rows=chunk_rows,
        chunk_prompt=chunk_prompt,
        sequential=bool(value.get("sequential", False)),
        answer_format=FORMAT_LINES
        if len(questions) <= LINES_FORMAT_QUESTION_LIMIT
        else FORMAT_INDEXED,
        first_fill=_first_fill(value.get("first_fill", FILL_PLACEHOLDER)),
    )


def parse_jev_request(
    body: Any, *, maximum_think_tokens: int
) -> tuple[Schema, str, int]:
    """Schema, state text, and seed from a Jev ``/v1/systemone`` body."""
    if not isinstance(body, dict):
        raise SchemaError("request body must be a JSON object")
    if body.get("images"):
        raise SchemaError("images: this service reads text states only")
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise SchemaError("questions: needs a non-empty map of id -> question")
    entries = []
    for qid, question in questions.items():
        if not isinstance(question, dict):
            raise SchemaError(f"question {qid!r}: must be an object")
        kind, criteria = question.get("type"), question.get("criteria")
        item: dict[str, Any] = {
            "id": qid,
            "type": kind,
            "instructions": question.get("instructions", ""),
        }
        if kind == QUESTION_NOUL:
            if criteria is not None and not isinstance(criteria, dict):
                raise SchemaError(
                    f"question {qid!r}: noul criteria must be an object with true and false"
                )
            item["criteria"] = criteria
        elif kind == QUESTION_CHOICE:
            if not isinstance(criteria, dict) or not criteria:
                raise SchemaError(
                    f"question {qid!r}: choice criteria must map option names to descriptions"
                )
            item["options"] = [
                {"name": str(name), "description": d} for name, d in criteria.items()
            ]
        elif kind == QUESTION_SCORE:
            if not isinstance(criteria, list):
                raise SchemaError(
                    f"question {qid!r}: score criteria must be an ordered list of levels"
                )
            item["levels"] = criteria
        else:
            raise SchemaError(f"question {qid!r}: unknown type {kind!r}")
        for key in ("depends_on", "ask_if", "alone"):
            if key in question:
                item[key] = question[key]
        entries.append(item)
    schema_value = {key: body[key] for key in SCHEMA_EXTENSIONS if key in body}
    schema_value["questions"] = entries
    schema = parse_schema(schema_value, maximum_think_tokens=maximum_think_tokens)
    state = body.get("state")
    if state is None:
        raise SchemaError("state: required")
    seed = body.get("seed", DEFAULT_SEED)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SchemaError("seed: must be an integer")
    return schema, state if isinstance(state, str) else json.dumps(state), seed


def system_text(
    schema: Schema, questions: Sequence[Question], *, chunked: bool = False
) -> str:
    text = PREAMBLE
    if schema.instructions:
        text += "\n" + schema.instructions.strip() + "\n"
    for question in questions:
        text += f"\nQuestion {question.id}: {question.instructions.strip()}\n"
        for (name, description), label in zip(
            question.choices, question.labels, strict=True
        ):
            if question.kind == QUESTION_NOUL:
                text += (
                    f"  {label}: {str(description).strip()}\n"
                    if description
                    else f"  {label}\n"
                )
            elif description:
                text += f"  {label}: {name} ({str(description).strip()})\n"
            else:
                text += f"  {label}: {name}\n"
    text += "\n" + FORMATS[schema.answer_format].instruction
    if chunked:
        text += CHUNKED_NOTE
    return text


def answer_text(
    questions: Sequence[Question], label_indices: Sequence[int], answer_format: str
) -> str:
    form = FORMATS[answer_format]
    return form.join.join(
        form.lead.format(id=question.id) + question.labels[index]
        for question, index in zip(questions, label_indices, strict=True)
    )


def resolve_template(
    questions: Sequence[Question],
    encode: Callable[[str], list[int]],
    lead: str,
    answer_format: str,
) -> Template:
    """Tokenize the answer template and find each question's slot.

    Every label of a question must change exactly one token, at one position shared by all
    of that question's labels, or this raises SchemaError."""
    base_indices = [0] * len(questions)
    base = encode(lead + answer_text(questions, base_indices, answer_format))
    slots = []
    for index, question in enumerate(questions):
        position = None
        label_ids = [0] * len(question.labels)
        for label_index in range(1, len(question.labels)):
            indices = list(base_indices)
            indices[index] = label_index
            tokens = encode(lead + answer_text(questions, indices, answer_format))
            if len(tokens) != len(base):
                raise SchemaError(
                    f"question {question.id!r}: label {question.labels[label_index]!r} is not a single token"
                )
            differences = [
                i for i, (a, b) in enumerate(zip(tokens, base, strict=True)) if a != b
            ]
            if len(differences) != 1 or (
                position is not None and differences[0] != position
            ):
                raise SchemaError(
                    f"question {question.id!r}: labels do not share one template slot"
                )
            position = differences[0]
            label_ids[label_index] = tokens[position]
        assert position is not None
        label_ids[0] = base[position]
        if len(set(label_ids)) != len(label_ids):
            raise SchemaError(
                f"question {question.id!r}: two labels tokenize to the same id"
            )
        placeholder = _placeholder_id(
            questions, index, position, base, encode, lead, answer_format
        )
        slots.append(
            Slot(position=position, label_ids=tuple(label_ids), placeholder=placeholder)
        )
    if [slot.position for slot in slots] != sorted({slot.position for slot in slots}):
        raise SchemaError("schema: answer slots must follow question order")
    return Template(tokens=tuple(base), slots=tuple(slots))


def _placeholder_id(
    questions: Sequence[Question],
    index: int,
    position: int,
    base: Sequence[int],
    encode: Callable[[str], list[int]],
    lead: str,
    answer_format: str,
) -> int | None:
    """The token of PLACEHOLDER_LABEL for question ``index``'s slot, or None.

    The template with the placeholder in place of the label gives the canonical token when
    it changes exactly that one slot token (" ?" after "id: "). A label the tokenizer keeps
    apart from its leading space (a digit after " ") takes the bare placeholder token."""
    form = FORMATS[answer_format]
    text = form.join.join(
        form.lead.format(id=question.id)
        + (PLACEHOLDER_LABEL if i == index else question.labels[0])
        for i, question in enumerate(questions)
    )
    tokens = encode(lead + text)
    if len(tokens) == len(base):
        differences = [
            i for i, (a, b) in enumerate(zip(tokens, base, strict=True)) if a != b
        ]
        if differences == [position]:
            return tokens[position]
    bare = encode(PLACEHOLDER_LABEL)
    return bare[0] if len(bare) == 1 else None


def chunk_groups(
    schema: Schema, questions: Sequence[Question], encode: Callable[[str], list[int]]
) -> tuple[tuple[Question, ...], ...]:
    """``questions`` split in order into the fewest reads that each hold at most
    MAXIMUM_SLOTS_PER_READ slots and, with ``chunk_rows``, that many template rows.
    A question marked alone gets its own read."""
    groups: list[tuple[Question, ...]] = []
    group: list[Question] = []
    for question in questions:
        if question.alone:
            if group:
                groups.append(tuple(group))
                group = []
            groups.append((question,))
            continue
        trial = [*group, question]
        too_many = len(trial) > MAXIMUM_SLOTS_PER_READ
        too_long = schema.chunk_rows is not None and (
            len(encode(answer_text(trial, [0] * len(trial), schema.answer_format)))
            > schema.chunk_rows
        )
        if group and (too_many or too_long):
            groups.append(tuple(group))
            group = [question]
        else:
            group = trial
    if group:
        groups.append(tuple(group))
    return tuple(groups)


def build_read(
    prefix: Sequence[int], template: Template, fills: Sequence[int]
) -> ReadRequest:
    """The prefill for one read: prefix, then the template with ``fills`` in its slots.

    It ends before the last slot's own token, since no read row sees that token or any
    after it. The read positions are the rows just before each slot."""
    tokens = list(prefix) + list(template.tokens)
    for slot, fill in zip(template.slots, fills, strict=True):
        tokens[len(prefix) + slot.position] = int(fill)
    end = len(prefix) + template.slots[-1].position
    return ReadRequest(
        token_ids=tuple(tokens[:end]),
        positions=tuple(len(prefix) + slot.position - 1 for slot in template.slots),
    )


def plan_read_batches(
    *,
    lengths: Sequence[int],
    row_counts: Sequence[int],
    width: int,
    minimum_by_bucket: dict[int, int],
    maximum_rows: int,
) -> tuple[tuple[int, ...], ...]:
    """Reads grouped into shared batched replays, shortest first.

    ``minimum_by_bucket`` maps each batched bucket to the fewest reads a replay of it must
    carry to beat reading them alone. A read joins the open group while the group has fewer
    than ``width`` reads and its rows still fit ``maximum_rows``. A group reads batched only
    when it meets its bucket's minimum; otherwise, and for a read longer than every bucket,
    each read goes alone. Returns index groups covering every read once."""
    buckets = sorted(minimum_by_bucket)

    def bucket_for(length: int) -> int | None:
        return next((bucket for bucket in buckets if bucket >= length), None)

    groups: list[list[int]] = []
    open_group: list[int] = []
    open_rows = 0
    for index in sorted(range(len(lengths)), key=lambda i: (lengths[i], i)):
        rows = row_counts[index]
        if bucket_for(lengths[index]) is None or rows > maximum_rows:
            groups.append([index])
            continue
        if open_group and (len(open_group) == width or open_rows + rows > maximum_rows):
            groups.append(open_group)
            open_group, open_rows = [], 0
        open_group.append(index)
        open_rows += rows
    if open_group:
        groups.append(open_group)
    planned: list[tuple[int, ...]] = []
    for group in groups:
        bucket = bucket_for(max(lengths[index] for index in group))
        if bucket is None or len(group) < minimum_by_bucket[bucket]:
            planned.extend((index,) for index in group)
        else:
            planned.append(tuple(group))
    return tuple(planned)


def first_fills(
    template: Template, seed: int, noise_upper: int, kind: str = FILL_PLACEHOLDER
) -> list[int]:
    """The first draw's slot fills: each slot's placeholder token (``kind`` placeholder, where
    one exists) or noise."""
    rng = random.Random(seed)
    return [
        slot.placeholder
        if kind == FILL_PLACEHOLDER and slot.placeholder is not None
        else rng.randrange(noise_upper)
        for slot in template.slots
    ]


def sampled_fills(
    template: Template, reads: Sequence[SlotRead], seed: int
) -> list[int]:
    """A later draw's slot fills: each slot's label sampled from the first draw's distribution,
    so a later slot reads under the earlier answers the model finds plausible."""
    rng = random.Random(seed)
    return [
        slot.label_ids[
            rng.choices(range(len(slot.label_ids)), weights=read.probabilities)[0]
        ]
        for slot, read in zip(template.slots, reads, strict=True)
    ]


def summarize_slot(
    logits: torch.Tensor, label_ids: Sequence[int], top_k: int = TOP_K
) -> SlotRead:
    """Label probabilities at one slot at temperature 1.

    The probabilities are the softmax over the labels' exact log-probabilities; the entropy
    is over the top-k tokens of the full distribution, as the Jev gate reads it."""
    log_probabilities = torch.log_softmax(logits.float().reshape(-1), dim=-1)
    label_log_probabilities = log_probabilities[list(label_ids)]
    probabilities = torch.softmax(label_log_probabilities, dim=-1)
    top = torch.topk(log_probabilities, top_k)
    entropy = -float((top.values.exp() * top.values).sum())
    return SlotRead(
        probabilities=tuple(float(p) for p in probabilities),
        label_mass=float(label_log_probabilities.exp().sum()),
        entropy=entropy,
        argmax_is_label=int(top.indices[0]) in set(label_ids),
    )


def _argmax(values: Sequence[float]) -> int:
    return max(range(len(values)), key=lambda index: values[index])


def _read_unique(
    ports: ReadPorts, requests: Sequence[ReadRequest], counter: _ReadCounter
) -> list[DeviceRead]:
    """Device reads for ``requests``, computing each distinct prefill once per decision."""
    for request in requests:
        if len(request.token_ids) > ports.maximum_sequence_length:
            raise SchemaError(
                f"the read needs {len(request.token_ids)} tokens; the service holds "
                f"{ports.maximum_sequence_length}"
            )
    missing = tuple(
        request for request in dict.fromkeys(requests) if request not in counter.cache
    )
    if missing:
        results = ports.read(missing)
        if len(results) != len(missing):
            raise RuntimeError("the device returned the wrong number of reads")
        counter.device_reads += len(missing)
        for request, result in zip(missing, results, strict=True):
            counter.device_seconds += result.seconds
            counter.routes.add(result.route)
            counter.cache[request] = result
    return [counter.cache[request] for request in requests]


def _draw_reads(
    ports: ReadPorts,
    prefix: Sequence[int],
    template: Template,
    initial_fills: Sequence[list[int]],
    steps: int,
    counter: _ReadCounter,
) -> list[list[SlotRead]]:
    """Slot reads for one draw per initial fill, refined over ``steps``."""
    fills = [list(fill) for fill in initial_fills]
    reads: list[list[SlotRead]] = [[] for _fill in fills]
    active = list(range(len(fills)))
    for step in range(steps):
        results = _read_unique(
            ports,
            [build_read(prefix, template, fills[draw]) for draw in active],
            counter,
        )
        still_active = []
        for draw, result in zip(active, results, strict=True):
            reads[draw] = [
                summarize_slot(result.logits[index], slot.label_ids)
                for index, slot in enumerate(template.slots)
            ]
            chosen = [
                slot.label_ids[_argmax(read.probabilities)]
                for slot, read in zip(template.slots, reads[draw], strict=True)
            ]
            # The last slot's token is past every read row, so only the earlier fills matter.
            if step + 1 < steps and chosen[:-1] != fills[draw][:-1]:
                fills[draw] = chosen
                still_active.append(draw)
        active = still_active
        if not active:
            break
    return reads


@dataclass(frozen=True)
class _GroupResult:
    answers: dict[str, dict[str, Any]]
    diagnostics: dict[str, Any]
    rows: int


def _decide_group(
    ports: ReadPorts,
    schema: Schema,
    group: Sequence[Question],
    prefix: list[int],
    lead: str,
    seed: int,
    thought: dict[str, Any] | None,
    counter: _ReadCounter,
) -> _GroupResult:
    started = time.monotonic()
    template = resolve_template(group, ports.encode, lead, schema.answer_format)
    policy = schema.policy
    extended = None
    first_entropy = None
    reads = _draw_reads(
        ports,
        prefix,
        template,
        [first_fills(template, seed, ports.noise_upper, schema.first_fill)],
        schema.steps,
        counter,
    )
    if policy.mode == POLICY_FIXED:
        later = policy.count - 1
    else:
        first_entropy = {
            question.id: read.entropy
            for question, read in zip(group, reads[0], strict=True)
        }
        # The first slot's row sees no earlier slot, so every draw repeats its distribution;
        # only a later slot's uncertainty can change with more draws.
        extended = policy.count > 1 and any(
            entropy > policy.threshold for entropy in list(first_entropy.values())[1:]
        )
        later = policy.count - 1 if extended else 0
    if later:
        # Later draws read once: refining them would lead each back to the first draw's fixed
        # point, and their spread is the point. Each reads under earlier answers sampled from
        # the refined first draw.
        fills = [
            sampled_fills(template, reads[0], seed + 1 + draw * DRAW_SEED_STRIDE)
            for draw in range(later)
        ]
        reads += _draw_reads(ports, prefix, template, fills, 1, counter)
    count = len(reads)
    answers: dict[str, dict[str, Any]] = {}
    questions_diagnostics: dict[str, Any] = {}
    for index, question in enumerate(group):
        per_draw = [read[index].probabilities for read in reads]
        mean = [
            sum(p[label] for p in per_draw) / count
            for label in range(len(question.labels))
        ]
        top = _argmax(mean)
        answer: dict[str, Any] = {
            "type": question.kind,
            "label": question.labels[top],
            "confidence": mean[top],
            "probabilities": {
                name: probability
                for (name, _d), probability in zip(question.choices, mean, strict=True)
            },
        }
        if question.kind == QUESTION_NOUL:
            answer["noul"] = mean[0]
        elif question.kind == QUESTION_CHOICE:
            answer["choice"] = question.choices[top][0]
        else:
            answer["score"] = sum((label + 1) * m for label, m in enumerate(mean))
            answer["level"] = question.choices[top][0]
        if count > 1:
            variance = sum((p[top] - mean[top]) ** 2 for p in per_draw) / (count - 1)
            answer["stderr"] = math.sqrt(variance / count)
            answer["agreement"] = sum(1 for p in per_draw if _argmax(p) == top) / count
        answers[question.id] = answer
        questions_diagnostics[question.id] = {
            "pos": template.slots[index].position,
            "entropy": [read[index].entropy for read in reads],
            "label_mass": reads[0][index].label_mass,
            "argmax_is_label": reads[0][index].argmax_is_label,
        }
    tops = [
        {
            question.id: [
                question.labels[_argmax(read[index].probabilities)],
                max(read[index].probabilities),
                read[index].entropy,
            ]
            for index, question in enumerate(group)
        }
        for read in reads
    ]
    return _GroupResult(
        answers=answers,
        diagnostics={
            "steps": schema.steps,
            "samples": {
                "n": count,
                "tops": tops,
                "policy": {
                    **policy.as_dict(),
                    "extended": extended,
                    "first_read_entropy": first_entropy,
                },
                "first_fill": [
                    FILL_PLACEHOLDER
                    if schema.first_fill == FILL_PLACEHOLDER
                    and slot.placeholder is not None
                    else FILL_NOISE
                    for slot in template.slots
                ],
            },
            "timing": {"total_ms": (time.monotonic() - started) * 1e3, "reads": count},
            "thought": thought,
            "prompt_tokens": len(prefix),
            "questions": questions_diagnostics,
            "engine": ENGINE_NAME,
        },
        rows=len(template.tokens) + 1 + (thought["tokens"] if thought else 0),
    )


def write_thought(
    ports: ReadPorts, system: str, state: str, budget: int
) -> tuple[list[int], dict[str, Any]]:
    """A read prefix ending with a thought the model wrote: the chat prompt with thinking
    on, up to ``budget`` generated tokens, and the canonical close of the thought block."""
    prompt = ports.chat_prompt(system, state, True)
    started = time.monotonic()
    generated = list(ports.think(prompt, budget))
    closed = ports.thought_close_id in generated
    if closed:
        generated = generated[: generated.index(ports.thought_close_id)]
    return prompt + generated + list(ports.thought_close_ids), {
        "tokens": len(generated),
        "closed": closed,
        "ms": (time.monotonic() - started) * 1e3,
        "text": ports.decode(generated),
    }


def answer_name(question: Question, answer: dict[str, Any] | None) -> str | None:
    """The answer as ask_if compares it: yes or no, an option name, or a level name."""
    if answer is None:
        return None
    if question.kind == QUESTION_NOUL:
        return answer["label"]
    return answer.get("choice", answer.get("level"))


def decide(
    ports: ReadPorts, schema: Schema, state: str, seed: int
) -> tuple[dict[str, Any], int]:
    """One decision: the answers and diagnostics, and the template rows it read.

    Questions run in stages by their dependencies. A stage is one joint read per chunk,
    with a question marked alone in its own read. Later stages continue the earlier
    answers as prompt tokens. A question whose ask_if condition failed is skipped and its
    answer is null."""
    started = time.monotonic()
    counter = _ReadCounter()
    asked_questions = [
        q for q in schema.questions if schema.ask is None or q.id in schema.ask
    ]
    levels = schedule(asked_questions)
    form = FORMATS[schema.answer_format]
    chained = len(levels) > 1 or schema.sequential
    system_full = system_text(schema, schema.questions)
    base_ids: list[int] | None = None
    thought: dict[str, Any] | None = None
    if chained:
        if schema.think:
            base_ids, thought = write_thought(ports, system_full, state, schema.think)
        else:
            base_ids = ports.chat_prompt(system_full, state, False)
    by_id = {question.id: question for question in schema.questions}
    answered: dict[str, dict[str, Any] | None] = {}
    lines: list[str] = []
    parts: list[_GroupResult] = []
    stages: list[list[str]] = []
    chunks: list[list[str]] = []
    skipped: dict[str, Any] = {}
    read_index = 0
    for level in levels:
        asked = []
        for question in level:
            failed = next(
                (
                    (dep, values)
                    for dep, values in question.ask_if
                    if answer_name(by_id[dep], answered.get(dep)) not in values
                ),
                None,
            )
            if failed is not None:
                answered[question.id] = None
                skipped[question.id] = {
                    "because": failed[0],
                    "was": answer_name(by_id[failed[0]], answered.get(failed[0])),
                    "wanted": list(failed[1]),
                }
                continue
            asked.append(question)
        if not asked:
            continue
        stages.append([question.id for question in asked])
        for group in chunk_groups(schema, asked, ports.encode):
            group_seed = seed + GROUP_SEED_STRIDE * read_index
            group_thought = None
            if lines and base_ids is not None:
                prefix, lead = base_ids + ports.encode(form.join.join(lines)), form.join
            elif base_ids is not None:
                prefix, lead = base_ids, ""
            else:
                sub_system = (
                    system_text(schema, schema.questions, chunked=True)
                    if schema.chunk_prompt == CHUNK_PROMPT_SHARED
                    else system_text(schema, group)
                )
                if schema.think:
                    prefix, group_thought = write_thought(
                        ports, sub_system, state, schema.think
                    )
                else:
                    prefix = ports.chat_prompt(sub_system, state, False)
                lead = ""
            part = _decide_group(
                ports, schema, group, prefix, lead, group_seed, group_thought, counter
            )
            parts.append(part)
            chunks.append([question.id for question in group])
            answered.update(part.answers)
            lines.append(
                answer_text(
                    group,
                    [
                        question.labels.index(part.answers[question.id]["label"])
                        for question in group
                    ],
                    schema.answer_format,
                )
            )
            read_index += 1
    answers = {question.id: answered.get(question.id) for question in asked_questions}
    questions_diagnostics: dict[str, Any] = {}
    for part in parts:
        questions_diagnostics.update(part.diagnostics["questions"])
    if thought is None:
        thoughts = [part.diagnostics["thought"] for part in parts]
        thought = (
            thoughts[0] if len(parts) == 1 else ([t for t in thoughts if t] or None)
        )
        extra_rows = 0
    else:
        extra_rows = thought["tokens"]
    single = len(parts) == 1 and not skipped
    diagnostics = {
        "steps": schema.steps,
        "stages": stages,
        "skipped": skipped,
        "chunks": chunks,
        "chunk_prompt": "full" if chained else schema.chunk_prompt,
        "sequential": schema.sequential,
        "conditioning": None
        if len(stages) <= 1 and not schema.sequential
        else "prefill",
        "thought": thought,
        "samples": (
            parts[0].diagnostics["samples"]
            if single
            else {
                "n": [part.diagnostics["samples"]["n"] for part in parts],
                "tops": [part.diagnostics["samples"]["tops"] for part in parts],
                "policy": [part.diagnostics["samples"]["policy"] for part in parts],
            }
        ),
        "timing": {
            "total_ms": (time.monotonic() - started) * 1e3,
            "reads": sum(part.diagnostics["timing"]["reads"] for part in parts),
            "device_reads": counter.device_reads,
            "device_ms": counter.device_seconds * 1e3,
            "routes": sorted(counter.routes),
        },
        "prompt_tokens": max(
            (part.diagnostics["prompt_tokens"] for part in parts), default=0
        )
        or None,
        "questions": questions_diagnostics,
        "engine": ENGINE_NAME,
    }
    return {"answers": answers, "diagnostics": diagnostics}, sum(
        part.rows for part in parts
    ) + extra_rows


def jev_answer(
    question: Question, answer: dict[str, Any] | None
) -> dict[str, Any] | None:
    """One answer in the Jev contract's shape."""
    if answer is None:
        return None
    if question.kind == QUESTION_NOUL:
        return {"type": QUESTION_NOUL, "noul": answer["noul"]}
    if question.kind == QUESTION_CHOICE:
        return {
            "type": QUESTION_CHOICE,
            "choice": answer["choice"],
            "probabilities": answer["probabilities"],
            "confidence": answer["confidence"],
        }
    names = [name for name, _description in question.choices]
    probabilities = {
        str(index): answer["probabilities"][name] for index, name in enumerate(names)
    }
    return {
        "type": QUESTION_SCORE,
        "score": sum(index * p for index, p in enumerate(probabilities.values())),
        "legend": {str(index): name for index, name in enumerate(names)},
        "probabilities": probabilities,
        "confidence": answer["confidence"],
    }


def jev_response(
    schema: Schema, result: dict[str, Any], output_rows: int, model_alias: str
) -> dict[str, Any]:
    """The ``/v1/systemone`` body for a decision."""
    asked = [q for q in schema.questions if schema.ask is None or q.id in schema.ask]
    return {
        "model": model_alias,
        "answers": {
            question.id: jev_answer(question, result["answers"][question.id])
            for question in asked
        },
        "usage": {
            "input_tokens": result["diagnostics"]["prompt_tokens"] or 0,
            "output_tokens": output_rows,
        },
        "diagnostics": result["diagnostics"],
    }

"""Serialized Clef SystemOne decisions: TT Qwen hidden states, released CPU joint head.

Run with ``python -m qwen36_service.clef --model-path ... --variant clef
--cache-path ...``. The model directory is a pinned, read-only checkpoint; the
separate TT tensor cache must be writable. No autoregressive decode is used.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import gc
import hashlib
import io
import json
import logging
import os
import signal
import tempfile
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch
from clef_release_model import JointSchemaHead, encode_record, systemone_answer
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoProcessor

MAX_TOKENS = 16384
DEFAULT_MAX_LENGTH = 2048
MAX_BODY_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class Variant:
    revision: str
    layers: int
    hidden: int
    devices: tuple[int, ...]
    head_sha256: str


VARIANTS = {
    "clef": Variant(
        revision="2f3de3dd85f379784083b0814d997ab627200f0c",
        layers=64,
        hidden=5120,
        devices=(0, 1),
        head_sha256="a010ac04f078e699988e4049cbea5e62c962393f59fec366640b64e8d69a4953",
    ),
    "clef-flash": Variant(
        revision="17f0b0ad64efb65d273590632833508766b2aae6",
        layers=32,
        hidden=4096,
        devices=(0,),
        head_sha256="19cdcec8c81dc9212be320fff47462ab342fbc1278be4368fb3da71241cf5ba0",
    ),
}


class RequestError(ValueError):
    """Invalid request; safe to return as HTTP 400."""


class ServiceStop(Exception):
    """Request normal cleanup on process termination."""


def _stop_service(signum: int, _frame: Any) -> None:
    raise ServiceStop(f"signal {signum}")


def _json_file(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise TypeError(f"{path.name} must be a JSON object")
    return result


def validate_checkpoint(path: Path, variant: str) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Reject wrong revisions and incompatible architectures before claiming any devices."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    spec = VARIANTS[variant]
    path = path.expanduser().resolve(strict=True)
    if not path.is_dir():
        raise ValueError("--model-path must name a checkpoint directory")
    marker = path / "revision"
    if marker.is_file():
        if marker.read_text(encoding="ascii").strip() != spec.revision:
            raise ValueError(f"checkpoint revision must be {spec.revision}")
    elif not (path.name == spec.revision and path.parent.name == "snapshots"):
        raise ValueError("checkpoint requires a matching revision file or pinned HF snapshot path")
    config = _json_file(path / "config.json")
    text = config.get("text_config", {})
    expected = {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "tie_word_embeddings": False,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("checkpoint is not a compatible Qwen3.5 conditional generation model")
    if (
        text.get("model_type") != "qwen3_5_text"
        or text.get("num_hidden_layers") != spec.layers
        or text.get("hidden_size") != spec.hidden
        or text.get("vocab_size") != 248320
        or len(text.get("layer_types", [])) != spec.layers
        or any(
            layer != ("full_attention" if (i + 1) % 4 == 0 else "linear_attention")
            for i, layer in enumerate(text["layer_types"])
        )
        or config.get("vision_config", {}).get("out_hidden_size") != spec.hidden
    ):
        raise ValueError(f"checkpoint architecture does not match {variant}")
    head_config = _json_file(path / "joint_head_config.json")
    if head_config != {
        "hidden_size": spec.hidden,
        "width": 1024,
        "routing_layers": 2,
        "layers": 4,
        "heads": 16,
        "feedforward": 4096,
    }:
        raise ValueError("joint schema head configuration does not match the pinned release")
    index = _json_file(path / "model.safetensors.index.json")
    shard = index.get("weight_map", {}).get("lm_head.weight")
    if not isinstance(shard, str) or Path(shard).name != shard or not shard.endswith(".safetensors"):
        raise ValueError("checkpoint has no valid lm_head.weight safetensors shard")
    if not (path / shard).is_file() or not (path / "joint_head.safetensors").is_file():
        raise ValueError("checkpoint is missing head or output embedding weights")
    with (path / "joint_head.safetensors").open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != spec.head_sha256:
        raise ValueError("joint schema head weights do not match the pinned release")
    return config, head_config, path / shard


def _load_head(path: Path, head_config: dict[str, Any], embedding_shard: Path) -> tuple[JointSchemaHead, torch.Tensor]:
    head = JointSchemaHead(**head_config).to(dtype=torch.bfloat16)
    head.load_state_dict(load_file(path / "joint_head.safetensors", device="cpu"), strict=True)
    head.eval()
    # Read only this tensor; never instantiate a host copy of the Qwen backbone.
    with safe_open(embedding_shard, framework="pt", device="cpu") as weights:
        lm_head = weights.get_tensor("lm_head.weight")
    if tuple(lm_head.shape) != (248320, head_config["hidden_size"]):
        raise ValueError("lm_head.weight shape does not match the selected variant")
    return head, lm_head


def _reference_visual(path: Path, config: dict[str, Any]) -> Any:
    """Load just the HF visual tower needed by the TT vision adapter."""
    from transformers import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    visual = Qwen3_5VisionModel(Qwen3_5Config.from_dict(config).vision_config)
    visual = visual.to(dtype=torch.bfloat16)
    index = _json_file(path / "model.safetensors.index.json")["weight_map"]
    prefix = "model.visual."
    keys = {key.removeprefix(prefix): key for key in index if key.startswith(prefix)}
    if not keys:
        raise ValueError("checkpoint index has no vision weights")
    state: dict[str, torch.Tensor] = {}
    for shard in sorted({index[key] for key in keys.values()}):
        if Path(shard).name != shard:
            raise ValueError("vision shard path escapes checkpoint")
        with safe_open(path / shard, framework="pt", device="cpu") as weights:
            for name, original in keys.items():
                if index[original] == shard:
                    state[name] = weights.get_tensor(original)
    visual.load_state_dict(state, strict=True)
    return visual.eval()


def _media_image(value: Any) -> Any:
    """Accept PIL images in Python, or base64 image data URLs over JSON HTTP."""
    from PIL import Image

    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if not isinstance(value, str) or not value.startswith("data:image/") or ";base64," not in value:
        raise RequestError("images and video frames must be PIL images or base64 image data URLs")
    try:
        payload = base64.b64decode(value.split(";base64,", 1)[1], validate=True)
        image = Image.open(io.BytesIO(payload))
        image.load()
        return image.convert("RGB")
    except (binascii.Error, ValueError, OSError) as error:
        raise RequestError("invalid base64 image data URL") from error


def _prepare_media(request: dict[str, Any]) -> dict[str, Any]:
    record = dict(request)
    images, videos = request.get("images"), request.get("videos")
    if images and videos:
        raise RequestError("mixed image/video records are unsupported by the TT vision splice")
    if images is not None:
        if not isinstance(images, list) or not images:
            raise RequestError("images must be a nonempty list of PIL images or image data URLs")
        record["images"] = [_media_image(image) for image in images]
    if videos is not None:
        if (
            not isinstance(videos, list)
            or not videos
            or any(not isinstance(video, list) or not video for video in videos)
        ):
            raise RequestError("videos must be a nonempty list of nonempty frame lists")
        record["videos"] = [[_media_image(frame) for frame in video] for video in videos]
    if "media_kwargs" in record and not isinstance(record["media_kwargs"], dict):
        raise RequestError("media_kwargs must be a JSON object")
    return record


def _validate_request(request: Any, variant: str) -> dict[str, Any]:
    if not isinstance(request, dict):
        raise RequestError("request must be a JSON object")
    release_name = "Clef" if variant == "clef" else "Clef-Flash"
    if request.get("model") not in (variant, f"Cloudflare/{variant}", f"Cloudflare/{release_name}"):
        raise RequestError(f"model must identify Cloudflare/{release_name}")
    if "state" not in request:
        raise RequestError("state is required")
    questions = request.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise RequestError("at least one question is required")
    for key, question in questions.items():
        if not isinstance(key, str) or not isinstance(question, dict):
            raise RequestError("questions must map string IDs to objects")
        kind = question.get("type")
        if kind not in ("noul", "choice", "score"):
            raise RequestError(f"{key}: type must be noul, choice, or score")
        criteria = question.get("criteria")
        if kind == "choice" and (
            not isinstance(criteria, dict)
            or not criteria
            or any(not isinstance(option, str) for option in criteria)
        ):
            raise RequestError(f"{key}: choice criteria must map option IDs to descriptions")
        if kind == "score" and (not isinstance(criteria, list) or not criteria):
            raise RequestError(f"{key}: score criteria must be a nonempty list")
        if kind == "noul" and criteria is not None and (
            not isinstance(criteria, dict) or set(criteria) - {"true", "false"}
        ):
            raise RequestError(f"{key}: noul criteria may describe only true and false")
    return _prepare_media(request)


class ClefEngine:
    """One request at a time: TT GDN, KV and vision state are mutable per request."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        head: JointSchemaHead,
        lm_head: torch.Tensor,
        variant: str,
        max_length: int = DEFAULT_MAX_LENGTH,
    ):
        if not 1 <= max_length <= MAX_TOKENS:
            raise ValueError(f"max_length must be between 1 and {MAX_TOKENS}")
        self.model = model
        self.processor = processor
        self.head = head
        self.lm_head = lm_head
        self.variant = variant
        self.max_length = max_length
        self._lock = threading.Lock()

    @torch.inference_mode()
    def answer(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            record = _validate_request(request, self.variant)
            try:
                encoded = encode_record(
                    self.processor.tokenizer, record, max_length=self.max_length, processor=self.processor
                )
            except (TypeError, ValueError, KeyError) as error:
                raise RequestError(f"record encoding failed: {error}") from error
            tokens = torch.tensor([encoded.input_ids], dtype=torch.long)
            if (record.get("images") or record.get("videos")) and not encoded.media:
                raise RequestError("processor omitted requested visual inputs")
            vision = None
            try:
                if encoded.media:
                    media = encoded.media
                    if record.get("images"):
                        if "pixel_values" not in media or "image_grid_thw" not in media:
                            raise RequestError("processor did not return image patches and grid")
                        if self.model.vision_model is None:
                            raise RuntimeError("TT vision tower was not initialized")
                        vision = self.model.get_image_features(
                            media["pixel_values"], media["image_grid_thw"]
                        )
                    elif record.get("videos"):
                        if "pixel_values_videos" not in media or "video_grid_thw" not in media:
                            raise RequestError("processor did not return video patches and grid")
                        if self.model.vision_model is None:
                            raise RuntimeError("TT vision tower was not initialized")
                        vision = self.model.get_video_features(
                            media["pixel_values_videos"], media["video_grid_thw"]
                        )
                hidden = self.model.prefill_hidden_states(
                    tokens, valid_len=tokens.shape[1], vision_tokens=vision
                )
                if tuple(hidden.shape) != (1, tokens.shape[1], VARIANTS[self.variant].hidden):
                    raise RuntimeError("TT backbone returned incompatible hidden states")
                attention_mask = torch.ones_like(tokens)
                logits = self.head(
                    hidden.to(dtype=self.head.hidden_norm.weight.dtype),
                    tokens,
                    attention_mask,
                    [encoded],
                    self.lm_head,
                )[0]
                if len(logits) != len(encoded.questions):
                    raise RuntimeError("joint head returned an incomplete decision")
                answers = {}
                for question, scores in zip(encoded.questions, logits):
                    if scores.numel() != len(question.option_ids) or not torch.isfinite(scores).all():
                        raise RuntimeError("joint head returned invalid option logits")
                    probabilities = dict(zip(question.option_ids, scores.float().softmax(-1).tolist()))
                    answers[question.question_id] = systemone_answer(
                        request["questions"][question.question_id], probabilities
                    )
                return {
                    "model": request["model"],
                    "answers": answers,
                    "usage": {"input_tokens": tokens.shape[1], "output_tokens": 0},
                }
            finally:
                try:
                    if vision is not None:
                        import ttnn

                        ttnn.deallocate(vision)
                finally:
                    self.model._req_image_grid_thw = None
                    self.model._req_video_grid_thw = None


def make_handler(engine: ClefEngine) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            if self.path != "/v1/systemone":
                self._send(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})
                return
            try:
                size = int(self.headers.get("Content-Length", ""))
            except (ValueError, OverflowError):
                self._send(HTTPStatus.BAD_REQUEST, {"error": "invalid Content-Length"})
                return
            try:
                if size < 1:
                    raise RequestError("Content-Length must be positive")
                if size > MAX_BODY_BYTES:
                    self._send(
                        HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                        {"error": f"request exceeds {MAX_BODY_BYTES} bytes"},
                    )
                    return
                request = json.loads(self.rfile.read(size))
                self._send(HTTPStatus.OK, engine.answer(request))
            except (RequestError, json.JSONDecodeError, UnicodeDecodeError) as error:
                self._send(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            except Exception:
                logging.exception("SystemOne inference failed")
                self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "inference failed"})

    return Handler


def _cache_directory(model_path: Path, cache_path: Path, revision: str) -> Path:
    cache_path = cache_path.expanduser().resolve(strict=True)
    if (
        not cache_path.is_dir()
        or cache_path == model_path
        or model_path in cache_path.parents
        or cache_path.name != revision
    ):
        raise ValueError(
            "--cache-path must be an existing writable, checkpoint-revision-named directory outside --model-path"
        )
    with tempfile.TemporaryFile(dir=cache_path):
        pass
    return cache_path


def serve(
    model_path: Path,
    variant: str,
    cache_path: Path,
    host: str = "127.0.0.1",
    port: int = 8000,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> None:
    """Validate checkpoint and devices; own mesh/fabric exclusively until HTTP shutdown."""
    config, head_config, shard = validate_checkpoint(model_path, variant)
    model_path = model_path.expanduser().resolve(strict=True)
    cache_path = _cache_directory(model_path, cache_path, VARIANTS[variant].revision)
    if not 1 <= max_length <= MAX_TOKENS:
        raise ValueError(f"--max-length must be between 1 and {MAX_TOKENS}")
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    head, lm_head = _load_head(model_path, head_config, shard)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)

    import ttnn
    from models.demos.blackhole.qwen36.tt.model import Qwen36Model

    spec = VARIANTS[variant]
    physical_ids = tuple(ttnn.get_pcie_device_ids())
    admitted_ids = (spec.devices,) if variant == "clef" else ((0,), (0, 1))
    if physical_ids not in admitted_ids:
        raise RuntimeError(f"{variant} requires physical PCIe IDs {admitted_ids}; found {physical_ids}")
    os.environ["HF_MODEL"] = str(model_path)
    os.environ["TT_CACHE_PATH"] = str(cache_path)
    mesh = model = server = None
    fabric_enabled = False
    try:
        if len(spec.devices) > 1:
            fabric_enabled = True
            ttnn.set_fabric_config(
                ttnn.FabricConfig.FABRIC_1D,
                ttnn.FabricReliabilityMode.STRICT_INIT,
                None,
                ttnn.FabricTensixConfig.DISABLED,
                ttnn.FabricUDMMode.DISABLED,
                ttnn.FabricManagerMode.DEFAULT,
            )
        mesh = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(1, len(spec.devices)),
            physical_device_ids=list(spec.devices),
            l1_small_size=24_576,
            num_command_queues=2,
        )
        if mesh.get_num_devices() != len(spec.devices):
            raise RuntimeError("mesh did not open the admitted physical devices")
        mesh.enable_program_cache()
        model = Qwen36Model.from_pretrained(mesh, max_batch_size=1, max_seq_len=max_length)
        if len(model.layers) != spec.layers or model.args.dim != spec.hidden:
            raise RuntimeError("loaded TT backbone differs from checkpoint variant")
        # The TT vision model expects an HF reference visual for patch/position steps.
        # Load only its weights, never the complete CPU language backbone.
        visual = _reference_visual(model_path, config)
        try:
            model.init_vision_model(reference_visual=visual)
        finally:
            del visual
        engine = ClefEngine(model, processor, head, lm_head, variant, max_length=max_length)
        server = ThreadingHTTPServer((host, port), make_handler(engine))
        server.daemon_threads = False
        print(json.dumps({"stage": "ready", "model": variant, "host": host, "port": port}), flush=True)
        server.serve_forever()
    finally:
        try:
            if server is not None:
                server.server_close()
        finally:
            try:
                if model is not None:
                    model.free_kv_caches()
                    del model
                    gc.collect()
            finally:
                try:
                    if mesh is not None:
                        for submesh in mesh.get_submeshes():
                            ttnn.close_mesh_device(submesh)
                        ttnn.close_mesh_device(mesh)
                finally:
                    if fabric_enabled:
                        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a pinned Cloudflare Clef SystemOne checkpoint on TT")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path(os.environ["TT_CACHE_PATH"]) if os.environ.get("TT_CACHE_PATH") else None,
    )
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.cache_path is None:
        parser.error("--cache-path (or TT_CACHE_PATH) is required for writable TT tensor cache")
    handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
    try:
        for signum in handlers:
            signal.signal(signum, _stop_service)
        try:
            serve(args.model_path, args.variant, args.cache_path, args.host, args.port, args.max_length)
        except ServiceStop:
            pass
        except (TypeError, ValueError, OSError, RuntimeError) as error:
            parser.exit(1, f"clef: {error}\n")
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    main()

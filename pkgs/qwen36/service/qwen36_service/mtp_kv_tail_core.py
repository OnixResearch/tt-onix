"""Plan rejected target-cache writes without device effects."""

from dataclasses import dataclass


@dataclass(frozen=True)
class KvTailRow:
    position: int
    page: int
    offset: int


@dataclass(frozen=True)
class KvTailPlan:
    next_position: int
    rows: tuple[KvTailRow, ...]


def plan_kv_tail(
    *,
    position: int,
    expected_position: int,
    draft_count: int,
    accepted: int,
    page_size: int,
    page_count: int,
    pages: tuple[int, ...],
) -> KvTailPlan:
    """Retain the seed and accepted inputs. Restore the remaining draft inputs."""
    integers = (
        position,
        expected_position,
        draft_count,
        accepted,
        page_size,
        page_count,
    )
    if any(type(value) is not int for value in integers):
        raise ValueError("KV tail dimensions and positions must be integers")
    if position < 0 or position != expected_position:
        raise ValueError("KV tail committed boundary changed")
    if draft_count < 1 or not 0 <= accepted <= draft_count:
        raise ValueError("KV tail acceptance count is invalid")
    if type(pages) is not tuple:
        raise ValueError("KV tail page map must be an immutable tuple")
    if page_size < 1 or page_count < 1 or not pages:
        raise ValueError("KV tail page geometry is empty")
    if any(type(page) is not int or not 0 <= page < page_count for page in pages):
        raise ValueError("KV tail page index is invalid")
    if len(set(pages)) != len(pages):
        raise ValueError("KV tail page aliases are not supported")
    stop = position + draft_count + 1
    if stop > len(pages) * page_size:
        raise ValueError("KV tail verification exceeds cache capacity")
    next_position = position + accepted + 1
    return KvTailPlan(
        next_position=next_position,
        rows=tuple(
            KvTailRow(index, pages[index // page_size], index % page_size)
            for index in range(next_position, stop)
        ),
    )

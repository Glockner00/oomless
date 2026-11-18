"""The "what fits" calculation: the largest batch size that fits on this GPU."""

from dataclasses import dataclass
from typing import Any

# Measured: the estimate was 3% low on GPT-2 with 1024-token texts, so keep 10% spare.
HEADROOM = 0.9


@dataclass
class Fit:
    """What the batch that crashed needs, and a batch size that fits."""

    batch: int
    peak: int
    capacity: int
    suggested: int | None
    suggested_peak: int


def what_fits(info: dict[str, Any], oom: dict[str, Any], in_use: int, held: int) -> Fit | None:
    """Estimate a batch size that fits, if the crash came in one of the first training steps."""
    step = info.get("step")
    if not step:
        return None
    capacity = min(info["limit"], held + oom["device_free"])
    batch = step["batch"]
    if step["end"] is None:
        # The forward pass did not finish: scale up by how many of the model's layers ran.
        per_sample = (
            (in_use + oom["size"] - step["start"]) * step["leaves"] / step["called"] / batch
        )
    else:
        per_sample = (step["end"] - step["start"]) / batch
    weights = step["weights"]
    state, temporary = _optimizer(step["optimizers"])
    fixed = step["start"] + weights - step["gradients"]
    if not step["steps_done"]:
        fixed += state * weights

    def peak(b: int) -> int:
        return int(fixed + max(per_sample * b, temporary * weights))

    if peak(batch) <= HEADROOM * capacity:
        return None
    suggested = next(
        (b for b in range(batch - 1, 0, -1) if batch % b == 0 and peak(b) <= HEADROOM * capacity),
        None,
    )
    return Fit(batch, peak(batch), capacity, suggested, peak(suggested or 1))


def _optimizer(optimizers: list[dict[str, Any]]) -> tuple[int, int]:
    """Copies of the weights the optimizer keeps, and the extra copies it makes while it steps."""
    # shortcut: only SGD is told apart, others count as Adam (2 copies); add more as the zoo grows
    state = max(
        ((1 if o["momentum"] else 0) if o["name"] == "SGD" else 2 for o in optimizers), default=2
    )
    temporary = 0 if optimizers and all(o["fused"] for o in optimizers) else 1
    return state, temporary

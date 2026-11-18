from typing import Any

import pytest

from oomless.estimate import Fit, _optimizer, what_fits

MiB = 2**20
SGD = {"name": "SGD", "momentum": 0.9, "fused": False}
OOM = {"size": 100 * MiB, "device_free": 10_000 * MiB}


def info(**step: Any) -> dict[str, Any]:
    measured = {
        "start": 100 * MiB,
        "gradients": 0,
        "end": None,
        "batch": 64,
        "leaves": 10,
        "called": 5,
        "steps_done": 0,
        "weights": 50 * MiB,
        "optimizers": [SGD],
    }
    return {"limit": 1000 * MiB, "step": {**measured, **step}}


def test_crash_in_the_forward_pass() -> None:
    # Half the layers ran: 700 MiB so far, so 1400 MiB for the whole pass.
    # Fixed: 100 at the start, 50 for gradients, 50 for momentum.
    fit = what_fits(info(), OOM, in_use=700 * MiB, held=900 * MiB)
    assert fit == Fit(64, 1600 * MiB, 1000 * MiB, 32, 900 * MiB)


def test_crash_after_the_forward_pass() -> None:
    fit = what_fits(info(end=900 * MiB), OOM, in_use=900 * MiB, held=900 * MiB)
    assert fit == Fit(64, 1000 * MiB, 1000 * MiB, 32, 600 * MiB)


def test_second_step_has_the_optimizer_memory_already() -> None:
    fit = what_fits(info(steps_done=1), OOM, in_use=700 * MiB, held=900 * MiB)
    assert fit == Fit(64, 1550 * MiB, 1000 * MiB, 32, 850 * MiB)


def test_nothing_fits() -> None:
    fit = what_fits(info(weights=400 * MiB), OOM, in_use=700 * MiB, held=900 * MiB)
    assert fit
    assert fit.suggested is None


def test_capacity_is_what_the_gpu_had_left() -> None:
    oom = {**OOM, "device_free": 0}
    fit = what_fits(info(), oom, in_use=700 * MiB, held=900 * MiB)
    assert fit
    assert (fit.capacity, fit.suggested) == (900 * MiB, 16)


@pytest.mark.parametrize(
    "measured",
    [{}, {"limit": 1, "step": None}, info(batch=2, called=10, leaves=10)],
    ids=["no-measurements", "later-step", "batch-would-fit"],
)
def test_no_estimate(measured: dict[str, Any]) -> None:
    assert what_fits(measured, OOM, in_use=100 * MiB, held=900 * MiB) is None


@pytest.mark.parametrize(
    ("optimizers", "factors"),
    [
        ([], (2, 1)),
        ([{**SGD, "momentum": 0}], (0, 1)),
        ([SGD], (1, 1)),
        ([{"name": "AdamW", "momentum": 0, "fused": True}], (2, 0)),
        ([{"name": "Adam", "momentum": 0, "fused": False}], (2, 1)),
    ],
)
def test_optimizer_memory(optimizers: list[dict[str, Any]], factors: tuple[int, int]) -> None:
    assert _optimizer(optimizers) == factors

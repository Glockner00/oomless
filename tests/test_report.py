from pathlib import Path
from typing import Any

import pytest
from syrupy.assertion import SnapshotAssertion

from oomless import report
from oomless.snapshot import load

ROOT = Path(__file__).parent.parent
CASES = sorted((ROOT / "tests" / "fixtures" / "snapshots").glob("*/*.pickle.gz"))
GiB = 2**30
MiB = 2**20
FRAME = {"filename": "train.py", "line": 1, "name": "<module>"}


def fake(
    in_use: int, free: list[int], request: int, device_free: int = 0, expandable: bool = False
) -> dict[str, Any]:
    blocks = [{"size": in_use, "state": "active_allocated", "frames": [FRAME]}]
    blocks += [{"size": n, "state": "inactive", "frames": []} for n in free]
    oom = {"action": "oom", "size": request, "device_free": device_free, "frames": [FRAME]}
    return {
        "segments": [{"device": 0, "is_expandable": expandable, "blocks": blocks}],
        "device_traces": [[oom]],
    }


@pytest.mark.parametrize("case", CASES, ids=lambda p: f"{p.parent.name}/{p.name.split('.')[0]}")
def test_real_crashes(
    case: Path, monkeypatch: pytest.MonkeyPatch, snapshot: SnapshotAssertion
) -> None:
    monkeypatch.chdir(ROOT)
    assert report.explain(load(case)) == snapshot


def test_fragmented() -> None:
    text = report.explain(fake(GiB, [300 * MiB] * 4, GiB))
    assert "The memory is fragmented: PyTorch had 1.17 GiB free" in text
    assert "but in pieces of at most 300 MiB." in text
    assert "PyTorch's" not in text


@pytest.mark.parametrize(
    "snap",
    [
        fake(GiB, [300 * MiB] * 4, GiB, expandable=True),
        fake(10 * GiB, [300 * MiB] * 4, GiB),
        fake(GiB, [2 * GiB], GiB),
    ],
    ids=["already-expandable", "small-share", "one-piece-big-enough"],
)
def test_not_fragmented(snap: dict[str, Any]) -> None:
    assert "fragmented" not in report.explain(snap)


def test_gpu_still_had_room() -> None:
    assert "The GPU itself still had 8.00 GiB free" in report.explain(fake(GiB, [], GiB, 8 * GiB))


def test_no_oom_in_the_recording() -> None:
    text = report.explain({"segments": [], "device_traces": [[]]})
    assert text.startswith("oomless: this recording has no out-of-memory error\n")


def test_shows_your_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.py").write_text("import torch\nx = torch.ones(9, device='cuda')\n")
    frame = {"filename": str(tmp_path / "train.py"), "line": 2, "name": "<module>"}
    snap = fake(GiB, [], 2 * GiB)
    snap["device_traces"][0][0]["frames"] = [frame]
    assert "  at train.py:2   x = torch.ones(9, device='cuda')\n" in report.explain(snap)


def test_what_fits() -> None:
    snap = fake(700 * MiB, [200 * MiB], 100 * MiB, device_free=10 * GiB)
    step: dict[str, Any] = {
        "start": 100 * MiB,
        "gradients": 0,
        "end": None,
        "batch": 64,
        "leaves": 10,
        "called": 5,
        "steps_done": 0,
        "weights": 50 * MiB,
        "optimizers": [],
    }
    snap["oomless"] = {"limit": 1000 * MiB, "step": step}
    assert (
        "\nWhat fits (an estimate)\n"
        "  Batch size 64 needs about 1.61 GiB. PyTorch can use 1000 MiB here.\n"
        "  Use batch size 16: it needs about 600 MiB.\n"
        "  To train as if the batch were still 64, add up the gradients of\n"
        "  4 batches before each optimizer step (gradient accumulation).\n"
    ) in report.explain(snap)

    step["weights"] = 400 * MiB
    assert (
        "  Even batch size 1 needs about 1.66 GiB: the model is too big for this GPU.\n"
    ) in report.explain(snap)

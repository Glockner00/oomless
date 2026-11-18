import json
from pathlib import Path

import pytest

from oomless import report, snapshot

CASES = sorted((Path(__file__).parent / "fixtures" / "snapshots").glob("*/*.json"))


@pytest.mark.parametrize("case", CASES, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_recording_holds_the_crash(case: Path) -> None:
    snap = snapshot.load(case.with_suffix(".pickle.gz"))
    events = [e for trace in snap["device_traces"] for e in trace]
    active = [b for s in snap["segments"] for b in s["blocks"] if b["state"] == "active_allocated"]
    assert any(e["action"] == "oom" for e in events)
    assert any(f["filename"].startswith("examples/") for b in active for f in b["frames"])


@pytest.mark.parametrize("case", CASES, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_batch_size_only_for_too_big(case: Path) -> None:
    text = report.explain(snapshot.load(case.with_suffix(".pickle.gz")))
    too_big = json.loads(case.read_text())["expected"] == "too-big"
    assert ("\n  Use batch size " in text) == too_big

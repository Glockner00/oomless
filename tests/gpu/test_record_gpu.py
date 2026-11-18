import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from oomless import snapshot

torch = pytest.importorskip("torch")
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
]

OOMLESS = shutil.which("oomless", path=str(Path(sys.executable).parent))
SCRIPT = """\
import torch
torch.cuda.set_per_process_memory_fraction(0.35)
kept = torch.empty(2**30, dtype=torch.uint8, device="cuda")
try:
    torch.empty(2**40, dtype=torch.uint8, device="cuda")
except torch.OutOfMemoryError:
    if {reraise}:
        raise
"""


@pytest.mark.parametrize(("reraise", "code"), [(True, 1), (False, 0)], ids=["crash", "caught"])
def test_saves_the_recording(tmp_path: Path, reraise: bool, code: int) -> None:
    assert OOMLESS
    script = tmp_path / "train.py"
    script.write_text(SCRIPT.format(reraise=reraise))
    done = subprocess.run([OOMLESS, "train.py"], cwd=tmp_path, check=False)  # noqa: S603
    assert done.returncode == code

    snap = snapshot.load(tmp_path / "oomless-report" / "snapshot.pickle")
    oom = [e for trace in snap["device_traces"] for e in trace if e["action"] == "oom"]
    assert [(e["size"], e["frames"][0]["line"]) for e in oom] == [(2**40, 5)]
    kept = [
        b["frames"][0]
        for s in snap["segments"]
        for b in s["blocks"]
        if b["state"] == "active_allocated" and b["size"] == 2**30
    ]
    assert kept == [{"filename": str(script), "line": 3, "name": "<module>"}]

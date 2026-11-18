import os
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
FRAGMENTED = """\
import torch
torch.cuda.set_per_process_memory_fraction(0.35)
MiB = 2**20
big = torch.empty(4096 * MiB, dtype=torch.uint8, device="cuda")
del big
pieces = [torch.empty(200 * MiB, dtype=torch.uint8, device="cuda") for _ in range(20)]
del pieces[::2]
torch.empty(2048 * MiB, dtype=torch.uint8, device="cuda")
"""


def run(tmp_path: Path, code: str, env: dict[str, str] | None = None) -> tuple[int, str]:
    assert OOMLESS
    (tmp_path / "train.py").write_text(code)
    done = subprocess.run(  # noqa: S603
        [OOMLESS, "train.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, **(env or {})},
    )
    return done.returncode, done.stderr


@pytest.mark.parametrize(("reraise", "code"), [(True, 1), (False, 0)], ids=["crash", "caught"])
def test_saves_the_recording(tmp_path: Path, reraise: bool, code: int) -> None:
    assert run(tmp_path, SCRIPT.format(reraise=reraise))[0] == code

    snap = snapshot.load(tmp_path / "oomless-report" / "snapshot.pickle")
    oom = [e for trace in snap["device_traces"] for e in trace if e["action"] == "oom"]
    assert [(e["size"], e["frames"][0]["line"]) for e in oom] == [(2**40, 5)]
    kept = [
        b["frames"][0]
        for s in snap["segments"]
        for b in s["blocks"]
        if b["state"] == "active_allocated" and b["size"] == 2**30
    ]
    assert kept == [{"filename": str(tmp_path / "train.py"), "line": 3, "name": "<module>"}]


def test_explains_the_crash(tmp_path: Path) -> None:
    code, err = run(tmp_path, SCRIPT.format(reraise=True))
    assert code == 1
    assert "\noomless: the GPU ran out of memory\n  at train.py:5   torch.empty(2**40," in err
    assert (
        '\n   1.00 GiB  train.py:3   kept = torch.empty(2**30, dtype=torch.uint8, device="cuda")\n'
        in err
    )
    assert "fragmented" not in err


def test_fragmented(tmp_path: Path) -> None:
    code, err = run(tmp_path, FRAGMENTED)
    assert code == 1
    assert "\nThe memory is fragmented: " in err
    assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in err

    code, err = run(tmp_path, FRAGMENTED, {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    assert (code, err) == (0, "")

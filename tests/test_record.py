import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from oomless import record, snapshot

torch = pytest.importorskip("torch")

OOMLESS = shutil.which("oomless", path=str(Path(sys.executable).parent))
SNAP: dict[str, Any] = {"segments": [], "device_traces": [[{"action": "oom", "size": 1}]]}


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> record.Recorder:
    monkeypatch.setattr(torch.cuda.memory, "_snapshot", lambda device=None: SNAP)
    return record.Recorder()


def test_starts_when_your_script_imports_torch(tmp_path: Path) -> None:
    assert OOMLESS
    (tmp_path / "train.py").write_text(
        "import importlib.util, sys\n"
        "print('torch' in sys.modules, importlib.util.find_spec('torch') is not None)\n"
        "import torch\n"
        "print([c.__qualname__ for c, _ in torch.cuda._queued_calls][-1])\n"
    )
    done = subprocess.run(  # noqa: S603
        [OOMLESS, "train.py"], cwd=tmp_path, capture_output=True, text=True, check=True
    )
    assert done.stdout == "False True\nRecorder._start\n"


def test_waits_until_cuda_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    queued: list[Any] = []
    history: list[dict[str, Any]] = []
    observers: list[Any] = []
    monkeypatch.setattr(torch.cuda, "_lazy_call", queued.append)
    monkeypatch.setattr(
        torch.cuda.memory, "_record_memory_history", lambda **kw: history.append(kw)
    )
    monkeypatch.setattr(
        torch._C, "_cuda_attach_out_of_memory_observer", observers.append, raising=False
    )
    rec = record.Recorder()

    rec._torch_imported()
    assert len(queued) == 1
    assert not history

    queued[0]()
    assert history == [{"max_entries": 100_000, "stacks": "python"}]
    assert observers == [rec._on_oom]


def test_saves_the_recording_from_the_oom_callback(
    recorder: record.Recorder, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    recorder._on_oom(0, 2, 3, 1)
    recorder.on_error(torch.OutOfMemoryError("CUDA out of memory"))
    recorder.save(tmp_path / "report")
    assert snapshot.load(tmp_path / "report" / "snapshot.pickle") == SNAP
    assert "Recording saved in" in capsys.readouterr().err


def test_takes_the_recording_if_the_callback_never_ran(
    recorder: record.Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    recorder.on_error(torch.OutOfMemoryError("CUDA out of memory"))
    assert recorder.snapshot == SNAP


@pytest.mark.parametrize(
    ("error", "cuda_started"),
    [(ValueError("boom"), True), (torch.OutOfMemoryError("CUDA out of memory"), False)],
)
def test_no_recording_without_a_gpu_oom(
    recorder: record.Recorder,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    error: Exception,
    cuda_started: bool,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: cuda_started)
    recorder.on_error(error)
    recorder.save(tmp_path / "report")
    assert not (tmp_path / "report").exists()
    assert capsys.readouterr().err == ""


def test_never_breaks_your_script(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def changed(**kwargs: Any) -> None:
        raise RuntimeError("changed in this PyTorch version")

    monkeypatch.setattr(torch.cuda.memory, "_record_memory_history", changed)
    record.Recorder()._start()
    assert capsys.readouterr().err == (
        "oomless: can't record GPU memory: RuntimeError('changed in this PyTorch version')\n"
    )

import importlib.util
from pathlib import Path

import pytest

from oomless import __version__, cli, record, report
from oomless.cli import main
from oomless.snapshot import load

RECORDING = (
    Path(__file__).parent / "fixtures/snapshots/amd-radeon-rx-9060-xt/cifar_too_big.pickle.gz"
)


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out == f"oomless {__version__}\n"


def test_no_arguments_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    main([])
    assert capsys.readouterr().out.startswith("usage: oomless")


def test_explain(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["explain", str(RECORDING)]) == 0
    assert capsys.readouterr().out.startswith("oomless: the GPU ran out of memory\n")


def test_explain_a_file_that_is_not_a_recording(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "x.pickle"
    path.write_bytes(b"not a pickle")
    assert main(["explain", str(path)]) == 1
    assert capsys.readouterr().err.startswith(f"oomless: can't read {path}: ")


def test_explain_needs_one_file() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["explain"])
    assert exc.value.code == 2


def test_doctor(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"oomless {__version__}\nPython ")
    assert "\nPyTorch" in out


def test_doctor_without_torch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main(["doctor"])
    assert capsys.readouterr().out.endswith(
        "\nPyTorch: not installed\nPYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n"
    )


def test_report_after_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    recorder = record.Recorder()
    recorder.snapshot = load(RECORDING)
    cli._report(recorder, recorder.snapshot)
    err = capsys.readouterr().err
    assert err.startswith("\noomless: the GPU ran out of memory\n")
    assert f"\nRecording saved in {Path('oomless-report', 'snapshot.pickle')}\n" in err
    assert load(tmp_path / "oomless-report" / "snapshot.pickle") == recorder.snapshot


def test_report_problems_keep_the_recording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken(snap: object) -> str:
        raise KeyError("segments")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(report, "explain", broken)
    recorder = record.Recorder()
    recorder.snapshot = load(RECORDING)
    cli._report(recorder, recorder.snapshot)
    err = capsys.readouterr().err
    assert err.startswith("oomless: can't explain the crash: KeyError('segments')\n")
    assert (tmp_path / "oomless-report" / "snapshot.pickle").exists()

import gzip
import pickle
from pathlib import Path

import pytest

from oomless import snapshot


def test_loads_plain_and_gzipped(tmp_path: Path) -> None:
    data = {"segments": [], "device_traces": [[{"action": "oom", "pool_id": (0, 0)}]]}
    (tmp_path / "a.pickle").write_bytes(pickle.dumps(data))
    with gzip.open(tmp_path / "a.pickle.gz", "wb") as f:
        pickle.dump(data, f)
    assert snapshot.load(tmp_path / "a.pickle") == data
    assert snapshot.load(tmp_path / "a.pickle.gz") == data


class Harmful:
    def __reduce__(self) -> tuple[object, tuple[str]]:
        return print, ("this ran",)


def test_refuses_a_file_that_runs_code(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "harmful.pickle"
    path.write_bytes(pickle.dumps({"segments": [Harmful()]}))
    with pytest.raises(pickle.UnpicklingError, match=r"refused to load builtins\.print"):
        snapshot.load(path)
    assert capsys.readouterr().out == ""


def test_refuses_what_is_not_a_recording(tmp_path: Path) -> None:
    path = tmp_path / "list.pickle"
    path.write_bytes(pickle.dumps([1, 2]))
    with pytest.raises(ValueError, match="not a PyTorch memory recording"):
        snapshot.load(path)


def frame(path: str) -> dict[str, object]:
    return {"filename": path, "line": 1, "name": "f"}


YOURS = frame("/home/you/train.py")
TORCH = frame("/venv/lib/python3.13/site-packages/torch/nn/functional.py")
TORCH_ENTRY = frame("/venv/lib/python3.13/site-packages/torch/_tensor.py")
HF = frame("/usr/local/lib/python3.11/dist-packages/transformers/trainer.py")
PYTHON = frame("/usr/lib/python3.13/runpy.py")


@pytest.mark.parametrize(
    ("frames", "expected"),
    [
        ([TORCH, HF, YOURS, PYTHON], (YOURS, HF)),
        ([TORCH, TORCH_ENTRY, YOURS], (YOURS, TORCH_ENTRY)),
        ([PYTHON, YOURS], (YOURS, None)),
        ([TORCH, PYTHON], (None, TORCH)),
        ([frame("<frozen runpy>"), frame(r"C:\Python313\Lib\threading.py")], (None, None)),
        (
            [frame(r"C:\venv\Lib\site-packages\torch\x.py"), YOURS],
            (YOURS, frame(r"C:\venv\Lib\site-packages\torch\x.py")),
        ),
        ([], (None, None)),
    ],
)
def test_where(frames: list[dict[str, object]], expected: tuple[object, object]) -> None:
    assert snapshot.where(frames) == expected

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

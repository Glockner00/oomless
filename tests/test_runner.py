import shutil
import subprocess
import sys
from pathlib import Path

import pytest

OOMLESS = shutil.which("oomless", path=str(Path(sys.executable).parent))

SHOW = """\
import sys
print(sys.argv, sys.path[0], __name__, __file__, "torch" in sys.modules)
sys.exit(int(sys.argv[1]))
"""


def run(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
    done = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)  # noqa: S603
    return done.returncode, done.stdout, done.stderr


@pytest.mark.parametrize(
    ("code", "args"),
    [
        (SHOW, ["0", "b"]),
        (SHOW, ["3", "--flag"]),
        ("print('out')\nraise ValueError('boom')\n", []),
        ("def broken(:\n", []),
        ("import helper\n", []),
    ],
)
@pytest.mark.parametrize(
    "target", [["pkg/main.py"], ["pkg/main"], ["-m", "pkg.main"]], ids=["py", "no-py", "module"]
)
def test_same_as_python(tmp_path: Path, code: str, args: list[str], target: list[str]) -> None:
    assert OOMLESS
    (tmp_path / "helper.py").write_text("print('top folder')\n")
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").touch()
    (pkg / "helper.py").write_text("print('pkg folder')\n")
    (pkg / "main.py").write_text(code)
    (pkg / "main").write_text(code)

    expected = run([sys.executable, *target, *args], tmp_path)
    assert run([OOMLESS, *target, *args], tmp_path) == expected


@pytest.mark.parametrize("target", [["missing.py"], ["-m", "missing"]])
def test_missing_file_or_module(tmp_path: Path, target: list[str]) -> None:
    assert OOMLESS
    code, out, err = run([sys.executable, *target], tmp_path)
    got_code, got_out, got_err = run([OOMLESS, *target], tmp_path)
    # Each starts the line with its own name.
    assert (got_code, got_out, got_err.partition(": ")[2]) == (code, out, err.partition(": ")[2])

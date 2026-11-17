import pytest

from oomless import __version__
from oomless.cli import main


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out == f"oomless {__version__}\n"


def test_no_arguments_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    main([])
    assert capsys.readouterr().out.startswith("usage: oomless")

"""Run a script or module the same way `python` does."""

import runpy
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

OnError = Callable[[BaseException], None] | None


def run_script(script: str, args: list[str], on_error: OnError = None) -> int:
    """Run `script` like `python script args` and return the exit code."""
    path = Path(script).absolute()
    try:
        source = path.read_bytes()
    except OSError as exc:
        sys.stderr.write(
            f"oomless: can't open file {str(path)!r}: [Errno {exc.errno}] {exc.strerror}\n"
        )
        return 2
    sys.argv = [script, *args]
    if not sys.flags.safe_path:
        sys.path[0] = str(path.resolve().parent)
    return _run(lambda: _exec_main(path, source), on_error)


def run_module(module: str, args: list[str], on_error: OnError = None) -> int:
    """Run `module` like `python -m module args` and return the exit code."""
    sys.argv = ["-m", *args]
    if not sys.flags.safe_path:
        sys.path[0] = str(Path.cwd())
    sys.modules["__main__"] = ModuleType("__main__")
    return _run(lambda: runpy._run_module_as_main(module), on_error)  # type: ignore[attr-defined]  # noqa: SLF001


def _exec_main(path: Path, source: bytes) -> None:
    code = compile(source, str(path), "exec", dont_inherit=True)
    main = ModuleType("__main__")
    main.__file__ = str(path)
    sys.modules["__main__"] = main
    exec(code, main.__dict__)  # noqa: S102


def _run(target: Callable[[], object], on_error: OnError) -> int:
    try:
        target()
    except Exception as exc:  # noqa: BLE001
        tb = exc.__traceback__
        while tb is not None and tb.tb_frame.f_code.co_filename == __file__:
            tb = tb.tb_next
        sys.excepthook(type(exc), exc.with_traceback(tb), tb)
        if on_error is not None:
            on_error(exc)
        return 1
    return 0

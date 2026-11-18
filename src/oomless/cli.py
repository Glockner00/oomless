"""The `oomless` command."""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from oomless import __version__, env, record, report, runner, snapshot

USAGE = """\
%(prog)s script.py [args ...]
       %(prog)s -m module [args ...]
       %(prog)s explain oomless-report/snapshot.pickle
       %(prog)s doctor"""


def main(argv: list[str] | None = None) -> int:
    """Run the `oomless` command and return its exit code."""
    args = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(
        prog="oomless",
        usage=USAGE,
        description="Tells you why your PyTorch training ran out of GPU memory, and how to fix it.",
    )
    if args[:1] == ["explain"]:
        if len(args) != 2:
            parser.error("explain takes one recording, e.g. oomless-report/snapshot.pickle")
        return _explain(Path(args[1]))
    if args[:1] == ["doctor"]:
        sys.stdout.write(env.doctor())
        return 0
    if args and not args[0].startswith("-"):
        return _record(runner.run_script, args[0], args[1:])
    if len(args) > 1 and args[0] == "-m":
        return _record(runner.run_module, args[1], args[2:])

    parser.add_argument("-m", metavar="module", help="run a module, like python -m")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.parse_args(args)
    parser.print_help()
    return 0


def _explain(path: Path) -> int:
    try:
        text = report.explain(snapshot.load(path))
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"oomless: can't read {path}: {exc}\n")
        return 1
    sys.stdout.write(text)
    return 0


def _record(
    run: Callable[[str, list[str], runner.OnError], int], target: str, args: list[str]
) -> int:
    recorder = record.start()
    try:
        return run(target, args, recorder.on_error)
    finally:
        if recorder.snapshot is not None:
            _report(recorder, recorder.snapshot)


def _report(recorder: record.Recorder, snap: dict[str, Any]) -> None:
    saved = recorder.save(Path("oomless-report"))
    try:
        sys.stderr.write("\n" + report.explain(snap))
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"oomless: can't explain the crash: {exc!r}\n")
    if saved:
        sys.stderr.write(
            f"\nRecording saved in {saved}\n"
            f"  (oomless explain {saved} shows this again;"
            " https://pytorch.org/memory_viz shows every allocation)\n"
        )

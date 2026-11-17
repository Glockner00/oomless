"""The `oomless` command."""

import argparse
import sys

from oomless import __version__, runner


def main(argv: list[str] | None = None) -> int:
    """Run the `oomless` command and return its exit code."""
    args = sys.argv[1:] if argv is None else argv
    if args and not args[0].startswith("-"):
        return runner.run_script(args[0], args[1:])
    if len(args) > 1 and args[0] == "-m":
        return runner.run_module(args[1], args[2:])

    parser = argparse.ArgumentParser(
        prog="oomless",
        usage="%(prog)s script.py [args ...]\n       %(prog)s -m module [args ...]",
        description="Tells you why your PyTorch training ran out of GPU memory, and how to fix it.",
    )
    parser.add_argument("-m", metavar="module", help="run a module, like python -m")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.parse_args(args)
    parser.print_help()
    return 0

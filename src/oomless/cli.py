"""The `oomless` command."""

import argparse

from oomless import __version__


def main(argv: list[str] | None = None) -> None:
    """Run the `oomless` command."""
    parser = argparse.ArgumentParser(
        prog="oomless",
        description="Tells you why your PyTorch training ran out of GPU memory, and how to fix it.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.parse_args(argv)
    parser.print_help()

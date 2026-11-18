"""Read memory recordings."""

import gzip
import pickle
import re
from pathlib import Path
from typing import Any, NoReturn

Frame = dict[str, Any]


class _PlainData(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> NoReturn:
        raise pickle.UnpicklingError(
            f"refused to load {module}.{name}: a memory recording holds only plain data"
        )


def load(path: Path) -> dict[str, Any]:
    """Load a recording (`.pickle` or `.pickle.gz`). Refuses files that would run code."""
    with gzip.open(path) if path.suffix == ".gz" else path.open("rb") as f:
        data = _PlainData(f).load()
    if not (isinstance(data, dict) and "segments" in data and "device_traces" in data):
        raise ValueError("not a PyTorch memory recording")
    return data


def where(frames: list[Frame]) -> tuple[Frame | None, Frame | None]:
    """Your code line, and the library line inside it that asked for the memory."""
    library = entry = None
    for frame in frames:  # innermost first
        kind = _kind(frame["filename"])
        if kind == "yours":
            return frame, library or entry
        if kind == "torch":
            entry = frame
        elif kind == "library" and library is None:
            library = frame
    return None, library or entry


def _kind(filename: str) -> str:
    path = filename.replace("\\", "/")
    if "-packages/torch/" in path:
        return "torch"
    if "-packages/" in path:
        return "library"
    if path.startswith("<") or re.search(r"/lib/python3\.\d+/|/Lib/[^/]+\.py$", path):
        return "python"
    return "yours"

"""Read memory recordings."""

import gzip
import pickle
from pathlib import Path
from typing import Any, NoReturn


class _PlainData(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> NoReturn:
        raise pickle.UnpicklingError(
            f"refused to load {module}.{name}: a memory recording holds only plain data"
        )


def load(path: Path) -> dict[str, Any]:
    """Load a recording (`.pickle` or `.pickle.gz`). Refuses files that would run code."""
    with gzip.open(path) if path.suffix == ".gz" else path.open("rb") as f:
        data: dict[str, Any] = _PlainData(f).load()
    return data

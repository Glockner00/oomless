"""Record GPU memory while your script runs, and keep the recording if it runs out.

Every private PyTorch call lives in this file.
"""

import functools
import importlib.util
import pickle
import sys
from collections.abc import Callable, Sequence
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import Any, ParamSpec, TypeVar

MAX_ENTRIES = 100_000

P = ParamSpec("P")
R = TypeVar("R")


def _never_raise(fn: Callable[P, R]) -> Callable[P, R | None]:
    @functools.wraps(fn)
    def run(*args: P.args, **kwargs: P.kwargs) -> R | None:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"oomless: can't record GPU memory: {exc!r}\n")
            return None

    return run


class Recorder:
    """Starts recording when your script first uses the GPU."""

    def __init__(self) -> None:
        self.snapshot: dict[str, Any] | None = None
        self._looking = False

    def find_spec(
        self, name: str, path: Sequence[str] | None, target: ModuleType | None = None
    ) -> ModuleSpec | None:
        """Wait for your script to import torch, so oomless never imports it first."""
        if name != "torch" or self._looking:
            return None
        self._looking = True
        try:
            spec = importlib.util.find_spec(name)
        finally:
            self._looking = False
        if spec is not None and spec.loader is not None:
            exec_module = spec.loader.exec_module

            # Other libraries may only check that torch is installed: wait for the real import.
            def run(module: ModuleType) -> None:
                if self in sys.meta_path:
                    sys.meta_path.remove(self)
                exec_module(module)
                self._torch_imported()

            spec.loader.exec_module = run  # type: ignore[method-assign]
        return spec

    @_never_raise
    def _torch_imported(self) -> None:
        import torch  # noqa: PLC0415

        torch.cuda._lazy_call(self._start)  # type: ignore[no-untyped-call]

    @_never_raise
    def _start(self) -> None:
        import torch  # noqa: PLC0415

        torch.cuda.memory._record_memory_history(max_entries=MAX_ENTRIES, stacks="python")
        attach = getattr(torch._C, "_cuda_attach_out_of_memory_observer", None)
        if attach is not None:
            attach(self._on_oom)

    @_never_raise
    def _on_oom(self, device: int, requested: int, limit: int, free: int) -> None:
        import torch  # noqa: PLC0415

        self.snapshot = torch.cuda.memory._snapshot()

    @_never_raise
    def on_error(self, exc: BaseException) -> None:
        """Take the recording now if the out-of-memory callback never ran."""
        torch = sys.modules.get("torch")
        if (
            self.snapshot is None
            and torch is not None
            and isinstance(exc, torch.OutOfMemoryError)
            and torch.cuda.is_initialized()
        ):
            self.snapshot = torch.cuda.memory._snapshot()

    @_never_raise
    def save(self, folder: Path) -> Path:
        """Save the recording in `folder`."""
        folder.mkdir(exist_ok=True)
        path = folder / "snapshot.pickle"
        with path.open("wb") as f:
            pickle.dump(self.snapshot, f)
        return path


def start() -> Recorder:
    """Start watching for `import torch`."""
    recorder = Recorder()
    sys.meta_path.insert(0, recorder)
    return recorder

"""Record GPU memory while your script runs, and keep the recording if it runs out.

Every private PyTorch call lives in this file.
"""

import functools
import gc
import importlib.util
import pickle
import sys
import weakref
from collections.abc import Callable, Iterable, Sequence
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


def _hook(fn: Callable[..., None]) -> Callable[..., None]:
    @functools.wraps(fn)
    def run(self: "Recorder", *args: Any) -> None:
        if self._done:
            return
        try:
            fn(self, *args)
        except Exception as exc:  # noqa: BLE001
            self._done = True
            sys.stderr.write(f"oomless: can't measure the training step: {exc!r}\n")

    return run


class Recorder:
    """Starts recording when your script first uses the GPU."""

    def __init__(self) -> None:
        self.snapshot: dict[str, Any] | None = None
        self._looking = False
        self._hooks: list[Any] = []
        self._done = False
        self._steps = 0
        self._depth = 0
        self._step: dict[str, Any] = {}
        self._called: set[int] = set()
        self._model: weakref.ref[Any] | None = None

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
        from torch.nn.modules.module import (  # noqa: PLC0415
            register_module_forward_hook,
            register_module_forward_pre_hook,
        )
        from torch.optim.optimizer import register_optimizer_step_post_hook  # noqa: PLC0415

        torch.cuda.memory._record_memory_history(max_entries=MAX_ENTRIES, stacks="python")
        attach = getattr(torch._C, "_cuda_attach_out_of_memory_observer", None)
        if attach is not None:
            attach(self._on_oom)
        self._hooks = [
            register_module_forward_pre_hook(self._before),
            register_module_forward_hook(self._after, always_call=True),
            register_optimizer_step_post_hook(self._stepped),
        ]

    def _before(self, module: Any, args: tuple[Any, ...]) -> None:
        if self._done:
            # Removed here, not in _stepped: PyTorch loops over the live optimizer hooks.
            for handle in self._hooks:
                handle.remove()
            self._hooks = []
            self._step = {}
            self._model = None
        else:
            self._measure(module, args)

    @_hook
    def _measure(self, module: Any, args: tuple[Any, ...]) -> None:
        import torch  # noqa: PLC0415

        # A loss function like nn.CrossEntropyLoss() has no weights: it is not the model.
        if self._depth == 0 and any(p.requires_grad for p in module.parameters()):
            self._step = {}
            if torch.is_grad_enabled():
                self._step = {
                    "start": torch.cuda.memory_allocated(),
                    "gradients": _bytes(p.grad for p in module.parameters() if p.grad is not None),
                    "end": None,
                    "batch": None,
                    "leaves": sum(1 for m in module.modules() if next(m.children(), None) is None),
                }
                self._called = set()
                self._model = weakref.ref(module)
        self._depth += 1
        if self._step and self._step["end"] is None and next(module.children(), None) is None:
            self._called.add(id(module))
            if self._step["batch"] is None:
                # shortcut: assumes the first layer's input is batch-first, recheck when the zoo has an RNN
                self._step["batch"] = next(
                    (a.shape[0] for a in args if isinstance(a, torch.Tensor) and a.ndim), None
                )

    @_hook
    def _after(self, module: Any, args: tuple[Any, ...], output: Any) -> None:
        import torch  # noqa: PLC0415

        self._depth -= 1
        # On an error inside the model, PyTorch calls this with output None.
        if self._depth == 0 and output is not None and self._step:
            self._step["end"] = torch.cuda.memory_allocated()

    @_hook
    def _stepped(self, optimizer: Any, args: Any, kwargs: Any) -> None:
        # Two steps are enough: by the second one the optimizer has made all its memory.
        self._steps += 1
        self._done = self._steps == 2

    @_never_raise
    def _on_oom(self, device: int, requested: int, limit: int, free: int) -> None:
        import torch  # noqa: PLC0415

        snap = torch.cuda.memory._snapshot()
        snap["oomless"] = {"limit": limit, "step": self._measured()}
        self.snapshot = snap

    def _measured(self) -> dict[str, Any] | None:
        """What oomless measured in the training step that crashed, if it was one of the first two."""
        import torch  # noqa: PLC0415

        model = self._model() if self._model else None
        if self._done or model is None or not self._step or not self._step["batch"]:
            return None
        trainable = [p for p in model.parameters() if p.requires_grad]
        found = (
            getattr(o, "optimizer", o)
            for o in gc.get_objects()
            if issubclass(type(o), torch.optim.Optimizer)
        )
        optimizers = list({id(o): o for o in found}.values())
        return {
            **self._step,
            "called": len(self._called),
            "steps_done": self._steps,
            "weights": _bytes(trainable),
            "optimizers": [
                {
                    "name": type(o).__name__,
                    "momentum": o.defaults.get("momentum", 0),
                    "fused": bool(o.defaults.get("fused")),
                }
                for o in optimizers
            ],
        }

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


def _bytes(tensors: Iterable[Any]) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


def start() -> Recorder:
    """Start watching for `import torch`."""
    recorder = Recorder()
    sys.meta_path.insert(0, recorder)
    return recorder

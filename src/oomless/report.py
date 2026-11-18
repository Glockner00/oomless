"""Turn a memory recording into a report you can read."""

import linecache
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from oomless import estimate
from oomless.snapshot import Frame, where

GiB = 1024**3
MiB = 1024**2
TOP = 3
# shortcut: 20% is a guess, tune it when the bug collection has a real fragmentation crash
FRAGMENTED_SHARE = 0.2


def explain(snap: dict[str, Any]) -> str:
    """Say where the GPU ran out of memory and what was using it."""
    ooms = [
        (device, e)
        for device, trace in enumerate(snap["device_traces"])
        for e in trace
        if e["action"] == "oom"
    ]
    device, oom = max(ooms, key=lambda d_e: d_e[1].get("time_us", 0)) if ooms else (None, None)
    segments = [s for s in snap["segments"] if device is None or s.get("device") == device]
    blocks = [b for s in segments for b in s["blocks"]]
    active = [b for b in blocks if b["state"] != "inactive"]
    in_use = sum(b["size"] for b in active)
    free = [b["size"] for b in blocks if b["state"] == "inactive"]
    held = in_use + sum(free)

    if oom is None:
        lines = ["oomless: this recording has no out-of-memory error"]
    else:
        requested = oom["size"]
        largest = max(free, default=0)
        lines = [
            "oomless: the GPU ran out of memory",
            f"  at {_place(oom['frames'])}",
            f"  It asked for {_size(requested)} more.",
            (
                f"  PyTorch held {_size(held)}: {_size(in_use)} in use, {_size(sum(free))} free"
                f" (largest free piece {_size(largest)})."
            ),
        ]
        if oom["device_free"] >= requested:
            lines += [
                f"  The GPU itself still had {_size(oom['device_free'])} free: the limit was PyTorch's",
                "  own, set with torch.cuda.set_per_process_memory_fraction.",
            ]
        expandable = snap.get("allocator_settings", {}).get("expandable_segments") or any(
            s.get("is_expandable") for s in segments
        )
        if (
            sum(free) >= requested > largest
            and sum(free) >= FRAGMENTED_SHARE * held
            and not expandable
        ):
            lines += [
                "",
                f"The memory is fragmented: PyTorch had {_size(sum(free))} free, enough for this,",
                f"but in pieces of at most {_size(largest)}.",
                "  Fix: set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True before you start Python.",
            ]
        if fit := estimate.what_fits(snap.get("oomless") or {}, oom, in_use, held):
            lines += ["", *_what_fits(fit)]

    lines += ["", "What uses the memory, by the line in your code"]
    by_line: Counter[str] = Counter()
    inside: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for block in active:
        yours, library = where(block["frames"])
        frame = yours or library
        line = _show(frame) if frame else "no Python code (PyTorch itself)"
        by_line[line] += block["size"]
        if yours and library:
            inside[line][_show(library)] += block["size"]
    for line, size in by_line.most_common(TOP):
        lines.append(f"  {_size(size):>9}  {line}")
        lines += [
            f"  {'':>9}    {_size(n):>9}  in {lib}" for lib, n in inside[line].most_common(TOP)
        ]
    if rest := in_use - sum(size for _, size in by_line.most_common(TOP)):
        lines.append(f"  {_size(rest):>9}  other lines")
    return "\n".join(lines) + "\n"


def _what_fits(fit: estimate.Fit) -> list[str]:
    need, can = _size(fit.peak), _size(fit.capacity)
    lines = [
        "What fits (an estimate)",
        f"  Batch size {fit.batch} needs about {need}. PyTorch can use {can} here.",
    ]
    if fit.suggested is None:
        need = _size(fit.suggested_peak)
        return [
            *lines,
            f"  Even batch size 1 needs about {need}: the model is too big for this GPU.",
        ]
    return [
        *lines,
        f"  Use batch size {fit.suggested}: it needs about {_size(fit.suggested_peak)}.",
        f"  To train as if the batch were still {fit.batch}, add up the gradients of",
        f"  {fit.batch // fit.suggested} batches before each optimizer step (gradient accumulation).",
    ]


def _place(frames: list[Frame]) -> str:
    yours, library = where(frames)
    if yours and library:
        return f"{_show(yours)}\n     inside {_show(library)}"
    frame = yours or library
    return _show(frame) if frame else "a place with no Python code (PyTorch itself)"


def _show(frame: Frame) -> str:
    filename = frame["filename"].replace("\\", "/")
    if "-packages/" in filename:
        return f"{filename.split('-packages/')[-1]}:{frame['line']} ({frame['name']})"
    code = linecache.getline(frame["filename"], frame["line"]).strip()
    path = Path(frame["filename"])
    if path.is_absolute() and path.is_relative_to(Path.cwd()):
        filename = path.relative_to(Path.cwd()).as_posix()
    return f"{filename}:{frame['line']}" + (f"   {code}" if code else "")


def _size(n: int) -> str:
    return f"{n / GiB:.2f} GiB" if n >= GiB else f"{n / MiB:.0f} MiB"

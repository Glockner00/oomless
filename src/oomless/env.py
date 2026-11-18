"""`oomless doctor`: what oomless sees on this computer."""

import importlib.util
import os
import platform

from oomless import __version__

GiB = 1024**3


def doctor() -> str:
    """Versions, graphics cards and free memory, for bug reports."""
    lines = [f"oomless {__version__}", f"Python {platform.python_version()} on {platform.system()}"]
    if importlib.util.find_spec("torch") is None:
        lines.append("PyTorch: not installed")
    else:
        import torch  # noqa: PLC0415

        build = (
            f"CUDA {torch.version.cuda}"
            if torch.version.cuda
            else f"ROCm {torch.version.hip}"
            if torch.version.hip
            else "CPU only"
        )
        lines.append(f"PyTorch {torch.__version__} ({build})")
        if not torch.cuda.is_available():
            lines.append("GPU: none found")
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            lines.append(
                f"GPU {i}: {torch.cuda.get_device_name(i)}, {total / GiB:.2f} GiB,"
                f" {free / GiB:.2f} GiB free"
            )
    lines += [
        f"{name}={os.environ[name]}"
        for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")
        if name in os.environ
    ]
    return "\n".join(lines) + "\n"

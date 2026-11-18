"""Run the practice trainings under oomless on a GPU and save their recordings for the tests.

Run from the repo root with the GPU environment: .venv-rocm/bin/python scripts/record_fixtures.py
"""

import functools
import gzip
import json
import pickle
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from oomless import snapshot

ROOT = Path(__file__).resolve().parent.parent
HOME = Path.home().resolve()
KEEP_TRACE = 5_000

CASES = {
    "cifar_too_big": (
        ["examples/train_cifar.py", "--batch-size", "4096", "--epochs", "1"],
        "too-big",
    ),
    "cifar_eval_no_grad": (
        ["examples/train_cifar.py", "--bug", "eval-no-grad", "--epochs", "1"],
        "eval-with-grad",
    ),
    "gpt2_too_big": (["examples/train_gpt2.py", "--batch-size", "32"], "too-big"),
    "gpt2_too_long": (["examples/train_gpt2.py", "--max-length", "1024"], "too-big"),
    "gpt2_keep_outputs": (
        ["examples/train_gpt2.py", "--bug", "keep-outputs"],
        "outputs-kept-on-gpu",
    ),
}


@functools.cache
def short(filename: str) -> str:
    path = Path(filename)
    if path.is_absolute():
        path = path.resolve()
    if path.is_relative_to(ROOT):
        return path.relative_to(ROOT).as_posix()
    if path.is_relative_to(HOME):
        return "~/" + path.relative_to(HOME).as_posix()
    return filename


def trim(snap: dict[str, Any]) -> dict[str, Any]:
    """Keep the fixture small and free of your home folder's path."""

    def frames(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{**f, "filename": short(f["filename"])} for f in items]

    for segment in snap["segments"]:
        if "frames" in segment:
            segment["frames"] = frames(segment["frames"])
        for block in segment["blocks"]:
            block["frames"] = (
                frames(block["frames"]) if block["state"] == "active_allocated" else []
            )
    snap["device_traces"] = [
        [{**e, "frames": frames(e["frames"])} for e in trace[-KEEP_TRACE:]]
        for trace in snap["device_traces"]
    ]
    return snap


def main() -> None:
    oomless = shutil.which("oomless", path=str(Path(sys.executable).parent))
    if oomless is None:
        sys.exit("install oomless into this environment first")
    gpu = torch.cuda.get_device_name(0)
    total = torch.cuda.mem_get_info()[1]
    slug = re.sub(r"[^a-z0-9]+", "-", gpu.lower()).strip("-")
    out = ROOT / "tests" / "fixtures" / "snapshots" / slug
    out.mkdir(parents=True, exist_ok=True)
    recording = ROOT / "oomless-report" / "snapshot.pickle"

    for name, (command, expected) in CASES.items():
        print(f"== {name}: oomless {' '.join(command)}", flush=True)
        recording.unlink(missing_ok=True)
        code = subprocess.run([oomless, *command], cwd=ROOT, check=False).returncode  # noqa: S603
        if not recording.exists():
            sys.exit(f"{name}: no recording saved (exit code {code})")
        data = pickle.dumps(trim(snapshot.load(recording)))
        if any(str(home).encode() in data for home in (Path.home(), HOME)):
            sys.exit(f"{name}: the recording still holds your home folder's path")
        with gzip.open(out / f"{name}.pickle.gz", "wb") as f:
            f.write(data)
        info = {
            "command": command,
            "expected": expected,
            "gpu": gpu,
            "total_bytes": total,
            "torch": torch.__version__,
        }
        (out / f"{name}.json").write_text(json.dumps(info, indent=2) + "\n")
        print(f"   saved {out.relative_to(ROOT)}/{name}.pickle.gz", flush=True)


if __name__ == "__main__":
    main()

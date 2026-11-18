# Changelog

Every change that matters to you is listed here, newest first.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Added

- `oomless script.py args` and `oomless -m module args` run your code exactly like
  `python` does: same output, same exit code.
- `oomless --version` prints the version.
- When your script runs out of GPU memory, oomless tells you where it happened, what used
  the memory (by the line in your code, and the line inside the library it called), and
  whether the memory was fragmented, with the fix.
- It also saves a recording of the GPU memory in `oomless-report/snapshot.pickle`. Open it at
  https://pytorch.org/memory_viz to see every allocation.
- `oomless explain oomless-report/snapshot.pickle` shows the report again, also for a recording
  someone sends you. It refuses files that would run code.
- `oomless doctor` prints the versions, your graphics cards and their free memory.

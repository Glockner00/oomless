# Changelog

Every change that matters to you is listed here, newest first.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Added

- `oomless script.py args` and `oomless -m module args` run your code exactly like
  `python` does: same output, same exit code.
- `oomless --version` prints the version.
- When your script runs out of GPU memory, oomless saves a recording of the GPU memory in
  `oomless-report/snapshot.pickle`. Open it at https://pytorch.org/memory_viz to see every
  allocation.

# oomless

Your PyTorch training crashed with "CUDA out of memory". oomless tells you why, and what to change.

## What it will do

Run your script with `oomless` instead of `python`:

```bash
oomless train.py --epochs 10
```

If your GPU runs out of memory, you see:

- what uses the memory: the model, the optimizer, or the batch it is working on
- which line in your code caused it
- the smallest change that makes it fit, for example "use batch size 8 instead of 16"

## Install


You need Python 3.11 or newer. oomless does not install PyTorch. It uses the one you already have.

## License

MIT

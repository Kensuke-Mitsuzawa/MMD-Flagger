# 🔧 Setup Instructions

## 🐍 Python Environment

This project uses Python 3.11.

## 📦 Dependency Installation

We use `uv` for dependency management. Install it and set up the environment:

```
uv sync
```

Poetry will automatically create and manage a virtual environment.

# Example

See `example.py` for the usage.

## Note

This package does not include codes to reproduce the reported scores in the paper.
This package covers the core algorithm of MMD-Flagger.

# Citation

```
@inproceedings{
mitsuzawa2026mmdflagger,
title={{MMD}-Flagger: Leveraging Maximum Mean Discrepancy to Detect Hallucinations},
author={Kensuke Mitsuzawa and Damien Garreau},
booktitle={Second Workshop on Safe AI},
year={2026},
url={https://openreview.net/forum?id=QfHXlS9vJN}
}
```

"""Backward-compatibility shim.

pre-training/tpu_smpd_train.py has been renamed to
pre-training/kaggle_tpu_smpd_train.py.
"""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "kaggle_tpu_smpd_train.py"
    runpy.run_path(str(target), run_name="__main__")

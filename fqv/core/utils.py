from __future__ import annotations

import os
import platform
import random
import sys

import numpy as np


def set_seed(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def environment_info() -> dict:
    """Сведения о программно-аппаратной среде для отчёта."""
    import torch, sklearn, pandas, scipy
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "cpu": platform.processor(),
            "cpu_count": os.cpu_count(), "torch": torch.__version__, "numpy": np.__version__,
            "pandas": pandas.__version__, "scikit-learn": sklearn.__version__, "scipy": scipy.__version__}
    for name in ("arch", "yfinance", "matplotlib"):
        try:
            info[name] = __import__(name).__version__
        except Exception:
            info[name] = None
    try:
        with open("/proc/cpuinfo") as f:
            info["cpu"] = next(l.split(":", 1)[1].strip() for l in f if l.startswith("model name"))
    except OSError:
        pass
    return info

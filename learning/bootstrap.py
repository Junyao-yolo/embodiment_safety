"""Load the project's pinned Diffusion Policy dependencies."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "third_party/diffusion_policy"


def configure():
    dependencies = ROOT / "tools/tool_hang_benchmark_v001"
    sys.path[:0] = [
        str(ROOT),
        str(dependencies / "official_deps"),
        str(dependencies / "deps"),
        str(VENDOR),
    ]
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, "4")

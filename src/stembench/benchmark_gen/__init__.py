"""Original bilingual (ru/en) procedural STEM benchmark generators.

Deterministic, seeded generation of >= 620 semantic question pairs (each pair
renders one parameter set through parallel hand-authored Russian and English
templates).  Answers are computed in code and independently re-verified by a
second code path (``verify.py``); ``qc.py`` enforces schema/pairing/distractor/
dedup/distribution gates and ``build.py`` writes the versioned dataset.

CLI:
    PYTHONPATH=src python -m stembench.benchmark_gen.build --out data/stembench_v1
"""

from __future__ import annotations

from ._core import DEFAULT_SEED, VERSION
from .build import assemble, build_dataset
from .qc import run_qc
from .verify import verify_pair

__all__ = [
    "DEFAULT_SEED",
    "VERSION",
    "assemble",
    "build_dataset",
    "run_qc",
    "verify_pair",
]

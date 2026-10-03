"""Reference predictors for the local scorer.

``empirical_mean`` loads the organizers' own ``empirical_mean/model.py`` from a
checkout of ``aims-foundations/paiec_baseline`` so the anchor score is the
unmodified baseline, not a re-implementation.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

DEFAULT_BASELINE_DIR = Path(__file__).resolve().parents[3] / "aims-foundations" / "paiec_baseline"


def baseline_dir() -> Path:
    return Path(os.environ.get("PAIEC_BASELINE_DIR", DEFAULT_BASELINE_DIR))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def constant(value: float = 0.5):
    def predict(input, labeled=None):
        return value
    return predict


def empirical_mean():
    return load_module(baseline_dir() / "empirical_mean" / "model.py", "paiec_empirical_mean").predict

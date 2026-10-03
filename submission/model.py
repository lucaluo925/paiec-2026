"""PAIEC submission entry point: ``predict(input, labeled) -> float``.

An online hierarchical IRT model (``pirt_online.py``) with parameters
calibrated offline on the public measurement-db (``params.json``). Each call
synchronises the posterior with the supplied labels, one label at a time, and
returns the posterior expected probability of success. No network access,
credentials or API calls are used.
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from pirt_online import shared_engine  # noqa: E402

ENGINE = shared_engine(_HERE / "params.json")


def predict(input, labeled=None):
    """Probability that ``input = [subject, item]`` is answered correctly."""
    return ENGINE.predict(input, labeled)

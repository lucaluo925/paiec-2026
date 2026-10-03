"""Streaming acquisition: Fisher-information-tilted selection sampling.

``acquisition_function(input, *, labeled=None, context=None) -> bool``. It
omits ``prediction`` so the evaluator makes no extra ``predict`` call; the
decision reads the same posterior engine as ``model.py`` (shared through
``pirt_online.shared_engine``). See ``labeling_core.py``.
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from labeling_core import make_acquisition  # noqa: E402
from pirt_online import shared_engine  # noqa: E402

acquisition_function = make_acquisition(shared_engine(_HERE / "params.json"), mode="fisher")

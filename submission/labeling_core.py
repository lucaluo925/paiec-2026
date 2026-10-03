"""Streaming label acquisition driven by the online IRT posterior.

The evaluator streams a pair's acquisition-pool items one at a time and asks
for a yes/no decision (``context`` gives ``labels_remaining`` and
``items_remaining``). Without a hook it queries with probability
q = labels_remaining / items_remaining (selection sampling).

``mode="fisher"``: the informativeness of a candidate is the expected
reduction in the posterior variance of the pair's ability on this benchmark
(the Fisher-information criterion of computerized adaptive testing, evaluated
under the current joint posterior, so uncertainty about the item's difficulty
lowers its value). The query probability is tilted by the ratio rho of that
gain to the gain of a generic unseen item: min(1, q * rho**kappa). When no
candidate is distinguishable (rho = 1) the rule is the evaluator's random
selection sampling. The remaining budget is always filled.

``mode="uncertainty"``: rho is p(1-p) of the candidate relative to a generic
item (uncertainty sampling), with the same tilt.
"""

from __future__ import annotations

import hashlib
import json
import math


def _uniform(context: dict, current_input) -> float:
    key = json.dumps(["paiec-acq", context, current_input], sort_keys=True, separators=(",", ":"), default=str)
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2**64


def make_acquisition(engine, mode: str = "fisher", kappa: float = 2.0, rho_cap: float = 4.0):
    from pirt_online import expected_sigmoid

    def acquisition_function(input, *, labeled=None, context=None) -> bool:
        if context is None:
            raise ValueError("this acquisition function needs the streaming context")
        remaining = int(context["labels_remaining"])
        items = int(context["items_remaining"])
        if remaining <= 0:
            return False
        if items <= remaining:
            return True
        q = remaining / items
        subject, item = input
        with engine.lock:
            engine.sync(labeled)
            state = engine.state_for(item)
            if not hasattr(state, "ability_info"):
                rho = 1.0  # numpy-free fallback: plain selection sampling
            elif mode == "fisher":
                gain, generic = state.ability_info(subject, item)
                rho = gain / generic if generic > 0 else 1.0
            elif mode == "uncertainty":
                m, v = state.logit_moments(subject, item)
                p = expected_sigmoid(m, v)
                p0 = expected_sigmoid(*state.logit_moments(subject, {"benchmark_id": item.get("benchmark_id")}))
                rho = (p * (1 - p)) / max(p0 * (1 - p0), 1e-9)
            else:
                raise ValueError(mode)
        rho = min(max(rho, 0.0), rho_cap) if math.isfinite(rho) else 1.0
        return _uniform(context, input) < min(1.0, q * rho ** kappa)

    return acquisition_function

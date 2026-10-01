"""Brier ALC exactly as the competition defines it, plus the budget protocol.

    ALC = 0.1*B0 + 0.2*B1 + 0.2*B3 + 0.2*B7 + 0.2*B15 + 0.1*B31

Bb is the Brier score at label budget b. For a (subject, benchmark) pair the items
are split 50/50 into an ACQUISITION pool and an EVALUATION pool when the benchmark
has 80+ items; b labels are revealed from the acquisition pool only, and the score
is computed on the evaluation pool only.

The predictor is called through the SAME signature the submission uses --
predict(input, labeled) with input = [subject_dict, item_dict] -- so what is
validated here is literally what gets submitted. An optional acquisition_function
with the official signature chooses which labels to reveal; without one they are
revealed at random, which is what the platform does.
"""
from __future__ import annotations
import numpy as np

BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)
SPLIT_MIN_ITEMS = 80


def brier(pred, truth):
    p = np.clip(np.asarray(pred, dtype=float), 0.0, 1.0)
    y = np.asarray(truth, dtype=float)
    return float(np.mean((p - y) ** 2))


def alc(by_budget: dict[int, float]) -> float:
    return float(sum(w * by_budget[b] for w, b in zip(WEIGHTS, BUDGETS)))


def pools(item_ids, rng, n_items_in_benchmark):
    """Acquisition / evaluation split. Only benchmarks with 80+ items are split."""
    ids = np.asarray(item_ids)
    if n_items_in_benchmark < SPLIT_MIN_ITEMS:
        return ids, ids                       # everything is both, per the published rule
    perm = rng.permutation(len(ids))
    half = len(ids) // 2
    return ids[perm[:half]], ids[perm[half:]]


def score_pair(predict, subject, items_by_id, acq_ids, ev_ids, truth_by_id, rng,
               acquisition_function=None):
    """Returns {budget: brier} for one (subject, benchmark) pair."""
    out = {}
    order = list(rng.permutation(acq_ids))
    for b in BUDGETS:
        if acquisition_function is None or b == 0:
            chosen = order[:b]
        else:
            chosen, pool = [], list(order)
            for k in range(b):
                ctx = {"subject_id": subject.get("_sid", ""),
                       "benchmark_id": items_by_id[ev_ids[0]].get("benchmark_id"),
                       "labels_remaining": b - k, "labels_acquired": k,
                       "max_labels": b, "items_remaining": len(pool)}
                labeled = [[[subject, items_by_id[i]], int(truth_by_id[i])] for i in chosen]
                scores = []
                for cand in pool:
                    pr = predict([subject, items_by_id[cand]], labeled)
                    scores.append(float(acquisition_function(
                        [subject, items_by_id[cand]], pr, labeled, ctx)))
                pick = pool[int(np.argmax(scores))]
                chosen.append(pick); pool.remove(pick)
        labeled = [[[subject, items_by_id[i]], int(truth_by_id[i])] for i in chosen]
        preds = [predict([subject, items_by_id[i]], labeled) for i in ev_ids]
        out[b] = brier(preds, [truth_by_id[i] for i in ev_ids])
    return out

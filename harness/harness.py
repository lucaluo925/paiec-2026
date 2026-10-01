"""Leave-one-BENCHMARK-out evaluation.

The competition's test benchmarks are private and unseen, so a split by subject
measures the wrong thing: it lets the model memorise per-item difficulty for items
it will never meet again. Holding out whole benchmarks reproduces the real
condition. With only three usable public benchmarks this gives three folds, each
trained on two -- so per-fold numbers are reported, never just their mean.
"""
from __future__ import annotations
import numpy as np
from alc import BUDGETS, alc, pools, score_pair


def leave_one_benchmark_out(data, build_predictor, seeds=(0, 1, 2),
                            acquisition_builder=None):
    """data: {benchmark: {"subjects": {sid: attrs}, "items": {iid: attrs},
                          "truth": {(sid, iid): 0/1}}}
       build_predictor(train_data) -> predict(input, labeled) -> float
    """
    benchmarks = sorted(data)
    rows = []
    for held in benchmarks:
        train = {b: data[b] for b in benchmarks if b != held}
        predict = build_predictor(train)
        acq = acquisition_builder(train) if acquisition_builder else None
        d = data[held]
        per_seed = []
        for sd in seeds:
            rng = np.random.default_rng(sd)
            acc = {b: [] for b in BUDGETS}
            for sid, subj in d["subjects"].items():
                ids = d["by_subject"][sid] if "by_subject" in d else [i for (s, i) in d["truth"] if s == sid]
                if len(ids) < 4:
                    continue
                a, e = pools(ids, rng, len(d["items"]))
                subj = dict(subj); subj["_sid"] = sid
                got = score_pair(predict, subj, d["items"], list(a), list(e),
                                 {i: d["truth"][(sid, i)] for i in ids}, rng, acq)
                for b in BUDGETS:
                    acc[b].append(got[b])
            per_seed.append({b: float(np.mean(acc[b])) for b in BUDGETS})
        mean = {b: float(np.mean([p[b] for p in per_seed])) for b in BUDGETS}
        rows.append({"held_out": held, "n_subjects": len(d["subjects"]),
                     **{f"B{b}": mean[b] for b in BUDGETS}, "ALC": alc(mean)})
    return rows

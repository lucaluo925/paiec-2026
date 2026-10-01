"""The test that justifies the whole design.

A predictor that memorises per-ITEM difficulty from training data is the natural
thing to build and the wrong thing to submit: the competition's test benchmarks
are private, so its items were never in training. If the validation protocol
splits by SUBJECT, that predictor looks excellent -- it meets the same items
again. Splitting by BENCHMARK takes the items away and the advantage should
vanish.

If both protocols give the same answer, the protocol choice does not matter and
this harness is over-engineering. The gap below is how much it matters.
"""
import numpy as np, sys
sys.path.insert(0, '.')
from alc import BUDGETS, alc, pools, score_pair
from harness import leave_one_benchmark_out
from world import make_world
from selftest import shrunk

def memoriser(train):
    """Per-item empirical difficulty, shrunk. Legitimate code, wrong protocol."""
    rate = float(np.mean([v for d in train.values() for v in d["truth"].values()]))
    per_item = {}
    for d in train.values():
        for (s, i), y in d["truth"].items():
            per_item.setdefault(i, []).append(y)
    diff = {i: (rate * 3 + sum(v)) / (3 + len(v)) for i, v in per_item.items()}
    def predict(inp, labeled):
        subject, item = inp
        base = diff.get(item.get("_iid"), rate)          # item prior, if seen
        ys = [y for _, y in (labeled or [])]
        if not ys:
            return float(min(max(base, 0.0), 1.0))
        # subject-level shift applied to the item prior, in log-odds
        sub = (rate * 3 + sum(ys)) / (3 + len(ys))
        lo = lambda p: np.log(np.clip(p, 1e-6, 1-1e-6) / (1 - np.clip(p, 1e-6, 1-1e-6)))
        z = lo(base) + (lo(sub) - lo(rate))
        return float(min(max(1 / (1 + np.exp(-z)), 0.0), 1.0))
    return predict


def leave_one_subject_group_out(data, build, seeds=(0, 1)):
    """The WRONG protocol: hold out subjects, keep every benchmark in training."""
    rows = []
    for held in sorted(data):
        d = data[held]
        sids = sorted(d["subjects"])
        cut = len(sids) // 2
        tr_ids, te_ids = set(sids[:cut]), sids[cut:]
        train = {b: dict(v) for b, v in data.items()}
        train[held] = {**d, "truth": {k: v for k, v in d["truth"].items() if k[0] in tr_ids}}
        predict = build(train)
        per_seed = []
        for sd in seeds:
            rng = np.random.default_rng(sd)
            acc = {b: [] for b in BUDGETS}
            for sid in te_ids:
                ids = [i for (s, i) in d["truth"] if s == sid]
                subj = dict(d["subjects"][sid]); subj["_sid"] = sid
                a, e = pools(ids, rng, len(d["items"]))
                got = score_pair(predict, subj, d["items"], list(a), list(e),
                                 {i: d["truth"][(sid, i)] for i in ids}, rng)
                for b in BUDGETS: acc[b].append(got[b])
            per_seed.append({b: float(np.mean(acc[b])) for b in BUDGETS})
        mean = {b: float(np.mean([p[b] for p in per_seed])) for b in BUDGETS}
        rows.append({"held_out": held, "ALC": alc(mean)})
    return rows


world = make_world(np.random.default_rng(7))
for d in world.values():                       # let the memoriser see item ids
    for iid, at in d["items"].items():
        at["_iid"] = iid

base_b = leave_one_benchmark_out(world, shrunk, seeds=(0, 1))
memo_b = leave_one_benchmark_out(world, memoriser, seeds=(0, 1))
base_s = leave_one_subject_group_out(world, shrunk)
memo_s = leave_one_subject_group_out(world, memoriser)

m = lambda rows: float(np.mean([r["ALC"] for r in rows]))
print("                        shrunk    memoriser    gain")
print(f"hold out SUBJECTS     {m(base_s):.4f}    {m(memo_s):.4f}     {m(base_s)-m(memo_s):+.4f}   <- items seen in training")
print(f"hold out BENCHMARKS   {m(base_b):.4f}    {m(memo_b):.4f}     {m(base_b)-m(memo_b):+.4f}   <- items never seen")
print()
print("per fold (benchmark holdout):")
for a, b in zip(base_b, memo_b):
    print(f"  {a['held_out']}   shrunk {a['ALC']:.4f}   memoriser {b['ALC']:.4f}   {a['ALC']-b['ALC']:+.4f}")

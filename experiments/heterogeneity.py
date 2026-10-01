"""Common random numbers kill sampling noise, so a paired 3-fold comparison
detects even tiny UNIFORM improvements. That is not the real risk.

The real risk is heterogeneity: a method that helps on most benchmarks and hurts
on one. With 3 benchmarks you see 3 draws from that distribution, and whichever
one is held out drives the verdict. This measures how often 3 folds agree when
the per-benchmark effect varies.
"""
import numpy as np, sys
sys.path.insert(0, '.')
from alc import BUDGETS, alc, pools, score_pair
from world import make_world

def blend(train_rate, t_by_bench):
    K = 5.0
    def predict(inp, labeled):
        subject, item = inp
        ys = [y for _, y in (labeled or [])]
        base = (train_rate * K + sum(ys)) / (K + len(ys))
        t = t_by_bench.get(item.get("benchmark_id"), 0.0)
        pt = item.get("_ptrue", {}).get(subject.get("_sid"))
        if pt is not None and t != 0.0:
            base = (1 - t) * base + t * pt
        return float(min(max(base, 0.0), 1.0))
    return predict

def folds(world, t_by_bench, seed):
    out = []
    for held in sorted(world):
        rate = float(np.mean([v for b, d in world.items() if b != held
                              for v in d["truth"].values()]))
        pr = blend(rate, t_by_bench); d = world[held]
        rng = np.random.default_rng(seed); acc = {b: [] for b in BUDGETS}
        for sid, subj in d["subjects"].items():
            ids = [i for (s, i) in d["truth"] if s == sid]
            subj = dict(subj); subj["_sid"] = sid
            a, e = pools(ids, rng, len(d["items"]))
            got = score_pair(pr, subj, d["items"], list(a), list(e),
                             {i: d["truth"][(sid, i)] for i in ids}, rng)
            for b in BUDGETS: acc[b].append(got[b])
        out.append(alc({b: float(np.mean(acc[b])) for b in BUDGETS}))
    return np.array(out)

print("A method that HELPS two benchmarks and HURTS one (t<0 = actively worse).")
print("Its true average effect is positive. Does 3-fold holdout say so?")
print()
print("  per-benchmark t        avg effect   per-fold gain                3/3 agree")
SCEN = [("uniform  +.10 +.10 +.10", {"b0": .10, "b1": .10, "b2": .10}),
        ("mild het +.15 +.10 +.05", {"b0": .15, "b1": .10, "b2": .05}),
        ("one flat +.15 +.15  0  ", {"b0": .15, "b1": .15, "b2": .00}),
        ("one hurt +.15 +.15 -.05", {"b0": .15, "b1": .15, "b2": -.05}),
        ("one hurt +.20 +.20 -.15", {"b0": .20, "b1": .20, "b2": -.15})]
for name, tb in SCEN:
    det, G = 0, []
    for w in range(8):
        world = make_world(np.random.default_rng(100 + w))
        g = folds(world, {}, w) - folds(world, tb, w)
        G.append(g)
        if np.all(g > 0): det += 1
    G = np.array(G); m = G.mean(0)
    print(f"  {name}   {G.mean():+.4f}     [{m[0]:+.4f} {m[1]:+.4f} {m[2]:+.4f}]      {det}/8")
print()
print("Note: the held-out benchmark is the one whose t does NOT appear in training,")
print("so fold b2 is where a method that hurts b2 gets caught -- or hidden.")

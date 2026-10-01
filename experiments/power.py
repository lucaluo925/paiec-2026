"""With only three usable benchmarks, what size of real improvement can a
leave-one-benchmark-out comparison actually detect?

Method: take a predictor, make a second one that is better by a KNOWN amount
(its probabilities are shrunk toward the true conditional mean by a factor t, so
the improvement is real, not noise), and ask how often the 3-fold comparison
gets the sign right. Both predictors see identical label draws -- common random
numbers -- because that mistake voided a T2 result once already.
"""
import numpy as np, sys
sys.path.insert(0, '.')
from alc import BUDGETS, alc, pools, score_pair
from world import make_world
from selftest import shrunk

def oracle_blend(train, t):
    """t=0 is the plain shrunk predictor; t>0 mixes in the true probability.
    The improvement is real and its size is controlled by t."""
    rate = float(np.mean([v for d in train.values() for v in d["truth"].values()]))
    K = 5.0
    def predict(inp, labeled):
        subject, item = inp
        ys = [y for _, y in (labeled or [])]
        base = (rate * K + sum(ys)) / (K + len(ys))
        p_true = item.get("_ptrue", {}).get(subject.get("_sid"))
        if t > 0 and p_true is not None:
            base = (1 - t) * base + t * p_true
        return float(min(max(base, 0.0), 1.0))
    return predict


def one_fold_set(world, t, seed):
    """Returns per-fold ALC for the blended predictor at strength t."""
    out = []
    for held in sorted(world):
        train = {b: world[b] for b in world if b != held}
        predict = oracle_blend(train, t)
        d = world[held]
        rng = np.random.default_rng(seed)          # common random numbers
        acc = {b: [] for b in BUDGETS}
        for sid, subj in d["subjects"].items():
            ids = [i for (s, i) in d["truth"] if s == sid]
            subj = dict(subj); subj["_sid"] = sid
            a, e = pools(ids, rng, len(d["items"]))
            got = score_pair(predict, subj, d["items"], list(a), list(e),
                             {i: d["truth"][(sid, i)] for i in ids}, rng)
            for b in BUDGETS: acc[b].append(got[b])
        out.append(alc({b: float(np.mean(acc[b])) for b in BUDGETS}))
    return np.array(out)


print("effect size = true ALC gain of the better predictor")
print("detected    = all 3 folds agree on the sign (the only honest 3-fold test)")
print()
print("   t     true gain    per-fold gains              3/3 agree?   worlds detected")
for t in (0.0, 0.05, 0.10, 0.20, 0.40):
    det, gains_all = 0, []
    for w in range(8):
        world = make_world(np.random.default_rng(100 + w))
        base = one_fold_set(world, 0.0, seed=w)
        cur  = one_fold_set(world, t,   seed=w)
        g = base - cur                      # positive = the blended one is better
        gains_all.append(g)
        if np.all(g > 0): det += 1
    G = np.array(gains_all)
    print(f"  {t:.2f}    {G.mean():+.4f}     [{G.mean(0)[0]:+.4f} {G.mean(0)[1]:+.4f} {G.mean(0)[2]:+.4f}]"
          f"      {det}/8")

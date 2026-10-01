"""Self-tests. A scorer nobody checked is worse than no scorer: it produces
numbers that look like evidence. Each test below has an answer known in advance
from the rules themselves, not from a previous run of this code.
"""
import numpy as np, sys
sys.path.insert(0, '.')
from alc import brier, alc, pools, BUDGETS, SPLIT_MIN_ITEMS
from harness import leave_one_benchmark_out

from world import make_world
def _unused_make_world(rng, n_bench=3, n_sub=40, n_item=100, gamma=0.2):
    """gamma 0.2 matches the interaction strength measured on the eligible data."""
    data = {}
    ability = rng.normal(0, 1, n_sub)
    fam = rng.integers(0, 5, n_sub)
    INTER = rng.normal(0, gamma, (5, 4))
    for b in range(n_bench):
        diff = rng.normal(rng.normal(0, 0.8), 1, n_item)
        itype = rng.integers(0, 4, n_item)
        subs = {f"s{b}_{k}": {"provider": f"fam{fam[k]}", "_proxy": ability[k] + rng.normal(0, .6)}
                for k in range(n_sub)}
        items = {f"i{b}_{j}": {"benchmark_id": f"b{b}", "item_content": f"item {j}",
                               "item_features": f"type={itype[j]}", "interactors": ""}
                 for j in range(n_item)}
        truth = {}
        for k, sid in enumerate(subs):
            lg = ability[k] - diff + INTER[fam[k], itype]
            p = 1 / (1 + np.exp(-lg))
            for j, iid in enumerate(items):
                truth[(sid, iid)] = int(rng.random() < p[j])
        data[f"b{b}"] = {"subjects": subs, "items": items, "truth": truth}
    return data

# ---- predictors ----------------------------------------------------------
def const_half(train):
    return lambda inp, labeled: 0.5

def empirical_mean(train):                      # the official baseline's shape
    def predict(inp, labeled):
        if not labeled:
            return 0.5
        ys = [y for _, y in labeled]
        return float(np.mean(ys))               # UNSHRUNK -- the known flaw
    return predict

def shrunk(train):
    rate = float(np.mean([v for d in train.values() for v in d["truth"].values()]))
    K = 5.0
    def predict(inp, labeled):
        ys = [y for _, y in (labeled or [])]
        return min(max((rate * K + sum(ys)) / (K + len(ys)), 0.0), 1.0)
    return predict

if __name__ != "__main__":
    import sys as _s; _s.modules[__name__].__dict__.setdefault("_skip", True)

def _run_tests():
    # ---- tests ---------------------------------------------------------------
    fails = []
    def check(name, got, want, tol=0.0):
        ok = abs(got - want) <= tol
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {got:.6f} (expect {want}{' ±'+str(tol) if tol else ''})")
        if not ok: fails.append(name)

    print("T1  Brier of a constant 0.5 is exactly 0.25, whatever the truth")
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000)
    check("brier(0.5)", brier(np.full(5000, 0.5), y), 0.25)

    print("T2  a perfect predictor scores 0, an inverted one scores 1")
    check("brier(perfect)", brier(y.astype(float), y), 0.0)
    check("brier(inverted)", brier(1.0 - y, y), 1.0)

    print("T3  ALC weights sum to 1, so a flat Brier passes straight through")
    check("alc(flat 0.25)", alc({b: 0.25 for b in BUDGETS}), 0.25, 1e-12)  # weights sum to 1-1e-16 in binary

    print(f"T4  pools: a benchmark under {SPLIT_MIN_ITEMS} items is NOT split")
    r = np.random.default_rng(1)
    a, e = pools([f"i{k}" for k in range(40)], r, 40)
    check("small: acq == eval size", float(len(a) == len(e) == 40), 1.0)
    a, e = pools([f"i{k}" for k in range(100)], r, 100)
    check("large: disjoint", float(len(set(a) & set(e))), 0.0)
    check("large: covers all", float(len(set(a) | set(e))), 100.0)

    print("T5  end-to-end on synthetic data: constant 0.5 scores 0.25 in EVERY fold")
    world = make_world(np.random.default_rng(7))
    rows = leave_one_benchmark_out(world, const_half, seeds=(0,))
    for r_ in rows:
        check(f"fold {r_['held_out']}", r_["ALC"], 0.25, 1e-12)

    print("T6  the official baseline's unshrunk mean is worse than 0.5 at budget 1")
    rows_em = leave_one_benchmark_out(world, empirical_mean, seeds=(0, 1))
    b1 = float(np.mean([r_["B1"] for r_ in rows_em]))
    em = float(np.mean([r_["ALC"] for r_ in rows_em]))
    print(f"       B1={b1:.4f}  ALC={em:.4f}")
    check("B1 > 0.25 (one label, returned raw)", float(b1 > 0.25), 1.0)
    check("ALC worse than constant 0.5", float(em > 0.25), 1.0)

    print("T7  shrinkage fixes it without any modelling")
    rows_sh = leave_one_benchmark_out(world, shrunk, seeds=(0, 1))
    sh = float(np.mean([r_["ALC"] for r_ in rows_sh]))
    print(f"       ALC={sh:.4f}  (baseline {em:.4f}, constant 0.2500)")
    check("beats the official baseline", float(sh < em), 1.0)
    check("beats constant 0.5", float(sh < 0.25), 1.0)

    print()
    print("FAILED:", fails if fails else "none")

if __name__ == "__main__":
    _run_tests()

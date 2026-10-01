"""Synthetic world for testing the harness. Returns the GENERATING probabilities
alongside the labels, so a test can construct a predictor that is better by a
known amount."""
import numpy as np

def make_world(rng, n_bench=3, n_sub=40, n_item=100, gamma=0.2):
    """gamma 0.2 matches the interaction strength measured on the eligible data."""
    data = {}
    ability = rng.normal(0, 1, n_sub)
    fam = rng.integers(0, 5, n_sub)
    INTER = rng.normal(0, gamma, (5, 4))
    for b in range(n_bench):
        diff = rng.normal(rng.normal(0, 0.8), 1, n_item)
        itype = rng.integers(0, 4, n_item)
        subs = {f"s{b}_{k}": {"provider": f"fam{fam[k]}",
                              "_proxy": ability[k] + rng.normal(0, .6)}
                for k in range(n_sub)}
        items = {f"i{b}_{j}": {"benchmark_id": f"b{b}", "item_content": f"item {j}",
                               "item_features": f"type={itype[j]}", "interactors": "",
                               "_iid": f"i{b}_{j}", "_ptrue": {}}
                 for j in range(n_item)}
        iids = list(items)
        truth = {}
        for k, sid in enumerate(subs):
            p = 1 / (1 + np.exp(-(ability[k] - diff + INTER[fam[k], itype])))
            for j, iid in enumerate(iids):
                truth[(sid, iid)] = int(rng.random() < p[j])
                items[iid]["_ptrue"][sid] = float(p[j])   # the generating probability
        data[f"b{b}"] = {"subjects": subs, "items": items, "truth": truth}
    return data

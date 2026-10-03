"""Two-way hold-out: benchmarks by fold, model identities held out globally.

The private test set consists of benchmarks that never appear in training, so
items are never split at random. Eligible benchmarks (>= 80 items) are assigned
to K folds; benchmarks that share item content are kept in the same fold so a
near-duplicate cannot leak difficulties. Independently, a fraction of model
identities (normalized names) is removed from every training fold; they occur
only as unseen subjects in the evaluated folds.

Training data for fold f = every benchmark outside fold f (eligible or not),
restricted to subjects whose identity is not held out.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .data import Benchmark, model_identity


@dataclass
class Split:
    folds: list  # list[list[bench_key]]
    heldout_identities: set
    seed: int

    def train_keys(self, fold: int) -> list:
        test = set(self.folds[fold])
        return [k for k in self.all_keys if k not in test]

    all_keys: list = None


def _content_groups(db: dict[str, Benchmark], keys: list[str]) -> list[list[str]]:
    """Union benchmarks whose item texts overlap by >= 5% of the smaller one."""
    parent = {k: k for k in keys}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    hashes = {}
    for k in keys:
        texts = {v["item_content"].strip().lower() for v in db[k].items.values() if v["item_content"].strip()}
        hashes[k] = {hashlib.sha1(t.encode()).hexdigest()[:16] for t in texts}
    owner = {}
    overlap = {}
    for k in keys:
        for h in hashes[k]:
            if h in owner and owner[h] != k:
                a, b = sorted((owner[h], k))
                overlap[(a, b)] = overlap.get((a, b), 0) + 1
            else:
                owner[h] = k
    for (a, b), n in overlap.items():
        if n >= 0.05 * max(1, min(len(hashes[a]), len(hashes[b]))):
            parent[find(a)] = find(b)
    groups = {}
    for k in keys:
        groups.setdefault(find(k), []).append(k)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])


def make_split(db: dict[str, Benchmark], k: int = 5, seed: int = 0, min_items: int = 80,
               heldout_frac: float = 0.2) -> Split:
    keys = sorted(db)
    eligible = [key for key in keys if db[key].n_items >= min_items]
    groups = _content_groups(db, eligible)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    # Greedy balance of response counts across folds.
    folds = [[] for _ in range(k)]
    load = np.zeros(k)
    for gi in sorted(order, key=lambda i: -sum(len(db[b].resp) for b in groups[i])):
        f = int(np.argmin(load))
        folds[f].extend(groups[gi])
        load[f] += sum(len(db[b].resp) for b in groups[gi])
    identities = sorted({model_identity(s) for b in db.values() for s in b.subjects.values()})
    n_hold = int(round(heldout_frac * len(identities)))
    held = set(rng.choice(identities, size=n_hold, replace=False).tolist()) if n_hold else set()
    return Split(folds=[sorted(f) for f in folds], heldout_identities=held, seed=seed, all_keys=keys)

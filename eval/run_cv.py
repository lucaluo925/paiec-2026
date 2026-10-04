"""Cross-validated local evaluation (benchmark folds x held-out model identities).

For each fold the offline calibration sees only the other folds' benchmarks and
never the held-out model identities; the fold's benchmarks are then scored with
the streaming protocol in ``local_scorer``. Records from all folds are pooled
before computing per-budget Brier scores and the ALC. Aggregation follows the
official rule: Brier is averaged within each subject-benchmark pair and then
across pairs with equal weight.

    python -m eval.run_cv --data data/raw --variants const0.5 empirical_mean irt \
        --out results/by_budget.csv
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "submission"))

from eval import baselines  # noqa: E402
from eval.data import load_db  # noqa: E402
from eval.local_scorer import (  # noqa: E402
    BUDGETS, ProtocolConfig, _alc, build_pairs, per_pair_brier, run_protocol,
)
from eval.splits import make_split  # noqa: E402
from train.fit_offline import fit  # noqa: E402

import pirt_online  # noqa: E402


def deep_update(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def apply_tuned(params: dict, tuned: dict | None) -> dict:
    """Apply a tuned_hyper*.json patch exactly as train_final.py does.

    Kept deliberately identical to train_final so that a CV fold evaluates the same
    configuration that gets shipped. NOTE (2026-10-01): the tuned values were selected
    using all folds, so applying them inside CV re-uses information from the evaluated
    fold. That is a real optimism bias; it is accepted here because the question this
    harness answers is "how does the SHIPPED model behave", not "what is an unbiased
    estimate of a freshly tuned model".
    """
    if not tuned:
        return params
    params = copy.deepcopy(params)
    for key, scale in tuned.get("hyper_scale", {}).items():
        if key not in params["hyper"]:
            raise SystemExit(f"tuned scale for unknown hyper-parameter {key!r}")
        params["hyper"][key] *= float(scale)
    params = deep_update(params, {k: v for k, v in tuned.items() if k in ("shrink", "tuning")})
    params["hyper_scale"] = tuned.get("hyper_scale", {})
    return params


@dataclass
class Variant:
    name: str
    kind: str  # const | empirical_mean | irt
    patch: dict = None
    scope: str = "global"  # labeled scope passed to the predictor
    acquisition: str | None = None  # None -> default random policy


def make_predictor(variant: Variant, params: dict | None, db=None, train_keys=None):
    if variant.kind == "const":
        return baselines.constant(0.5), None
    if variant.kind == "empirical_mean":
        return baselines.empirical_mean(), None
    if variant.kind == "irt":
        engine = pirt_online.Engine(deep_update(params, variant.patch or {}))
        acq = None
        if variant.acquisition:
            import labeling_core
            acq = labeling_core.make_acquisition(engine, variant.acquisition)
        return engine.predict, acq
    if variant.kind in ("irt_textres", "irt_textres_shuffle"):
        # 预注册候选，docs/PREREG_text_residual.md。lam 只在训练 benchmark 上估。
        from . import text_residual
        cfg = deep_update(params, variant.patch or {})
        lam = make_predictor._lam_cache.get(id(params))
        if lam is None:
            lam = text_residual.fit_lambda(
                db, train_keys, cfg,
                engine_factory=lambda: pirt_online.Engine(copy.deepcopy(cfg)))
            make_predictor._lam_cache[id(params)] = lam
            print(f"  [textres] lam fitted on training benchmarks only = {lam:.5f}")
        engine = pirt_online.Engine(cfg)
        ref = pirt_online.Engine(copy.deepcopy(cfg))   # pristine，只用来算离线残差
        pred = text_residual.Predictor(
            engine, lam, shuffle=variant.kind.endswith("shuffle"), seed=0, ref_engine=ref)
        return pred, None
    raise ValueError(variant.kind)


make_predictor._lam_cache = {}


def pooled_metrics(chunks: list) -> dict:
    """chunks: list of (records, pairs, heldout identities) per fold. Returns per-budget metrics.

    ``brier`` is the official aggregation: average within each subject-benchmark
    pair, then across pairs with equal weight. ``brier_micro`` keeps the plain
    per-prediction average for reference. Seen/unseen splits pairs by whether the
    subject's model identity was held out of the offline calibration.
    """
    out = {}
    # Pair ids must be unique across folds before the folds are pooled.
    offsets, n_pairs = [], 0
    for _, pairs, _ in chunks:
        offsets.append(n_pairs)
        n_pairs += len(pairs)
    unseen = np.zeros(n_pairs, dtype=bool)
    for off, (_, pairs, heldout) in zip(offsets, chunks):
        for k, pair in enumerate(pairs):
            unseen[off + k] = pair.identity in heldout
    for b in BUDGETS:
        ps, ys, ids = [], [], []
        for off, (records, _, _) in zip(offsets, chunks):
            p, y, pi = records[b]
            ps.append(p)
            ys.append(y)
            ids.append(np.asarray(pi) + off)
        p, y, pid = np.concatenate(ps), np.concatenate(ys), np.concatenate(ids)
        sq = (p - y) ** 2
        per_pair, has = per_pair_brier(sq, pid, n_pairs)
        seen_sel, unseen_sel = has & ~unseen, has & unseen
        out[b] = {
            "brier": float(per_pair[has].mean()),
            "brier_seen": float(per_pair[seen_sel].mean()) if seen_sel.any() else float("nan"),
            "brier_unseen": float(per_pair[unseen_sel].mean()) if unseen_sel.any() else float("nan"),
            "brier_micro": float(sq.mean()),
            "n": int(len(y)), "n_pairs": int(has.sum()), "n_pairs_unseen": int(unseen_sel.sum()),
            "n_unseen": int(unseen[pid].sum()),
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heldout-frac", type=float, default=0.2)
    ap.add_argument("--only-folds", type=int, nargs="*")
    ap.add_argument("--variants", nargs="+", default=["const0.5", "empirical_mean", "irt"])
    ap.add_argument("--variant-file", help="JSON list of extra variants {name, kind, patch, scope, acquisition}")
    ap.add_argument("--tuned", help="tuned_hyper*.json: apply the SAME hyper_scale and shrink patch that "
                                    "train_final.py bakes into the shipped params, so the folds evaluate the "
                                    "configuration that is actually submitted. Without it, fit() returns "
                                    "shrink.weights={'0': 0.0} and the run measures a model with shrinkage off.")
    ap.add_argument("--labeled-scope", default="global", choices=["global", "pair"])
    ap.add_argument("--split-mode", default="pair", choices=["benchmark", "pair"])
    ap.add_argument("--max-subjects", type=int, default=40)
    ap.add_argument("--max-eval", type=int, default=60)
    ap.add_argument("--max-stream", type=int, default=400)
    ap.add_argument("--out", default="results/by_budget.csv")
    ap.add_argument("--records", help="optional .npz to store pooled predictions")
    args = ap.parse_args(argv)

    known = {
        "const0.5": Variant("const0.5", "const"),
        "empirical_mean": Variant("empirical_mean", "empirical_mean", scope="pair"),
        "irt": Variant("irt", "irt"),
        "irt_pairscope": Variant("irt_pairscope", "irt", scope="pair"),
        "irt_textres": Variant("irt_textres", "irt_textres"),
        "irt_textres_shuffle": Variant("irt_textres_shuffle", "irt_textres_shuffle"),
    }
    variants = []
    extra = {}
    if args.variant_file:
        for v in json.loads(Path(args.variant_file).read_text()):
            extra[v["name"]] = Variant(v["name"], v["kind"], v.get("patch"), v.get("scope", "global"),
                                       v.get("acquisition"))
    for name in args.variants:
        variants.append(extra.get(name) or known[name])

    tuned = json.loads(Path(args.tuned).read_text()) if args.tuned else None
    if tuned is None:
        print("WARNING: --tuned not given; shrinkage is OFF (fit() returns weights {'0': 0.0}) and tuned "
              "hyper scaling is not applied, so these folds do NOT measure the shipped configuration.")

    t0 = time.time()
    db = load_db(args.data)
    split = make_split(db, k=args.folds, seed=args.seed, heldout_frac=args.heldout_frac)
    print(f"loaded {len(db)} eligible benchmarks in {time.time() - t0:.0f}s; folds {[len(f) for f in split.folds]}; "
          f"{len(split.heldout_identities)} held-out identities")
    folds = args.only_folds if args.only_folds else range(args.folds)
    chunks = {v.name: [] for v in variants}
    for f in folds:
        if not split.folds[f]:
            continue
        t1 = time.time()
        params, _ = fit(db, split.train_keys(f), split.heldout_identities)
        params = apply_tuned(params, tuned)
        print(f"fold {f}: fit {time.time() - t1:.0f}s  hyper={json.dumps({k: round(v, 3) for k, v in params['hyper'].items()})}")
        for v in variants:
            cfg = ProtocolConfig(split_mode=args.split_mode, labeled_scope=v.scope if v.scope == "pair" else args.labeled_scope,
                                 max_subjects_per_benchmark=args.max_subjects, max_eval_per_pair=args.max_eval,
                                 max_stream_per_pair=args.max_stream, seed=args.seed)
            pairs = build_pairs(db, split.folds[f], cfg)
            predict, acq = make_predictor(v, params, db, split.train_keys(f))
            res = run_protocol(pairs, predict, acq, cfg)
            chunks[v.name].append((res.records, pairs, split.heldout_identities))
            dropped = getattr(predict, "dropped", 0)
            print(f"  {v.name:<28} ALC {res.alc_pair_macro:.5f}  "
                  + " ".join(f"B{b}={res.brier_pair_macro[b]:.4f}" for b in BUDGETS)
                  + f"  ({res.seconds:.0f}s, {res.n_pairs} pairs, {res.n_targets} targets)"
                  + (f"  [!! dropped {dropped} labeled entries]" if dropped else ""))

    rows = []
    for v in variants:
        m = pooled_metrics(chunks[v.name])
        brier = {b: m[b]["brier"] for b in BUDGETS}
        row = {"variant": v.name, "ALC": _alc(brier)}
        for b in BUDGETS:
            row[f"B{b}"] = brier[b]
        row["ALC_seen"] = _alc({b: m[b]["brier_seen"] for b in BUDGETS})
        row["ALC_unseen"] = _alc({b: m[b]["brier_unseen"] for b in BUDGETS})
        row["ALC_micro"] = _alc({b: m[b]["brier_micro"] for b in BUDGETS})
        row["n_pairs"] = m[0]["n_pairs"]
        row["n_pairs_unseen"] = m[0]["n_pairs_unseen"]
        row["n_targets"] = m[0]["n"]
        row["n_targets_unseen"] = m[0]["n_unseen"]
        row["acquisition"] = v.acquisition or "random(default)"
        row["labeled_scope"] = v.scope if v.scope == "pair" else args.labeled_scope
        row["split_mode"] = args.split_mode
        rows.append(row)
    import csv
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{x:.6f}" if isinstance(x, float) else x) for k, x in r.items()})
    print("\n" + "variant".ljust(30) + "ALC      " + "  ".join(f"B{b:<6}" for b in BUDGETS))
    for r in rows:
        print(r["variant"].ljust(30) + f"{r['ALC']:.5f}  " + "  ".join(f"{r[f'B{b}']:.5f}" for b in BUDGETS))
    if args.records:
        store = {}
        for v in variants:
            # Fold-offset pair ids so the figures can use the same pair-macro
            # aggregation as the CSV instead of a plain per-prediction average.
            offsets, total = [], 0
            for _, pairs, _ in chunks[v.name]:
                offsets.append(total)
                total += len(pairs)
            for b in BUDGETS:
                store[f"{v.name}|{b}|p"] = np.concatenate([c[0][b][0] for c in chunks[v.name]])
                store[f"{v.name}|{b}|y"] = np.concatenate([c[0][b][1] for c in chunks[v.name]])
                store[f"{v.name}|{b}|pid"] = np.concatenate(
                    [np.asarray(c[0][b][2]) + off for off, c in zip(offsets, chunks[v.name])])
        np.savez_compressed(args.records, **store)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

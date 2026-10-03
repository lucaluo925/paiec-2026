"""Pre-registered A/B of the acquisition (label-sampling) function.

Runs the baseline predictor once per fold with the official random policy and
once per acquisition variant, over the same pairs, then pools the folds and
compares them with a paired bootstrap under the official pair-macro metric.

Decision rule, fixed before the run: an acquisition variant is shipped only if
its ALC is lower than random AND the 95% interval of the paired bootstrap over
benchmarks excludes 0. Otherwise the submission keeps the random policy.

    python -m eval.ab --data data/raw --out results/acquisition_ab.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "submission"))

from eval.data import load_db  # noqa: E402
from eval.local_scorer import (  # noqa: E402
    BUDGETS, ProtocolConfig, bootstrap_diff, build_pairs, pair_macro_alc, run_protocol,
)
from eval.run_cv import Variant, make_predictor  # noqa: E402
from eval.tune import apply_scales  # noqa: E402
from eval.splits import make_split  # noqa: E402
from train.fit_offline import fit  # noqa: E402


def pooled(runs) -> tuple[float, dict]:
    """Pair-macro ALC and per-budget Brier over every fold in ``runs``."""
    p, y, pid = {}, {}, {}
    offsets, n_pairs = [], 0
    for _, pairs in runs:
        offsets.append(n_pairs)
        n_pairs += len(pairs)
    for b in BUDGETS:
        p[b] = np.concatenate([res.records[b][0] for res, _ in runs])
        y[b] = np.concatenate([res.records[b][1] for res, _ in runs])
        pid[b] = np.concatenate([res.records[b][2] + off for off, (res, _) in zip(offsets, runs)])
    return pair_macro_alc(p, y, pid, n_pairs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heldout-frac", type=float, default=0.2)
    ap.add_argument("--baseline", default="irt", help="variant run with the official random policy")
    ap.add_argument("--variants", nargs="+", default=["irt+fisher", "irt+uncertainty"])
    ap.add_argument("--variant-file", default="eval/variants_acquisition.json")
    ap.add_argument("--tuned", help="tuned_hyper.json to apply to every variant")
    ap.add_argument("--max-subjects", type=int, default=40)
    ap.add_argument("--max-eval", type=int, default=60)
    ap.add_argument("--max-stream", type=int, default=400)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="results/acquisition_ab.md")
    ap.add_argument("--csv", default="results/acquisition_ab.csv")
    ap.add_argument("--records", default="results/ab_records.npz")
    args = ap.parse_args(argv)

    known = {"irt": Variant("irt", "irt")}
    for v in json.loads(Path(args.variant_file).read_text()):
        known[v["name"]] = Variant(v["name"], v["kind"], v.get("patch"), v.get("scope", "global"),
                                   v.get("acquisition"))
    names = [args.baseline] + args.variants
    variants = [known[n] for n in names]

    t0 = time.time()
    db = load_db(args.data)
    split = make_split(db, k=args.folds, seed=args.seed, heldout_frac=args.heldout_frac)
    cfg = ProtocolConfig(max_subjects_per_benchmark=args.max_subjects, max_eval_per_pair=args.max_eval,
                         max_stream_per_pair=args.max_stream, seed=args.seed)
    print(f"loaded {len(db)} benchmarks in {time.time() - t0:.0f}s; folds {[len(f) for f in split.folds]}")

    tuned = json.loads(Path(args.tuned).read_text()) if args.tuned else None
    runs = {n: [] for n in names}
    for f in range(args.folds):
        if not split.folds[f]:
            continue
        params, _ = fit(db, split.train_keys(f), split.heldout_identities)
        if tuned:
            params = apply_scales(params, tuned["hyper_scale"], tuned["shrink"]["weights"])
        for v in variants:
            # run_protocol mutates pairs (acquired labels, stream position), so each
            # variant needs its own build_pairs; same db/keys/cfg/seed makes them identical.
            pairs = build_pairs(db, split.folds[f], cfg)
            predict, acq = make_predictor(v, params)
            res = run_protocol(pairs, predict, acq, cfg)
            runs[v.name].append((res, pairs))
            print(f"  fold {f} {v.name:<20} ALC {pooled([(res, pairs)])[0]:.5f} ({res.seconds:.0f}s)")

    rows, lines = [], []
    base_alc, base_brier = pooled(runs[args.baseline])
    lines.append("# Acquisition A/B (pre-registered, single run)\n")
    lines.append(f"Data: `{args.data}` | folds {[len(f) for f in split.folds]} | "
                 f"{sum(len(p) for _, p in runs[args.baseline])} subject-benchmark pairs | "
                 f"{args.n_boot} bootstrap draws\n")
    lines.append("Metric: official pair-macro Brier ALC. Lower is better.\n")
    lines.append("| variant | ALC | " + " | ".join(f"B{b}" for b in BUDGETS) + " | dALC vs random |"
                 " 95% CI (benchmark clusters) | 95% CI (pair clusters, optimistic) | ships? |")
    lines.append("|---|---|" + "---|" * len(BUDGETS) + "---|---|---|---|")
    lines.append(f"| {args.baseline} (random) | {base_alc:.5f} | "
                 + " | ".join(f"{base_brier[b]:.4f}" for b in BUDGETS) + " | - | - | - | baseline |")
    rows.append({"variant": f"{args.baseline} (random)", "ALC": base_alc,
                 **{f"B{b}": base_brier[b] for b in BUDGETS}})
    for n in args.variants:
        alc, brier = pooled(runs[n])
        d, lo, hi = bootstrap_diff(runs[n], runs[args.baseline], n_boot=args.n_boot, seed=args.seed,
                                   cluster="benchmark")
        _, plo, phi = bootstrap_diff(runs[n], runs[args.baseline], n_boot=args.n_boot, seed=args.seed,
                                     cluster="pair")
        ships = bool(d < 0 and hi < 0)
        lines.append(f"| {n} | {alc:.5f} | " + " | ".join(f"{brier[b]:.4f}" for b in BUDGETS)
                     + f" | {d:+.5f} | [{lo:+.5f}, {hi:+.5f}] | [{plo:+.5f}, {phi:+.5f}] |"
                     + (" **yes**" if ships else " no") + " |")
        rows.append({"variant": n, "ALC": alc, **{f"B{b}": brier[b] for b in BUDGETS},
                     "delta_alc": d, "ci_lo_benchmark": lo, "ci_hi_benchmark": hi,
                     "ci_lo_pair": plo, "ci_hi_pair": phi, "ships": ships})
    winners = [r["variant"] for r in rows if r.get("ships")]
    lines.append("\n## Decision\n")
    lines.append("Rule fixed before the run: ship only if dALC < 0 and the benchmark-cluster "
                 "95% CI excludes 0.\n")
    if winners:
        lines.append(f"**Ship {winners[0]}** (`--with-labeling`).\n")
    else:
        lines.append("**Keep the random policy.** No variant met the rule, so the submission "
                     "ships without `labeling.py`.\n")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")

    import csv
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[-1]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{x:.6f}" if isinstance(x, float) else x) for k, x in r.items()})
    if args.records:
        store = {}
        for n in names:
            offsets, total = [], 0
            for _, pairs in runs[n]:
                offsets.append(total)
                total += len(pairs)
            for b in BUDGETS:
                store[f"{n}|{b}|p"] = np.concatenate([res.records[b][0] for res, _ in runs[n]])
                store[f"{n}|{b}|y"] = np.concatenate([res.records[b][1] for res, _ in runs[n]])
                store[f"{n}|{b}|pid"] = np.concatenate(
                    [res.records[b][2] + off for off, (res, _) in zip(offsets, runs[n])])
        np.savez_compressed(args.records, **store)
    print("\n".join(lines[3:]))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

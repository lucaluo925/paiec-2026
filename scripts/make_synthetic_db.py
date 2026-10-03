"""Write a synthetic measurement-db copy for testing the pipeline without data.

The tables use the public HF layout (``<dir>/{benchmarks,subjects,items,
response}.parquet``) and are generated from a known hierarchical 2PL model, so
the local scorer and the offline fit can be checked end to end before the real
corpus is available. Nothing here is competition data.

    python scripts/make_synthetic_db.py --out data/synthetic --seed 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROVIDERS = ["Alpha", "Beacon", "Cirrus", "Delta", "Ember", "Fjord", "Gamma", "Helix", "Iris"]
WORDS_EASY = "list name find add count read copy match pick sort".split()
WORDS_HARD = "prove optimize derive integrate refactor theorem asymptotic concurrency inverse eigen".split()
FILLER = "the a of to and in with for on by from this that item task value result answer".split()


def _h(*parts) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-benchmarks", type=int, default=40)
    ap.add_argument("--n-models", type=int, default=90)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Models: ability grows with release date; size in the name adds signal.
    models = []
    prov_eff = {p: rng.normal(0, 0.4) for p in PROVIDERS}
    for m in range(args.n_models):
        prov = PROVIDERS[m % len(PROVIDERS)]
        year = rng.uniform(2023.0, 2026.5)
        size = int(rng.choice([1, 3, 7, 8, 13, 34, 70, 405]))
        theta = 0.7 * (year - 2024.7) + 0.35 * np.log(size / 13) + prov_eff[prov] + rng.normal(0, 0.6)
        month = int((year % 1) * 12) + 1
        models.append(
            {
                "normalized_name": f"{prov} Model-{m} {size}B",
                "provider": prov,
                "release_date": f"{int(year)}-{month:02d}-15",
                "theta": float(theta),
            }
        )

    truth = {"models": models, "benchmarks": {}}
    for b in range(args.n_benchmarks):
        key = f"synth_bench_{b:02d}"
        d = out / key
        d.mkdir(exist_ok=True)
        n_items = int(np.exp(rng.uniform(np.log(40), np.log(1500))))
        beta = rng.normal(0.0, 1.2)
        alpha = float(np.exp(rng.normal(0, 0.35)))
        tau = 0.5
        has_tiers = rng.random() < 0.5
        binary = not (b % 13 == 12)  # a few non-binary benchmarks exercise eligibility
        n_sub = int(rng.integers(10, 60))
        chosen = rng.choice(len(models), size=min(n_sub, len(models)), replace=False)
        access = f"2026-0{int(rng.integers(1, 9))}-01"
        harness = "" if rng.random() < 0.7 else "agent-harness"

        subj_rows, subj_theta = [], {}
        for mi in chosen:
            m = models[mi]
            effort = "" if rng.random() < 0.8 else str(rng.choice(["low", "high"]))
            eff_bonus = {"": 0.0, "low": -0.3, "high": 0.4}[effort]
            row = {
                "display_name": m["normalized_name"].replace(" ", "-").lower(),
                "normalized_name": m["normalized_name"] if rng.random() > 0.05 else None,
                "provider": m["provider"],
                "release_date": m["release_date"],
                "access_date": access,
                "harness": harness or None,
                "reasoning_effort": effort or None,
                "harness_version": None,
                "subject_features_extra": None,
            }
            sid = _h(key, *row.values())
            row = {"subject_id": sid, **row}
            subj_rows.append(row)
            subj_theta[sid] = alpha * (m["theta"] + eff_bonus) + rng.normal(0, tau)

        item_rows, deltas = [], {}
        for i in range(n_items):
            tier = rng.integers(0, 3) if has_tiers else None
            n_hard = rng.poisson(1.0)
            n_easy = rng.poisson(1.0)
            words = list(rng.choice(FILLER, size=int(rng.integers(8, 40))))
            words += list(rng.choice(WORDS_HARD, size=n_hard)) + list(rng.choice(WORDS_EASY, size=n_easy))
            rng.shuffle(words)
            content = f"Q{i}: " + " ".join(words) + "?"
            delta = beta + 0.5 * (n_hard - n_easy) + rng.normal(0, 1.1)
            feats = ""
            if tier is not None:
                delta += [-1.0, 0.0, 1.0][tier]
                feats = f"difficulty={['easy', 'medium', 'hard'][tier]}"
            iid = _h(key, i, content)
            deltas[iid] = float(delta)
            item_rows.append(
                {"item_id": iid, "benchmark_id": key, "raw_item_id": str(i), "content": content,
                 "item_features": feats or None}
            )

        coverage = rng.uniform(0.6, 1.0)
        resp_rows = []
        for srow in subj_rows:
            sid = srow["subject_id"]
            for irow in item_rows:
                if rng.random() > coverage:
                    continue
                iid = irow["item_id"]
                p = 1 / (1 + np.exp(-(subj_theta[sid] - deltas[iid])))
                trials = 2 if rng.random() < 0.05 else 1
                for t in range(1, trials + 1):
                    y = float(rng.random() < p) if binary else float(rng.integers(1, 6))
                    resp_rows.append(
                        {"subject_id": sid, "item_id": iid, "response_id": _h(sid, iid, t),
                         "benchmark_id": key, "trial": t, "test_condition": None,
                         "interactors": None, "response": y}
                    )
        pd.DataFrame(
            [{"benchmark_id": key, "name": key.replace("_", " ").title(), "license": "CC-BY-4.0",
              "source_url": "synthetic", "description": "Synthetic benchmark for pipeline tests.",
              "modality": ["text"], "domain": ["general"], "multi_single_turn": "single_turn",
              "response_type": "binary" if binary else "likert_5", "granularity": "item",
              "n_subjects": len(subj_rows), "n_items": n_items, "n_responses": len(resp_rows)}]
        ).to_parquet(d / "benchmarks.parquet", index=False)
        pd.DataFrame(subj_rows).to_parquet(d / "subjects.parquet", index=False)
        pd.DataFrame(item_rows).to_parquet(d / "items.parquet", index=False)
        pd.DataFrame(resp_rows).to_parquet(d / "response.parquet", index=False)
        truth["benchmarks"][key] = {"beta": beta, "alpha": alpha, "n_items": n_items, "binary": binary}
    (out / "truth.json").write_text(json.dumps(truth, indent=1))
    print(f"wrote {args.n_benchmarks} synthetic benchmarks to {out}")


if __name__ == "__main__":
    main()

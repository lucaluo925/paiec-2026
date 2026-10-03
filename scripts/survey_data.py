"""Summarise the downloaded measurement-db corpus (Task 1.3).

    python scripts/survey_data.py --data data/raw --out results/data_survey.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.data import SUBJECT_FIELDS, list_benchmark_dirs, load_benchmark, model_identity, read_meta  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/data_survey.md")
    args = ap.parse_args()
    root = Path(args.data)
    dirs = list_benchmark_dirs(root)
    manifest = json.loads((root / "corpus_manifest.json").read_text()) if (root / "corpus_manifest.json").exists() else {}
    all_rows = manifest.get("benchmark_rows", {})
    rt = Counter(v.get("response_type") for v in all_rows.values())
    gran = Counter(v.get("granularity") for v in all_rows.values())

    rows, subj_fill, identities, names = [], Counter(), set(), set()
    n_subject_rows = 0
    item_feat, inter, content_len = [], [], []
    per_bench_ident = {}
    for d in dirs:
        meta = read_meta(d)
        bench = load_benchmark(d)
        if bench is None:
            continue
        r = bench.resp
        n_sub, n_item = r["subject_id"].nunique(), r["item_id"].nunique()
        rows.append({
            "benchmark": bench.key, "domain": ",".join(meta.get("domain") or []) if isinstance(meta.get("domain"), list) else meta.get("domain"),
            "subjects": n_sub, "items": n_item, "responses": len(r), "density": len(r) / max(n_sub * n_item, 1),
            "base_rate": float(r["y"].mean()), "item_features_nonempty": float(np.mean([bool(v["item_features"]) for v in bench.items.values()])),
            "interactors_nonempty": float((r["interactors"] != "").mean()),
            "median_content_chars": float(np.median([len(v["item_content"]) for v in bench.items.values()])),
        })
        used = set(r["subject_id"].unique())
        ids = set()
        for sid, s in bench.subjects.items():
            if sid not in used:
                continue
            n_subject_rows += 1
            for f in SUBJECT_FIELDS:
                subj_fill[f] += bool(s[f])
            identities.add(model_identity(s))
            ids.add(model_identity(s))
            if s["normalized_name"]:
                names.add(s["normalized_name"])
        per_bench_ident[bench.key] = ids
        item_feat += [v["item_features"] for v in bench.items.values() if v["item_features"]]
        inter += [x for x in r["interactors"].unique() if x]
        content_len += [len(v["item_content"]) for v in bench.items.values()]
    df = pd.DataFrame(rows).sort_values("responses", ascending=False)
    ident_bench_counts = Counter(i for ids in per_bench_ident.values() for i in ids)
    feat_keys = Counter(part.split("=", 1)[0] for f in item_feat for part in f.split(";") if part)

    lines = [f"# measurement-db survey", "",
             f"Source: `{manifest.get('db_repo', '?')}` revision `{manifest.get('revision', '?')}`", "",
             f"* benchmark directories on HF: {len(all_rows) or 'n/a'}; response_type {dict(rt)}; granularity {dict(gran)}",
             f"* eligible (binary, item-level) loaded: **{len(df)}**",
             f"* eligible with >= 80 items (evaluable): **{int((df['items'] >= 80).sum())}**",
             f"* responses (one per subject x item x interactors): **{df['responses'].sum():,}**",
             f"* items: **{df['items'].sum():,}**; subject rows (configurations): **{n_subject_rows:,}**; "
             f"distinct model identities: **{len(identities):,}** ({len(names):,} with a normalized name)",
             f"* identities appearing in >= 2 / >= 5 / >= 10 benchmarks: "
             f"{sum(c >= 2 for c in ident_bench_counts.values())} / {sum(c >= 5 for c in ident_bench_counts.values())} / "
             f"{sum(c >= 10 for c in ident_bench_counts.values())}",
             f"* density (responses / subjects x items): median {df['density'].median():.3f}, "
             f"overall {df['responses'].sum() / (df['subjects'] * df['items']).sum():.3f}",
             f"* base rate: pooled {np.average(df['base_rate'], weights=df['responses']):.3f}, "
             f"per-benchmark median {df['base_rate'].median():.3f} (IQR {df['base_rate'].quantile(.25):.3f}-{df['base_rate'].quantile(.75):.3f})",
             f"* items per benchmark: median {df['items'].median():.0f}, max {df['items'].max():,}; "
             f"subjects per benchmark: median {df['subjects'].median():.0f}, max {df['subjects'].max():,}",
             "", "## Subject attribute fill rates", "",
             "| field | non-empty |", "|---|---|"]
    for f in SUBJECT_FIELDS:
        lines.append(f"| {f} | {subj_fill[f] / max(n_subject_rows, 1):.1%} |")
    lines += ["", "## Item attributes", "",
              f"* item_content: median {np.median(content_len):.0f} chars, 90th pct {np.percentile(content_len, 90):.0f}",
              f"* item_features non-empty in {len(item_feat):,} items; most common keys: "
              + ", ".join(f"`{k}` ({c})" for k, c in feat_keys.most_common(15)),
              f"* distinct non-empty interactors values: {len(set(inter))}",
              "", "## Largest benchmarks", "", df.head(40).to_markdown(index=False, floatfmt=".3f")]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    df.to_csv(out.with_suffix(".csv"), index=False)
    print("\n".join(lines[:20]))


if __name__ == "__main__":
    main()

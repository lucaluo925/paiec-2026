"""measurement-db -> harness shape. Column names VERIFIED against the real files
on 2026-09-29; the data has since been deleted, so do not "fix" these from memory.

Three mismatches with the submission interface, handled here so model code never
sees them:
  items.content      -> item_content   (the submission's field name)
  interactors        lives on RESPONSE, not on the item; empty in all four sources
  trial              repeated measurements of the same (subject, item)

On `trial`: the competition scores ONE realised outcome, so averaging over trials
would understate Brier. One trial is sampled per (subject, item) with a fixed
seed; trial_mode="first" gives a deterministic alternative for A/B runs.

Verified schema:
  subjects   subject_id display_name normalized_name provider release_date
             access_date harness reasoning_effort harness_version subject_features_extra
  items      item_id benchmark_id raw_item_id content asset_manifest
             grading_criterion verifier content_hash item_features
  response   subject_id item_id response_id benchmark_id trial test_condition
             interactors response(double, strictly {0,1} in all four sources)
  benchmarks 25 cols incl. response_type granularity n_subjects n_items coverage

Verified counts (eligible sources, one trial per cell):
  multi_swebench     82 subj  2126 items  57808 cells  base 0.148
  real_webagents     33 subj   233 items   3759 cells  base 0.372
  researchcodebench  31 subj   212 items   6572 cells  base 0.353
  swe_rebench         1 subj  6306 items   6306 cells  base 0.478   <- no subject effect
"""
from __future__ import annotations
import numpy as np, pyarrow.parquet as pq

ELIGIBLE = ("multi_swebench", "real_webagents", "researchcodebench", "swe_rebench")
# swe_rebench has ONE subject: keep it for item difficulty, exclude it from any
# cross-subject fit and from leave-one-benchmark-out.
CROSS_SUBJECT = ("multi_swebench", "real_webagents", "researchcodebench")

def load(root, sources=ELIGIBLE, seed=0, trial_mode="sample", with_content=False):
    rng = np.random.default_rng(seed); data = {}
    for s in sources:
        b = pq.read_table(f"{root}/{s}/benchmarks.parquet").to_pandas()
        assert (b.response_type == "binary").all() and (b.granularity == "item").all(), s
        subj = pq.read_table(f"{root}/{s}/subjects.parquet").to_pandas()
        cols = ["item_id", "benchmark_id", "item_features"] + (["content"] if with_content else [])
        it = pq.read_table(f"{root}/{s}/items.parquet", columns=cols).to_pandas()
        rs = pq.read_table(f"{root}/{s}/response.parquet",
                           columns=["subject_id", "item_id", "benchmark_id", "trial", "response"]).to_pandas()
        assert set(rs.response.unique()) <= {0., 1.}, s
        if trial_mode == "first":
            rs = rs.sort_values("trial").drop_duplicates(["subject_id", "item_id"], keep="first")
        else:
            rs = rs.iloc[rng.permutation(len(rs))].drop_duplicates(["subject_id", "item_id"], keep="first")
        subjects = {r.subject_id: {"normalized_name": r.normalized_name, "provider": r.provider,
                    "release_date": r.release_date, "access_date": r.access_date, "harness": r.harness,
                    "reasoning_effort": r.reasoning_effort, "harness_version": r.harness_version,
                    "subject_features_extra": r.subject_features_extra} for r in subj.itertuples()}
        items = {r.item_id: {"benchmark_id": r.benchmark_id,
                 "item_content": (r.content if with_content else ""),
                 "item_features": r.item_features, "interactors": ""} for r in it.itertuples()}
        truth, by_subject = {}, {}
        for r in rs.itertuples():
            truth[(r.subject_id, r.item_id)] = int(r.response)
            by_subject.setdefault(r.subject_id, []).append(r.item_id)
        data[s] = {"subjects": subjects, "items": items, "truth": truth, "by_subject": by_subject}
    return data

"""Download the competition-eligible tables of the public measurement-db.

Mirrors ``paiec_baseline/tools/prepare_data.py``: resolve the current ``main``
revision of ``aims-foundations/measurement-db`` once, keep benchmarks whose
``response_type`` is ``binary`` and ``granularity`` is ``item``, and copy their
``benchmarks``, ``subjects``, ``items`` and response tables. Traces and
embeddings are not needed and are skipped. The revision is recorded in
``corpus_manifest.json`` for provenance.

    python scripts/download_measurement_db.py --out data/raw
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import shutil
import time
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

REPO = "aims-foundations/measurement-db"
TABLES = ("benchmarks", "subjects", "items", "response")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    info = HfApi().repo_info(REPO, repo_type="dataset", revision="main")
    revision = info.sha
    files = {s.rfilename for s in info.siblings}
    dirs = sorted({f.rsplit("/", 1)[0] for f in files if f.endswith("/benchmarks.parquet")})
    print(f"{REPO}@{revision}: {len(dirs)} benchmark directories")

    def fetch(name):
        return hf_hub_download(REPO, name, repo_type="dataset", revision=revision)

    def eligible(d):
        row = pd.read_parquet(fetch(f"{d}/benchmarks.parquet")).iloc[0]
        return d, bool(row["response_type"] == "binary" and row["granularity"] == "item"), row.to_dict()

    metas, keep = {}, []
    with cf.ThreadPoolExecutor(args.workers) as ex:
        for d, ok, row in ex.map(eligible, dirs):
            metas[d] = {k: str(v) for k, v in row.items()}
            if ok:
                keep.append(d)
    print(f"{len(keep)} eligible (binary, item granularity)")

    def copy_dir(d):
        got = []
        (out / d).mkdir(parents=True, exist_ok=True)
        for t in TABLES:
            name = f"{d}/{t}.parquet"
            if t == "response" and name not in files:
                name = f"{d}/responses.parquet"
            if name not in files:
                continue
            dst = out / d / f"{t}.parquet"
            src = fetch(name)
            if not dst.exists() or dst.stat().st_size != Path(src).stat().st_size:
                shutil.copyfile(src, dst)
            got.append(name)
        return got

    t0 = time.time()
    copied = []
    with cf.ThreadPoolExecutor(args.workers) as ex:
        for i, got in enumerate(ex.map(copy_dir, keep), 1):
            copied += got
            if i % 20 == 0 or i == len(keep):
                print(f"  {i}/{len(keep)} ({time.time() - t0:.0f}s)")
    (out / "corpus_manifest.json").write_text(json.dumps(
        {"db_repo": REPO, "revision": revision, "generated_unix": int(time.time()),
         "all_dirs": dirs, "eligible_dirs": keep, "files": sorted(copied), "benchmark_rows": metas},
        indent=1))
    size = sum(p.stat().st_size for p in out.rglob("*.parquet")) / 1e9
    print(f"done: {len(keep)} benchmarks, {size:.2f} GB in {out}")


if __name__ == "__main__":
    main()

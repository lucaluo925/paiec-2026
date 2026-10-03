"""Fit the submission parameters on every eligible public benchmark.

    python -m train.train_final --data data/raw --out submission/params.json \
        [--tuned results/tuned_hyper.json]

``--tuned`` merges hyper-parameters and shrinkage weights selected by local
cross-validation (``eval/tune.py``) over the offline estimates.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.data import load_db  # noqa: E402
from eval.run_cv import deep_update  # noqa: E402
from train.fit_offline import fit  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=str(ROOT / "submission" / "params.json"))
    ap.add_argument("--tuned", help="JSON patch with tuned 'hyper' and 'shrink' entries")
    args = ap.parse_args(argv)
    t0 = time.time()
    db = load_db(args.data)
    params, _ = fit(db, sorted(db))
    if args.tuned:
        tuned = json.loads(Path(args.tuned).read_text())
        for key, scale in tuned.get("hyper_scale", {}).items():
            if key not in params["hyper"]:
                raise SystemExit(f"tuned scale for unknown hyper-parameter {key!r}")
            params["hyper"][key] *= float(scale)
        params = deep_update(params, {k: v for k, v in tuned.items() if k in ("shrink", "tuning")})
        params["hyper_scale"] = tuned.get("hyper_scale", {})
    params["provenance"] = {
        "training_data": "aims-foundations/measurement-db (public), eligible binary item-level benchmarks",
        "n_benchmarks": len(db),
        "n_responses": int(sum(len(b.resp) for b in db.values())),
    }
    if (Path(args.data) / "truth.json").exists():
        params["provenance"]["synthetic"] = True  # build_submission_zip refuses these parameters
    manifest = Path(args.data) / "corpus_manifest.json"
    if manifest.exists():
        params["provenance"]["revision"] = json.loads(manifest.read_text()).get("revision")
    out = Path(args.out)
    out.write_text(json.dumps(params, indent=1, sort_keys=True))
    print(f"wrote {out} ({out.stat().st_size:,} bytes) from {len(db)} benchmarks in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

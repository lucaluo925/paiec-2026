"""Load measurement-db benchmark directories into compact evaluation tables.

Each directory follows the public layout
``<benchmark_dir>/{benchmarks,subjects,items,response}.parquet`` (the upstream
builder spelling ``responses.parquet`` is also accepted). Only benchmarks with
``response_type == "binary"`` and ``granularity == "item"`` are competition
eligible, matching ``paiec_baseline/tools/prepare_data.py``.

Inputs are converted to the competition's visible format:

* subject: the eight string attributes listed in ``SUBJECT_FIELDS``
  (missing values become ``""``, as in the organizers' smoke inputs);
* item: ``item_content``, ``item_features``, ``interactors`` and an anonymous
  ``benchmark_id``.

Responses are reduced to one binary observation per (subject, item,
interactors) triple: the lowest (test_condition, trial) row with a 0/1 grade.
The visible input cannot distinguish trials or test conditions, so keeping one
row avoids duplicate targets with identical inputs.
"""

from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "submission"))
from pirt_online import model_identity  # noqa: E402,F401  (one identity rule for training and runtime)

SUBJECT_FIELDS = (
    "normalized_name",
    "provider",
    "release_date",
    "access_date",
    "harness",
    "reasoning_effort",
    "harness_version",
    "subject_features_extra",
)


def _s(value) -> str:
    """String form of a table cell; None, NaN and pd.NA become ""."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):  # array-like cells
        pass
    return str(value)


def _str_codes(series: pd.Series) -> pd.Categorical:
    """Categorical of the column as strings, with missing values (None/NaN/NA) as ""."""
    return pd.Categorical(series.astype("string").fillna("").astype(str))


def anon_id(prefix: str, key: str, salt: str = "paiec-local") -> str:
    digest = hashlib.sha256(f"{salt}:{prefix}:{key}".encode()).hexdigest()
    return f"{prefix}_{int(digest[:12], 16) % 1_000_000:06d}"


@dataclass
class Benchmark:
    key: str
    anon: str
    meta: dict
    subjects: dict  # subject_id -> visible subject dict
    items: dict  # item_id -> {"item_content", "item_features"}
    # One row per (subject_id, item_id, interactors), categorical ids; y is 0/1 int8.
    resp: pd.DataFrame = field(repr=False)

    @property
    def n_items(self) -> int:
        return int(self.resp["item_id"].nunique())

    def item_input(self, item_id: str, interactors: str) -> dict:
        item = self.items[item_id]
        return {
            "item_content": item["item_content"],
            "item_features": item["item_features"],
            "interactors": interactors,
            "benchmark_id": self.anon,
        }


def _response_path(directory: Path) -> Path | None:
    for name in ("response.parquet", "responses.parquet"):
        if (directory / name).exists():
            return directory / name
    return None


def read_meta(directory: Path) -> dict | None:
    path = directory / "benchmarks.parquet"
    if not path.exists():
        return None
    row = pd.read_parquet(path).iloc[0].to_dict()
    out = {}
    for k, v in row.items():
        if isinstance(v, np.ndarray):
            v = [str(x) for x in v.tolist()]
        elif isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.floating,)):
            v = float(v)
        elif isinstance(v, (np.bool_,)):
            v = bool(v)
        out[k] = v
    return out


def is_eligible(meta: dict | None) -> bool:
    return bool(meta) and meta.get("response_type") == "binary" and meta.get("granularity") == "item"


def load_benchmark(directory: str | os.PathLike) -> Benchmark | None:
    directory = Path(directory)
    meta = read_meta(directory)
    if not is_eligible(meta):
        return None
    rpath = _response_path(directory)
    if rpath is None:
        return None
    resp = pd.read_parquet(
        rpath, columns=["subject_id", "item_id", "trial", "test_condition", "interactors", "response"]
    )
    resp = resp[resp["response"].isin([0.0, 1.0])]
    if resp.empty:
        return None
    # Categorical codes keep multi-million-row tables small and make the
    # de-duplication sort integer-based.
    cols = {}
    for c in ("subject_id", "item_id", "interactors", "test_condition"):
        cols[c] = _str_codes(resp[c])
    order = np.lexsort((
        resp["trial"].to_numpy(),
        cols["test_condition"].codes,
        cols["interactors"].codes,
        cols["item_id"].codes,
        cols["subject_id"].codes,
    ))
    key = np.stack([cols[c].codes[order].astype(np.int64) for c in ("subject_id", "item_id", "interactors")])
    first = np.ones(len(order), dtype=bool)
    first[1:] = (key[:, 1:] != key[:, :-1]).any(axis=0)
    keep = order[first]
    resp = pd.DataFrame(
        {
            "subject_id": cols["subject_id"][keep],
            "item_id": cols["item_id"][keep],
            "interactors": cols["interactors"][keep],
            "y": resp["response"].to_numpy()[keep].astype(np.int8),
        }
    )

    subj = pd.read_parquet(directory / "subjects.parquet")
    subjects = {}
    for row in subj.to_dict("records"):
        subjects[str(row["subject_id"])] = {f: _s(row.get(f)) for f in SUBJECT_FIELDS}
    items_df = pd.read_parquet(directory / "items.parquet", columns=["item_id", "content", "item_features"])
    used = set(resp["item_id"].cat.categories)
    items = {}
    for iid, content, feats in zip(items_df["item_id"], items_df["content"], items_df["item_features"]):
        iid = str(iid)
        if iid in used:
            items[iid] = {"item_content": _s(content), "item_features": _s(feats)}
    resp = resp[resp["item_id"].isin(items.keys()) & resp["subject_id"].isin(subjects.keys())]
    resp = resp.reset_index(drop=True)
    for c in ("subject_id", "item_id", "interactors"):
        resp[c] = resp[c].cat.remove_unused_categories()
    if resp.empty:
        return None
    key = directory.name
    return Benchmark(key=key, anon=anon_id("benchmark", key), meta=meta, subjects=subjects, items=items, resp=resp)


def list_benchmark_dirs(root: str | os.PathLike) -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "benchmarks.parquet").exists())


def load_db(root: str | os.PathLike, only: list[str] | None = None, verbose: bool = False) -> dict[str, Benchmark]:
    db = {}
    for directory in list_benchmark_dirs(root):
        if only is not None and directory.name not in only:
            continue
        bench = load_benchmark(directory)
        if bench is not None:
            db[bench.key] = bench
            if verbose:
                print(f"  loaded {bench.key}: {len(bench.resp):,} responses, {bench.n_items:,} items")
    return db

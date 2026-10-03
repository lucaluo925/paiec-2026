"""Local replica of the competition's streaming Brier-ALC evaluation.

Protocol (``streaming_alc_v1`` as implemented by the organizers'
``paiec_baseline/tools/streaming_ingestion.py``, plus the published scoring
rule):

* Label budgets 0, 1, 3, 7, 15, 31 with ALC weights
  0.1, 0.2, 0.2, 0.2, 0.2, 0.1.
* Only benchmarks with at least 80 items are evaluated. Each benchmark's items
  are split 50/50 into an acquisition pool and an evaluation pool.
* For every (subject, benchmark) pair the coordinator streams the pair's
  acquisition-pool items one at a time. ``max_labels`` is 31. Without a
  ``labeling.py`` the default policy queries an item when
  ``sha256(json([context, input]))`` maps below
  ``labels_remaining / items_remaining`` (selection sampling). A streaming hook
  ``acquisition_function(input, prediction, labeled, context) -> bool`` decides
  instead; a one-argument hook returns a static priority used to rank the pool.
* Every pair is advanced to budget b before the checkpoint at b. All
  evaluation-pool targets are then predicted with the labels acquired so far.
  By default ``labeled`` holds the labels of all pairs (the baselines filter it
  by subject and benchmark); ``labeled_scope="pair"`` passes only the target
  pair's labels, as a pessimistic check.

Items the coordinator never sends are never scored, so a prediction is always a
function of attributes plus acquired labels.

Unverified details (the rules page was not reachable from this environment)
are exposed as ``ProtocolConfig`` options, and every report records them.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from .data import Benchmark, anon_id, model_identity

BUDGETS = (0, 1, 3, 7, 15, 31)
# ALC weights in tenths, so a constant 0.5 predictor scores exactly 0.25.
WEIGHTS_X10 = {0: 1, 1: 2, 3: 2, 7: 2, 15: 2, 31: 1}
WEIGHTS = {b: w / 10 for b, w in WEIGHTS_X10.items()}
MAX_LABELS = 31
assert sum(WEIGHTS_X10.values()) == 10


@dataclass
class ProtocolConfig:
    min_items: int = 80
    # Official rules: "For each subject-benchmark pair with at least 80 distinct
    # items, items are randomly split 50/50 into acquisition and evaluation pools."
    split_mode: str = "pair"  # one 50/50 split per subject-benchmark pair, or "benchmark" to share it
    labeled_scope: str = "global"  # "global" (all acquired labels) or "pair"
    max_subjects_per_benchmark: int | None = 40
    max_eval_per_pair: int | None = 60
    max_stream_per_pair: int | None = 400
    seed: int = 0


@dataclass
class Pair:
    bench_key: str
    bench_anon: str
    subject_id: str
    subject_anon: str
    subject: dict
    identity: str
    stream: list  # [(item_input, y)] in coordinator order
    targets: list  # [(item_input, y)]
    pos: int = 0
    acquired: list = field(default_factory=list)  # [[[subject, item], y], ...]


def _seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def build_pairs(db: dict[str, Benchmark], bench_keys, cfg: ProtocolConfig) -> list[Pair]:
    pairs = []
    for key in sorted(bench_keys):
        bench = db[key]
        items = np.array(sorted(bench.resp["item_id"].unique().astype(str)))
        if len(items) < cfg.min_items:
            continue
        rng = np.random.default_rng(_seed(cfg.seed, "split", key))
        perm = rng.permutation(len(items))
        acq_items = set(items[perm[: len(items) // 2]])

        by_subject = {str(sid): g for sid, g in bench.resp.groupby("subject_id", sort=True, observed=True)}
        sids = sorted(by_subject)
        if cfg.max_subjects_per_benchmark and len(sids) > cfg.max_subjects_per_benchmark:
            srng = np.random.default_rng(_seed(cfg.seed, "subjects", key))
            sids = sorted(srng.choice(sids, size=cfg.max_subjects_per_benchmark, replace=False))
        for sid in sids:
            g = by_subject[sid]
            if cfg.split_mode == "pair":
                prng = np.random.default_rng(_seed(cfg.seed, "pair-split", key, sid))
                pperm = prng.permutation(len(items))
                acq = set(items[pperm[: len(items) // 2]])
            else:
                acq = acq_items
            rows = list(zip(g["item_id"].astype(str).tolist(), g["interactors"].astype(str).tolist(),
                            g["y"].tolist()))
            s_rows = [r for r in rows if r[0] in acq]
            e_rows = [r for r in rows if r[0] not in acq]
            if not e_rows:
                continue
            prng = np.random.default_rng(_seed(cfg.seed, "order", key, sid))
            s_rows = [s_rows[i] for i in prng.permutation(len(s_rows))]
            if cfg.max_stream_per_pair:
                s_rows = s_rows[: cfg.max_stream_per_pair]
            if cfg.max_eval_per_pair and len(e_rows) > cfg.max_eval_per_pair:
                idx = prng.choice(len(e_rows), size=cfg.max_eval_per_pair, replace=False)
                e_rows = [e_rows[i] for i in sorted(idx)]
            subject = dict(bench.subjects[sid])
            pairs.append(
                Pair(
                    bench_key=key,
                    bench_anon=bench.anon,
                    subject_id=sid,
                    subject_anon=anon_id("subject", f"{key}:{sid}"),
                    subject=subject,
                    identity=model_identity(subject),
                    stream=[(bench.item_input(i, it), int(y)) for i, it, y in s_rows],
                    targets=[(bench.item_input(i, it), int(y)) for i, it, y in e_rows],
                )
            )
    return pairs


def random_policy_query(context: dict, current_input: list) -> bool:
    """Byte-for-byte the organizers' default deterministic random policy."""
    key = json.dumps([context, current_input], sort_keys=True, separators=(",", ":"))
    uniform = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2**64
    return uniform < min(1, context["labels_remaining"] / context["items_remaining"])


class Hook:
    """Call-shape detection copied from ``AcquisitionHook`` (organizer code).

    Unlike the organizer hook it does not deep-copy arguments; local runs pass
    the same objects and ``tests`` check that predictors do not mutate them.
    """

    def __init__(self, function: Callable | None):
        self.function = function
        self.mode, self.positional, self.keywords = "random", False, ()
        if function is None:
            return
        signature = inspect.signature(function)
        names = ("prediction", "labeled", "context")
        has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
        keywords = tuple(
            n for n in names
            if has_kwargs or (n in signature.parameters
                              and signature.parameters[n].kind != inspect.Parameter.POSITIONAL_ONLY)
        )
        shapes = [(False, names), (True, ())]
        if keywords and keywords != names:
            shapes.append((False, keywords))
        for positional, supplied in shapes:
            try:
                if positional:
                    signature.bind(None, None, None, None)
                else:
                    signature.bind(None, **{n: None for n in supplied})
            except TypeError:
                continue
            self.mode, self.positional, self.keywords = "stream", positional, supplied
            return
        signature.bind(None)
        self.mode = "legacy_rank"

    @property
    def needs_prediction(self) -> bool:
        return self.positional or "prediction" in self.keywords

    def __call__(self, input, prediction=None, labeled=None, context=None):
        if self.mode == "legacy_rank":
            return float(self.function(input))
        values = {"prediction": prediction, "labeled": labeled, "context": context}
        if self.positional:
            result = self.function(input, prediction, labeled, context)
        else:
            result = self.function(input, **{k: values[k] for k in self.keywords})
        if type(result) is not bool:
            raise ValueError("streaming acquisition_function must return a bool")
        return result


def _validate(p) -> float:
    if isinstance(p, bool):
        raise ValueError("predict() returned a bool")
    p = float(p)
    if not math.isfinite(p) or p < 0.0 or p > 1.0:
        raise ValueError(f"predict() returned {p!r}, outside [0, 1]")
    return p


@dataclass
class RunResult:
    config: dict
    policy: str
    brier: dict  # budget -> micro Brier
    brier_pair_macro: dict
    brier_bench_macro: dict
    logloss: dict
    alc: float
    alc_pair_macro: float
    alc_bench_macro: float
    n_targets: int
    n_pairs: int
    n_benchmarks: int
    acquired_mean: dict
    seconds: float
    records: dict = field(default_factory=dict, repr=False)  # budget -> arrays
    trajectory: tuple = field(default=None, repr=False)  # (labeled, labeled_at)

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("records")
        d.pop("trajectory")
        return d


def per_pair_brier(sq, pair_ids, n_pairs: int):
    """Per-pair mean squared error plus a mask of pairs that have any target.

    The official metric averages Brier within each subject-benchmark pair and
    then across pairs with equal weight, so this is the building block for it.
    """
    counts = np.bincount(pair_ids, minlength=n_pairs)
    totals = np.bincount(pair_ids, weights=sq, minlength=n_pairs)
    return totals / np.maximum(counts, 1), counts > 0


def pair_macro_alc(p, y, pair_ids, n_pairs: int, budgets=BUDGETS) -> tuple[float, dict]:
    """ALC and per-budget Brier under the official pair-macro aggregation.

    ``p``/``y``/``pair_ids`` are dicts keyed by budget; pair ids must be unique
    across every fold that is pooled together.
    """
    brier = {}
    for b in budgets:
        per_pair, has = per_pair_brier((p[b] - y[b]) ** 2, pair_ids[b], n_pairs)
        brier[b] = float(per_pair[has].mean())
    return _alc(brier), brier


def _alc(brier: dict) -> float:
    return math.fsum(WEIGHTS_X10[b] * brier[b] for b in BUDGETS) / 10


def run_protocol(
    pairs: list[Pair],
    predict: Callable,
    acquisition: Callable | None = None,
    cfg: ProtocolConfig | None = None,
    keep_records: bool = True,
    progress: bool = False,
) -> RunResult:
    cfg = cfg or ProtocolConfig()
    hook = Hook(acquisition)
    t0 = time.time()
    for p in pairs:
        p.pos, p.acquired = 0, []
    labeled: list = []
    order = [pairs[i] for i in np.random.default_rng(_seed(cfg.seed, "pair-order")).permutation(len(pairs))]

    # Coordinator order of each pair's stream; a one-argument (legacy) hook ranks
    # the pool once by its static priority. Pairs themselves are not modified.
    streams = {}
    for p in pairs:
        stream = p.stream
        if hook.mode == "legacy_rank":
            scores = [hook([p.subject, item]) for item, _ in stream]
            stream = [stream[i] for i in sorted(range(len(stream)), key=lambda i: (-scores[i], i))]
        streams[id(p)] = stream

    records = {}
    labeled_at = {}
    for b in BUDGETS:
        if b > 0:
            for p in order:
                view = labeled if cfg.labeled_scope == "global" else p.acquired
                stream = streams[id(p)]
                while len(p.acquired) < b and p.pos < len(stream):
                    item, y = stream[p.pos]
                    current = [p.subject, item]
                    if hook.mode == "legacy_rank":
                        query = True
                    else:
                        context = {
                            "subject_id": p.subject_anon,
                            "benchmark_id": p.bench_anon,
                            "labels_remaining": MAX_LABELS - len(p.acquired),
                            "labels_acquired": len(p.acquired),
                            "max_labels": MAX_LABELS,
                            "items_remaining": len(stream) - p.pos,
                        }
                        if hook.mode == "random":
                            query = random_policy_query(context, current)
                        else:
                            prediction = _validate(predict(current, view)) if hook.needs_prediction else None
                            query = hook(current, prediction, view, context)
                    p.pos += 1
                    if query:
                        entry = [[p.subject, item], int(y)]
                        p.acquired.append(entry)
                        labeled.append(entry)
                        view = labeled if cfg.labeled_scope == "global" else p.acquired
        labeled_at[b] = len(labeled)
        probs, ys, pair_idx = [], [], []
        for k, p in enumerate(pairs):
            view = labeled if cfg.labeled_scope == "global" else p.acquired
            for item, y in p.targets:
                probs.append(_validate(predict([p.subject, item], view)))
                ys.append(y)
                pair_idx.append(k)
        records[b] = (np.array(probs), np.array(ys, dtype=np.int8), np.array(pair_idx))
        if progress:
            pb = np.array(probs)
            print(f"  budget {b:>2}: Brier {np.mean((pb - np.array(ys)) ** 2):.5f}  ({time.time() - t0:.0f}s)")

    return _summarise(pairs, records, cfg, hook.mode if acquisition is not None else "random(default)",
                      time.time() - t0, keep_records, labeled, labeled_at)


def evaluate_fixed(pairs: list[Pair], trajectory: tuple, predict: Callable, cfg: ProtocolConfig | None = None,
                   keep_records: bool = True) -> RunResult:
    """Re-score a recorded acquisition trajectory with another predictor.

    ``trajectory = (labeled, labeled_at)`` from a previous ``run_protocol`` with a
    predictor-independent (random) policy; each pair's ``acquired`` list must be
    intact. Gives exactly the predictions a fresh ``run_protocol`` would.
    """
    cfg = cfg or ProtocolConfig()
    labeled, labeled_at = trajectory
    t0 = time.time()
    records = {}
    for b in BUDGETS:
        view_global = labeled[: labeled_at[b]]
        probs, ys, pair_idx = [], [], []
        for k, p in enumerate(pairs):
            view = view_global if cfg.labeled_scope == "global" else p.acquired[:b]
            for item, y in p.targets:
                probs.append(_validate(predict([p.subject, item], view)))
                ys.append(y)
                pair_idx.append(k)
        records[b] = (np.array(probs), np.array(ys, dtype=np.int8), np.array(pair_idx))
    return _summarise(pairs, records, cfg, "replayed", time.time() - t0, keep_records, labeled, labeled_at)


def _summarise(pairs, records, cfg, policy, seconds, keep_records, labeled, labeled_at) -> "RunResult":
    bench_of = np.array([p.bench_key for p in pairs])
    brier, pmacro, bmacro, logloss, acq_mean = {}, {}, {}, {}, {}
    for b in BUDGETS:
        pr, y, pi = records[b]
        sq = (pr - y) ** 2
        brier[b] = float(sq.mean())
        per_pair, has = per_pair_brier(sq, pi, len(pairs))
        pmacro[b] = float(per_pair[has].mean())
        benches = bench_of[pi]
        bmacro[b] = float(np.mean([sq[benches == k].mean() for k in np.unique(benches)]))
        pc = np.clip(pr, 1e-6, 1 - 1e-6)
        logloss[b] = float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean())
    # Acquired counts are cumulative, so report them at the final checkpoint.
    for b in BUDGETS:
        acq_mean[b] = float(np.mean([min(len(p.acquired), b) for p in pairs]))
    return RunResult(
        config=asdict(cfg),
        policy=policy,
        brier=brier,
        brier_pair_macro=pmacro,
        brier_bench_macro=bmacro,
        logloss=logloss,
        alc=_alc(brier),
        alc_pair_macro=_alc(pmacro),
        alc_bench_macro=_alc(bmacro),
        n_targets=int(len(records[0][0])),
        n_pairs=len(pairs),
        n_benchmarks=len(set(p.bench_key for p in pairs)),
        acquired_mean=acq_mean,
        seconds=seconds,
        records=records if keep_records else {},
        trajectory=(labeled, labeled_at),
    )


def bootstrap_diff(runs_a, runs_b, n_boot: int = 2000, seed: int = 0, cluster: str = "benchmark"):
    """Paired bootstrap of ALC(a) - ALC(b) under the official pair-macro metric.

    ``runs_a`` / ``runs_b`` are lists of ``(RunResult, pairs)``, one entry per
    fold, in the same fold order and over identical pair lists, so the two
    variants line up pair by pair and the bootstrap stays paired.

    ``cluster="benchmark"`` resamples whole benchmarks: that is the
    pre-registered decision rule, and the only level that respects the
    correlation between pairs of the same benchmark. ``cluster="pair"``
    resamples subject-benchmark pairs; it is far finer but ignores that
    correlation, so its interval is optimistic and is a diagnostic only.
    Returns ``(delta, lo, hi)``; delta is negative when a beats b.
    """
    if [len(pp) for _, pp in runs_a] != [len(pp) for _, pp in runs_b]:
        raise ValueError("runs_a and runs_b must cover the same folds and pairs")

    def terms(runs):
        sse = {b: [] for b in BUDGETS}
        cnt = {b: [] for b in BUDGETS}
        for res, pairs in runs:
            for b in BUDGETS:
                pr, y, pi = res.records[b]
                sse[b].append(np.bincount(pi, weights=(pr - y) ** 2, minlength=len(pairs)))
                cnt[b].append(np.bincount(pi, minlength=len(pairs)))
        return ({b: np.concatenate(v) for b, v in sse.items()},
                {b: np.concatenate(v) for b, v in cnt.items()})

    bench = np.array([p.bench_key for _, pairs in runs_a for p in pairs])
    sse_a, cnt_a = terms(runs_a)
    sse_b, cnt_b = terms(runs_b)
    has = cnt_a[BUDGETS[0]] > 0

    def alc_of(sse, cnt, sel):
        return _alc({b: float(np.mean(sse[b][sel] / np.maximum(cnt[b][sel], 1))) for b in BUDGETS})

    if cluster == "benchmark":
        groups = [np.flatnonzero((bench == k) & has) for k in np.unique(bench)]
    elif cluster == "pair":
        groups = [np.array([i]) for i in np.flatnonzero(has)]
    else:
        raise ValueError(cluster)
    groups = [g for g in groups if len(g)]
    if not groups:
        raise ValueError("no pairs with targets to compare")
    every = np.concatenate(groups)
    delta = alc_of(sse_a, cnt_a, every) - alc_of(sse_b, cnt_b, every)

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        sel = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        diffs[i] = alc_of(sse_a, cnt_a, sel) - alc_of(sse_b, cnt_b, sel)
    return float(delta), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

"""Select hyper-parameter scales and shrinkage weights on local validation folds.

The random acquisition policy does not depend on the predictor, so one
trajectory per fold is recorded and every candidate is re-scored on the same
labels (``evaluate_fixed``). Variance hyper-parameters are tuned as
multiplicative scales of the offline estimates by coordinate search on the
pooled ALC, aggregated the official way (Brier averaged within each
subject-benchmark pair, then across pairs with equal weight). The final safety net mixes the prediction with the training base
rate, p' = (1 - w_b) p + w_b * base_rate; w_b is the least-squares optimum for
each label count b on the pooled validation predictions, clipped to [0, 1].

``--crossfit`` also reports an honest estimate: folds are split in two halves,
each half is scored with settings tuned on the other half.

    python -m eval.tune --data data/raw --out results/tuned_hyper.json
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

from eval import baselines  # noqa: E402
from eval.data import load_db  # noqa: E402
from eval.local_scorer import (  # noqa: E402
    BUDGETS, ProtocolConfig, build_pairs, evaluate_fixed, pair_macro_alc, run_protocol,
)
from eval.run_cv import deep_update  # noqa: E402
from eval.splits import make_split  # noqa: E402
from train.fit_offline import fit  # noqa: E402

import pirt_online  # noqa: E402

TUNED_KEYS = ("delta_var", "tau2", "beta_var", "a_var", "w_var")
FACTORS = (0.35, 0.6, 1.0, 1.6, 2.6)


def ship_base_rate(db) -> float:
    """train_final 打包时会用的那个 base_rate：全量数据、不排除任何 identity，
    按 subject-benchmark pair 等权。算法与 fit_offline.fit() 的那两行逐行一致。

    为什么需要它：w_b 是对**某一个** base 的最小二乘最优解
    (w = sum omega*d*(p-y) / sum omega*d^2, d = p - base)。原来 CV 里每折用的是
    该折训练集自己的 base_rate（实测 0.25162 / 0.25407 / 0.27447 / 0.34892），
    而出厂 params.json 配的是 0.28159 —— 选出来的权重和它要乘的那个常数
    不是在同一个取值下定下来的，w_0 = 0.535 时 B0 的落点能差 0.04。
    """
    from train.fit_offline import aggregate
    rows, _ = aggregate(db, sorted(db))
    per = [r[3] / r[2] for r in rows if r[2] > 0]
    return float(sum(per) / len(per)) if per else 0.5


def apply_scales(params: dict, scales: dict, shrink: dict | None = None,
                 base_rate: float | None = None) -> dict:
    hyper = {k: (v * scales.get(k, 1.0)) for k, v in params["hyper"].items()}
    sh: dict = {"weights": shrink or {"0": 0.0}}
    if base_rate is not None:
        sh["base_rate"] = float(base_rate)
    patch = {"hyper": hyper, "shrink": sh}
    return deep_update(params, patch)


class Folds:
    """base_rate=None 保留旧行为（每折用自己的 base_rate），用作 no-op 对照。

    传入 base_rate 时所有折共用打包会用的那个常数，于是 fit_shrink 选出来的 w
    与出厂配置里它要乘的那个 base 是同一个值。代价与 run_cv 的 --tuned 相同：
    该常数用到了被评估折的数据，是真实的乐观偏差；接受它，是因为这个台子回答的
    问题是「出厂模型表现如何」，不是「新调参模型的无偏估计」。
    """

    def __init__(self, db, split, folds, cfg, base_rate=None):
        self.base_rate = base_rate
        self.items = []
        # Pair-id offsets are fixed here so ids stay unique whatever subset is scored.
        self.n_pairs = 0
        for f in folds:
            if not split.folds[f]:
                continue
            params, _ = fit(db, split.train_keys(f), split.heldout_identities)
            pairs = build_pairs(db, split.folds[f], cfg)
            traj = run_protocol(pairs, baselines.constant(0.5), cfg=cfg, keep_records=False).trajectory
            self.items.append((f, params, pairs, traj, self.n_pairs))
            self.n_pairs += len(pairs)

    def score(self, scales: dict, shrink: dict | None = None, subset=None, cfg=None):
        recs = {b: ([], [], []) for b in BUDGETS}
        base = []
        for f, params, pairs, traj, offset in self.items:
            if subset is not None and f not in subset:
                continue
            br = (self.base_rate if self.base_rate is not None
                  else params["shrink"]["base_rate"])
            engine = pirt_online.Engine(
                apply_scales(params, scales, shrink, base_rate=self.base_rate))
            res = evaluate_fixed(pairs, traj, engine.predict, cfg)
            for b in BUDGETS:
                recs[b][0].append(res.records[b][0])
                recs[b][1].append(res.records[b][1])
                recs[b][2].append(np.asarray(res.records[b][2]) + offset)
            # 收缩目标必须与**引擎实际用的**那个 base 一致，否则 fit_shrink 解出来的
            # w 对应的是另一个常数。
            base.append(np.full(len(res.records[0][0]), br))
        p = {b: np.concatenate(recs[b][0]) for b in BUDGETS}
        y = {b: np.concatenate(recs[b][1]) for b in BUDGETS}
        pid = {b: np.concatenate(recs[b][2]) for b in BUDGETS}
        alc, brier = pair_macro_alc(p, y, pid, self.n_pairs)
        return alc, brier, p, y, np.concatenate(base), pid


def fit_shrink(p: dict, y: dict, base: np.ndarray, pid: dict) -> dict:
    """Least-squares shrinkage weight per budget, weighted the way the metric averages.

    The official score averages squared error inside a subject-benchmark pair and
    then across pairs with equal weight, so each prediction of a pair with n
    targets contributes 1/n. Solving the same weighted least squares here keeps
    the selected w consistent with what it is scored on; plain pooling would let
    the pairs with the most targets choose w for everyone.
    """
    weights = {}
    for b in BUDGETS:
        d = p[b] - base
        n = np.bincount(pid[b], minlength=int(pid[b].max()) + 1)[pid[b]]
        omega = 1.0 / np.maximum(n, 1)
        den = float(np.sum(omega * d * d))
        w = float(np.sum(omega * d * (p[b] - y[b])) / den) if den > 0 else 0.0
        weights[str(b)] = min(max(w, 0.0), 1.0)
    return weights


def coordinate_search(folds: Folds, subset=None, rounds: int = 2, cfg=None, log=print, start=None):
    scales = dict(start) if start else {k: 1.0 for k in TUNED_KEYS}
    best, *_ = folds.score(scales, subset=subset, cfg=cfg)
    log(f"  start ALC {best:.5f}")
    for r in range(rounds):
        changed = False
        for key in TUNED_KEYS:
            cur = scales[key]
            for fac in FACTORS:
                if fac == 1.0:
                    continue
                trial = dict(scales, **{key: cur * fac})
                alc, *_ = folds.score(trial, subset=subset, cfg=cfg)
                if alc < best - 1e-6:
                    best, scales, changed = alc, trial, True
            log(f"  round {r} {key:<10} scale {scales[key]:.3f}  ALC {best:.5f}")
        if not changed:
            break
    return scales, best


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heldout-frac", type=float, default=0.2)
    ap.add_argument("--max-subjects", type=int, default=40)
    ap.add_argument("--max-eval", type=int, default=60)
    ap.add_argument("--max-stream", type=int, default=400)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--scales", help="tuned_hyper.json to start (and with --rounds 0, keep) the scales from. "
                                     "Use when the protocol is too expensive to search but the shrinkage "
                                     "weights still have to be fitted and cross-fitted under it.")
    ap.add_argument("--crossfit", action="store_true")
    ap.add_argument("--legacy-fold-base-rate", action="store_true",
                    help="no-op 对照：每折仍用自己训练集的 base_rate（修复前的行为）。"
                         "只用来验证改动在旧口径下逐位复现，不要用它产出结论。")
    ap.add_argument("--out", default="results/tuned_hyper.json")
    args = ap.parse_args(argv)
    t0 = time.time()
    db = load_db(args.data)
    split = make_split(db, k=args.folds, seed=args.seed, heldout_frac=args.heldout_frac)
    cfg = ProtocolConfig(max_subjects_per_benchmark=args.max_subjects, max_eval_per_pair=args.max_eval,
                         max_stream_per_pair=args.max_stream, seed=args.seed)
    br = None if args.legacy_fold_base_rate else ship_base_rate(db)
    folds = Folds(db, split, range(args.folds), cfg, base_rate=br)
    print(f"prepared {len(folds.items)} folds in {time.time() - t0:.0f}s")
    print("收缩目标 base_rate = "
          + ("每折自己的（legacy 对照）" if br is None else f"{br:.6f}（出厂口径）"))

    start = json.loads(Path(args.scales).read_text())["hyper_scale"] if args.scales else None
    default_alc, default_brier, *_ = (folds.score(start or {}, cfg=cfg))
    scales, tuned_alc = coordinate_search(folds, rounds=args.rounds, cfg=cfg, start=start)
    _, _, p, y, base, pid = folds.score(scales, cfg=cfg)
    shrink = fit_shrink(p, y, base, pid)
    final_alc, final_brier, *_ = folds.score(scales, shrink, cfg=cfg)
    result = {
        "hyper_scale": scales,
        # base_rate 和 weights 必须成对使用：w 是对这个 base 解出来的。
        # train_final 会 deep_update 整个 shrink，所以写进来即可随包出厂。
        "shrink": ({"weights": shrink} if br is None
                   else {"weights": shrink, "base_rate": br}),
        "tuning": {
            "alc_offline_defaults": default_alc, "brier_offline_defaults": default_brier,
            "alc_tuned_scales": tuned_alc, "alc_tuned_scales_and_shrink": final_alc,
            "brier_tuned": final_brier, "protocol": vars(args),
        },
    }
    if args.crossfit:
        ids = [f for f, *_ in folds.items]
        halves = [set(ids[::2]), set(ids[1::2])]
        cross = {b: ([], [], []) for b in BUDGETS}
        for tune_on, test_on in ((halves[0], halves[1]), (halves[1], halves[0])):
            s, _ = coordinate_search(folds, subset=tune_on, rounds=args.rounds, cfg=cfg,
                                     log=lambda *_: None, start=start)
            _, _, pt, yt, bt, pidt = folds.score(s, subset=tune_on, cfg=cfg)
            w = fit_shrink(pt, yt, bt, pidt)
            _, _, pe, ye, _, pide = folds.score(s, w, subset=test_on, cfg=cfg)
            for b in BUDGETS:
                cross[b][0].append(pe[b])
                cross[b][1].append(ye[b])
                cross[b][2].append(pide[b])
        # The two halves score disjoint folds, so their pair ids never collide.
        cp = {b: np.concatenate(cross[b][0]) for b in BUDGETS}
        cy = {b: np.concatenate(cross[b][1]) for b in BUDGETS}
        cpid = {b: np.concatenate(cross[b][2]) for b in BUDGETS}
        crossfit_alc, brier = pair_macro_alc(cp, cy, cpid, folds.n_pairs)
        result["tuning"]["crossfit_alc"] = crossfit_alc
        result["tuning"]["crossfit_brier"] = brier
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps(result["tuning"] | {"hyper_scale": scales, "shrink": shrink}, indent=1, default=str))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

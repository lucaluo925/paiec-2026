"""Offline calibration on public measurement-db benchmarks.

Produces the parameter file consumed by ``submission/pirt_online.py``:

1. Subject ability. Per (benchmark, subject unit) accuracies are modelled as
   logit(acc) = a_B * theta_u - b_B + noise (weighted alternating least
   squares). A unit is a model name plus its harness/reasoning/extra settings;
   theta_u is shrunk towards its model's ability plus configuration effects.
2. Attribute model. Ridge regression of model abilities on provider, release
   date, parameter count and name tokens, used for models absent from training.
3. Item side. For every training benchmark, an item-level model
   eta = a * theta + v_s - beta - d_i is fitted by block-coordinate Newton with
   EM variance updates. Its per-benchmark discrimination, difficulty,
   subject-deviation and item-difficulty spreads define the hyperpriors used for
   a new benchmark, and the item difficulties are training targets for the
   optional text prior.
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "submission"))

from eval.data import Benchmark, model_identity  # noqa: E402
from pirt_online import (  # noqa: E402
    SubjectPrior, config_tokens, name_tokens, parse_size_b, parse_year,
)

DEFAULT_HYPER = {
    "a_mean": 1.0, "a_var": 0.25,
    "beta_mean": 0.0, "beta_var": 2.0,
    "tau2": 0.3, "delta_var": 2.0, "w_var": 0.5,
}


def _logit_counts(k, n):
    return np.log((k + 0.5) / (n - k + 0.5)), 1.0 / (k + 0.5) + 1.0 / (n - k + 0.5)


def aggregate(db: dict[str, Benchmark], train_keys, exclude_identities=frozenset()):
    """Rows of (benchmark, unit, identity, n, k) plus unit metadata."""
    rows = []
    units = {}
    for key in train_keys:
        bench = db[key]
        g = bench.resp.groupby("subject_id", observed=True)["y"].agg(["sum", "count"])
        for sid, (k, n) in g.iterrows():
            subject = bench.subjects[sid]
            ident = model_identity(subject)
            if ident in exclude_identities:
                continue
            # Unnamed subjects are their own unit, so unrelated models are never pooled.
            ukey = SubjectPrior.unit_key(subject) or ident
            units.setdefault(ukey, {"identity": ident, "subject": subject, "tokens": config_tokens(subject)})
            rows.append((key, ukey, int(n), int(k)))
    return rows, units


def fit_abilities(rows, units, n_iter: int = 60, sigma_res2: float = 0.15, unit_var: float = 0.15,
                  cfg_ridge: float = 4.0, verbose: bool = False):
    bench_keys = sorted({r[0] for r in rows})
    unit_keys = sorted(units)
    bi = {k: i for i, k in enumerate(bench_keys)}
    ui = {k: i for i, k in enumerate(unit_keys)}
    idents = sorted({units[u]["identity"] for u in unit_keys})
    ii = {k: i for i, k in enumerate(idents)}
    toks = sorted({t for u in unit_keys for t in units[u]["tokens"]})
    ti = {k: i for i, k in enumerate(toks)}

    B = np.array([bi[r[0]] for r in rows])
    U = np.array([ui[r[1]] for r in rows])
    n = np.array([r[2] for r in rows], dtype=float)
    k = np.array([r[3] for r in rows], dtype=float)
    ell, var_l = _logit_counts(k, n)
    w = 1.0 / (var_l + sigma_res2)
    unit_ident = np.array([ii[units[u]["identity"]] for u in unit_keys])
    T = np.zeros((len(unit_keys), len(toks)))
    for u in unit_keys:
        for t in units[u]["tokens"]:
            T[ui[u], ti[t]] = 1.0

    # Initialise with within-benchmark z-scores.
    z = np.zeros(len(rows))
    for b in range(len(bench_keys)):
        m = B == b
        sd = ell[m].std()
        z[m] = (ell[m] - ell[m].mean()) / (sd if sd > 1e-6 else 1.0)
    theta = np.bincount(U, weights=z, minlength=len(unit_keys)) / np.maximum(
        np.bincount(U, minlength=len(unit_keys)), 1)
    ident = np.zeros(len(idents))
    c = np.zeros(len(toks))
    a = np.ones(len(bench_keys))
    b0 = np.zeros(len(bench_keys))
    for it in range(n_iter):
        # (a_B, b_B) by weighted regression of ell on theta, ridge a -> 1.
        x = theta[U]
        sw = np.bincount(B, weights=w, minlength=len(bench_keys))
        swx = np.bincount(B, weights=w * x, minlength=len(bench_keys))
        swxx = np.bincount(B, weights=w * x * x, minlength=len(bench_keys))
        swy = np.bincount(B, weights=w * ell, minlength=len(bench_keys))
        swxy = np.bincount(B, weights=w * x * ell, minlength=len(bench_keys))
        lam = 1.0
        # Solve [[swxx+lam, -swx], [-swx, sw]] [a, b] = [swxy+lam, -swy]
        det = (swxx + lam) * sw - swx * swx
        a = ((swxy + lam) * sw - swx * swy) / det
        b0 = ((swxx + lam) * (-swy) + swx * (swxy + lam)) / det
        a = np.maximum(a, 0.05)
        # theta_u with prior mean ident + config effects.
        pm = ident[unit_ident] + T @ c
        num = np.bincount(U, weights=w * a[B] * (ell + b0[B]), minlength=len(unit_keys)) + pm / unit_var
        den = np.bincount(U, weights=w * a[B] ** 2, minlength=len(unit_keys)) + 1.0 / unit_var
        theta = num / den
        # Identity means (prior N(0, 1)) and config effects (ridge).
        resid = theta - T @ c
        ident = np.bincount(unit_ident, weights=resid / unit_var, minlength=len(idents)) / (
            np.bincount(unit_ident, minlength=len(idents)) / unit_var + 1.0)
        r2 = theta - ident[unit_ident]
        if len(toks):
            c = np.linalg.solve(T.T @ T + cfg_ridge * np.eye(len(toks)), T.T @ r2)
        # Fix location/scale on identities: mean 0, sd 1.
        mu, sd = ident.mean(), ident.std()
        sd = sd if sd > 1e-6 else 1.0
        ident = (ident - mu) / sd
        theta = (theta - mu) / sd
        c = c / sd
        if verbose and (it % 20 == 0 or it == n_iter - 1):
            pred = a[B] * theta[U] - b0[B]
            print(f"    ALS {it}: weighted rmse {np.sqrt(np.average((ell - pred) ** 2, weights=w)):.4f}")
    theta_var = 1.0 / den / sd ** 2
    n_units_per_ident = np.bincount(unit_ident, minlength=len(idents))
    ident_var = 1.0 / (n_units_per_ident / unit_var + 1.0)
    return {
        "bench_keys": bench_keys, "a": a, "b": b0,
        "units": {u: (float(theta[ui[u]]), float(theta_var[ui[u]])) for u in unit_keys},
        "identities": {name: (float(ident[ii[name]]), float(ident_var[ii[name]] + unit_var)) for name in idents},
        "config_effects": {t: float(c[ti[t]]) for t in toks},
        "config_var": float(unit_var),
        "unit_meta": units,
    }


def fit_attribute_model(abilities: dict, ridge: float = 3.0, token_min_count: int = 3):
    """theta_identity ~ provider + release year + log2 size + name tokens."""
    units = abilities["unit_meta"]
    by_ident = {}
    for u, meta in units.items():
        by_ident.setdefault(meta["identity"], meta["subject"])
    names = sorted(by_ident)
    y = np.array([abilities["identities"][nm][0] for nm in names])
    years = [parse_year(by_ident[nm].get("release_date")) for nm in names]
    sizes = [parse_size_b(by_ident[nm].get("normalized_name")) for nm in names]
    yv = np.array([v for v in years if v is not None])
    sv = np.array([math.log2(v) for v in sizes if v is not None])
    year_center = float(np.median(yv)) if len(yv) else 2024.5
    year_min = float(yv.min()) if len(yv) else 2020.0
    year_max = float(yv.max()) if len(yv) else 2027.0
    size_center = float(np.median(sv)) if len(sv) else 4.0
    providers = sorted({str(by_ident[nm].get("provider") or "").strip().lower() for nm in names} - {""})
    tok_count = defaultdict(int)
    for nm in names:
        for t in name_tokens(by_ident[nm].get("normalized_name")):
            tok_count[t] += 1
    tokens = sorted(t for t, cnt in tok_count.items() if cnt >= token_min_count)
    cols = ["year", "year_missing", "size", "size_missing"] + ["p:" + p for p in providers] + ["t:" + t for t in tokens]
    ci = {c: i for i, c in enumerate(cols)}
    X = np.zeros((len(names), len(cols)))
    for r, nm in enumerate(names):
        s = by_ident[nm]
        if years[r] is None:
            X[r, ci["year_missing"]] = 1.0
        else:
            X[r, ci["year"]] = years[r] - year_center
        if sizes[r] is None:
            X[r, ci["size_missing"]] = 1.0
        else:
            X[r, ci["size"]] = math.log2(sizes[r]) - size_center
        p = str(s.get("provider") or "").strip().lower()
        if p:
            X[r, ci["p:" + p]] = 1.0
        for t in name_tokens(s.get("normalized_name")):
            if "t:" + t in ci:
                X[r, ci["t:" + t]] = 1.0
    pen = np.full(len(cols), ridge)
    pen[ci["year"]] = pen[ci["size"]] = 0.1
    intercept = float(y.mean())
    coef = np.linalg.solve(X.T @ X + np.diag(pen), X.T @ (y - intercept))
    resid = y - intercept - X @ coef
    dof = max(len(names) - min(len(cols), len(names) // 2), 1)
    residual_var = float(np.sum(resid ** 2) / dof) if len(names) > 1 else 1.0
    return {
        "intercept": intercept,
        "year_slope": float(coef[ci["year"]]), "year_missing": float(coef[ci["year_missing"]]),
        "year_center": year_center, "year_min": year_min, "year_max": year_max,
        "size_slope": float(coef[ci["size"]]), "size_missing": float(coef[ci["size_missing"]]),
        "size_center": size_center,
        "provider": {p: float(coef[ci["p:" + p]]) for p in providers},
        "name_tokens": {t: float(coef[ci["t:" + t]]) for t in tokens},
        "residual_var": max(residual_var, 0.05),
        "n_identities": len(names),
    }


def fit_item_level(bench: Benchmark, theta_of, n_iter: int = 40, init_tau2: float = 0.3,
                   init_delta_var: float = 2.0):
    """Block-coordinate Newton MAP with EM variance updates for one benchmark."""
    resp = bench.resp
    sids = resp["subject_id"].astype(str).to_numpy()
    iids = resp["item_id"].astype(str).to_numpy()
    y = resp["y"].to_numpy().astype(float)
    su, S = np.unique(sids, return_inverse=True)
    iu, I = np.unique(iids, return_inverse=True)
    th = np.array([theta_of(bench.subjects[s])[0] for s in su])
    x = th[S]
    a, beta = 1.0, -float(np.log((y.mean() + 1e-3) / (1 - y.mean() + 1e-3)))
    v = np.zeros(len(su))
    d = np.zeros(len(iu))
    tau2, dvar = init_tau2, init_delta_var
    ns, ni = len(su), len(iu)
    for it in range(n_iter):
        for _ in range(2):
            eta = a * x + v[S] - beta - d[I]
            p = 1 / (1 + np.exp(-eta))
            ww = p * (1 - p)
            g = np.bincount(I, weights=-(y - p), minlength=ni) - d / dvar
            h = np.bincount(I, weights=ww, minlength=ni) + 1 / dvar
            d = np.clip(d + g / h, -12, 12)
            eta = a * x + v[S] - beta - d[I]
            p = 1 / (1 + np.exp(-eta))
            ww = p * (1 - p)
            g = np.bincount(S, weights=(y - p), minlength=ns) - v / tau2
            hv = np.bincount(S, weights=ww, minlength=ns) + 1 / tau2
            v = np.clip(v + g / hv, -12, 12)
            eta = a * x + v[S] - beta - d[I]
            p = 1 / (1 + np.exp(-eta))
            ww = p * (1 - p)
            ga = np.sum((y - p) * x) - (a - 1.0) / 1.0
            gb = -np.sum(y - p)
            haa = np.sum(ww * x * x) + 1.0
            hbb = np.sum(ww) + 1e-3
            hab = -np.sum(ww * x)
            det = haa * hbb - hab * hab
            a = a + (hbb * ga - hab * gb) / det
            beta = beta + (-hab * ga + haa * gb) / det
        # EM variance updates with Laplace posterior variances.
        dvar = float(np.mean(d ** 2 + 1 / h))
        tau2 = float(np.mean(v ** 2 + 1 / hv))
        dvar = min(max(dvar, 0.05), 25.0)
        tau2 = min(max(tau2, 0.01), 9.0)
    return {
        "a": float(a), "beta": float(beta), "tau2": tau2, "delta_var": dvar,
        "n_subjects": ns, "n_items": ni, "n_responses": int(len(y)),
        "item_ids": iu, "d": d, "d_var": 1 / h,
    }


def fit(db: dict[str, Benchmark], train_keys, exclude_identities=frozenset(), verbose: bool = False,
        item_level: bool = True, min_item_level_subjects: int = 5, max_item_level_subjects: int = 300):
    rows, units = aggregate(db, train_keys, exclude_identities)
    ab = fit_abilities(rows, units, verbose=verbose)
    attr = fit_attribute_model(ab)
    params = {
        "version": 1,
        "subject_prior": {
            "units": {u: list(v) for u, v in ab["units"].items()},
            "identities": {k: list(v) for k, v in ab["identities"].items()},
            "config_effects": ab["config_effects"],
            "config_var": ab["config_var"],
            "attribute_model": attr,
        },
        "hyper": dict(DEFAULT_HYPER),
        "shrink": {"weights": {"0": 0.0}, "base_rate": 0.5},
    }
    # Base rate over the training responses of non-held-out subjects only, averaged
    # with equal weight per subject-benchmark pair because that is how the official
    # metric averages. Pooling responses instead lets the largest benchmark set the
    # target: here it would give 0.21 where the pair-weighted rate is 0.31.
    per_pair = [r[3] / r[2] for r in rows if r[2] > 0]
    params["shrink"]["base_rate"] = float(sum(per_pair) / len(per_pair)) if per_pair else 0.5
    item_fits = {}
    if item_level:
        prior = SubjectPrior(params)
        for key in train_keys:
            bench = db[key]
            allowed = [s for s in bench.resp["subject_id"].cat.categories
                       if model_identity(bench.subjects[s]) not in exclude_identities]
            if len(allowed) > max_item_level_subjects:
                rng = np.random.default_rng(0)
                allowed = sorted(rng.choice(allowed, size=max_item_level_subjects, replace=False))
            sub = bench.resp[bench.resp["subject_id"].isin(allowed)]
            if sub["subject_id"].nunique() < min_item_level_subjects or sub["item_id"].nunique() < 10:
                continue
            tmp = Benchmark(key=bench.key, anon=bench.anon, meta=bench.meta, subjects=bench.subjects,
                            items=bench.items, resp=sub.reset_index(drop=True))
            item_fits[key] = fit_item_level(tmp, prior)
        if item_fits:
            wts = np.array([math.sqrt(f["n_subjects"]) for f in item_fits.values()])
            A = np.array([f["a"] for f in item_fits.values()])
            Bt = np.array([f["beta"] for f in item_fits.values()])
            T2 = np.array([f["tau2"] for f in item_fits.values()])
            DV = np.array([f["delta_var"] for f in item_fits.values()])
            hyper = params["hyper"]
            hyper["a_mean"] = float(np.average(A, weights=wts))
            hyper["a_var"] = float(np.average((A - hyper["a_mean"]) ** 2, weights=wts))
            hyper["beta_mean"] = float(np.average(Bt, weights=wts))
            hyper["beta_var"] = float(np.average((Bt - hyper["beta_mean"]) ** 2, weights=wts))
            hyper["tau2"] = float(np.median(T2))
            hyper["delta_var"] = float(np.median(DV))
    return params, {"abilities": ab, "attr": attr, "item_fits": item_fits}

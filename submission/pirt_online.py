"""Online hierarchical IRT for the Predictive AI Evaluation Competition.

For a benchmark B (anonymous ``benchmark_id``) the log-odds that subject s
answers item i correctly is

    eta = a * theta_s + v_s - beta - d_i - sum_f w_f

* ``theta_s``: the subject's general ability. Its prior comes from offline
  calibration on measurement-db: known models (by normalized name) have a
  fitted ability; unknown models get a prior from their attributes (provider,
  release date, parameter count, reasoning effort, harness).
* ``a``, ``beta``: the benchmark's discrimination of general ability and its
  overall difficulty. They are unknown for a new benchmark and have broad
  priors fitted across training benchmarks.
* ``v_s``: the subject's benchmark-specific deviation.
* ``d_i``: item difficulty deviation. ``w_f``: effects of the item's visible
  ``item_features`` / ``interactors`` tokens, learned within the benchmark.

All latents of one benchmark share one joint Gaussian posterior. Each acquired
label is absorbed by an assumed-density-filtering (ADF) step: the logistic
likelihood is approximated by a probit, and the Gaussian is updated with a
closed-form rank-one correction, one label at a time. No refitting occurs.
Labels acquired for other subjects on the same benchmark inform beta, a, the
shared item difficulties and feature effects, so they are used when the
evaluator supplies them.

A prediction is the posterior expectation of the success probability,
E[sigmoid(eta)], computed with the probit approximation, then shrunk towards
the global base rate by a factor chosen on local validation.
"""

from __future__ import annotations

import json
import math
import re
import threading
from pathlib import Path

try:  # numpy powers the joint posterior; a scalar fallback keeps the entry point alive without it
    import numpy as np
except ImportError:  # pragma: no cover - exercised by tests through HAVE_NUMPY=False
    np = None
HAVE_NUMPY = np is not None

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
PROBIT_VAR = 8.0 / math.pi  # logistic(x) ~= Phi(x / sqrt(8 / pi))
_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)


# ----------------------------------------------------------------------------
# Numerics


def _mills_inverse(z: float) -> float:
    """phi(z) / Phi(z), stable for very negative z."""
    if z > -8.0:
        return math.exp(-0.5 * z * z) / _SQRT2PI / (0.5 * math.erfc(-z / _SQRT2))
    x = -z  # continued fraction for the Mills ratio (1 - Phi(x)) / phi(x)
    cf = x
    for k in range(40, 0, -1):
        cf = x + k / cf
    return cf


def _log_ncdf(z: float) -> float:
    if z > -8.0:
        return math.log(0.5 * math.erfc(-z / _SQRT2))
    return -0.5 * z * z - math.log(_SQRT2PI) - math.log(_mills_inverse(z))


def _expected_reduction(m: float, v_u: float, cov: float) -> float:
    """E_y[Var(c) - Var(c | y)] when y ~ Bernoulli(Phi(u / sqrt(PROBIT_VAR))), u ~ N(m, v_u)."""
    denom2 = PROBIT_VAR + v_u
    z1 = m / math.sqrt(denom2)
    p1 = 0.5 * math.erfc(-z1 / _SQRT2)
    total = 0.0
    for z, p in ((z1, p1), (-z1, 1.0 - p1)):
        if p <= 0.0:
            continue
        r = _mills_inverse(z)
        total += p * r * (z + r)
    return cov * cov * total / denom2


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def expected_sigmoid(m: float, v: float) -> float:
    """E[sigmoid(x)] for x ~ N(m, v), probit approximation."""
    return sigmoid(m / math.sqrt(1.0 + v / PROBIT_VAR))


# ----------------------------------------------------------------------------
# Visible attributes


def _s(value) -> str:
    return "" if value is None else str(value)


def subject_key(subject: dict) -> tuple:
    """Hashable identity of a visible subject: all eight attributes."""
    return tuple(_s(subject.get(f)) for f in SUBJECT_FIELDS)


def model_identity(subject: dict) -> str:
    """Model identity: the lower-cased normalized name, or the full configuration if unnamed."""
    name = _s(subject.get("normalized_name")).strip().lower()
    if name:
        return name
    return "cfg:" + json.dumps([_s(subject.get(f)).strip().lower() for f in SUBJECT_FIELDS], separators=(",", ":"))


def item_key(item: dict) -> tuple:
    """Hashable identity of an item within a benchmark: text plus features."""
    return (_s(item.get("item_content")), _s(item.get("item_features")))


def feature_tokens(item: dict) -> list:
    """``k=v`` tokens of item_features and interactors (sorted, de-duplicated)."""
    tokens = set()
    for prefix, field in (("f", "item_features"), ("x", "interactors")):
        text = _s(item.get(field)).strip()
        if not text:
            continue
        for part in text.split(";"):
            part = part.strip()
            if part:
                tokens.add(f"{prefix}:{part[:200]}")
    return sorted(tokens)


_SIZE_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])")
_DATE_RE = re.compile(r"^(\d{4})(?:-(\d{1,2}))?")


def parse_year(text: str) -> float | None:
    m = _DATE_RE.match(_s(text).strip())
    if not m:
        return None
    year = int(m.group(1))
    month = int(m.group(2)) if m.group(2) else 6
    if not (1990 <= year <= 2100 and 1 <= month <= 12):
        return None
    return year + (month - 0.5) / 12.0


def parse_size_b(name: str) -> float | None:
    sizes = [float(x) for x in _SIZE_RE.findall(_s(name))]
    sizes = [x for x in sizes if 0.01 <= x <= 5000]
    return max(sizes) if sizes else None


def name_tokens(name: str) -> list:
    return sorted({t for t in re.split(r"[^a-z0-9.]+", _s(name).lower()) if t and not t.replace(".", "").isdigit()})


def config_tokens(subject: dict) -> list:
    tokens = []
    effort = _s(subject.get("reasoning_effort")).strip().lower()
    tokens.append("effort:" + (effort or "<none>"))
    harness = _s(subject.get("harness")).strip().lower()
    tokens.append("harness:" + (harness or "<none>"))
    if harness:
        tokens.append("harness:<any>")
    extra = _s(subject.get("subject_features_extra")).strip().lower()
    for part in extra.split(";"):
        if part.strip():
            tokens.append("extra:" + part.strip()[:100])
    return tokens


class SubjectPrior:
    """theta prior from offline calibration: known units, known models, attributes."""

    def __init__(self, params: dict):
        sp = params["subject_prior"]
        self.units = sp.get("units", {})  # unit key -> [mean, var]
        self.identities = sp.get("identities", {})  # normalized name -> [mean, var]
        self.cfg = sp.get("config_effects", {})
        self.cfg_var = float(sp.get("config_var", 0.1))
        self.attr = sp["attribute_model"]
        self._cache = {}

    @staticmethod
    def unit_key(subject: dict) -> str | None:
        """Model name plus settings; None for unnamed subjects (their identity is the unit)."""
        if not _s(subject.get("normalized_name")).strip():
            return None
        return json.dumps(
            [_s(subject.get(f)).strip().lower() for f in
             ("normalized_name", "harness", "reasoning_effort", "harness_version", "subject_features_extra")],
            separators=(",", ":"),
        )

    def attribute_prior(self, subject: dict) -> tuple:
        a = self.attr
        x = a["intercept"]
        prov = _s(subject.get("provider")).strip().lower()
        x += a["provider"].get(prov, 0.0)
        year = parse_year(subject.get("release_date"))
        if year is None:
            x += a["year_missing"]
        else:
            x += a["year_slope"] * (min(max(year, a["year_min"]), a["year_max"]) - a["year_center"])
        size = parse_size_b(subject.get("normalized_name"))
        if size is None:
            x += a["size_missing"]
        else:
            x += a["size_slope"] * (math.log2(size) - a["size_center"])
        for tok in name_tokens(subject.get("normalized_name")):
            x += a["name_tokens"].get(tok, 0.0)
        return x, float(a["residual_var"])

    def __call__(self, subject: dict, key: tuple | None = None) -> tuple:
        key = subject_key(subject) if key is None else key
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        ukey = self.unit_key(subject)
        unit = self.units.get(ukey) if ukey is not None else None
        cfg_shift = sum(self.cfg.get(t, 0.0) for t in config_tokens(subject))
        if unit is not None:
            out = (float(unit[0]), float(unit[1]))
        else:
            ident = self.identities.get(model_identity(subject))
            if ident is not None:
                out = (float(ident[0]) + cfg_shift, float(ident[1]) + self.cfg_var)
            else:
                m, v = self.attribute_prior(subject)
                out = (m + cfg_shift, v + self.cfg_var)
        self._cache[key] = out
        return out


# ----------------------------------------------------------------------------
# Joint Gaussian with rank-one ADF updates


class Gaussian:
    def __init__(self, capacity: int = 64):
        self.mu = np.zeros(capacity)
        self.cov = np.zeros((capacity, capacity))
        self.n = 0
        self.index = {}

    def add(self, name, mean: float, var: float) -> int:
        idx = self.index.get(name)
        if idx is not None:
            return idx
        if self.n == len(self.mu):
            cap = 2 * len(self.mu)
            mu = np.zeros(cap)
            mu[: self.n] = self.mu[: self.n]
            cov = np.zeros((cap, cap))
            cov[: self.n, : self.n] = self.cov[: self.n, : self.n]
            self.mu, self.cov = mu, cov
        idx = self.n
        self.mu[idx] = mean
        self.cov[idx, : self.n + 1] = 0.0
        self.cov[: self.n + 1, idx] = 0.0
        self.cov[idx, idx] = var
        self.index[name] = idx
        self.n += 1
        return idx

    def moments(self, idx: np.ndarray, val: np.ndarray) -> tuple:
        m = float(self.mu[idx] @ val)
        v = float(val @ self.cov[idx[:, None], idx] @ val)
        return m, max(v, 1e-12)

    def update(self, idx: np.ndarray, val: np.ndarray, y: int, extra_mean: float = 0.0,
               extra_var: float = 0.0) -> float:
        """Absorb one Bernoulli(sigmoid(h'z + extra)) observation; returns log Z."""
        n = self.n
        sh = self.cov[:n, idx] @ val  # Sigma h
        m = float(self.mu[idx] @ val)
        v = float(sh[idx] @ val) if len(idx) else 0.0
        v = max(v, 1e-12)
        s = 1.0 if y else -1.0
        denom = math.sqrt(PROBIT_VAR + v + extra_var)
        z = s * (m + extra_mean) / denom
        r = _mills_inverse(z)
        dm = s * v * r / denom
        dv = v * v * r * (z + r) / (denom * denom)
        self.mu[:n] += sh * (dm / v)
        self.cov[:n, :n] -= np.outer(sh, sh) * (dv / (v * v))
        return _log_ncdf(z)


# ----------------------------------------------------------------------------
# Per-benchmark state


class BenchState:
    def __init__(self, hp: dict, prior: SubjectPrior):
        self.hp = hp
        self.prior = prior
        self.g = Gaussian()
        self.g.add("a", hp["a_mean"], hp["a_var"])
        self.g.add("beta", hp["beta_mean"], hp["beta_var"])
        self.n_labels = 0
        self.subject_counts = {}
        self.n_pos = 0          # observed successes in this benchmark (candidate-B base rate)
        self.log_evidence = 0.0

    def _v_prior(self, theta_var: float) -> float:
        hp = self.hp
        return hp["tau2"] + (hp["a_mean"] ** 2 + hp["a_var"]) * theta_var

    def design(self, subject: dict, item: dict, create: bool, skey: tuple | None = None):
        """Sparse design vector h, plus prior mean/var of latents not in the state."""
        g = self.g
        skey = subject_key(subject) if skey is None else skey
        theta, theta_var = self.prior(subject, skey)
        skey = ("v", skey)
        ikey = ("d",) + item_key(item)
        toks = [("w", t) for t in feature_tokens(item)]
        names = [("a", theta), ("beta", -1.0), (skey, 1.0), (ikey, -1.0)] + [(t, -1.0) for t in toks]
        priors = {skey: (0.0, self._v_prior(theta_var)), ikey: (0.0, self.hp["delta_var"])}
        for t in toks:
            priors[t] = (0.0, self.hp["w_var"])
        idx, val = [], []
        extra_mean, extra_var = 0.0, 0.0
        for name, coef in names:
            j = g.index.get(name)
            if j is None and create:
                j = g.add(name, *priors[name])
            if j is None:
                pm, pv = priors[name]
                extra_mean += coef * pm
                extra_var += coef * coef * pv
            else:
                idx.append(j)
                val.append(coef)
        return np.array(idx, dtype=np.int64), np.array(val), extra_mean, extra_var

    def observe(self, subject: dict, item: dict, y: int) -> None:
        key = subject_key(subject)
        idx, val, em, ev = self.design(subject, item, create=True, skey=key)
        self.log_evidence += self.g.update(idx, val, y, em, ev)
        self.n_labels += 1
        self.n_pos += y
        self.subject_counts[key] = self.subject_counts.get(key, 0) + 1

    def predict(self, subject: dict, item: dict, skey: tuple | None = None) -> float:
        return expected_sigmoid(*self.logit_moments(subject, item, skey))

    def logit_moments(self, subject: dict, item: dict, skey: tuple | None = None) -> tuple:
        idx, val, em, ev = self.design(subject, item, create=False, skey=skey)
        m, v = self.g.moments(idx, val)
        return m + em, v + ev

    def ability_info(self, subject: dict, item: dict) -> tuple:
        """Expected variance reduction of the pair ability c = a*theta + v_s - beta.

        Returns (gain for this item, gain for a generic unseen item without
        features). Both use the probit ADF update, averaged over the predictive
        distribution of the label: the Fisher-information criterion of
        adaptive testing, with the item's difficulty uncertainty included.
        """
        g = self.g
        key = subject_key(subject)
        theta, theta_var = self.prior(subject, key)
        skey = ("v", key)
        c_names = [("a", theta), ("beta", -1.0), (skey, 1.0)]
        v_prior = self._v_prior(theta_var)
        c_idx, c_val, c_var_extra = [], [], 0.0
        for name, coef in c_names:
            j = g.index.get(name)
            if j is None:
                c_var_extra += coef * coef * v_prior
            else:
                c_idx.append(j)
                c_val.append(coef)
        c_idx = np.array(c_idx, dtype=np.int64)
        c_val = np.array(c_val)
        c_mean = float(g.mu[c_idx] @ c_val)
        c_var = float(c_val @ g.cov[c_idx[:, None], c_idx] @ c_val) + c_var_extra

        idx, val, em, ev = self.design(subject, item, create=False, skey=key)
        m = float(g.mu[idx] @ val) + em
        v_u = float(val @ g.cov[idx[:, None], idx] @ val) + ev
        cov = float(c_val @ g.cov[c_idx[:, None], idx] @ val)
        if skey not in g.index:
            cov += v_prior  # the unseen v_s is shared by c and the candidate logit
        gain = _expected_reduction(m, v_u, cov)
        generic = _expected_reduction(c_mean, c_var + self.hp["delta_var"], c_var)
        return gain, generic


class ScalarBenchState:
    """Numpy-free fallback: an independent scalar posterior per subject.

    Tracks c_s = a*theta_s + v_s - beta for each subject with 1-D ADF updates,
    marginalising item difficulty and feature effects. It ignores labels of
    other subjects, so it is weaker than ``BenchState``; it exists only so that
    a missing numpy cannot crash the entry point.
    """

    def __init__(self, hp: dict, prior: SubjectPrior):
        self.hp = hp
        self.prior = prior
        self.post = {}
        self.subject_counts = {}
        self.n_pos = 0          # observed successes in this benchmark (candidate-B base rate)
        self.n_labels = 0

    def _item_var(self, item: dict) -> float:
        return self.hp["delta_var"] + self.hp["w_var"] * len(feature_tokens(item))

    def _get(self, subject: dict, key: tuple | None = None):
        key = subject_key(subject) if key is None else key
        cur = self.post.get(key)
        if cur is None:
            theta, theta_var = self.prior(subject, key)
            hp = self.hp
            mean = hp["a_mean"] * theta - hp["beta_mean"]
            var = (hp["a_var"] * theta * theta + hp["beta_var"] + hp["tau2"]
                   + (hp["a_mean"] ** 2 + hp["a_var"]) * theta_var)
            cur = [mean, var]
        return key, cur

    def observe(self, subject: dict, item: dict, y: int) -> None:
        key, (m, v) = self._get(subject)
        s = 1.0 if y else -1.0
        denom = math.sqrt(PROBIT_VAR + v + self._item_var(item))
        z = s * m / denom
        r = _mills_inverse(z)
        m = m + s * v * r / denom
        v = max(v - v * v * r * (z + r) / (denom * denom), 1e-9)
        self.post[key] = [m, v]
        self.subject_counts[key] = self.subject_counts.get(key, 0) + 1
        self.n_pos += y
        self.n_labels += 1

    def logit_moments(self, subject: dict, item: dict, skey: tuple | None = None) -> tuple:
        _, (m, v) = self._get(subject, skey)
        return m, v + self._item_var(item)

    def predict(self, subject: dict, item: dict, skey: tuple | None = None) -> float:
        return expected_sigmoid(*self.logit_moments(subject, item, skey))


def new_state(hp: dict, prior: SubjectPrior):
    return BenchState(hp, prior) if HAVE_NUMPY else ScalarBenchState(hp, prior)


# ----------------------------------------------------------------------------
# Engine: incremental sync with the evaluator's labeled list


def _entry_fingerprint(entry) -> str:
    try:
        return json.dumps(entry, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        return repr(entry)


class Engine:
    """Posterior states per benchmark, kept in step with the evaluator's label list.

    The list is assumed to grow by appending between calls (as the streaming
    evaluator's does); any other change is detected and triggers a rebuild.
    A re-entrant lock serialises calls in case the host uses threads.
    """

    def __init__(self, params: dict):
        self.hp = params["hyper"]
        self.prior = SubjectPrior(params)
        self.shrink = params.get("shrink", {"weights": {"0": 0.0}, "base_rate": 0.5})
        self.lock = threading.RLock()
        self._shrink_keys = sorted(int(k) for k in self.shrink["weights"])
        # "const": the offline-fitted pool mean (previous and default behaviour).
        # "bench": estimate the target benchmark's own success rate from the labels seen so
        #   far, falling back to `cold_rate` while that benchmark has no labels at all.
        #   Rationale (2026-10-01): the live B0/B1 Brier was worse than predicting 0.5 because
        #   a single fitted constant (0.28158 = public-pool mean) does not transfer to a
        #   benchmark whose true success rate is elsewhere; the public pool itself already
        #   spans 0.148-0.475 across benchmarks.
        self.shrink_target = str(self.shrink.get("target", "const"))
        self.cold_rate = float(self.shrink.get("cold_rate", 0.5))
        self.bench_pseudo = float(self.shrink.get("bench_pseudo", 2.0))
        self._reset()

    def _reset(self):
        self.benches = {}
        self.n_synced = 0
        self.n_skipped = 0  # malformed label entries, ignored rather than failing the run
        self._marks = {}
        self._last_ref = None  # held so its id() cannot be reused by another list
        self._empty = None

    def _bench(self, key: str) -> BenchState:
        state = self.benches.get(key)
        if state is None:
            state = self.benches[key] = new_state(self.hp, self.prior)
        return state

    def _consistent(self, labeled: list) -> bool:
        if len(labeled) < self.n_synced:
            return False
        for pos, fp in self._marks.items():
            if _entry_fingerprint(labeled[pos]) != fp:
                return False
        return True

    @staticmethod
    def _parse(entry):
        """(subject, item, y) for a well-formed ``[[subject, item], y]`` entry, else None."""
        try:
            (subject, item), y = entry
            binary = not isinstance(y, str) and y in (0, 1)
        except (TypeError, ValueError):
            return None
        if not binary or not isinstance(subject, dict) or not isinstance(item, dict):
            return None
        return subject, item, 1 if y == 1 else 0

    def sync(self, labeled) -> None:
        with self.lock:
            labeled = [] if labeled is None else labeled
            if labeled is self._last_ref and len(labeled) == self.n_synced:
                return
            if not self._consistent(labeled):
                self._reset()
            for pos in range(self.n_synced, len(labeled)):
                parsed = self._parse(labeled[pos])
                if parsed is None:
                    self.n_skipped += 1
                else:
                    subject, item, y = parsed
                    self._bench(_s(item.get("benchmark_id"))).observe(subject, item, y)
                self.n_synced = pos + 1  # stays consistent if interrupted mid-way
            n = len(labeled)
            self._marks = {}
            for pos in sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1} if n else ()):
                self._marks[pos] = _entry_fingerprint(labeled[pos])
            self._last_ref = labeled

    def state_for(self, item: dict):
        """Posterior state for the item's benchmark (a shared empty one if unseen)."""
        state = self.benches.get(_s(item.get("benchmark_id")))
        if state is None:
            if self._empty is None:
                self._empty = new_state(self.hp, self.prior)
            state = self._empty
        return state

    def base_rate(self, state) -> float:
        """Shrinkage target for this benchmark."""
        if self.shrink_target != "bench":
            return float(self.shrink["base_rate"])
        n = getattr(state, "n_labels", 0)
        if n <= 0:
            return self.cold_rate
        # Laplace-style smoothing towards the cold rate; with few labels the estimate stays
        # near cold_rate, and it converges to the observed rate as labels accumulate.
        k = self.bench_pseudo
        return (getattr(state, "n_pos", 0) + k * self.cold_rate) / (n + k)

    def shrink_weight(self, n_pair: int) -> float:
        best = None
        for k in self._shrink_keys:
            if k <= n_pair:
                best = k
        return float(self.shrink["weights"][str(best)]) if best is not None else 0.0

    def predict(self, input, labeled=None) -> float:
        subject, item = input
        with self.lock:
            self.sync(labeled)
            state = self.state_for(item)
            skey = subject_key(subject)
            p = state.predict(subject, item, skey)
            n_pair = state.subject_counts.get(skey, 0)
            # read inside the lock: base_rate() now reads mutable counters (n_labels, n_pos)
            # that observe() updates, so reading it outside would allow a torn read between
            # the two increments.
            base = self.base_rate(state)
        w = self.shrink_weight(n_pair)
        p = (1.0 - w) * p + w * base
        if not math.isfinite(p):  # numerical failure: fall back to the base rate
            p = base
        return min(max(p, 0.0), 1.0)


_ENGINES = {}  # params path -> Engine; shared by model.py and labeling.py


def shared_engine(params_path) -> Engine:
    """One engine per parameter file per process, whatever name the host imports us under."""
    key = str(Path(params_path).resolve())
    engine = _ENGINES.get(key)
    if engine is None:
        with open(key, encoding="utf-8") as fh:
            engine = _ENGINES[key] = Engine(json.load(fh))
    return engine

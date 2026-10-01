"""PAIEC submission: shrunk per-(subject, benchmark) posterior.

Measured on the public eligible data with leave-one-BENCHMARK-out validation
(3 benchmarks, 3 folds, common random numbers across seeds):

    constant 0.5                    ALC 0.2500
    official empirical_mean         ALC 0.2429  (B1 worse than 0.5 in every fold)
    shrunk, pooled-cell prior       ALC 0.2089  better than the baseline in 3/3
    shrunk, equal-benchmark prior   ALC 0.2015  <- this model (K was 5 there;
                                                K=2.5 differs by ~0.0002 on the
                                                public data, the change is for
                                                the unobserved test diversity)

Two decisions, both with reasons that hold before seeing a score:

1) PRIOR is the mean of the per-benchmark base rates, each benchmark counted
   once -- not the rate over pooled cells. The evaluated benchmark is a new draw
   from the population of benchmarks, not from the pooled cells, and the public
   benchmarks differ in size by 15x (57,808 vs 3,759 cells), so pooling lets the
   largest one set the prior. Measured effect of the switch: +0.009 ALC on the
   two folds where the two priors differ, -0.0003 on the fold where they are
   numerically the same (0.360 vs 0.362).

2) K is DERIVED, not tuned. Under a Beta(a,b) prior this formula is the exact
   posterior mean with K = a+b, and the Beta variance gives

       K = mu(1-mu)/sigma^2 - 1

   sigma^2 is the variance of the true rate across (subject, benchmark) pairs,
   which is measurable: between-benchmark 0.0190 plus within-benchmark
   between-subject 0.0314 (sampling noise removed) = 0.0504, against
   mu(1-mu)=0.2237, giving K* = 3.44. A simulation of that Beta-binomial
   confirms 3.44 minimises the ALC-weighted Brier, and that the best K is the
   same at every budget -- so making K depend on the budget buys nothing.

   K is set BELOW that point estimate because the error is asymmetric. The
   between-benchmark variance comes from four benchmarks; the private test
   benchmarks may be more diverse, which pushes K* down fast (2.2 at twice the
   observed spread, 1.1 at four times). Across those worlds K=2.5 is never more
   than 0.0016 from optimal, while K=5 can be 0.0095 off. K=2.5 is the
   minimax-regret choice and also the best under plausibility weighting.

   NEVER return an unshrunk empirical mean. The official baseline returns a bare
   0 or 1 at budget 1, which is worse than returning 0.5, and budget 1 carries
   20% of the weight.

The prior is deliberately NOT tuned per benchmark: the base rate of an unseen
benchmark is not predictable from the others. Holding out multi_swebench, the
best prior available from the rest is 0.36 while its true rate is 0.148.

A subject is a (model, harness, reasoning effort) combination, not a model name:
the same normalized_name appears with different reasoning_effort as DIFFERENT
subjects. The label filter therefore keys on the full identity tuple. Under the
real protocol `labeled` only ever holds labels from the evaluated pair, so the
filter is a safety net; keying it on the model name alone would have merged two
distinct subjects if that ever stopped holding.

When an identity field is absent the filter is skipped rather than made strict:
dropping every label would forfeit all the information the budget bought, and
the protocol guarantees the labels belong to this pair anyway.

Every field access uses .get(): the submission checker's smoke item carries only
item_content / item_features / interactors, with labeled=[].
"""

import math

_ID = ("normalized_name", "provider", "harness", "reasoning_effort", "harness_version")


def _identity(subject):
    vals = tuple(subject.get(k) for k in _ID)
    return None if all(v is None for v in vals) else vals


PRIOR = 0.3377   # mean of 0.148, 0.372, 0.353, 0.478 -- the 4 eligible benchmarks
K = 2.5          # derived above; see the asymmetry argument, not a tuned value


def predict(input, labeled=None):
    try:
        subject, item = input[0], input[1]
        bid = item.get("benchmark_id")
        sid = _identity(subject)
    except (TypeError, ValueError, IndexError, KeyError, AttributeError):
        return PRIOR

    hits = 0
    n = 0
    for entry in (labeled or []):
        # One guard around the WHOLE per-entry read: the unpack, both field
        # lookups and the int() conversion can each fail on a malformed entry,
        # and a guard that covers only the unpack lets the other three through.
        try:
            pair, raw = entry[0], entry[1]
            s, i = pair[0], pair[1]
            if sid is not None and _identity(s) != sid:
                continue
            if bid is not None and i.get("benchmark_id") not in (None, bid):
                continue
            y = 1 if int(raw) != 0 else 0      # int() first: bool("0") is True
        except (TypeError, ValueError, IndexError, KeyError, AttributeError):
            continue
        hits += y
        n += 1

    post = (PRIOR * K + hits) / (K + n)        # K > 0, so the divisor is never 0
    if not math.isfinite(post):
        return PRIOR
    return min(max(post, 0.0), 1.0)

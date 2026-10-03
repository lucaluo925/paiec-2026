# Online Hierarchical IRT with Assumed-Density Filtering for Predictive AI Evaluation

**Haoyuan Luo** — University of California, Davis — `zixluo@ucdavis.edu`

*Technical report — Predictive AI Evaluation Competition, NeurIPS 2026.*
Codabench submission **955485** (`paiec_irt.zip`, sha256 `98555787…2cfe0f`).
Code: `https://github.com/lucaluo925/paiec-2026`

## Abstract

A decision maker choosing between AI systems can rarely afford to evaluate them
on every item. This competition asks how well a system's per-item success can be
predicted on a benchmark never seen during development, given a budget of 0 to
31 labels for the subject–benchmark pair under test. We model the response as a
hierarchical item-response model — logit P(correct) = a·θ_s + v_s − β − d_i −
Σ w_f — with subject ability θ_s calibrated offline on the public
measurement-db and predicted from provider, release date, parameter count and
name tokens for subjects absent from calibration. At evaluation time we hold one
joint Gaussian posterior per anonymous benchmark and absorb each acquired label
with an assumed-density-filtering step (probit approximation, closed-form
rank-one update). Labels of *other* subjects on the same benchmark inform the
shared latents, which is where most of the gain at small budgets comes from.

Under leave-one-benchmark-out validation on four public benchmarks (147
subject–benchmark pairs, 19,621 evaluation targets), the fully out-of-fold model
reaches Brier ALC **0.16875** against **0.23110** for the organizers' empirical-mean
baseline and 0.25000 for a constant 0.5; cross-fitting the shrinkage weights
under the same protocol gives an honest **0.16515**. The advantage does not
depend on having met the model before: on the 27 pairs whose identity
calibration never saw, ALC is 0.18424 against 0.24960.

Two negative results are reported as fully as the positive one. First, the
competition's formative feedback is far noisier than it appears: five
byte-identical submissions of the same artefact scored 0.20624, 0.20098,
0.18124, 0.18685 and 0.19461 — a standard deviation of **0.0102** on ALC and
0.014–0.023 per budget, because each evaluation redraws only eight or nine
subject–benchmark pairs. Of everything that feedback could be asked, only three
statements survive it, and we state which. Second, three pre-registered
mechanism-level fixes for the zero-label prior — the one budget where the live
result clearly exceeds our local estimate — were all falsified on
leave-one-benchmark-out, and the shipped configuration was left unchanged. We
also record a sequencing error of our own: the noise calibration was run after,
not before, the first live score had been used to draw conclusions.

**Keywords:** predictive evaluation; item response theory; assumed density
filtering; online Bayesian updating; calibration; label-efficient evaluation;
Brier score.

> **Scope of the numbers.** Every figure outside §5b is a local hold-out
> estimate on the public measurement-db, produced by the scripts in this
> repository. §5b reports the live Codabench scores and treats them as a
> repeated measurement of noise, not as a selection signal; the leaderboard was
> never consulted for model selection.

## 1. Task and scoring

A predictor receives a subject (eight visible attributes: `normalized_name`,
`provider`, `release_date`, `access_date`, `harness`, `reasoning_effort`,
`harness_version`, `subject_features_extra`), an item (`item_content`,
`item_features`, `interactors`, anonymous `benchmark_id`) and the labels acquired
so far, and outputs P(correct). Brier scores at label budgets
b ∈ {0, 1, 3, 7, 15, 31} are combined as
ALC = 0.1·B₀ + 0.2·B₁ + 0.2·B₃ + 0.2·B₇ + 0.2·B₁₅ + 0.1·B₃₁,
so 80 % of the weight lies at 1–15 labels. The official rules average Brier
within each subject–benchmark pair first and then across pairs with equal weight,
"regardless of its number of responses"; all numbers in this report use that
pair-macro aggregation. Test benchmarks are private and unseen during development.

## 2. Data

* **Training corpus.** The public `aims-foundations/measurement-db` (Hugging
  Face, dataset revision `bc8204d811823da849c6686bf124d4ca9f82e4de`), restricted
  — like the organizers' preparation script — to benchmarks with
  `response_type = binary` and `granularity = item`.
* **What that leaves.** The repository holds 10 benchmark directories
  (`response_type`: 6 binary, 2 mixed, 2 fraction). Two of the six "binary"
  directories are `reproduction_checks/.../tables` smoke artifacts carrying 1 and
  2 responses and re-using their parent's `benchmark_id`; they are not
  independent benchmarks and are excluded (they would also fail the ≥ 80-item
  rule). **Four usable benchmarks remain**, and neither of the two excluded
  top-level benchmarks (`matharena`, `mmdocrag`) is replaced, so the training
  pool contains no mathematics and no document-understanding benchmark.

  | benchmark | domain | subjects | items | responses | density | base rate |
  |---|---|---|---|---|---|---|
  | `multi_swebench` | software engineering, agents | 82 | 2 126 | 57 808 | 0.332 | 0.148 |
  | `researchcodebench` | software engineering, ML engineering | 31 | 212 | 6 572 | 1.000 | 0.352 |
  | `swe_rebench` | software engineering, agents | 1 | 6 306 | 6 306 | 1.000 | 0.475 |
  | `real_webagents` | agents and tool use | 33 | 233 | 3 759 | 0.489 | 0.375 |

  In total 74 445 responses, 8 877 items, 147 subject configurations and 60
  distinct model identities. 17 identities appear in ≥ 2 benchmarks and **none in
  ≥ 5**, so the cross-benchmark ability calibration rests on 17 models.
* **Attribute coverage.** `normalized_name` and `provider` 100 %,
  `release_date` 93.9 %, `access_date` and `harness` 55.8 %,
  `harness_version` 0.7 %, **`reasoning_effort` and `subject_features_extra`
  0 %**. The configuration-effect terms for reasoning effort therefore have no
  support in this corpus. `item_features` is non-empty for 2 571 items
  (most common keys `lang`, `website`, `paper`); `interactors` is empty
  everywhere.
* **Preprocessing.** One response per (subject, item, interactors): the
  lowest (test condition, trial) row with a 0/1 grade, because the visible input
  cannot distinguish trials or conditions. Missing attributes become `""`.
* **No other data.** No external data, benchmark results, model cards or
  web sources were used. The model registry used for normalized names is the one
  already embedded in measurement-db subject tables.

## 3. Local evaluation protocol

All model selection used a local replica of the evaluator (`eval/local_scorer.py`),
never the Codabench leaderboard.

* **Streaming protocol.** Re-implements `streaming_alc_v1` from the organizers'
  `tools/streaming_ingestion.py`: benchmarks with ≥ 80 items; a 50/50
  acquisition/evaluation item split **per subject–benchmark pair**; each pair's
  acquisition items are streamed one at a time with `max_labels = 31`; the
  default policy queries when a SHA-256 uniform falls below
  `labels_remaining / items_remaining`; all pairs reach budget b before the
  checkpoint at b; `labeled` contains the labels of all sampled pairs.
* **Checked against the official rules page** (aimslab.stanford.edu/competition):
  the per-pair 50/50 split, the pair-macro averaging, the ALC weights, the
  budgets, `max_labels = 31`, the `context` fields (`subject_id`,
  `benchmark_id`, `labels_acquired`, `labels_remaining`, `max_labels`,
  `items_remaining` including the current candidate), and the global scope of
  `labeled` all match. Two of our earlier assumptions did **not** match and were
  corrected: the split had been made once per benchmark and shared across
  subjects, and Brier had been averaged over predictions rather than over pairs.
  Both corrections are locked by regression tests
  (`test_official_protocol_defaults`, `test_pair_macro_weights_pairs_equally`).
* **Verification.** (i) A constant 0.5 predictor scores exactly 0.25 at every
  budget and in ALC (weights are integers in tenths and summed with `math.fsum`).
  (ii) The organizers' own streaming client (`run_streaming`) driven by our
  coordinator reproduces our acquisitions and predictions exactly (unit test).
  (iii) Label sets are nested across budgets.
* **Hold-out.** Two-way: benchmarks are split into 5 folds (benchmarks sharing
  ≥ 5 % item texts are kept together), and 20 % of model identities are removed
  from every training fold and appear only as unseen subjects. Items are never
  split at random. With four benchmarks the 5-fold split degenerates to
  **leave-one-benchmark-out** (fold sizes 1, 1, 1, 1, 0). Local evaluation uses
  every subject configuration in the corpus: 147 pairs and 19 621 evaluation
  targets, of which 27 pairs / 3 556 targets belong to the 12 held-out
  identities.
* **Still unverified.** The rules page does not state the evaluation worker
  concurrency or the size of the private test set. The engine holds an `RLock`
  and is safe under threads; process-level parallelism only re-creates state.
  The caps we pass locally (≤ 200 subjects per benchmark, ≤ 200 evaluation
  items per pair, ≤ 800 streamed items per pair) are ours, not the rules'; at
  these values none of them binds on this corpus, so the reported numbers use
  the whole public dataset. The rules' own cap — 1 000 unique subject–item
  pairs — applies to formative feedback runs only.

## 4. Method

### 4.1 Model
For benchmark B, subject s and item i,

  logit P(y=1) = a·θ_s + v_s − β − d_i − Σ_f w_f

* θ_s — general ability, **calibrated offline**. Known models use their fitted
  ability; unseen models use a ridge regression on provider, release date,
  parameter count (parsed from the name) and name tokens, plus learned effects of
  reasoning effort / harness settings.
* a, β — the benchmark's discrimination of general ability and its difficulty;
  priors are the across-benchmark distribution of item-level fits.
* v_s ~ N(0, τ² + (ā² + var a)·var θ_s) — subject × benchmark deviation.
* d_i ~ N(0, σ_δ²) — item difficulty; w_f ~ N(0, σ_w²) — effects of the
  item's `item_features` / `interactors` tokens, learned within the benchmark.

### 4.2 Offline calibration (`train/fit_offline.py`)
1. Subject abilities by weighted alternating least squares on per-(benchmark,
   subject) logit accuracies: logit(acc) = a_B θ_u − b_B, with θ of a
   configuration unit shrunk towards its model's ability plus configuration
   effects.
2. Item side: for every training benchmark, an item-level model
   (a, β, v_s, d_i with variance components τ², σ_δ²) by block-coordinate Newton
   with EM variance updates. The distribution of these item-side parameters across
   benchmarks gives the priors for a new benchmark. Item-level parameters are
   never transferred directly, because the test benchmarks are new.
3. Hyper-parameter scales and the shrinkage weights are selected on the local
   folds (`eval/tune.py`), with a cross-fitted estimate for honesty. Selected
   scales: `delta_var` 1.00, `tau2` 0.21, `beta_var` 6.76, `a_var` 0.60,
   `w_var` 4.16.
   The coordinate search over the scales was run on a 40-subject-per-benchmark
   subsample, because a search under the full protocol costs about 20 s per
   candidate and does not fit the compute available here; the shrinkage weights
   are then fitted **and cross-fitted under the full protocol**
   (`--scales ... --rounds 0 --crossfit`). This is disclosed because the scales
   are therefore selected on a smaller sample than the one they are reported on.

### 4.3 Online update (`submission/pirt_online.py`)
One joint Gaussian over all latents of a benchmark. Each label is absorbed by an
assumed-density-filtering step: the logistic likelihood is approximated by a
probit, whose moments are closed-form, followed by a rank-one covariance update.
Labels are processed one at a time in the order supplied; the state is cached
and extended incrementally when the evaluator's label list grows. Labels of other
subjects on the same benchmark update the shared latents (β, a, d_i, w_f). A
numpy-free scalar fallback runs if numpy is unavailable.

### 4.4 Prediction and safety net
p = E[σ(logit)] under the posterior (probit approximation), then
p' = (1 − w_n)·p + w_n·p̄ with p̄ the training base rate and w_n chosen per
label count on local validation. Selected weights: w₀ = 0.535, w₁ = 0.141,
w₃ = 0.000, w₇ = 0.015, w₁₅ = 0.004, w₃₁ = 0.003; final p̄ = 0.2816.

w_n is the weighted least-squares optimum in which each prediction of a pair
with n targets carries weight 1/n, matching the way the metric averages; pooling
the predictions instead would let the largest pairs choose w for every pair. On
this corpus the two agree to within 1e-5 of ALC, so the alignment is a
correctness property rather than a source of gain.

p̄ is the mean over training subject–benchmark pairs of each pair's accuracy,
not the pooled response rate. This matters: pooling lets the largest benchmark
set the target, which here would give 0.21 against a pair-weighted 0.31, and cost
about 0.003 of B₀.

### 4.5 Acquisition (`labeling.py`) — not shipped
Streaming decision per candidate. The value of a candidate is the expected
reduction of the posterior variance of the pair ability c = a θ_s + v_s − β
(Fisher information under the current posterior, including the candidate's
difficulty uncertainty), relative to a generic unseen item: ρ. The query
probability is min(1, q·ρ²) with q = labels_remaining / items_remaining; with
ρ ≡ 1 it is exactly the evaluator's random policy.

A single pre-registered A/B (`eval/ab.py`, `results/acquisition_ab.md`) compared
the random policy against Fisher information and against posterior uncertainty,
with the decision rule fixed beforehand: ship only if ALC is lower **and** the
95 % interval of a paired bootstrap over benchmarks excludes 0.

| policy | ALC | ΔALC vs random | 95 % CI (benchmark clusters) |
|---|---|---|---|
| random (official) | 0.16875 | — | — |
| Fisher information | 0.16981 | +0.00106 | [−0.00071, +0.00345] |
| posterior uncertainty | 0.16846 | −0.00029 | [−0.00321, +0.00448] |

Neither interval excludes 0, so **the submission ships without `labeling.py`**
and uses the official random policy. The A/B was first run on the
40-subject subsample (Fisher +0.00127, uncertainty +0.00156, both intervals
straddling 0) and then re-run unchanged on the full protocol, which is the table
above; the decision is the same under both, and the rule was not altered after
seeing either. Posterior uncertainty turns marginally negative on the larger
sample but its interval is four times wider than the effect. Fisher information
is better at B₁ and B₃₁ and worse at B₃–B₁₅, where the ALC weight is
concentrated; we report this as an observation, not as grounds for a third
attempt.

## 5. Results (local hold-out)

Cross-validated, leave-one-benchmark-out, pair-macro Brier; lower is better;
147 pairs and 19 621 evaluation targets. The rows below are the fully
out-of-fold model **before** hyper-parameter tuning, so no selection of any kind
enters them.

| Predictor | ALC | B₀ | B₁ | B₃ | B₇ | B₁₅ | B₃₁ |
|---|---|---|---|---|---|---|---|
| Constant 0.5 | 0.25000 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 |
| Empirical mean (organizers) | 0.23110 | 0.2500 | 0.3449 | 0.2310 | 0.1902 | 0.1774 | 0.1740 |
| **Online IRT** | **0.16875** | 0.2211 | 0.1932 | 0.1771 | 0.1593 | 0.1412 | 0.1248 |
| Online IRT, own-pair labels only | 0.19450 | 0.2211 | 0.2117 | 0.1985 | 0.1875 | 0.1777 | 0.1731 |

That is 27 % better than the organizers' baseline on ALC. With the shrinkage
weights fitted and cross-fitted under the same protocol, the **cross-fitted
(honest) estimate** is **ALC 0.16515**, per-budget Brier
0.2061 / 0.1897 / 0.1724 / 0.1574 / 0.1408 / 0.1248.

Unseen model identities (27 of 147 pairs, calibration never saw the identity):
Online IRT 0.18424, empirical mean 0.24960, constant 0.5 0.25000. Seen
identities: 0.16526 / 0.22693 / 0.25000. The advantage does not depend on having
met the model before.

Per fold the ALC is 0.1509, 0.1943, 0.2728 and 0.1860; the worst fold is
`swe_rebench`, which contributes exactly one pair.

Calibration improves with labels: expected calibration error
0.122 → 0.072 → 0.058 → 0.023 → 0.027 → 0.016 across the six budgets
(`figures/calibration.pdf`). At zero labels the predictor is visibly
over-confident; this is what the shrinkage weight w₀ = 0.535 absorbs.

Figures: `figures/learning_curves.pdf`, `figures/calibration.pdf`;
acquisition A/B: `results/acquisition_ab.md`; data survey:
`results/data_survey.md`; code review notes: `results/ocr_review*.md`.

## 5b. Results (live, Codabench formative feedback)

The shipped artefact (`dist/paiec_irt.zip`, sha256 `98555787…2cfe0f`, 15,436
bytes) was submitted **five times unchanged**, three on 2026-10-01 (955485,
955728, 955767) and two on 2026-10-02 (957572, 957580). The rules state that
*"each formative evaluation uses a newly sampled subset of the hidden test
data, capped at 1,000 unique subject–item pairs"*, so the five runs differ only
in which subject–benchmark pairs were drawn. We report them as a repeated
measurement, because a single live score turns out to be far noisier than we
expected, and the noise is the main thing we learned from them.

| | 955485 | 955728 | 955767 | 957572 | 957580 | mean | **SD** | local | gap vs local |
|---|---|---|---|---|---|---|---|---|---|
| **Brier ALC** | 0.20624 | 0.20098 | 0.18124 | 0.18685 | 0.19461 | 0.19398 | **0.0102** | 0.17540 | +0.0186 (1.8 σ) |
| B₀ | 0.30845 | 0.27314 | 0.25431 | 0.30584 | 0.28153 | 0.28465 | 0.0228 | 0.21634 | **+0.0683 (3.0 σ)** |
| B₁ | 0.25933 | 0.22918 | 0.22100 | 0.24012 | 0.21610 | 0.23315 | 0.0172 | 0.20171 | +0.0314 (1.8 σ) |
| B₃ | 0.19287 | 0.20820 | 0.17058 | 0.20076 | 0.19303 | 0.19309 | 0.0141 | 0.18436 | +0.0087 (0.6 σ) |
| B₇ | 0.17314 | 0.17904 | 0.15904 | 0.14204 | 0.17071 | 0.16479 | 0.0147 | 0.16570 | −0.0009 (−0.1 σ) |
| B₁₅ | 0.16742 | 0.17175 | 0.15398 | 0.13154 | 0.16892 | 0.15872 | 0.0167 | 0.15042 | +0.0083 (0.5 σ) |
| B₃₁ | 0.16840 | 0.16031 | 0.14893 | 0.13378 | 0.16701 | 0.15569 | 0.0145 | 0.13330 | +0.0224 (1.6 σ) |
| ECE | 0.16324 | 0.13985 | 0.13641 | 0.19109 | 0.14631 | 0.15538 | 0.0225 | — | — |

Each run drew 8 or 9 subject–benchmark pairs (8, 8, 8, 9, 9), and in every one
the unweighted mean of the per-pair ALCs reproduces the reported total to six
decimals, confirming that the official aggregation is an equal-weight mean over
pairs. Run 3's eight pairs share no subject with run 1's. The sampled subjects —
and, as runs 4 and 5 show, the number of them — are where the variance comes
from: with eight or nine pairs per run, the identity of the draw dominates.

**The measurement noise is the headline result.** A single formative score has
a standard deviation of about **0.010** on ALC, **0.014–0.023** per budget, and
**0.022** on ECE. The spread across five identical submissions is 0.0250,
2.5 SD. This is larger than most of the differences one would want to act on.
What the evidence does and does not support:

1. **The live result is worse than the local estimate, but not significantly
   so.** The mean gap is +0.019 on ALC, 1.8 SD. All five runs lie above the
   local estimate, so the direction is consistent across every draw; the
   magnitude is not established.
2. **Only the zero-label budget survives the noise.** B₀ is +0.068 above local
   at **3.0 SD**, consistent in all five runs, and it is the one per-budget
   reading that strengthened as runs were added (2.3 σ at n=3, 2.6 σ at n=4,
   3.0 σ at n=5). B₁ did not: it was 1.7 σ at n=3 and is 1.8 σ now, with the
   fifth run's 0.2161 nearly level with the local 0.2017. **The shortfall is at
   the zero-label prior specifically, not at "the low budgets" generally.**
3. **The middle budgets are indistinguishable, and we can show it rather than
   assert it.** B₇ read +0.0047 (worse than local) at n=3, flipped sign to
   −0.0024 at n=4, and sits at −0.0009 (−0.1 SD) at n=5. **One additional draw
   reversed the direction of a per-budget conclusion.** B₃ and B₁₅ are likewise
   within 0.6 SD. Any statement about an individual budget other than B₀ is
   reading the draw, not the method.
4. **At zero labels the method is at best level with predicting 0.5.** B₀ was
   above the constant-0.5 value of 0.2500 in all five runs (minimum 0.2543),
   but the mean margin is +0.035, only 1.5 SD, and the smallest is 0.2 SD. We
   report this as a consistent direction rather than a confirmed deficit.
5. **Leaderboard position is not a stable quantity here, but the gap to the
   leader is.** The five scores span 0.1812–0.2062, a range that covered
   roughly four places on the 21-entry public table at the time of measurement,
   so we do not quote a rank. Against the organisers' own entry at 0.180113 our
   mean is +0.014 (1.4 SD) — worse in all five runs but not significantly so,
   the closest run being only 0.0011 above it. Against the best entry at
   0.117238 the gap is +0.077, **7.6 SD**: that one is established. The 27 %
   margin over the empirical-mean baseline in §5 is a statement about that
   baseline method on local hold-out data and should not be read as
   competitiveness.

Two further remarks on procedure. First, the noise calibration should have
been run before any inference was drawn from a live score, not after. Our first
live number was used to diagnose a deficit, propose three fixes and revise
protocol conclusions, all before we knew its standard deviation was 0.010 —
with hindsight that first "finding" was under 3 SD and should have been held.
Second, σ̂ itself is estimated from five points and carries roughly 35 %
relative error, so it sits close to the 0.01 threshold we pre-registered for
discounting single-score inferences. We have not spent further submissions
trying to resolve which side of that threshold it falls on: at this magnitude
the noise is of the same order as every effect we would act on, and the third
decimal of σ̂ does not change that.

We pre-registered and tested three mechanism-level fixes for the zero-label
prior: shrinking towards 0.5, estimating the target benchmark's success rate
online with a 0.5 cold start, and the same with the original constant as the
cold start. Gates were fixed before running; **all three failed** on
leave-one-benchmark-out, each making B₀ and/or B₁ worse. The constant was fitted
on exactly the four public benchmarks, so for them it is close to the best
available prior rather than a wrong one. The live shortfall at B₀ — now the one
per-budget effect that clears the noise — appears to involve a benchmark whose
success rate lies outside the 0.148–0.475 range the public pool spans, which no
local protocol can reproduce, so we record the hypothesis as **untested rather
than refuted**. The shipped configuration is unchanged.

**No model choice in this report was made using live feedback.** Formative
evaluation resamples on every submission and the summative evaluation uses a
separate common subset, so selecting on it would be selecting on noise. All
five runs above were spent on measuring that noise, not on choosing anything.
Of the ten submissions of this artefact, two failed outright on the platform
and three have remained queued without producing output (one for over 33
hours); we report the five that scored.

## 6. Limitations

* **The training pool is very small and narrow.** Four benchmarks, all software
  engineering / agent tasks; no mathematics, no document understanding. The
  private test benchmarks may well include domains with no representative here,
  and the across-benchmark priors that the method depends on are estimated from
  four points.
* **The local estimate is coarse.** Leave-one-benchmark-out over four benchmarks
  gives per-fold ALC of 0.151, 0.194, 0.273 and 0.186. Any difference smaller
  than that spread — including the 0.004 gained by hyper-parameter tuning — is
  not resolvable with this data. `swe_rebench` contributes a single
  subject–benchmark pair, and on that fold the model (0.273) is worse than a
  constant 0.5; a benchmark with one subject gives the offline calibration
  nothing to learn from.
* **The live estimate is coarse too, and for a different reason.** Each
  formative evaluation draws eight or nine subject–benchmark pairs afresh, so a
  single live score carries a standard deviation of about 0.010 on ALC and
  0.014–0.023 per budget (§5b, five identical submissions). That is the same
  order as every effect in this report we might want to act on, so the live
  feedback supports only three statements: the gap to the leading entry is real
  (7.6 SD), the zero-label shortfall against our local estimate is real
  (3.0 SD), and nothing else is resolved — including whether we are behind the
  organizers' own entry (1.4 SD). We measured this only after drawing
  conclusions from the first live score, which was a mistake in sequencing, not
  just in arithmetic.
* **The pre-registered bootstrap is weak by construction.** Resampling four
  benchmark clusters yields a coarse distribution. We kept the rule rather than
  changing it after seeing the data, and report a pair-clustered interval
  alongside as a diagnostic; both agree that the acquisition variants are not
  distinguishable from random.
* **Zero-label prediction is the weakest budget, but it does beat a constant.**
  B₀ is 0.2061 cross-fitted. The right comparison is a constant chosen the same
  way the model is fitted — the base rate of the training folds only — which
  scores 0.2122; the model is 0.006 better. An earlier draft of this report
  claimed the opposite by comparing against the constant that minimises error on
  the evaluation pool itself (0.2160 at the smaller protocol), which is not a
  baseline any predictor could have chosen in advance. This is a statement about
  local hold-out data against a base-rate constant, and it does not carry over
  to the live setting: §5b finds live B₀ above the constant-0.5 value of 0.2500
  in all five runs, and above our own local B₀ by 3.0 SD. The zero-label prior
  is the one place where local and live disagree beyond the noise.
  The weighted least squares for w_n described in §4.4 was aligned with the metric after that
  finding and changed the result by less than 1e-5 of ALC, confirming that the
  earlier gap was an artefact of the unfair baseline, not a weighting bug.
* **`reasoning_effort` has no support.** The configuration-effect machinery for
  it is inert on this corpus; if the private subjects carry that field, its
  effect is untrained.
* Items of the evaluation pool are never labeled, so item-level variation comes
  only from visible features; the text of items is not used by the model beyond
  feature tokens. The text-difficulty prior (Task 4 of our plan) was not built.
* One latent ability dimension plus benchmark-specific deviations; skills that
  do not correlate with general ability are captured only through v_s.
* ADF is order-dependent and approximates the posterior.
* Evaluation worker concurrency is unstated in the rules; the engine is
  thread-safe but has not been tested under the organizers' actual harness.

## 7. Reproducibility

```
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements-lock.txt
git clone --depth 1 https://github.com/aims-foundations/paiec_baseline   # pinned at 82d330d
export PAIEC_BASELINE_DIR=<path to paiec_baseline>
export HF_TOKEN=<read token; the dataset is gated>
CAPS="--max-subjects 200 --max-eval 200 --max-stream 800"
python scripts/download_measurement_db.py --out data/raw
python scripts/survey_data.py --data data/raw --out results/data_survey.md
python -m eval.run_cv --data data/raw --variants const0.5 empirical_mean irt irt_pairscope \
    $CAPS --out results/by_budget.csv --records results/cv_records.npz
python -m eval.tune --data data/raw --crossfit --out results/tuned_hyper.json      # scale search, 40-subject subsample
python -m eval.tune --data data/raw --crossfit --rounds 0 --scales results/tuned_hyper.json \
    $CAPS --out results/tuned_hyper_full.json                                      # shrinkage under the full protocol
python -m eval.ab  --data data/raw $CAPS --out results/acquisition_ab.md
python tools/make_figures.py --records results/cv_records.npz --curves irt empirical_mean irt_pairscope \
    --calibrate irt --ab "irt:random" "irt+uncertainty:Posterior uncertainty" --ab-records results/ab_records.npz
python -m train.train_final --data data/raw --tuned results/tuned_hyper_full.json
python tools/build_submission_zip.py
python $PAIEC_BASELINE_DIR/check_submission_zip.py dist/paiec_irt.zip
```

Seeds are fixed (split seed 0, deterministic SHA-256 hashing). Verified on
Python 3.11.16 with the locked dependency versions: two `run_cv` runs produce
identical CSVs, two `train_final` runs produce identical `params.json`, and two
ZIP builds are byte-identical
(sha256 `98555787478009fc9e53b05a7a532912db37b716e24ed9abac121bac33c2fe0f`).
The organizers' `check_submission_zip.py` passes without `--static-only`.
Extracted into an empty directory and imported in a fresh process, the model
predicts in single-digit microseconds per call (6.7 µs with numpy, 2.8 µs on the
numpy-free fallback), which is negligible against the 8-hour run limit;
malformed labels are skipped and an empty label list returns the prior.

The submission ZIP contains `model.py`, `pirt_online.py`, `params.json`,
`requirements.txt` and `README.md`. No `models.txt` is needed: the submission
declares no Hugging Face model. The build script scans every packaged file for
credentials and refuses parameters fitted on synthetic data.

## 8. Disclosures

* **Pretrained models:** none. The submission contains no neural network,
  embedding model or LLM, and makes no API calls. Nothing in the evaluation path
  touches the network.
* **Data:** only the public (access-gated) `aims-foundations/measurement-db` at
  revision `bc8204d811823da849c6686bf124d4ca9f82e4de`; see §2.
* **LLM coding assistant:** Claude Opus 5 (Anthropic), used through the Claude
  desktop app's agentic coding mode, to write the code and draft this report
  under the author's direction; all results were produced by the scripts in this
  repository.

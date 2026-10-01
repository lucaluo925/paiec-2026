# What Transfers Across Benchmarks? A Negative Result and a Calibrated Baseline for Predicting AI Evaluation Outcomes

**PAIEC 2026 technical report — DRAFT**

---

## Summary

The competition asks for the probability that a given AI system answers a given item correctly, on **benchmarks that were never seen during development**, with a label budget of 0 to 31 per subject–benchmark pair.

We set out to build a latent-trait model. We did not submit one. Three candidate sources of cross-benchmark signal were each measured and each failed, for three different and identifiable reasons:

1. **Benchmark difficulty is not predictable.** Holding out `multi_swebench`, the best prior the remaining data supports is 0.36; its true base rate is 0.148.
2. **Subject ability transfers but cannot be used.** Relative model strength correlates across benchmarks (Pearson +0.56 to +0.85, all three pairs same sign). Applying it as a prior shift makes the score *worse*, because the unknown benchmark-level offset is as large as the subject signal itself — a ratio of 1.70 on the fold where it hurts.
3. **Item difficulty is the largest variance component and has no transferable predictor.** It accounts for 28–40% of response variance after removing binomial noise — more than subject identity does. But `item_features` vocabularies are disjoint across benchmarks; generic content features reverse sign between benchmarks; and a ridge model on the official embeddings, which predicts difficulty *within* a benchmark at r = 0.23–0.49, transfers at r ≈ 0.

The consequence is that for an unseen benchmark the acquired labels are very nearly the only usable information. We therefore submit a two-parameter posterior: a prior equal to the mean of per-benchmark base rates, updated by the acquired labels with a shrinkage constant **derived from the measured variance decomposition rather than tuned**. It scores **ALC 0.2015** under leave-one-benchmark-out validation, against 0.2429 for the official `empirical_mean` baseline and 0.2500 for a constant 0.5.

We also report a validation-protocol result we believe matters more than our score: **holding out subjects instead of benchmarks inflates the apparent gain of a per-item difficulty model by a factor of twenty** (+0.0266 vs +0.0013). Any result in this competition validated by a subject-level split should be treated as unmeasured.

---

## 1. Data

### 1.1 Eligibility

`tools/prepare_data.py` restricts the competition to tables with `response_type == "binary"` and `granularity == "item"`. Four of the six public sources qualify. `matharena` (`mixed`) and `mmdocrag` (`fraction`) do not, and together they are **76% of all 571,921 responses**.

| source | subjects | items | cells | base rate |
|---|---:|---:|---:|---:|
| multi_swebench | 82 | 2,126 | 57,808 | 0.148 |
| real_webagents | 33 | 233 | 3,759 | 0.372 |
| researchcodebench | 31 | 212 | 6,572 | 0.353 |
| swe_rebench | **1** | 6,306 | 6,306 | 0.478 |

`swe_rebench` has a single subject and therefore carries no information about how performance varies across systems. **Three benchmarks are available for cross-subject learning.** This is the binding constraint on the whole problem: any claim of the form "method A generalises to a new benchmark better than method B" rests on three observations.

### 1.2 Repeated trials

`response.parquet` carries a `trial` column; the same (subject, item) cell is measured more than once. Repetition rates differ sharply: `swe_rebench` 90.6%, `real_webagents` 15.5%, `researchcodebench` 3.1%, `multi_swebench` 0.3%.

The competition scores **one realised outcome**, so we sample a single trial per cell rather than averaging. Averaging would reduce target variance and understate Brier, producing a local score that cannot be compared with the official one.

### 1.3 Download footprint

The gated `measurement-db` repository holds 2,181 files. The four eligible sources' `{subjects, items, benchmarks, response}.parquet` — 16 files, **58 MB** — are sufficient. `traces.parquet` accounts for 405 MB of the 463 MB in those sources and is not needed for the prediction task.

---

## 2. Validation protocol

### 2.1 Hold out benchmarks, not subjects

The test benchmarks are private. Their items were never in training. A validation split that holds out *subjects* leaves the items in training, so a model that memorises per-item difficulty meets the same items again at test time.

We measured the size of that error. A per-item empirical difficulty model — the most natural thing to build — was evaluated both ways:

| protocol | shrunk baseline | memoriser | apparent gain |
|---|---:|---:|---:|
| hold out **subjects** | 0.2375 | 0.2109 | **+0.0266** |
| hold out **benchmarks** | 0.2399 | 0.2385 | **+0.0013** |

**95% of the apparent gain is an artefact of the split.** Under benchmark holdout one of the three folds is negative.

### 2.2 The decision rule: all folds, not the mean

With three benchmarks, leave-one-out gives three folds. We accept a change only when **no fold shows a real loss**, and we do not use the mean.

This is not conservatism for its own sake. On a synthetic replica of the problem, with common random numbers across compared methods, a *uniform* improvement of +0.006 ALC is detected in 8 of 8 worlds — sampling noise is not the difficulty. But a method that helps two benchmarks and hurts one, with a true mean effect of **+0.0092**, produces per-fold gains of +0.0158 / +0.0188 / −0.0069. **You do not receive the mean. You receive one draw — the private test benchmark — and there is a one-in-three chance it is the one where the method loses.**

A negative fold is exempted only when it is not a real loss, which requires both a magnitude far below the positive folds *and* a stated mechanism. We applied this exemption once (§4.1) and refused it twice (§4.2, §4.4).

### 2.3 Common random numbers

Every paired comparison reuses the same label-acquisition stream across methods. In earlier work on a different competition we voided a significant result that turned out to be seed noise; here the per-fold standard deviation across 8 seeds is 0.0008–0.0011, small enough that a deterministic difference is distinguishable from a stochastic one.

---

## 3. The prior

### 3.1 Benchmark base rates are not predictable

The most direct evidence for the whole paper. Under leave-one-benchmark-out, the prior available from the remaining benchmarks versus the truth:

| held out | pooled-cell prior | equal-benchmark prior | **true base rate** |
|---|---:|---:|---:|
| multi_swebench | 0.360 | 0.362 | **0.148** |
| real_webagents | 0.169 | 0.251 | **0.372** |
| researchcodebench | 0.162 | 0.260 | **0.353** |

No estimator built from two benchmarks comes close to the third. Consequently **budget 0, which carries 10% of the ALC weight, is close to unwinnable**: it is an information problem, not a modelling one.

### 3.2 Weight benchmarks equally, not cells

The public benchmarks differ in size by a factor of fifteen (57,808 vs 3,759 cells). A prior computed over pooled cells is therefore set almost entirely by `multi_swebench`. But the evaluated benchmark is **a new draw from the population of benchmarks, not from the pooled cell population**, so each benchmark should count once.

Measured effect, K fixed at 5, 8 seeds, common random numbers:

| held out | gain | sd over 8 seeds |
|---|---:|---:|
| multi_swebench | −0.0003 | 0.0000 |
| real_webagents | +0.0092 | 0.0008 |
| researchcodebench | +0.0098 | 0.0011 |

The negative fold is the one exemption we allow under §2.2: its standard deviation is zero — it is deterministic, not noise — and the mechanism is explicit. On that fold the two priors are 0.360 and 0.362, numerically the same estimator; the truth is 0.148, far below both. The −0.0003 records which of two badly wrong numbers is 0.002 less wrong, and is thirty times smaller than the gains on the folds where the two priors actually differ.

---

## 4. What we tried to add, and why none of it survived

### 4.1 Subject ability

Relative model strength does transfer. Per-benchmark logit deviation from the benchmark mean, for models appearing in both benchmarks:

| pair | n | Pearson | Spearman |
|---|---:|---|---|
| multi_swebench × real_webagents | 11 | +0.616 (p=0.044) | +0.618 |
| multi_swebench × researchcodebench | 9 | +0.849 (p=0.004) | +0.933 |
| real_webagents × researchcodebench | 13 | +0.559 (p=0.047) | +0.621 |

All three same sign; sample sizes are 9–13, and a Bonferroni correction over three tests leaves only the middle pair.

Applying this as a prior shift, at shrinkage 0, 0.5, 0.75 and 1.0, **fails at every setting**. The mechanism is a single ratio — the benchmark-level prior error against the between-subject spread, both in logit:

| held out | level error | subject spread | ratio | ALC |
|---|---:|---:|---:|---|
| multi_swebench | −1.18 | 0.70 | **1.70** | **worse**, 0.1665 → 0.1784 |
| real_webagents | +0.57 | 1.28 | 0.45 | slightly better |
| researchcodebench | +0.44 | 0.94 | 0.47 | flat |

**The ratio predicts the sign of each fold.** This is a real loss with a mechanism that will recur, not an artefact, so §2.2 rejects it.

The underlying reason is structural: the subject prior and the acquired labels estimate *the same quantity* — this subject's rate on this benchmark. The labels are unbiased. The prior carries an unknown benchmark-level bias of the same magnitude as its signal.

### 4.2 Item difficulty

Item difficulty is the larger component. After subtracting binomial sampling noise:

| | real item variance | share of total | subject share, for comparison |
|---|---:|---:|---:|
| multi_swebench | 0.0361 | **28.5%** | 22.1% |
| real_webagents | 0.0721 | **30.9%** | 19.6% |
| researchcodebench | 0.0908 | **39.7%** | 10.4% |

Three families of item-side features were tested.

**`item_features` does not transfer at all.** The key vocabularies are disjoint: `lang` in `multi_swebench`, `website` in `real_webagents`, `paper` in `researchcodebench`. On an unseen benchmark the field is effectively absent.

**Generic content features reverse sign.**

| | log length | code fences |
|---|---|---|
| multi_swebench | −0.004 (p=0.86) | −0.007 |
| real_webagents | **−0.264** (p<0.001), longer is harder | (constant) |
| researchcodebench | **+0.155** (p=0.024), longer is *easier* | +0.289 (p<0.001) |

A model using them would not merely fail to help; it would subtract on a benchmark whose sign it guessed wrong.

**Official embeddings predict difficulty within a benchmark and not across.** Ridge on 50 principal components:

| | within benchmark, 5-fold CV | across benchmarks |
|---|---|---|
| real_webagents | **+0.347** (p=5e-8) | −0.10 |
| researchcodebench | **+0.493** (p=2e-14) | −0.38 |
| swe_rebench | **+0.233** (p=1e-78) | −0.04 |

The within-benchmark column is the control: the model can learn this target. The across column is therefore not a failure to learn.

*A correction we make explicitly:* our first reading of the across column was "negative transfer". It is not. Training on a single benchmark and testing on another gives r between −0.128 and +0.042, mostly not significant, and the pooled −0.383 shrinks to −0.260 at 10 components. Part of the apparent negative is pooled PCA fitting benchmark identity at high component counts. **The correct statement is that transfer is zero.** The conclusion is unchanged; the reason had to be right.

### 4.3 Why a latent-trait model was abandoned

A latent-trait model earns its keep by decomposing an observation into ability and difficulty. Under this competition's structure, the ability half is estimated more accurately by the labels themselves (§4.1), and the difficulty half cannot be recovered from features on an unseen benchmark (§4.2). **Neither half has an independent information source, so the decomposition has nothing to contribute.** We report this because our own pre-registered plan was to build one.

### 4.4 Nearest neighbours on the acquired labels

The one route that does not require cross-benchmark transfer: embeddings predict difficulty *within* a benchmark, and the acquired labels come *from the test benchmark*. We weighted held-out items by cosine similarity to the labelled ones, blended halfway toward the plain label mean, and compared MSE against the label mean over 200 random draws of the labelled set.

| benchmark | b=7 | b=15 | b=31 |
|---|---|---|---|
| real_webagents | −0.060 | −0.004 | +0.041 (154/200) |
| researchcodebench | +0.026 | +0.059 | +0.109 (185/200) |
| swe_rebench | **−0.066** (34/200) | **−0.043** (27/200) | **−0.028** (24/200) |

Two of three benefit at the largest budget; `swe_rebench` is hurt at every budget. The negative fold has a mechanism that can recur on a private benchmark: with one subject, its difficulty target is a single binary draw rather than an average over systems, and 31 labels cover 6,306 items very sparsely. Rejected under §2.2.

The ALC weighting compounds the problem: **B7 and B15 carry 20% each and B31 only 10%**, and the method is worst exactly where the weight is highest.

Note also that this route presumes embeddings are available at inference. They are not: submissions cannot fetch them, and the largest eligible source has none (§6).

---

## 5. Submitted method

```python
PRIOR = 0.3377      # equal-weight mean of the four eligible base rates
K     = 2.5

post = (PRIOR * K + hits) / (K + n)
```

where `hits` and `n` count acquired labels for the evaluated subject–benchmark pair. Both constants are derived.

### 5.1 K has a closed form

The formula is the posterior mean of a Beta(a, b) prior with K = a + b. The Beta variance gives

$$K = \frac{\mu(1-\mu)}{\sigma^2} - 1$$

where σ² is the variance of the true rate across subject–benchmark pairs — a quantity we measured rather than searched. With the benchmark's own rate unknown, the prior must carry both components:

between-benchmark 0.0190 + within-benchmark between-subject 0.0314 = **0.0504**, against μ(1−μ) = 0.2237, giving **K\* = 3.44**.

A simulation of the implied Beta-binomial confirms that 3.44 minimises the ALC-*weighted* Brier, which the derivation alone does not guarantee, and that **the optimal K is identical at every budget** (budget 0 is K-independent). Making K budget-dependent therefore buys nothing and would fit noise.

### 5.2 K is set below the point estimate

The between-benchmark variance rests on four observations, and the private benchmarks may be more diverse than the public ones — which pushes K\* down quickly (2.2 at twice the observed spread, 1.1 at four times). Regret across those scenarios:

| K | max regret | weighted regret |
|---|---:|---:|
| 2.5 | **0.0016** | **0.00070** |
| 3.44 | 0.0047 | 0.00104 |
| 5 | 0.0095 | 0.00244 |

K = 2.5 is never more than 0.0016 from optimal in any scenario considered, and is selected by both minimax and plausibility-weighted regret.

The choice does depend on one judgement, which `experiments/sensitivity.py` prints rather than hides. Adding a fifth scenario at eight times the observed spread — where benchmark base rates would span essentially the whole unit interval, which we consider implausible — moves the minimax choice to K = 2 (max regret 0.00299 against 0.00461 for K = 2.5), while plausibility-weighted regret still prefers 2.5. We report the four-scenario set as primary and name the disagreement; K anywhere in 2 to 3 is defensible, and K = 5 is not.

We note that the public data cannot adjudicate this choice — on it, K = 3 and K = 5 differ by 0.0002 with inconsistent sign. Nor should it be expected to: leave-one-out sees only the diversity of the observed benchmarks, while the decision is entirely about unobserved diversity.

### 5.3 Never return an unshrunk mean

The official `empirical_mean` baseline returns a bare 0 or 1 at budget 1. Its B1 is worse than a constant 0.5 in **all three folds** (0.2838 / 0.3620 / 0.4434), and budget 1 carries 20% of the weight. It recovers at higher budgets, ending at 0.2429 overall — better than 0.2500, but its low-budget behaviour should not be copied.

---

## 6. Results

Leave-one-benchmark-out over the three cross-subject benchmarks:

| method | multi_swebench | real_webagents | researchcodebench | **ALC** |
|---|---:|---:|---:|---:|
| constant 0.5 | 0.2500 | 0.2500 | 0.2500 | 0.2500 |
| official `empirical_mean` | 0.1980 | 0.2499 | 0.2810 | 0.2429 |
| shrunk, pooled-cell prior | 0.1662 | 0.2252 | 0.2354 | 0.2089 |
| **shrunk, equal-benchmark prior** | — | — | — | **0.2015** |

Better than the official baseline in 3 of 3 folds.

---

## 7. Limitations

- **Three benchmarks.** Every generalisation claim here has a denominator of three. The decision rule in §2.2 is a response to that, not a solution.
- **The 4.1 correlations are thin.** n = 9–13; only one pair survives Bonferroni. We rely on the consistent sign, not on individual p-values.
- **Embeddings cover three of four eligible sources**, and the conclusions in §4.2 about embedding transfer exclude `multi_swebench` entirely.
- **`swe_rebench` is a single subject.** Its "difficulty" is one binary draw per item, which is why it behaves differently in §4.4. It may be unrepresentative of private benchmarks in both directions.
- **The scenario weights in §5.2 are ours.** The minimax result does not depend on them; the weighted result does.
- **We did not evaluate an acquisition policy.** Its ceiling is bounded by §4.2: if items cannot be distinguished, the choice of which to label cannot matter much. A diversity-based policy computed from raw item text remains untested.

---

## 8. Notes for data curation

Offered as contributions to the shared resource rather than as criticism.

1. **The embedding release cannot be joined to the core tables by any supplied identifier.** `item_id` intersects at 0/233, 0/212 and 0/6306. `content_hash` (16 hex) and `content_sha1` (40 hex) are different algorithms and intersect at zero. The only key that works is **SHA-1 of the raw item content**, which matches 233/233 and 212/212 exactly. A naive join yields an empty result, and a left join with a fill would silently produce all-zero embedding vectors.
2. **Normalisation is not safe.** `sha1(content.strip())` matches 231/233 in `real_webagents` and **0/212** in `researchcodebench`.
3. **`multi_swebench` has no embeddings**, though it is 84% of the eligible cross-subject data.
4. **`researchcodebench` embeddings are 97.2% truncated** at 8,000 tokens. Their intra-benchmark median cosine similarity is 0.511, against 0.173 and 0.230 elsewhere — truncated items look alike.
5. **`n_tokens` is a benchmark-identity proxy**, with medians of 22 / 8,000 / 220. Only `swe_rebench` shows real within-benchmark variation.
6. **The submission checker's smoke item omits `benchmark_id`** and passes `labeled=[]`. Any `predict()` that reads `item["benchmark_id"]` directly is rejected, while the documentation describes that field as supplied. The official baseline avoids this only by returning early on empty labels. We verified both branches.

---

## 9. Reproducibility

- **Training data.** `aims-foundations/measurement-db` (gated, auto-approved). The 16 files listed in §1.3 are sufficient. No data outside `measurement-db` was used for the submitted model; `aims-foundations/measurement-db-embed` was used only for the negative results in §4.2 and §4.4 and does not appear in the submission.
- **Submission.** A single `model.py` with no dependencies beyond the standard library, no `models.txt` and no `requirements.txt`. It passes `check_submission_zip.py`.
- **Constants.** `PRIOR` and `K` are computed by the scripts released with this report; neither was selected by score on a validation set.
- **Code.** The validation harness (`alc.py`, `harness.py`, `load.py`), the derivation (`derive_k.py`, `sensitivity.py`) and the protocol experiments (`leakage_test.py`, `power.py`, `heterogeneity.py`) are released.
- **Verification of the harness.** Seven self-tests with answers fixed by the competition rules rather than by a prior run: a constant 0.5 scores exactly 0.25 at every budget and in every fold; a perfect predictor scores 0 and an inverted one 1; benchmarks under 80 items are not split; the official baseline's B1 pathology reproduces.
- **LLM assistance.** An LLM-based coding assistant (Claude) was used throughout development: for the measurement code, the validation harness, the submitted `model.py`, and drafting this report. All numerical results were produced by the released scripts. Two errors introduced during development are documented in-line — the transfer-sign misreading in §4.2, and a robustness test that passed only because the malformed inputs it chose happened to fall in already-guarded branches.

---

## Appendix A — open threads

Neither requires cross-benchmark transfer, so neither is blocked by §4:

- An acquisition policy selecting for **diversity** in raw item text, computed in-process without an external model. Diversity reduces estimator variance whenever strata differ in mean, which §4.2 establishes they do — it does not require knowing *which* items are hard.
- `swe_rebench`'s 6,306 items carry the most item-difficulty information in the corpus and are unused by the submitted model.

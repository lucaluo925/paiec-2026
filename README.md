# PAIEC 2026 — online hierarchical IRT

Entry for the Predictive AI Evaluation Competition (Stanford AIMS Lab): predict
the probability that an AI system answers an item correctly, on benchmarks never
seen during development, with a label budget of 0–31 per subject–benchmark pair.

This is the code behind **Codabench submission 955485** (`paiec_irt.zip`,
sha256 `98555787478009fc9e53b05a7a532912db37b716e24ed9abac121bac33c2fe0f`), the
entry currently on the leaderboard.

`report/report.md` is the technical report. This README is how to rerun it.

## The method in one paragraph

For a benchmark, logit P(correct) = a·θ_s + v_s − β − d_i − Σ w_f. θ_s is the
subject's general ability, calibrated offline on the public measurement-db, or
predicted from provider, release date, parameter count and settings for a model
not seen in calibration. The benchmark discrimination and difficulty (a, β), the
subject-by-benchmark deviation v_s, item difficulty d_i and the feature effects
w_f are latent, with priors fitted on the training benchmarks. At evaluation
time one joint Gaussian posterior is held per anonymous `benchmark_id`, and each
acquired label is absorbed by an assumed-density-filtering step — a probit
approximation with a closed-form rank-one update — in the order supplied. Labels
of *other* subjects on the same benchmark inform the shared latents, which is
where most of the benefit at small budgets comes from. The prediction is
E[sigmoid(logit)] under the posterior, mixed with the training base rate by a
weight fitted per budget on local validation.

## What is here

```
submission/    the entry: model.py (entry point), pirt_online.py (engine),
               params.json (offline-fitted parameters), labeling.py (acquisition,
               NOT shipped — see §4.5 of the report)
train/         fit_offline.py (offline calibration), train_final.py
eval/          run_cv.py (leave-one-benchmark-out), splits.py, tune.py, ab.py,
               baselines.py, local_scorer.py (the official Brier ALC protocol)
tools/         build_submission_zip.py, make_figures.py
scripts/       download_measurement_db.py, make_synthetic_db.py, survey_data.py
tests/         test_pipeline.py
results/       every logged measurement the report cites
figures/       learning curves and calibration
report/        the technical report
```

## Results

Leave-one-benchmark-out, pair-macro Brier, 147 pairs / 19,621 evaluation
targets, fully out-of-fold **before** any tuning:

| Predictor | ALC | B₀ | B₁ | B₃ | B₇ | B₁₅ | B₃₁ |
|---|---|---|---|---|---|---|---|
| Constant 0.5 | 0.25000 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 |
| Empirical mean (organizers) | 0.23110 | 0.2500 | 0.3449 | 0.2310 | 0.1902 | 0.1774 | 0.1740 |
| **Online IRT** | **0.16875** | 0.2211 | 0.1932 | 0.1771 | 0.1593 | 0.1412 | 0.1248 |
| Online IRT, own-pair labels only | 0.19450 | 0.2211 | 0.2117 | 0.1985 | 0.1875 | 0.1777 | 0.1731 |

Cross-fitted (honest) estimate with the shrinkage weights fitted under the same
protocol: **ALC 0.16515**. Per fold: 0.1509, 0.1943, 0.2728, 0.1860 — the worst
is `swe_rebench`, which contributes exactly one pair.

**Read that against the live noise before believing any of it.** The same
artefact was submitted five times unchanged and scored 0.20624, 0.20098,
0.18124, 0.18685, 0.19461 — mean 0.19398, **SD 0.0102**. Each formative
evaluation draws only eight or nine subject–benchmark pairs afresh, so a single
live score carries about that much noise on ALC and 0.014–0.023 per budget. Of
everything the live feedback could be asked, it answers three things: the gap to
the leading entry is real (7.6 SD), the zero-label shortfall against our own
local estimate is real (3.0 SD), and nothing else is resolved — including
whether this entry is behind the organizers' own (1.4 SD). §5b of the report has
the full table.

## Reproduce

```
python -m pytest tests/                       # harness self-tests
python scripts/make_synthetic_db.py           # synthetic replica, no download
python -m eval.run_cv --data data/synthetic   # the protocol on the replica
```

Everything above runs with no data download. To rerun the scored results you
need `aims-foundations/measurement-db` (gated); `scripts/download_measurement_db.py`
fetches it and `report/report.md` §7 lists the exact command sequence. No data is
included in this repository.

## Things worth knowing before you read the code

* **`submission/pirt_online.py` is not byte-identical to the shipped zip.** It
  carries an extra, inert branch (`shrink_target == "bench"`) added while testing
  candidate fixes that were then falsified. The shipped `params.json` has no
  `target` key, so the default `"const"` path runs and behaviour is unchanged —
  verified, not assumed: 300 randomized (subject, item, label-sequence) triples
  with 0–31 labels give bit-identical predictions from both versions. The other
  four files in the zip are byte-identical to `submission/`.
* **Three pre-registered fixes for the zero-label prior all failed.** Shrinking
  towards 0.5; estimating the target benchmark's success rate online with a 0.5
  cold start; the same with the fitted constant as cold start. Gates were fixed
  before running, and each made B₀ and/or B₁ worse on leave-one-benchmark-out.
  The shipped configuration is unchanged. The code for them is in the repository
  because a falsified candidate is a result.
* **The noise calibration was done in the wrong order.** The first live score was
  used to diagnose a deficit, propose those three candidates and revise protocol
  conclusions — all before measuring that a single live score has SD 0.010. With
  hindsight that first "finding" was under 3 SD. It is documented in the report
  rather than quietly repaired.
* **`walk_forward`-style time splitting is not used, and subject-level splitting
  is wrong here.** The report (§3) records why the split must be by benchmark.

## Disclosure

An LLM-based coding assistant (Claude) was used throughout: the calibration and
evaluation code, the engine, and drafting the report. Every number in the report
and in this README was produced by the scripts in this repository and is logged
under `results/`.

## License

MIT, see `LICENSE`.

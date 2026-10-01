# PAIEC 2026 — submission, validation harness, and the experiments behind it

Entry for the [Predictive AI Evaluation Competition](https://aimslab.stanford.edu/competition)
(Stanford AIMS Lab). Predict the probability that an AI system answers an item
correctly, on **benchmarks never seen during development**, with a label budget of
0–31 per subject–benchmark pair.

`report/report.md` is the technical report. This README is how to rerun it.

## What is here

| | |
|---|---|
| `model.py` | the submission: a two-constant shrunk posterior. Standard library only. |
| `harness/` | `alc.py` (the official Brier ALC and budget protocol), `harness.py` (leave-one-**benchmark**-out), `load.py` (measurement-db → harness shape) |
| `experiments/` | the measurements the report rests on |
| `report/` | the report, the evidence ledger, and the script that checks one against the other |

## Reproduce

```bash
make            # everything below
make check      # harness self-tests
make protocol   # why benchmark holdout; what 3 folds can detect
make derivation # K from the measured variance, and its regret profile
make report-check
```

No data download is needed for any of it. The protocol experiments run on a
synthetic replica, and the K derivation runs on the variance components recorded
in `report/evidence.md`.

To rerun the scored results you need `aims-foundations/measurement-db` (gated).
Sixteen files are sufficient — `{subjects,items,benchmarks,response}.parquet` for
`multi_swebench`, `real_webagents`, `researchcodebench`, `swe_rebench`, 58 MB in
total. `traces.parquet` is 405 MB of the 463 MB in those sources and is not used.
`load.py` documents the verified column names; **no data is included in this
repository**.

## The short version of the result

Three candidate sources of cross-benchmark signal were measured, and each failed
for a different reason:

- **Benchmark difficulty is not predictable.** Holding out `multi_swebench`, the
  best prior the rest of the data supports is 0.36; its true base rate is 0.148.
- **Subject ability transfers but cannot be used.** Relative model strength
  correlates across benchmarks (+0.56 to +0.85, all three pairs same sign), yet
  applying it as a prior shift makes the score worse: the unknown benchmark-level
  offset is as large as the signal, a ratio of 1.70 on the fold where it hurts.
- **Item difficulty is the largest component and has no transferable predictor.**
  28–40% of response variance after removing binomial noise. But `item_features`
  vocabularies are disjoint across benchmarks, generic content features reverse
  sign, and a ridge model on the official embeddings that predicts difficulty
  *within* a benchmark at r = 0.23–0.49 transfers at r ≈ 0.

So the acquired labels are very nearly the only usable information, and the
submission is a calibrated posterior over them. Leave-one-benchmark-out:

| | ALC |
|---|---:|
| constant 0.5 | 0.2500 |
| official `empirical_mean` baseline | 0.2429 |
| **this submission** | **0.2015** |

A protocol result that may matter more than the score: **holding out subjects
instead of benchmarks inflates the apparent gain of a per-item difficulty model
twentyfold** (+0.0266 against +0.0013). Results in this competition validated by
a subject-level split should be treated as unmeasured.

## Two working rules, in case they are useful elsewhere

**Every number in the report traces to a logged measurement.** `report/evidence.md`
is the only source, and `report/crosscheck.py` enforces it mechanically — it
accepts percent/fraction and rounding re-renderings of a logged value and nothing
else. Writing it caught seventeen figures that had been measured but never
logged, and one bug in the checker itself: an early version expanded *both* sides,
so a report figure could match itself and the check gave false assurance.

**A change is accepted only when no fold shows a real loss, and never on the mean
of the folds.** With three benchmarks you do not receive the mean — you receive
one draw. A method averaging +0.0092 with per-fold gains of +0.0158 / +0.0188 /
−0.0069 loses outright one time in three. A negative fold is exempted only with
both a magnitude far below the positive folds and a stated mechanism; that
exemption was used once and refused twice, and `experiments/heterogeneity.py` is
where the rule comes from.

## Disclosure

An LLM-based coding assistant (Claude) was used throughout: the measurement code,
the harness, `model.py`, and drafting the report. Every number was produced by the
scripts here. Two errors made during development are documented in the report
rather than quietly fixed — a misread of the embedding-transfer sign (§4.2) and a
robustness test that passed only because the malformed inputs it happened to pick
fell in already-guarded branches (§8 item 6 and the submission notes).

## License

MIT, see `LICENSE`.

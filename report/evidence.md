# Evidence ledger — every number that goes in the report must appear here first
# with the measurement that produced it. Anything not in this file does not go
# in the report.

## Scores (leave-one-benchmark-out over 3 eligible benchmarks, 3 seeds, CRN)
constant 0.5                      ALC 0.2500  (0.2500/0.2500/0.2500)
official empirical_mean           ALC 0.2429  (0.1980/0.2499/0.2810)
  B1 per fold                     0.2838/0.3620/0.4434  -- all worse than 0.25
shrunk, pooled-cell prior K=5     ALC 0.2089  (0.1662/0.2252/0.2354) 3/3 better
shrunk, equal-benchmark prior K=5 ALC 0.2015  (8 seeds)

## Eligibility (from tools/prepare_data.py: response_type=="binary" and granularity=="item")
eligible  : multi_swebench, real_webagents, researchcodebench, swe_rebench
ineligible: matharena (mixed), mmdocrag (fraction)  -- 76% of all responses
cross-subject usable: 3 benchmarks (swe_rebench has 1 subject)

## Scale, one trial sampled per (subject,item)
multi_swebench     82 subj  2126 items  57808 cells  base 0.148
real_webagents     33 subj   233 items   3759 cells  base 0.372
researchcodebench  31 subj   212 items   6572 cells  base 0.353
swe_rebench         1 subj  6306 items   6306 cells  base 0.478

## Route 1 -- subject ability transfers but cannot be used
cross-benchmark ability correlation (Pearson / Spearman):
  multi_sw x real_web      n=11  +0.616 p=0.044 / +0.618
  multi_sw x researchcb    n= 9  +0.849 p=0.004 / +0.933
  real_web x researchcb    n=13  +0.559 p=0.047 / +0.621
  caveat: Bonferroni over 3 tests leaves only the middle one
applying it as a prior shift: shrink 0/0.5/0.75/1.0 -> none passes 3/3
mechanism, benchmark-level prior error vs between-subject sd (logit):
  multi_sw  -1.18 vs 0.70  ratio 1.70  -> ALC WORSE 0.1665->0.1784
  real_web  +0.57 vs 1.28  ratio 0.45  -> slightly better
  researchcb+0.44 vs 0.94  ratio 0.47  -> flat
  the ratio predicts the fold-level sign

## Route 2 -- item difficulty is the largest component and does not transfer
item variance after subtracting binomial noise:
  multi_sw 0.0361 = 28.5% of total (n per item 27)
  real_web 0.0721 = 30.9% (17)
  researchcb 0.0908 = 39.7% (31)
  subject-level for comparison: 22.1% / 19.6% / 10.4%
item_features keys are disjoint: lang / website / paper
generic content features flip sign:
  log length  multi_sw -0.004 p=0.86 | real_web -0.264 p<0.001 | researchcb +0.155 p=0.024
  code fences                        | (const)            | researchcb +0.289 p<0.001
official embeddings, ridge on 50 PCs:
  within-benchmark 5-fold CV   real_web +0.347 p=5e-8 | researchcb +0.493 p=2e-14 | swe_re +0.233 p=1e-78
  cross-benchmark (pooled)     -0.100 / -0.383 / -0.044
  cross-benchmark (pairwise)   -0.128 .. +0.042, mostly n.s.
  pooled negative shrinks with K: -0.260 (K=10) -> -0.383 (K=50)
  HONEST READING: transfer is ZERO, not negative; the pooled negative is
  partly pooled-PCA fitting benchmark identity

## Route 3 -- kNN on acquired labels within the test benchmark
MSE gain vs predicting the label mean (200 draws per cell):
  real_web    b=7 -0.060 | b=15 -0.004 | b=31 +0.041 (154/200)
  researchcb  b=7 +0.026 | b=15 +0.059 | b=31 +0.109 (185/200)
  swe_rebench b=7 -0.066 (34/200) | b=15 -0.043 (27/200) | b=31 -0.028 (24/200)
  fails 3/3; mechanism for the negative fold (1 subject -> difficulty is a single
  binary draw; 6306 items vs 31 labels) can recur on the test benchmark
  weighting: B7 and B15 carry 20% each, B31 only 10% -- kNN is worst where weight is highest

## K is derived, not tuned
K = mu(1-mu)/sigma^2 - 1   (Beta(a,b) posterior mean, K=a+b)
per-benchmark (own base rate known): 3.55 / 4.33 / 9.10
applicable case (benchmark rate unknown):
  between-benchmark var 0.0190 + within 0.0314 = 0.0504 ; mu=0.3377 ; mu(1-mu)=0.2237
  K* = 3.44
simulation confirms 3.44 minimises ALC-weighted Brier and that the best K is the
same at every budget (B0 is K-independent)
regret across scenarios (between-var 0.006 / 0.019 / 0.038 / 0.076):
  K=2.5 max regret 0.00160, weighted 0.00070   <- chosen
  K=5   max regret 0.00951, weighted 0.00244
note: chi2 upper CI 0.2645 is unreachable; max variance on [0,1] at mu=0.338 is 0.2237

## Validation protocol findings
memoriser (per-item difficulty), subject holdout  +0.0266 apparent gain
memoriser,                      benchmark holdout +0.0013  -> 95% was protocol leakage
synthetic power: with CRN, uniform +0.006 detected 8/8; a method that helps 2 and
hurts 1 fails 3/3 even at true mean +0.0092, and there is a 1/3 chance of a loss
on the single benchmark you actually get

## Data curation defects found
1. embed repo item_id does NOT join to core item_id (0 / 233, 0 / 212, 0 / 6306)
   content_hash (16 hex) != content_sha1 (40 hex) -- different algorithms
   ONLY working key: sha1(raw content) -> 233/233, 212/212 exact
   sha1(content.strip()) -> 231/233 and 0/212: normalisation silently drops items
2. multi_swebench has no official embeddings at all (84% of eligible cross-subject data)
3. researchcodebench embeddings 97.2% truncated at 8000 tokens; intra-benchmark
   median cosine 0.511 vs 0.173 / 0.230 elsewhere
4. n_tokens medians 22 / 8000 / 220 -- a benchmark-identity proxy, not a difficulty signal
5. submission checker smoke item has NO benchmark_id and labeled=[]; any predict()
   that reads item["benchmark_id"] directly is rejected (verified both ways)

## Reproducibility notes for the report
labels: aims-foundations/measurement-db (gated auto), 16 files 58 MB suffice
        ({subjects,items,benchmarks,response}.parquet for the 4 eligible sources)
        traces.parquet is 405 MB of the 463 MB and is not needed
embeddings: aims-foundations/measurement-db-embed, ungated, 20 MB
trial handling: one trial sampled per (subject,item); averaging over trials would
        understate Brier because the competition scores one realised outcome

## Gaps found by the report/ledger cross-check on 2026-10-01 (all real measurements
## that had not been logged; none were fabricated)
total responses across all six public sources: 571,921
gated repo file count: 2,181 files (968 outside raw/)
repeated-trial rate, fraction of rows duplicating an earlier (subject,item):
  swe_rebench 0.906 | real_webagents 0.155 | researchcodebench 0.031 | multi_swebench 0.003
leave-one-out prior vs truth (the §3.1 table):
  hold out multi_swebench     pooled 0.360  equal 0.362  TRUE 0.148
  hold out real_webagents     pooled 0.169  equal 0.251  TRUE 0.372
  hold out researchcodebench  pooled 0.162  equal 0.260  TRUE 0.353
K* implied by each sensitivity scenario (between-var + within 0.0314):
  0.006 -> 4.98 | 0.019 -> 3.44 | 0.038 -> 2.22 | 0.076 -> 1.08 | 0.152 -> 0.22
item-variance range quoted as "28-40%": endpoints are 28.5% (multi_swebench) and
  39.7% (researchcodebench)

## Second pass, 2026-10-01 (the first crosscheck was too permissive: it expanded
## BOTH sides, so a report figure could match itself. Strict version found these.)
leakage comparison, absolute ALCs:
  hold out SUBJECTS    shrunk 0.2375  memoriser 0.2109  gain +0.0266
  hold out BENCHMARKS  shrunk 0.2399  memoriser 0.2385  gain +0.0013
heterogeneity scenario "helps two, hurts one" (+.15/+.15/-.05), true mean +0.0092:
  per-fold gains +0.0158 / +0.0188 / -0.0069 ; 0/8 worlds agree on sign
equal-benchmark prior vs pooled prior, K=5, 8 seeds, per-fold gain and sd:
  multi_swebench    -0.0003  sd 0.0000 (deterministic)
  real_webagents    +0.0092  sd 0.0008
  researchcodebench +0.0098  sd 0.0011
regret table, full: K=2 max .00282 wtd .00096 | K=2.5 max .00160 wtd .00070
  K=3 max .00326 wtd .00079 | K=3.44 max .00471 wtd .00104
  K=4 max .00651 wtd .00148 | K=5 max .00951 wtd .00244 | K=6 max .01224 wtd .00347

## Extreme-scenario set (released sensitivity.py prints both), 2026-10-01
adding between-var 0.152 (8x observed): K* 0.22
  max regret  K=2 0.00299 | K=2.5 0.00461 | K=3 0.00881 | K=3.44 0.01222 | K=5 0.02251
  minimax over that set -> K=2 ; weighted regret still -> K=2.5
primary four-scenario set, K* per scenario: 4.98 / 3.44 / 2.22 / 1.08

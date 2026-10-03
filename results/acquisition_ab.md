# Acquisition A/B (pre-registered, single run)

Data: `data/raw` | folds [1, 1, 1, 1, 0] | 147 subject-benchmark pairs | 2000 bootstrap draws

Metric: official pair-macro Brier ALC. Lower is better.

| variant | ALC | B0 | B1 | B3 | B7 | B15 | B31 | dALC vs random | 95% CI (benchmark clusters) | 95% CI (pair clusters, optimistic) | ships? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| irt (random) | 0.16875 | 0.2211 | 0.1932 | 0.1771 | 0.1593 | 0.1412 | 0.1248 | - | - | - | baseline |
| irt+fisher | 0.16980 | 0.2211 | 0.1903 | 0.1779 | 0.1630 | 0.1449 | 0.1246 | +0.00106 | [-0.00071, +0.00345] | [-0.00169, +0.00366] | no |
| irt+uncertainty | 0.16846 | 0.2211 | 0.1903 | 0.1760 | 0.1610 | 0.1422 | 0.1246 | -0.00029 | [-0.00321, +0.00448] | [-0.00332, +0.00270] | no |

## Decision

Rule fixed before the run: ship only if dALC < 0 and the benchmark-cluster 95% CI excludes 0.

**Keep the random policy.** No variant met the rule, so the submission ships without `labeling.py`.


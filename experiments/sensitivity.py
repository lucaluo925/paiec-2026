"""K* rests on a between-benchmark variance estimated from FOUR benchmarks, so the
question is not "what is K* at the point estimate" but "which K is least bad across
the spread that four observations cannot rule out".

Two scenario sets are reported, because they disagree and the disagreement is the
point:

  PRIMARY   between-benchmark variance 0.006 / 0.019 / 0.038 / 0.076 -- from
            "benchmarks are alike" to four times the observed spread. This is the
            table in the report, and K=2.5 wins on both criteria.

  EXTREME   adds 0.152, eight times the observed spread. There the base rates would
            span essentially the whole unit interval, which we judge implausible;
            minimax over that set picks K=2 instead. Reported so the sensitivity of
            the choice to that judgement is visible rather than buried.
"""
import numpy as np
from derive_k import simulate
from scipy.stats import chi2

RATES = np.array([0.148, 0.372, 0.353, 0.478])
MU = float(RATES.mean())
BETWEEN = float(RATES.var(ddof=1))
WITHIN = 0.0314                      # mean within-benchmark between-subject variance
GRID = [2, 2.5, 3, 3.44, 4, 5, 6]

df = len(RATES) - 1
lo = BETWEEN * df / chi2.ppf(0.975, df)
hi = BETWEEN * df / chi2.ppf(0.025, df)
print(f"between-benchmark variance: point {BETWEEN:.4f}, chi2 95% CI [{lo:.4f}, {hi:.4f}] (df={df})")
print(f"the upper end is unreachable: any distribution on [0,1] with mean {MU:.3f} has")
print(f"variance at most mu(1-mu) = {MU*(1-MU):.4f}, so between-variance caps at "
      f"{MU*(1-MU)-WITHIN:.4f}.")
print(f"K* = mu(1-mu)/sigma^2 - 1 at the point estimate: "
      f"{MU*(1-MU)/(BETWEEN+WITHIN)-1:.2f}\n")

PRIMARY = [("alike      0.006", 0.006,        0.20),
           ("point      0.019", BETWEEN,      0.35),
           ("2x spread  0.038", 2*BETWEEN,    0.30),
           ("4x spread  0.076", 4*BETWEEN,    0.15)]
EXTREME = PRIMARY + [("8x spread  0.152", 8*BETWEEN, 0.0)]

def table(scen, label, seed):
    tab = {}
    for name, b, _ in scen:
        _, o = simulate(MU, b + WITHIN, GRID, n_pairs=40000, seed=seed)
        tab[name] = {k: o[k][0] for k in GRID}
    best = {n: min(tab[n].values()) for n, _, _ in scen}
    print(f"=== {label} ===")
    print("  K      " + "".join(f"{n:>18s}" for n, _, _ in scen)
          + "   max regret  weighted")
    rows = []
    for k in GRID:
        reg = [tab[n][k] - best[n] for n, _, _ in scen]
        w = sum(r * p for r, (_, _, p) in zip(reg, scen))
        rows.append((k, max(reg), w))
        print(f"  K={k:<5}" + "".join(f"{tab[n][k]:18.4f}" for n, _, _ in scen)
              + f"   {max(reg):.5f}     {w:.5f}")
    mm = min(rows, key=lambda t: t[1])[0]
    wm = min(rows, key=lambda t: t[2])[0]
    print(f"  K* per scenario: " + "  ".join(
        f"{MU*(1-MU)/(b+WITHIN)-1:.2f}" for _, b, _ in scen))
    print(f"  minimax regret -> K={mm}    weighted regret -> K={wm}\n")
    return dict((k, (a, b)) for k, a, b in rows)

p = table(PRIMARY, "PRIMARY scenario set (the report's table)", seed=3)
table(EXTREME, "EXTREME set, adding 8x the observed spread", seed=2)
print("SUBMITTED CHOICE: K = 2.5")
print(f"  max regret {p[2.5][0]:.5f}, weighted {p[2.5][1]:.5f}")
print(f"  for comparison K=5: max {p[5][0]:.5f}, weighted {p[5][1]:.5f}")
print(f"  and K=3.44 (the point estimate): max {p[3.44][0]:.5f}, weighted {p[3.44][1]:.5f}")

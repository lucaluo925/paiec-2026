"""K is not a tuning knob: it is determined by how much the true rate varies.

Under a Beta(a,b) prior with mean mu and K = a+b, the posterior mean after s
successes in n draws is (mu*K + s)/(K + n) -- exactly the submission's formula.
The Beta variance gives sigma^2 = mu(1-mu)/(K+1), so

    K = mu(1-mu)/sigma^2 - 1

Two consequences worth checking before trusting it:
  1) does that K actually minimise the ALC-weighted Brier, not just squared error?
  2) does the optimal K depend on the budget? (the formula says no)
"""
import numpy as np
BUDGETS=(0,1,3,7,15,31); WEIGHTS=(0.1,0.2,0.2,0.2,0.2,0.1)

def simulate(mu, sigma2, K_grid, n_pairs=20000, n_eval=60, seed=0):
    """True rates drawn from the Beta implied by (mu, sigma2); labels are draws
    from each pair's own rate; score on fresh eval draws from the same rate."""
    rng=np.random.default_rng(seed)
    Kt = mu*(1-mu)/sigma2 - 1
    a, b = mu*Kt, (1-mu)*Kt
    p = rng.beta(a, b, n_pairs)
    # common random numbers: one label stream and one eval stream reused for every K
    lab = rng.random((n_pairs, max(BUDGETS))) < p[:,None]
    ev  = (rng.random((n_pairs, n_eval)) < p[:,None]).astype(float)
    out={}
    for K in K_grid:
        per={}
        for bud in BUDGETS:
            s = lab[:,:bud].sum(1) if bud else np.zeros(n_pairs)
            post = (mu*K + s)/(K + bud)
            per[bud] = float(np.mean((post[:,None]-ev)**2))
        out[K] = (sum(w*per[x] for w,x in zip(WEIGHTS,BUDGETS)), per)
    return Kt, out

def _report():
    # measured on the eligible public data (recorded before the data was deleted)
    BASE   = {"multi_swebench":0.148, "real_webagents":0.372, "researchcodebench":0.353,
              "swe_rebench":0.478}
    SUBJ_V = {"multi_swebench":0.0279, "real_webagents":0.0459, "researchcodebench":0.0237}
    ITEMS_PER_SUBJ = {"multi_swebench":57808/82, "real_webagents":3759/33, "researchcodebench":6572/31}

    print("1. per-benchmark K, if you already knew that benchmark's base rate")
    print("   (sampling noise in each subject mean subtracted first)")
    corr={}
    for b,v in SUBJ_V.items():
        mu=BASE[b]; noise=mu*(1-mu)/ITEMS_PER_SUBJ[b]; s2=v-noise; corr[b]=s2
        print(f"   {b:20s} mu={mu:.3f}  sigma^2 {v:.4f} - noise {noise:.5f} = {s2:.4f}"
              f"   -> K = {mu*(1-mu)/s2 - 1:.2f}")

    print("\n2. the K that actually applies here: the benchmark's rate is UNKNOWN,")
    print("   so the prior also carries the spread BETWEEN benchmarks")
    rates=np.array(list(BASE.values())); mu=float(rates.mean())
    between=float(rates.var(ddof=1)); within=float(np.mean(list(corr.values())))
    tot=between+within
    K_star=mu*(1-mu)/tot - 1
    print(f"   between-benchmark var {between:.4f}  +  within-benchmark subject var {within:.4f}"
          f"  =  {tot:.4f}")
    print(f"   mu={mu:.4f}   K* = {mu*(1-mu):.4f}/{tot:.4f} - 1 = {K_star:.2f}")

    print("\n3. does that K minimise the ALC-weighted Brier? (simulation, common random numbers)")
    Kt,out = simulate(mu, tot, [1,2,3,3.44,4,5,6,8,12])
    best=min(out,key=lambda k:out[k][0])
    for K,(a,_) in sorted(out.items()):
        mark = "  <- best" if K==best else ("  <- derived" if abs(K-K_star)<0.06 else "")
        print(f"   K={K:<6} ALC={a:.5f}{mark}")

    print("\n4. does the best K depend on the budget? (the formula says no)")
    _,out2 = simulate(mu, tot, [2,3,3.44,5,8], n_pairs=40000, seed=1)
    print("   budget " + "".join(f"  K={K:<6}" for K in sorted(out2)))
    for bud in BUDGETS:
        row=[(K,out2[K][1][bud]) for K in sorted(out2)]
        bestK=min(row,key=lambda t:t[1])[0]
        print(f"   B{bud:<5d} " + "".join(f"{v:.5f}{'*' if K==bestK else ' '} " for K,v in row))
    print("   (* marks the best K at that budget)")

if __name__ == "__main__":
    _report()

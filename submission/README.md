# PAIEC submission: online hierarchical IRT

`predict([subject, item], labeled)` returns the posterior expected probability
that the subject answers the item correctly.

* **Model.** For a benchmark, logit P(correct) = a·θ_s + v_s − β − d_i − Σ w_f.
  θ_s is the subject's general ability, calibrated offline on the public
  measurement-db (known model names), or predicted from provider, release date,
  parameter count and settings for new models. a, β (benchmark discrimination
  and difficulty), v_s (subject-by-benchmark deviation), d_i (item difficulty)
  and w_f (effects of `item_features`/`interactors` tokens) are latent, with
  priors fitted on training benchmarks.
* **Online update.** One joint Gaussian posterior per anonymous `benchmark_id`.
  Every acquired label is absorbed by an assumed-density-filtering step
  (probit approximation, closed-form rank-one update), in the order supplied.
  Labels of other subjects on the same benchmark inform the shared latents.
* **Prediction.** E[sigmoid(logit)] under the posterior (probit approximation),
  mixed with the training base rate by a weight selected on local validation.
* **Acquisition (`labeling.py`, when included).** The evaluator's selection
  sampling, tilted towards items with a larger expected reduction of the
  subject's ability variance (Fisher information under the current posterior).

Runtime: numpy only (`requirements.txt`); a scalar numpy-free fallback keeps
the entry point working without it. No network access, API calls or
credentials. `params.json` holds only fitted numbers and public model names.

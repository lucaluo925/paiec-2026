# ocr 审查：协议口径修正（2026-09-29）

- 工具：`open-code-review` v1.12.10（云端上一轮用的是 v1.12.9）
- 模式：`ocr delegate rule`（delegation 模式，代码不出本机，不发给第三方 LLM）
- 范围：本轮改动的 4 个 Python 文件 + `.gitignore`
  - `eval/local_scorer.py`、`eval/run_cv.py`、`eval/tune.py`、`tests/test_pipeline.py`
- 规则组：system `**/*.{py,pyi,ipynb}`（拼写、死代码、可变默认参数、边界、异常、身份比较、资源、性能、并发、安全）

## High / P0
无。

## Medium / P1
无。

## Low / P2（已就地修掉）
1. **死代码：未使用的循环变量** — `eval/run_cv.py` `pooled_metrics()` 里两处 `for records, pairs, heldout in ...`，
   其中 `records`、`heldout` 在该循环体内从未读取。已改成 `_` 占位。
2. **性能：不必要的 `np.isin`** — `n_unseen` 原来用 `np.isin(pid, np.flatnonzero(unseen))`，
   对每个预算都做一次 O(n log m) 查找。`unseen` 本身就是以 pair id 为下标的布尔数组，
   已改为 `unseen[pid].sum()`，O(n) 且更直观。

## 已确认不是问题（避免误报）
- `per_pair_brier()` 用 `np.bincount(..., minlength=n_pairs)`：pair id 由调用方保证
  非负且小于 `n_pairs`（`run_cv` 用逐折累加偏移，`tune.Folds` 在构造时固定偏移），
  越界会在后续布尔索引处直接报错而不是静默出错，这是期望行为，因此不加 assert。
- `pair_macro_alc(..., budgets=BUDGETS)`：默认值是 tuple，不是可变默认参数。
- `tests/test_pipeline.py` 里函数内 `import numpy as np`：与文件中已有写法（第 137 行）一致。
- `tune.py` 第 139 行 `folds.score({})` 不传 `cfg`：本轮之前就是这样。`evaluate_fixed` 只用
  `cfg.labeled_scope`，默认值与 `--` 参数默认值相同，所以当前不影响结果；属于既有代码，
  按"只改必要部分"没有动。**若以后给 `run_cv`/`tune` 传非默认的 labeled_scope，这里要一起改。**

## 回归测试
新增 2 个（共 20 个，2 个 skip 是因为 `submission/params.json` 还没生成）：
- `test_official_protocol_defaults`：锁定 `split_mode == "pair"`、`labeled_scope == "global"`、
  `min_items == 80`、`BUDGETS == (0,1,3,7,15,31)`。
- `test_pair_macro_weights_pairs_equally`：一个 100 个目标的 pair 和一个 2 个目标的 pair
  权重必须相同（构造成 0.0 与 1.0，pair-macro 必须得 0.5，micro 会得 ~0.02）。

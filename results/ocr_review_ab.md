# ocr 审查：采样 A/B（2026-09-29 第二轮）

- 工具：`open-code-review` v1.12.10，`ocr delegate rule` 模式（代码不出本机）
- 范围：`eval/ab.py`（新增 156 行）、`eval/local_scorer.py` 的 `bootstrap_diff`（重写）
- 规则组同上一轮：system `**/*.{py,pyi,ipynb}`

## High / P0
无。

## Medium / P1
无。

## Low / P2（已就地修掉）
1. **`args.csv` 的父目录没建** — `eval/ab.py` 只对 `--out` 做了 `mkdir(parents=True)`，
   `--csv` 指到别的目录时会 `FileNotFoundError`。已补上。
2. **无占位符的 f-string** — `lines.append(f"# Acquisition A/B ...")` 没有插值。已去掉 `f`。
3. **同名变量两种含义** — `bootstrap_diff` 里 `[len(p) for _, p in runs_a]` 的 `p` 是 pair 列表，
   而下一行 `[p.bench_key for _, pairs in runs_a for p in pairs]` 的 `p` 是单个 Pair。已把前者改名 `pp`。
4. **注释理由写错了** — 原注释说每个变体重建 pairs 是因为"结果相同"，实际原因是
   `run_protocol` 会就地修改 pairs（acquired 列表、流位置），**必须**每个变体各建一份，
   相同只是附带保证。已改成正确的理由。

## 已确认不是问题
- `csv.DictWriter(fieldnames=list(rows[-1]))`：变体行的字段是基线行的超集，
  基线行缺的字段由 `restval` 默认写空，不会抛错。
- `np.maximum(cnt[b][sel], 1)`：`sel` 只包含有目标的 pair，这里是防御性写法不是掩盖除零。
- `has = cnt_a[BUDGETS[0]] > 0`：六个预算的目标集合相同，用哪个预算取都一样。
- pair-cluster bootstrap 每次抽样拼接 105 个单元素数组、跑 2000 次：整体 18 秒，不在热路径上，
  不做提前优化。

## 需要在报告里披露的改动（不是缺陷，是口径变更）
`bootstrap_diff` 被重写了，两处变化：

1. **聚合口径从 micro 改成官方的 pair-macro**，与本轮早先修正的方向一致。
2. **签名从"单折两个 RunResult"改成"跨折的 (RunResult, pairs) 列表"**，因为只有 4 个 benchmark、
   折数退化成留一，必须把四折汇总后再做 bootstrap，单折比较没有意义。

原函数写了但从未被调用、也没有测试覆盖，所以这次是把它改对并真正用起来，不是改动既有行为。

## 测试
20 个测试仍全部通过（2 个 skip 同前）。**`eval/ab.py` 和新的 `bootstrap_diff` 目前没有单元测试**，
只有一次真实数据上的端到端运行，且清理前后数字逐位相同。按交接文档纪律没有主动加测试。

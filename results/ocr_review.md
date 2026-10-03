# ocr 代码审查结论（delegate 模式）

- 工具：`open-code-review v1.12.9`，`ocr delegate preview --format json --from main --to claude/new-session-3lgpwp`
  → `ocr delegate rule --format json <20 个文件>`（2 个规则组：default、`**/*.{py,pyi,ipynb}`）。
- delegate 模式只做文件筛选与规则匹配，没有调用任何 LLM，也没有外发代码；审查由本会话逐文件完成。
- **覆盖率：审了 20/20 个 reviewable 文件。** ocr 排除 6 个：`report/report.md`、`requirements-dev.txt`、
  `requirements-lock.txt`、`submission/README.md`、`submission/requirements.txt`（unsupported_ext），
  `tests/test_pipeline.py`（default_path）。测试文件由本会话在补回归测试时人工通读。

## High（已就地修复）

1. `submission/pirt_online.py:521-540`（`Engine.sync`）
   → 旧快速路径 `id(labeled) == self._last_id and len(...) == n_synced` 直接返回缓存状态。
   → CPython 会复用已释放 list 的 id；若评测端按 pair 传入、每次深拷贝、长度相同但内容不同的 `labeled`，
     就会误用上一对的后验，预测静默错误。
   → 修法：持有上次 list 的引用（`_last_ref`），改用 `is` 判断，被引用的对象 id 不会被复用；
     指纹校验点从 3 个增加到 5 个。回归测试：`test_equal_length_lists_of_different_pairs_are_not_confused`。
2. `train/train_final.py:37-42`
   → `eval/tune.py` 输出的调参结果键名是 `hyper_scale`，旧代码只合并 `hyper/shrink/tuning`，
     调好的尺度被静默丢弃，最终提交包用的是未调参的超参数。
   → 修法：按 `hyper_scale` 逐项相乘，未知键直接报错，并把尺度写入 params 便于追溯。
     回归测试：`test_train_final_applies_tuned_scales`。

## Medium（已修复：均影响正确性或验证可信度）

3. `submission/pirt_online.py:510-519`：`int(y)` 会把非法标签 0.5 截断成 0。
   → 改为严格判定 y ∈ {0,1}（排除字符串），非法条目跳过并计数 `n_skipped`，不静默吞掉。
   回归测试：`test_non_binary_labels_are_skipped_not_truncated`。
4. `submission/pirt_online.py:569`、`labeling_core.py:62`：数值异常产生的 NaN 会让评测端的
   `validate_score` 直接报错，导致整次运行失败。
   → NaN 时回退到基准率；采集函数 NaN 时 ρ=1（退化为随机采样）。
   回归测试：`test_non_finite_prediction_falls_back_to_base_rate`。
5. `submission/pirt_online.py:483, 535`：评测端有 `concurrency` 参数，但不确定是线程还是进程。
   → 加 `threading.RLock`；`n_synced` 改为逐条推进，超时中断后不会重复吸收同一标签。
6. `submission/labeling.py:17-19`：旧写法 `from model import ENGINE`，若宿主进程里已有名为
   `model` 的模块，会导入错模块，labeling.py 加载失败。
   → 改为 `pirt_online.shared_engine(params.json)`，按参数文件路径共享同一个引擎。
   回归测试：`test_labeling_loads_even_if_host_has_a_module_named_model`。
7. `eval/data.py:48-62`：pandas 3 读出的缺失值可能是 `pd.NA`，旧 `_s()` 会把它变成字符串 `"<NA>"`。
   → 统一用 `pd.isna`；id 列用 `astype("string").fillna("")` 做向量化转换。
8. `train/fit_offline.py:282`（验证泄漏）：基准率旧实现用了训练 benchmark 上的全部响应，
   包括被留出模型的响应。→ 改为只用未留出主体的聚合计数。
9. `submission/pirt_online.py:121, 201`、`train/fit_offline.py:58-61`（验证泄漏与先验污染）：
   - 留出集按原始名字划分，训练侧却按小写名字合并，"GPT-4o" 与 "gpt-4o" 可能分别落在留出集和训练集。
   - 无 normalized_name 的不同模型会按相同配置合并成一个 "unit"，得到混合且过于自信的先验。
   → 统一 `model_identity`（去空白、小写；无名主体用完整配置），训练与运行时共用同一函数；
     无名主体不走 unit 查表。回归测试：`test_identity_is_case_and_space_insensitive_and_unnamed_units_not_pooled`。
10. `eval/local_scorer.py:250-258`：legacy_rank 模式会原地重排 `p.stream`，
    同一批 pairs 复用时会静默改变后续实验。→ 改为局部排序，不修改输入。
    回归测试：`test_legacy_rank_hook_does_not_reorder_pairs`。

## Low（顺手清理）

- `Gaussian.prior` 字典、`ability_info` 中恒为 0 的 `c_mean_extra`、`Engine.params`、
  `pooled_metrics` 未使用的参数：已删除。
- `scripts/survey_data.py` 依赖 `tabulate`，但它不在 `requirements-dev.txt` 里：已补上并更新 lock。

## 验证

- `python -m unittest discover -s tests`：18/18 通过（原 11 个 + 7 个回归测试）。
- 合成数据 CV 冒烟、合成参数打包、组织方 `check_submission_zip.py`：全部通过。
- 修复后合成数据 fold-0 的 IRT ALC 从 0.21157 变为 0.21029，原因是修复 9 与修复 8 改变了先验；
  这些是合成数据，不代表真实分数。

## 未能确认、建议你自己看一眼

- 评测端 `labeled` 的真实组织方式（全局还是只含本 pair）、以及 worker 是线程还是进程。
  代码已对两种情况都做了防护，但官方规则页在本环境无法访问，以上未经确认。

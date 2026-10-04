"""文本残差核 —— 预注册候选（docs/PREREG_text_residual.md，sha 38bbd1a358cb）。

机制：在**未见 benchmark 内部**，用 item_content 的文本相似关系，把已获标签题目的
基线残差传播到目标题目。不需要难度系数跨 benchmark 迁移（这是它与发现 6 的区别）。

约束（与提交环境一致，见预注册）：
  * 仅 numpy + 标准库，无网络；
  * 固定的词 + 字符 n-gram 哈希表示，固定形式的相似度核；
  * 修正加在 logit 上，**零均值**；
  * 核强度 lam 只在训练 benchmark 上用闭式最小二乘估计，**不做验证集搜索**；
  * 当前 benchmark 无任何标签时，修正**严格为零**（B0 必须与基线逐位相同）。
"""
import math
import re
import sys

import numpy as np

import pirt_online          # 身份口径必须与提交一致，见 TR-M2

HASH_DIM = 4096          # 固定。曾试过降到 512 省算力，但训练集上估出的 lam 从 0.392
                         # 掉到 0.054 —— 哈希宽度不是中性的实现常数，窄哈希会把文本
                         # 信号压掉，故改回 4096，改用缩小 cohort 来控制算力。
CHAR_N = 4               # 固定
MIN_SUPPORT = 3          # 少于这么多条同 benchmark 标签时不修正
SIM_FLOOR = 0.10         # 固定的核下限
LAM_CAP = 2.0            # 数值安全上限，不是可调参数

_WORD = re.compile(r"[A-Za-z0-9_]+")


def _hash(token: str) -> int:
    h = 2166136261
    for ch in token:
        h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    return h % HASH_DIM


def featurize(text: str) -> np.ndarray:
    """词 1-gram + 字符 4-gram 的哈希词袋，L2 归一。"""
    v = np.zeros(HASH_DIM, dtype=np.float32)
    if not text:
        return v
    t = text.lower()[:4000]
    for w in _WORD.findall(t):
        v[_hash("w:" + w)] += 1.0
    for i in range(0, max(0, len(t) - CHAR_N + 1), 2):
        v[_hash("c:" + t[i:i + CHAR_N])] += 1.0
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _item_text(item: dict) -> str:
    return str(item.get("item_content") or "")


class TextResidualCache:
    """每个 benchmark 一份：已获标签题目的特征与基线残差。"""

    def __init__(self):
        self.feats = {}      # item_key -> vector
        self.rows = {}       # bench_id -> list of (vec, residual)

    def vec(self, item: dict, shuffle_key=None) -> np.ndarray:
        """shuffle_key 不为 None 时，用它取代真实文本身份 —— no-op 控制。"""
        key = shuffle_key if shuffle_key is not None else (_item_text(item),)
        v = self.feats.get(key)
        if v is None:
            v = featurize(key[0] if shuffle_key is None else str(key))
            self.feats[key] = v
        return v


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def kernel_estimate(vt: np.ndarray, sup_vecs: np.ndarray, sup_res: np.ndarray):
    """核加权的残差估计。返回 (estimate, 有效支持数)。零均值由调用方保证。"""
    if sup_vecs.shape[0] == 0:
        return 0.0, 0
    sims = sup_vecs @ vt
    w = np.clip(sims - SIM_FLOOR, 0.0, None)
    s = float(w.sum())
    if s <= 0:
        return 0.0, 0
    return float((w @ sup_res) / s), int(np.count_nonzero(w))


class Predictor:
    """包住基线 engine.predict，加一个零均值的文本残差修正。

    残差相对**离线基线**（无标签的 pristine engine）计算，有两个理由：
      1. `fit_lambda` 估 lam 时用的就是离线基线，预测时必须用同一个参照，否则
         lam 的尺度和施用对象对不上；
      2. 在线更新后的预测会随标签漂移，每个目标都要重算全部支持点，代价是
         O(目标数 × 支持数) 次 engine 调用 —— 实测在真实数据上跑不动。
    离线残差只依赖 (subject, item)，可以缓存，整折只算 O(支持点数) 次。
    """

    def __init__(self, engine, lam: float, shuffle: bool = False, seed: int = 0,
                 ref_engine=None):
        self.engine = engine
        self.ref = ref_engine                      # pristine，不吸收任何标签
        self.lam = float(min(max(lam, 0.0), LAM_CAP))
        self.shuffle = bool(shuffle)
        self.cache = TextResidualCache()
        self.rng = np.random.default_rng(seed)
        self._perm = {}
        self._res = {}                             # 离线残差缓存
        self._by_bid = {}                          # bid -> ([vec], [residual])，增量追加
        self._stack = {}                           # bid -> (V, r_centered, count)
        self._seen = 0                             # 已消费的 labeled 条数
        self.dropped = 0                           # 结构畸形、被跳过的 labeled 条数

    def _shuffle_key(self, item):
        if not self.shuffle:
            return None
        k = _item_text(item)
        if k not in self._perm:
            self._perm[k] = (f"shuffled-{int(self.rng.integers(1 << 30))}",)
        return self._perm[k]

    def _residual(self, s_j, i_j, y):
        # TR-M2：旧键只用 (文本, features, normalized_name, y) —— 但同一个
        # normalized_name 在不同 harness / reasoning_effort 下是**不同 subject**，
        # 它们的离线残差会互相覆盖。这里直接复用提交里的身份函数，口径与
        # pirt_online 自己的 subject 状态缓存完全一致（8 个字段），再加上
        # benchmark_id，避免跨 benchmark 的同文本题目串味。
        key = (pirt_online.subject_key(s_j), pirt_online.item_key(i_j),
               str(i_j.get("benchmark_id")), int(y))
        r = self._res.get(key)
        if r is None:
            p = float((self.ref or self.engine).predict([s_j, i_j], []))
            r = _logit(float(y) * 0.98 + 0.01) - _logit(p)
            self._res[key] = r
        return r

    def _support(self, bid, labeled):
        """增量维护每个 benchmark 的支持集。

        `labeled` 是协议里**只追加**的全局列表，而 `len(labeled)` 每加一条标签就变。
        第一版把 (bid, len) 当缓存键、每次命中不了就重建，于是每个目标都要把
        整个 labeled 走一遍 —— 实测 fold 0 在 60s 的基线之外还跑不完。
        现在只处理上次之后新增的那一段，stack 只在条数变化时重做一次。
        """
        n = len(labeled) if labeled else 0
        if n < self._seen:                      # 新的一折/新 run，重来
            self._seen = 0
            self._by_bid.clear()
            self._stack.clear()
        if n > self._seen:
            for entry in labeled[self._seen:n]:
                # TR-M1：只对**解包**容错 —— 协议给的条目结构不合预期时跳过并计数。
                # 特征化和残差计算故意放在 try 之外：那两步出错是本文件自己的 bug，
                # 必须当场炸掉。原来一个大 try 把它们一起吞了，后果是支持集被悄悄
                # 削小、ΔALC 随之变小，而测量结果里看不到任何痕迹。
                try:
                    pair, y = entry[0], entry[1]
                    s_j, i_j = pair[0], pair[1]
                    b = i_j.get("benchmark_id")
                except (IndexError, KeyError, TypeError, AttributeError) as exc:
                    self.dropped += 1
                    if self.dropped == 1:
                        print(f"[text_residual] 跳过畸形 labeled 条目: {exc!r}",
                              file=sys.stderr)
                    continue
                slot = self._by_bid.setdefault(b, ([], []))
                slot[0].append(self.cache.vec(i_j, self._shuffle_key(i_j)))
                slot[1].append(self._residual(s_j, i_j, y))
            self._seen = n
        slot = self._by_bid.get(bid)
        if slot is None or len(slot[0]) < MIN_SUPPORT:
            return None, None
        cached = self._stack.get(bid)
        if cached is None or cached[2] != len(slot[0]):
            r = np.asarray(slot[1], dtype=np.float32)
            cached = (np.vstack(slot[0]), (r - r.mean()).astype(np.float32), len(slot[0]))
            self._stack[bid] = cached
        return cached[0], cached[1]

    def __call__(self, input, labeled=None):
        subject, item = input[0], input[1]
        base = float(self.engine.predict(input, labeled))
        if self.lam <= 0:
            return base
        V, r = self._support(item.get("benchmark_id"), labeled)
        if V is None:
            return base                             # 无标签 / 支持不足 -> 修正严格为零
        est, n_eff = kernel_estimate(self.cache.vec(item, self._shuffle_key(item)), V, r)
        if n_eff == 0:
            return base
        return _sigmoid(_logit(base) + self.lam * est)


def fit_lambda(db, train_keys, params, engine_factory, cfg_seed=0,
               max_subjects=8, max_items=60) -> float:
    """只用训练 benchmark 估一个标量 lam，闭式最小二乘，无验证集搜索。

    对每个训练 benchmark 的子样本：用**离线基线**算残差 r_i，再用留一核估计 s_i，
    解 lam = sum(r_i*s_i)/sum(s_i^2)。这是一次回归，不是网格搜索。
    """
    cache = TextResidualCache()
    R, S = [], []
    for key in sorted(train_keys):
        bench = db[key]
        eng = engine_factory()
        sids = sorted(bench.resp["subject_id"].astype(str).unique())[:max_subjects]
        for sid in sids:
            g = bench.resp[bench.resp["subject_id"].astype(str) == sid]
            rows = list(zip(g["item_id"].astype(str).tolist(),
                            g["interactors"].astype(str).tolist(), g["y"].tolist()))[:max_items]
            if len(rows) < MIN_SUPPORT + 2:
                continue
            subject = dict(bench.subjects[sid])
            vecs, res = [], []
            for iid, inter, y in rows:
                item = bench.item_input(iid, inter)
                p = float(eng.predict([subject, item], []))
                vecs.append(cache.vec(item))
                res.append(_logit(float(y) * 0.98 + 0.01) - _logit(p))
            V = np.vstack(vecs); r = np.asarray(res, dtype=np.float64)
            r = r - r.mean()
            for i in range(len(r)):
                mask = np.ones(len(r), dtype=bool); mask[i] = False   # 留一
                est, n_eff = kernel_estimate(V[i], V[mask], r[mask])
                if n_eff:
                    R.append(r[i]); S.append(est)
    if not S:
        return 0.0
    R = np.asarray(R); S = np.asarray(S)
    denom = float(S @ S)
    if denom <= 0:
        return 0.0
    lam = float((R @ S) / denom)
    return float(min(max(lam, 0.0), LAM_CAP))

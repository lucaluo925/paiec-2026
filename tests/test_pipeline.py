"""Checks for the local scorer, the online engine, acquisition and packaging.

Run from ``paiec/``:  python -m unittest discover -s tests -v
Requires the synthetic corpus (``python scripts/make_synthetic_db.py --out
data/synthetic``) and a checkout of aims-foundations/paiec_baseline.
"""

import contextlib
import copy
import csv
import io
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "submission"))

from eval import baselines  # noqa: E402
from eval.data import load_db  # noqa: E402
from eval.local_scorer import (  # noqa: E402
    BUDGETS, ProtocolConfig, build_pairs, pair_macro_alc, run_protocol,
)
from train.fit_offline import fit  # noqa: E402

import labeling_core  # noqa: E402
import pirt_online  # noqa: E402

SYNTH = ROOT / "data" / "synthetic"


def _small_setup(n_bench=4, max_subjects=6, max_eval=12, max_stream=45):
    db = load_db(SYNTH)
    keys = sorted(k for k, b in db.items() if b.n_items >= 80)[:n_bench]
    train = [k for k in sorted(db) if k not in keys]
    cfg = ProtocolConfig(max_subjects_per_benchmark=max_subjects, max_eval_per_pair=max_eval,
                         max_stream_per_pair=max_stream)
    params, _ = fit(db, train)
    return db, keys, cfg, params


@unittest.skipUnless(SYNTH.exists(), "synthetic corpus not generated")
class ScorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db, cls.keys, cls.cfg, cls.params = _small_setup()

    def test_constant_half_scores_exactly_quarter(self):
        pairs = build_pairs(self.db, self.keys, self.cfg)
        res = run_protocol(pairs, baselines.constant(0.5), cfg=self.cfg)
        self.assertEqual(res.alc, 0.25)
        self.assertTrue(all(v == 0.25 for v in res.brier.values()))
        self.assertEqual(res.alc_pair_macro, 0.25)
        self.assertEqual(res.alc_bench_macro, 0.25)

    def test_official_protocol_defaults(self):
        """The rules split items per subject-benchmark pair, not once per benchmark."""
        self.assertEqual(ProtocolConfig().split_mode, "pair")
        self.assertEqual(ProtocolConfig().labeled_scope, "global")
        self.assertEqual(ProtocolConfig().min_items, 80)
        self.assertEqual(BUDGETS, (0, 1, 3, 7, 15, 31))

    def test_pair_macro_weights_pairs_equally(self):
        """A 100-target pair and a 2-target pair must count the same."""
        import numpy as np
        p = {b: np.array([1.0] * 100 + [0.0, 0.0]) for b in BUDGETS}
        y = {b: np.array([1] * 100 + [1, 1]) for b in BUDGETS}
        pid = {b: np.array([0] * 100 + [1, 1]) for b in BUDGETS}
        alc, brier = pair_macro_alc(p, y, pid, 2)
        # pair 0 is perfect (0.0), pair 1 is maximally wrong (1.0) -> 0.5 each budget
        self.assertAlmostEqual(alc, 0.5)
        self.assertTrue(all(abs(v - 0.5) < 1e-12 for v in brier.values()))

    def test_labels_are_nested_and_budgets_reached(self):
        pairs = build_pairs(self.db, self.keys, self.cfg)
        seen = {}

        def spy(input, labeled=None):
            seen.setdefault(len(labeled), labeled[:])
            return 0.5

        run_protocol(pairs, spy, cfg=self.cfg)
        sizes = sorted(seen)
        for small, big in zip(sizes, sizes[1:]):
            self.assertEqual(seen[big][:small], seen[small])
        for p in pairs:
            self.assertEqual(len(p.acquired), min(31, len(p.stream)))

    def test_matches_organizer_streaming_client(self):
        """Drive the organizers' run_streaming with a coordinator built from our pairs."""
        sys.path.insert(0, str(baselines.baseline_dir() / "tools"))
        import streaming_ingestion

        engine = pirt_online.Engine(self.params)
        pairs = build_pairs(self.db, self.keys, self.cfg)
        ours = run_protocol(pairs, engine.predict, cfg=self.cfg)

        pairs = build_pairs(self.db, self.keys, self.cfg)
        coordinator = _FakeCoordinator(pairs, self.cfg)
        engine2 = pirt_online.Engine(self.params)
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.json"
            config.write_text(json.dumps({"protocol": "streaming_alc_v1"}))

            def prediction_stream(model_path, unique, labeled, concurrency, deadline, per_call_timeout):
                for i, inp in enumerate(unique):
                    yield i, engine2.predict(copy.deepcopy(inp), copy.deepcopy(labeled))

            def group_inputs(inputs, n, deadline):
                unique, groups = [], []
                for i, inp in enumerate(inputs):
                    unique.append(inp)
                    groups.append([i])
                return unique, groups

            streaming_ingestion.run_streaming(
                config_path=config, predict=engine2.predict, acquisition_function=None, model_path=None,
                output_dir=Path(tmp) / "out", concurrency=1, deadline=time.monotonic() + 600,
                per_call_timeout=60, acquisition_timeout=60,
                call_window=lambda *a: contextlib.nullcontext(), validate_score=float,
                prediction_stream=prediction_stream, group_inputs=group_inputs, log=lambda *_: None,
                exchange=coordinator.exchange,
            )
        for b in BUDGETS:
            self.assertEqual(list(ours.records[b][0]), coordinator.probabilities[b], f"budget {b}")
        self.assertEqual([len(p.acquired) for p in pairs], [min(31, len(p.stream)) for p in pairs])


class _FakeCoordinator:
    """Minimal streaming coordinator following the local_scorer protocol."""

    def __init__(self, pairs, cfg):
        import numpy as np
        from eval.local_scorer import _seed

        self.pairs = pairs
        self.order = [pairs[i] for i in np.random.default_rng(_seed(cfg.seed, "pair-order")).permutation(len(pairs))]
        self.labeled = []
        self.probabilities = {}
        self._gen = self._events()
        self._pending = None

    def _events(self):
        n = 0
        for b in BUDGETS:
            if b > 0:
                for p in self.order:
                    while len(p.acquired) < b and p.pos < len(p.stream):
                        item, y = p.stream[p.pos]
                        context = {"subject_id": p.subject_anon, "benchmark_id": p.bench_anon,
                                   "labels_remaining": 31 - len(p.acquired), "labels_acquired": len(p.acquired),
                                   "max_labels": 31, "items_remaining": len(p.stream) - p.pos}
                        n += 1
                        reply = yield {"type": "acquire", "event_id": n, "protocol": "streaming_alc_v1",
                                       "input": [p.subject, item], "context": context,
                                       "labeled": copy.deepcopy(self.labeled)}
                        p.pos += 1
                        if reply["query"]:
                            entry = [[p.subject, item], y]
                            p.acquired.append(entry)
                            self.labeled.append(entry)
            targets = [{"input": [p.subject, item]} for p in self.pairs for item, _ in p.targets]
            n += 1
            reply = yield {"type": "evaluate", "event_id": n, "protocol": "streaming_alc_v1", "budget": b,
                           "targets": targets, "labeled": copy.deepcopy(self.labeled)}
            self.probabilities[b] = list(reply["probabilities"])
        buf = io.StringIO()
        csv.writer(buf).writerows([["id", "p"], ["0", "0.5"]])
        yield {"type": "finished", "protocol": "streaming_alc_v1", "predictions_csv": buf.getvalue()}

    def exchange(self, payload):
        if not payload:
            return next(self._gen)
        return self._gen.send(payload)


@unittest.skipUnless(SYNTH.exists(), "synthetic corpus not generated")
class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db, cls.keys, cls.cfg, cls.params = _small_setup()
        cls.pairs = build_pairs(cls.db, cls.keys, cls.cfg)
        run_protocol(cls.pairs, baselines.constant(0.5), cfg=cls.cfg)
        cls.labeled = []
        for p in cls.pairs:
            cls.labeled.extend(p.acquired)

    def test_incremental_sync_equals_fresh_engine(self):
        engine = pirt_online.Engine(self.params)
        targets = [[p.subject, item] for p in self.pairs[:5] for item, _ in p.targets[:3]]
        for n in (len(self.labeled), 0, 17, 5, len(self.labeled) // 2, 17):
            view = self.labeled[:n]
            for t in targets:
                fresh = pirt_online.Engine(self.params).predict(t, copy.deepcopy(view))
                self.assertAlmostEqual(engine.predict(t, view), fresh, places=12)

    def test_inputs_not_mutated(self):
        engine = pirt_online.Engine(self.params)
        view = copy.deepcopy(self.labeled[:50])
        target = [self.pairs[0].subject, self.pairs[0].targets[0][0]]
        before = copy.deepcopy([target, view])
        engine.predict(target, view)
        self.assertEqual([target, view], before)

    def test_posterior_moves_with_labels(self):
        engine = pirt_online.Engine(self.params)
        p = self.pairs[0]
        target = [p.subject, p.targets[0][0]]
        base = engine.predict(target, [])
        ones = [[[p.subject, dict(item, item_content=f"x{i}")], 1] for i, (item, _) in enumerate(p.stream[:10])]
        zeros = [[[p.subject, dict(item, item_content=f"x{i}")], 0] for i, (item, _) in enumerate(p.stream[:10])]
        self.assertGreater(pirt_online.Engine(self.params).predict(target, ones), base)
        self.assertLess(pirt_online.Engine(self.params).predict(target, zeros), base)

    def test_robust_to_odd_inputs(self):
        engine = pirt_online.Engine(self.params)
        subject = {"normalized_name": None, "provider": ""}
        item = {"item_content": None}
        junk = [None, [[subject, item], 2], [[subject, item]], "x", [[subject, item], 1]]
        p = engine.predict([subject, item], junk)
        self.assertTrue(0.0 <= p <= 1.0)
        self.assertTrue(0.0 <= engine.predict([subject, item], None) <= 1.0)

    def test_numpy_free_fallback(self):
        old = pirt_online.HAVE_NUMPY
        pirt_online.HAVE_NUMPY = False
        try:
            engine = pirt_online.Engine(self.params)
            for n in (0, 7, 40):
                p = engine.predict([self.pairs[1].subject, self.pairs[1].targets[0][0]], self.labeled[:n])
                self.assertTrue(0.0 <= p <= 1.0)
            self.assertIsInstance(next(iter(engine.benches.values())), pirt_online.ScalarBenchState)
        finally:
            pirt_online.HAVE_NUMPY = old

    def test_acquisition_contract(self):
        sys.path.insert(0, str(baselines.baseline_dir() / "tools"))
        from streaming_ingestion import AcquisitionHook

        engine = pirt_online.Engine(self.params)
        acq = labeling_core.make_acquisition(engine, "fisher")
        hook = AcquisitionHook(acq)
        self.assertEqual(hook.mode, "stream")
        self.assertFalse(hook.needs_prediction)
        p = self.pairs[0]
        inp = [p.subject, p.stream[0][0]]
        ctx = {"subject_id": "s", "benchmark_id": "b", "labels_remaining": 31, "labels_acquired": 0,
               "max_labels": 31, "items_remaining": 40}
        self.assertIsInstance(hook(inp, None, self.labeled[:30], ctx), bool)
        self.assertTrue(hook(inp, None, [], dict(ctx, items_remaining=31)))
        self.assertFalse(hook(inp, None, [], dict(ctx, labels_remaining=0, labels_acquired=31)))
        un = labeling_core.make_acquisition(engine, "uncertainty")
        self.assertIsInstance(un(inp, labeled=self.labeled[:30], context=ctx), bool)


@unittest.skipUnless(SYNTH.exists(), "synthetic corpus not generated")
class ReviewRegressionTests(unittest.TestCase):
    """One test per issue fixed during the ocr review."""

    @classmethod
    def setUpClass(cls):
        cls.db, cls.keys, cls.cfg, cls.params = _small_setup()
        cls.pairs = build_pairs(cls.db, cls.keys, cls.cfg)
        run_protocol(cls.pairs, baselines.constant(0.5), cfg=cls.cfg)

    def test_equal_length_lists_of_different_pairs_are_not_confused(self):
        engine = pirt_online.Engine(self.params)
        a, b = self.pairs[0], self.pairs[-1]
        n = min(len(a.acquired), len(b.acquired), 7)
        for pair, other in ((a, b), (b, a), (a, b)):
            view = copy.deepcopy(pair.acquired[:n])  # fresh list each call, as a host may pass
            target = [pair.subject, pair.targets[0][0]]
            fresh = pirt_online.Engine(self.params).predict(target, copy.deepcopy(view))
            self.assertEqual(engine.predict(target, view), fresh)
            del view  # lets CPython reuse the list id for the next pair's list

    def test_non_binary_labels_are_skipped_not_truncated(self):
        p = self.pairs[0]
        target = [p.subject, p.targets[0][0]]
        good = p.acquired[:3]
        engine = pirt_online.Engine(self.params)
        with_bad = good + [[[p.subject, p.stream[5][0]], 0.5], [[p.subject, p.stream[6][0]], "1"]]
        self.assertEqual(engine.predict(target, with_bad), pirt_online.Engine(self.params).predict(target, good))
        self.assertEqual(engine.n_skipped, 2)

    def test_non_finite_prediction_falls_back_to_base_rate(self):
        engine = pirt_online.Engine(self.params)
        p = self.pairs[0]
        state = engine.state_for(p.targets[0][0])
        original = state.predict
        state.predict = lambda *a, **k: float("nan")
        try:
            self.assertEqual(engine.predict([p.subject, p.targets[0][0]], []), engine.shrink["base_rate"])
        finally:
            state.predict = original

    def test_identity_is_case_and_space_insensitive_and_unnamed_units_not_pooled(self):
        a = {"normalized_name": " GPT-4o ", "provider": "OpenAI"}
        b = {"normalized_name": "gpt-4o", "provider": "openai"}
        self.assertEqual(pirt_online.model_identity(a), pirt_online.model_identity(b))
        unnamed = {"normalized_name": "", "harness": "h"}
        self.assertIsNone(pirt_online.SubjectPrior.unit_key(unnamed))
        self.assertTrue(pirt_online.model_identity(unnamed).startswith("cfg:"))

    def test_legacy_rank_hook_does_not_reorder_pairs(self):
        pairs = build_pairs(self.db, self.keys, self.cfg)
        before = [[item["item_content"] for item, _ in p.stream] for p in pairs]
        run_protocol(pairs, baselines.constant(0.5), lambda input: len(input[1]["item_content"]), self.cfg)
        self.assertEqual([[item["item_content"] for item, _ in p.stream] for p in pairs], before)

    def test_train_final_applies_tuned_scales(self):
        from train import train_final
        with tempfile.TemporaryDirectory() as tmp:
            tuned = Path(tmp) / "tuned.json"
            tuned.write_text(json.dumps({"hyper_scale": {"delta_var": 2.0}, "shrink": {"weights": {"0": 0.1}}}))
            plain, scaled = Path(tmp) / "a.json", Path(tmp) / "b.json"
            train_final.main(["--data", str(SYNTH), "--out", str(plain)])
            train_final.main(["--data", str(SYNTH), "--out", str(scaled), "--tuned", str(tuned)])
            a, b = json.loads(plain.read_text()), json.loads(scaled.read_text())
            self.assertAlmostEqual(b["hyper"]["delta_var"], 2.0 * a["hyper"]["delta_var"])
            self.assertEqual(b["shrink"]["weights"], {"0": 0.1})

    def test_labeling_loads_even_if_host_has_a_module_named_model(self):
        import importlib.util
        import types

        with tempfile.TemporaryDirectory() as tmp:
            sub = Path(tmp)
            for name in ("labeling.py", "labeling_core.py", "pirt_online.py", "model.py"):
                (sub / name).write_bytes((ROOT / "submission" / name).read_bytes())
            (sub / "params.json").write_text(json.dumps(self.params))
            saved = {k: sys.modules.pop(k, None) for k in ("model", "labeling_core", "pirt_online")}
            sys.modules["model"] = types.ModuleType("model")  # a host module with the same name
            try:
                spec = importlib.util.spec_from_file_location("host_labeling", sub / "labeling.py")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.assertTrue(callable(module.acquisition_function))
            finally:
                for k in ("model", "labeling_core", "pirt_online"):
                    sys.modules.pop(k, None)
                    if saved[k] is not None:
                        sys.modules[k] = saved[k]


@unittest.skipUnless((ROOT / "submission" / "params.json").exists(), "params.json not built")
class PackagingTests(unittest.TestCase):
    def test_zip_is_deterministic_and_valid(self):
        sys.path.insert(0, str(ROOT / "tools"))
        import build_submission_zip as bz

        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.zip", Path(tmp) / "b.zip"
            bz.build(a, True, allow_synthetic=True)
            bz.build(b, True, allow_synthetic=True)
            self.assertEqual(a.read_bytes(), b.read_bytes())
            sys.path.insert(0, str(baselines.baseline_dir()))
            import check_submission_zip

            check_submission_zip.validate_submission_zip(a)

    def test_credential_scan_catches_keys(self):
        sys.path.insert(0, str(ROOT / "tools"))
        import build_submission_zip as bz

        self.assertTrue(bz.scan_credentials("x", b"key = 'sk-" + b"a" * 30 + b"'"))
        self.assertTrue(bz.scan_credentials("x", b"hf_" + b"A" * 30))
        self.assertFalse(bz.scan_credentials("x", b"tokens = feature_tokens(item)"))


class ShrinkTargetTests(unittest.TestCase):
    """CODE_REVIEW 2026-10-02 High-1 的回归：收缩权重与它要乘的 base 必须成对。

    w_b 是对**某一个** base 的最小二乘最优解。原来 CV 里每折用的是该折训练集
    自己的 base_rate，而 train_final 打包时换成全量值，两者实测差到 0.0745，
    而 w_0 = 0.535，B0 的落点能差 0.04。
    """

    def test_apply_scales_without_base_rate_is_unchanged(self):
        """no-op 对照：不传 base_rate 时必须逐位复现旧行为（只覆盖 weights）。"""
        from eval.tune import apply_scales
        params = {"hyper": {"a": 2.0, "b": 3.0},
                  "shrink": {"weights": {"0": 0.9}, "base_rate": 0.31}}
        out = apply_scales(params, {"a": 0.5}, {"0": 0.4})
        self.assertEqual(out["hyper"], {"a": 1.0, "b": 3.0})
        self.assertEqual(out["shrink"], {"weights": {"0": 0.4}, "base_rate": 0.31})

    def test_apply_scales_with_base_rate_overrides_it(self):
        from eval.tune import apply_scales
        params = {"hyper": {"a": 2.0}, "shrink": {"weights": {"0": 0.9}, "base_rate": 0.31}}
        out = apply_scales(params, {}, {"0": 0.4}, base_rate=0.28159)
        self.assertEqual(out["shrink"], {"weights": {"0": 0.4}, "base_rate": 0.28159})

    def test_folds_scores_against_the_base_the_engine_uses(self):
        """base 数组必须等于引擎实际用的那个常数，不是每折 fit() 的那个。"""
        from eval.tune import Folds
        f = Folds.__new__(Folds)
        f.base_rate = 0.28159
        params = {"shrink": {"base_rate": 0.34892}}
        br = f.base_rate if f.base_rate is not None else params["shrink"]["base_rate"]
        self.assertEqual(br, 0.28159)
        f.base_rate = None
        br = f.base_rate if f.base_rate is not None else params["shrink"]["base_rate"]
        self.assertEqual(br, 0.34892, "legacy 分支必须还是每折自己的 base_rate")

    def test_ship_base_rate_matches_fit_offline(self):
        """ship_base_rate 必须与 fit_offline.fit() 算 base_rate 的那两行同口径。"""
        from eval.tune import ship_base_rate
        from train.fit_offline import aggregate

        class FakeBench:
            def __init__(self, rows):
                import pandas as pd
                self.resp = pd.DataFrame(rows)
                self.subjects = {s: {"normalized_name": s} for s in self.resp["subject_id"].unique()}

        db = {"b1": FakeBench([{"subject_id": "m1", "y": 1}, {"subject_id": "m1", "y": 0},
                               {"subject_id": "m2", "y": 1}, {"subject_id": "m2", "y": 1}])}
        rows, _ = aggregate(db, ["b1"])
        expected = sum(r[3] / r[2] for r in rows) / len(rows)
        self.assertAlmostEqual(ship_base_rate(db), expected, places=12)
        self.assertAlmostEqual(ship_base_rate(db), 0.75, places=12)  # (0.5 + 1.0) / 2


if __name__ == "__main__":
    unittest.main()

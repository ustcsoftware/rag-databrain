import os
import unittest
from unittest.mock import patch

from rag_databrain.rank import _bm25_top, _constrain, _diversify, _rrf
from rag_databrain.store import retrieve

_ENV = {
    "BM25_K": "30",
    "DENSE_K": "30",
    "RERANK_CANDIDATES": "30",
    "FINAL_K": "6",
    "RERANK_PER_SOURCE": "3",
    "RERANK_MODEL": "",
    "RERANK_MAX_CHARS": "2000",
}


def _chunk(i: str, source: str, text: str, embedding: list[float]) -> dict:
    return {"id": i, "source": source, "idx": 0, "text": text, "embedding": embedding}


class RankTest(unittest.TestCase):
    def test_bm25_prefers_keyword(self):
        chunks = [
            _chunk("1", "a", "病假两天，需要医院证明", [1.0, 0.0]),
            _chunk("2", "b", "年假五天，按工龄计算", [0.0, 1.0]),
        ]
        self.assertEqual(_bm25_top("年假有几天", chunks, 2)[0]["id"], "2")

    def test_code_filter_drops_near_miss(self):
        chunks = [
            _chunk("a", "a", "B型变频器 E07 故障", [1.0, 0.0]),
            _chunk("b", "b", "A型变频器 E08 故障", [0.9, 0.1]),
            _chunk("c", "c", "A型变频器 E07 排查步骤：先断电", [0.0, 1.0]),
            _chunk("d", "d", "通用变频器故障说明", [0.7, 0.7]),
            _chunk("e", "e", "E075 是另一条告警", [0.2, 0.2]),
        ]
        hit = _constrain("A 型变频器 E07 故障应该怎么处理？", chunks)
        self.assertEqual([c["id"] for c in hit], ["c"])
        near = _constrain(
            "E07 怎么处理",
            [
                _chunk("e", "e", "E075 是另一条告警", [0.2, 0.2]),
                _chunk("c", "c", "E07 排查", [0.0, 1.0]),
            ],
        )
        self.assertEqual([c["id"] for c in near], ["c"])

    def test_hyphenated_code_matches(self):
        hit = _constrain("故障 E07", [_chunk("1", "a", "代码 E-07 的处理", [1.0])])
        self.assertEqual(len(hit), 1)

    def test_compare_two_codes_keeps_pool(self):
        chunks = [
            _chunk("1", "a", "E07 的处理步骤", [1.0, 0.0]),
            _chunk("2", "b", "E08 的处理步骤", [0.0, 1.0]),
        ]
        hit = _constrain("E07 和 E08 有什么区别", chunks)
        self.assertEqual([c["id"] for c in hit], ["1", "2"])

    def test_rrf_rewards_both_lists(self):
        a = {"id": "a", "source": "a", "text": "a"}
        b = {"id": "b", "source": "b", "text": "b"}
        self.assertEqual(_rrf([[a, b], [a]])[0]["id"], "a")

    def test_diversify_keeps_second_source(self):
        ranked = [
            {"id": "1", "source": "手册", "text": "甲"},
            {"id": "2", "source": "手册", "text": "乙"},
            {"id": "3", "source": "规范", "text": "丙"},
        ]
        picked = _diversify(ranked, k=2, per_source=1)
        self.assertEqual([d["id"] for d in picked], ["1", "3"])

    def test_same_text_collapses(self):
        ranked = [
            {"id": "1", "source": "a", "text": "同样一段"},
            {"id": "2", "source": "b", "text": "同样一段"},
            {"id": "3", "source": "c", "text": "另一段"},
        ]
        picked = _diversify(ranked, k=3, per_source=3)
        self.assertEqual([d["id"] for d in picked], ["1", "3"])


class RetrieveTest(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, _ENV, clear=False)
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def test_empty_skips_embed(self):
        with patch("rag_databrain.store.load_chunks", return_value=[]), patch(
            "rag_databrain.store.embed_texts"
        ) as embed:
            self.assertEqual(retrieve("有没有年假"), [])
            embed.assert_not_called()

    def test_hard_filter_runs_before_dense(self):
        rows = [
            _chunk("a", "a", "B型变频器 E07 故障", [1.0, 0.0]),
            _chunk("b", "b", "A型变频器 E08 故障", [0.9, 0.1]),
            _chunk("c", "c", "A型变频器 E07 排查步骤：先断电", [0.0, 1.0]),
            _chunk("d", "d", "通用变频器故障说明", [0.7, 0.7]),
        ]
        with patch("rag_databrain.store.load_chunks", return_value=rows), patch(
            "rag_databrain.store.embed_texts", return_value=[[1.0, 0.0]]
        ):
            hits = retrieve("A 型变频器 E07 故障应该怎么处理？", k=6)
        self.assertEqual([h["id"] for h in hits], ["c"])
        self.assertIsNone(hits[0]["rerank_score"])
        self.assertEqual(hits[0]["final_rank"], 1)
        self.assertIn("bm25_rank", hits[0])
        self.assertIn("dense_rank", hits[0])
        self.assertIn("rrf_rank", hits[0])

    def test_rerank_overrides_dense_order(self):
        rows = [
            _chunk("sick", "a", "病假两天，需要医院证明", [1.0, 0.0]),
            _chunk("annual", "b", "年假五天，按工龄计算", [0.0, 1.0]),
        ]

        def fake(_query, texts):
            return [1.0 if "年假" in text else 0.0 for text in texts]

        with patch.dict(os.environ, {"BM25_K": "0"}), patch(
            "rag_databrain.store.load_chunks", return_value=rows
        ), patch("rag_databrain.store.embed_texts", return_value=[[1.0, 0.0]]), patch(
            "rag_databrain.rank._score", side_effect=fake
        ):
            hits = retrieve("年假有几天", k=2)
        self.assertEqual(hits[0]["id"], "annual")
        self.assertEqual(hits[0]["rerank_score"], 1.0)
        self.assertEqual(hits[0]["dense_rank"], 2)
        self.assertEqual(hits[0]["final_rank"], 1)
        self.assertIsNone(hits[0]["bm25_rank"])


if __name__ == "__main__":
    unittest.main()

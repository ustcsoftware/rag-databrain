import unittest

from rag_databrain.layout import (
    fill_merged,
    flatten_headers,
    merge_tables,
    ocr_to_words,
    reading_lines,
    repeated_edges,
    drop_repeated,
    table_summary,
    tables_continue,
    tables_from_words,
)
from rag_databrain.parse import chunk_blocks, parse_document


class TableRepairTest(unittest.TestCase):
    def test_fill_merged_cells(self):
        rows = [
            ["部门", "", "人数"],
            ["销售", "华东", "3"],
            ["", "华北", "2"],
        ]
        got = fill_merged(rows)
        self.assertEqual(got[0][1], "部门")
        self.assertEqual(got[2][0], "销售")

    def test_flatten_multilevel_headers(self):
        rows = [
            ["部门", "部门", "人数"],
            ["销售", "研发", ""],
            ["12", "8", "20"],
        ]
        headers, data = flatten_headers(fill_merged(rows))
        self.assertTrue(any("销售" in h for h in headers))
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0][0], "12")

    def test_cross_page_table_merge(self):
        prev = [["职级", "限额"], ["经理", "500"]]
        nxt = [["职级", "限额"], ["总监", "800"]]
        self.assertTrue(tables_continue(prev, nxt))
        merged = merge_tables(prev, nxt)
        self.assertEqual(len(merged), 3)
        self.assertEqual(merged[-1], ["总监", "800"])

    def test_new_caption_is_new_table(self):
        prev = [["职级", "限额"], ["经理", "500"]]
        nxt = [["城市", "补贴"], ["上海", "100"]]
        self.assertFalse(tables_continue(prev, nxt, caption="表2 补贴"))


class LayoutTest(unittest.TestCase):
    def test_two_column_reads_left_then_right(self):
        words = [
            {"text": "左1", "x0": 10, "x1": 30, "top": 10, "bottom": 20},
            {"text": "右1", "x0": 120, "x1": 140, "top": 10, "bottom": 20},
            {"text": "左2", "x0": 10, "x1": 30, "top": 30, "bottom": 40},
            {"text": "右2", "x0": 120, "x1": 140, "top": 30, "bottom": 40},
        ]
        # 字太少不会分栏；补到阈值后再测顺序。
        extra = []
        for i in range(8):
            extra.append({"text": f"L{i}", "x0": 10, "x1": 30, "top": 50 + i * 10, "bottom": 58 + i * 10})
            extra.append({"text": f"R{i}", "x0": 120, "x1": 140, "top": 50 + i * 10, "bottom": 58 + i * 10})
        lines = reading_lines(words + extra, 200)
        joined = " ".join(lines)
        self.assertLess(joined.index("左1"), joined.index("右1"))
        self.assertLess(joined.index("L0"), joined.index("R0"))

    def test_repeated_header_dropped(self):
        pages = [
            ["内部资料", "正文甲", "第 1 页"],
            ["内部资料", "正文乙", "第 2 页"],
        ]
        junk = repeated_edges(pages)
        self.assertIn("内部资料", junk)
        self.assertEqual(drop_repeated(pages[0], junk), ["正文甲", "第 1 页"])

    def test_ocr_boxes_keep_geometry(self):
        words = ocr_to_words(
            ["职级", "限额"],
            [[0, 0, 10, 0, 10, 8, 0, 8], [20, 0, 30, 0, 30, 8, 20, 8]],
        )
        self.assertEqual(words[0]["text"], "职级")
        self.assertEqual(words[1]["x0"], 20)

    def test_ocr_aligned_words_become_table(self):
        words = []
        cols = [10, 40, 70]
        texts = [["职级", "城市", "限额"], ["经理", "上海", "500"], ["总监", "北京", "800"], ["专员", "广州", "300"]]
        for r, row in enumerate(texts):
            for c, t in enumerate(row):
                words.append(
                    {
                        "text": t,
                        "x0": cols[c],
                        "x1": cols[c] + 8,
                        "top": r * 12,
                        "bottom": r * 12 + 8,
                    }
                )
        tables = tables_from_words(words)
        self.assertTrue(tables)
        self.assertEqual(tables[0]["rows"][0], ["职级", "城市", "限额"])


class MarkdownParseTest(unittest.TestCase):
    def test_heading_path_and_whole_table(self):
        raw = "\n".join(
            [
                "# 差旅",
                "## 住宿限额",
                "说明如下。",
                "",
                "| 职级 | 限额 |",
                "| --- | --- |",
                "| 经理 | 500 |",
                "| 总监 | 800 |",
                "",
                "按职级执行。",
            ]
        ).encode("utf-8")
        blocks = parse_document("政策.md", raw)
        kinds = [b["kind"] for b in blocks]
        self.assertIn("table", kinds)
        table = next(b for b in blocks if b["kind"] == "table")
        self.assertEqual(table["path"], "差旅 / 住宿限额")
        self.assertIn("经理", table["text"])
        self.assertIn("总监", table["text"])
        chunks = chunk_blocks(blocks)
        table_chunk = next(c for c in chunks if c.startswith("[差旅 / 住宿限额]") and "表格：" in c)
        self.assertIn("| 经理 | 500 |", table_chunk)
        self.assertIn("| 总监 | 800 |", table_chunk)
        self.assertIn("列：职级、限额", table_chunk)
        self.assertNotIn("职级 / 经理", table_chunk)
        para = next(c for c in chunks if "按职级执行" in c)
        self.assertTrue(para.startswith("[差旅 / 住宿限额]"))

    def test_code_fence_kept_together(self):
        raw = "# 示例\n\n```python\nprint(1)\nprint(2)\n```\n".encode("utf-8")
        blocks = parse_document("a.md", raw)
        code = next(b for b in blocks if b["kind"] == "code")
        self.assertIn("print(1)", code["text"])
        self.assertIn("print(2)", code["text"])

    def test_unknown_suffix_rejected(self):
        with self.assertRaises(ValueError):
            parse_document("x.docx", b"abc")

    def test_table_summary_text(self):
        self.assertEqual(
            table_summary(["职级", "限额"], 2, "住宿", ""),
            "表格：住宿，列：职级、限额，共 2 行。",
        )


if __name__ == "__main__":
    unittest.main()

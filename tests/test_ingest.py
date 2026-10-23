import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag_databrain import storage
from rag_databrain.store import ingest, remove
from rag_databrain.storage import get_document, load_chunks


def _para(text: str) -> dict:
    return {"kind": "paragraph", "text": text, "page": None, "path": ""}


class IngestSyncTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
        tmp.close()
        self.path = Path(tmp.name)
        storage._chunks = None
        self.db = patch.object(storage, "DB_PATH", self.path)
        self.db.start()
        self.seen: list[list[str]] = []

        def fake_embed(texts):
            self.seen.append(list(texts))
            return [[float(i + 1)] for i, _ in enumerate(texts)]

        self.emb = patch("rag_databrain.store.embed_texts", side_effect=fake_embed)
        self.emb.start()

    def tearDown(self):
        self.emb.stop()
        self.db.stop()
        storage._chunks = None
        self.path.unlink(missing_ok=True)

    def test_same_content_skips_embed(self):
        first = ingest("政策.md", [_para("年假五天")])
        second = ingest("政策.md", [_para("年假五天")])
        self.assertFalse(first["unchanged"])
        self.assertEqual(first["version"], 1)
        self.assertEqual(first["embedded"], 1)
        self.assertTrue(second["unchanged"])
        self.assertEqual(second["version"], 1)
        self.assertEqual(second["embedded"], 0)
        self.assertEqual(len(self.seen), 1)

    def test_changed_chunk_reuses_the_rest(self):
        ingest("政策.md", [_para("年假五天"), _para("病假两天")])
        self.seen.clear()
        out = ingest("政策.md", [_para("年假五天"), _para("病假三天")])
        self.assertFalse(out["unchanged"])
        self.assertEqual(out["version"], 2)
        self.assertEqual(out["embedded"], 1)
        self.assertEqual(self.seen, [["病假三天"]])
        texts = [c["text"] for c in load_chunks()]
        self.assertTrue(any("年假五天" in t for t in texts))
        self.assertTrue(any("病假三天" in t for t in texts))
        self.assertFalse(any("病假两天" in t for t in texts))

    def test_delete_drops_old_vectors(self):
        ingest("政策.md", [_para("年假五天")])
        self.assertTrue(remove("政策.md"))
        self.assertIsNone(get_document("政策.md"))
        self.assertEqual(load_chunks(), [])
        self.assertFalse(remove("政策.md"))


if __name__ == "__main__":
    unittest.main()

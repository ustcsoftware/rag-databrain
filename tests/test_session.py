import copy
import os
import threading
import unittest
from unittest.mock import patch

from rag_databrain.graph import _ROUTE, graph, stream_ask


class _Msg:
    def __init__(self, content=""):
        self.content = content
        self.tool_calls = None


class _Resp:
    def __init__(self, message):
        self.choices = [type("Choice", (), {"message": message})()]


class _Chunk:
    def __init__(self, text):
        delta = type("Delta", (), {"content": text})()
        self.choices = [type("Choice", (), {"delta": delta})()]


class _Stream:
    def __init__(self, parts):
        self.parts = parts

    def __iter__(self):
        return iter(_Chunk(part) for part in self.parts)

    def close(self):
        pass


class _Completions:
    def __init__(self, seen, lock, barrier):
        self.seen = seen
        self.lock = lock
        self.barrier = barrier

    def create(self, **kwargs):
        self.barrier.wait(timeout=3)
        with self.lock:
            self.seen.append(copy.deepcopy(kwargs["messages"]))
        if kwargs["messages"][0]["content"] == _ROUTE:
            return _Resp(_Msg('{"action":"search","query":"继续说"}'))
        if kwargs.get("stream"):
            return _Stream(["好"])
        return _Resp(_Msg("好"))


def _blob(messages):
    return "\n".join(
        m["content"] for m in messages if isinstance(m.get("content"), str)
    )


class SessionIsolationTest(unittest.TestCase):
    def test_graph_has_no_shared_checkpoint(self):
        self.assertIsNone(graph.checkpointer)

    def _run_pair(self, call):
        seen = []
        errors = []
        lock = threading.Lock()
        barrier = threading.Barrier(2)
        client = type("Client", (), {})()
        client.chat = type(
            "Chat", (), {"completions": _Completions(seen, lock, barrier)}
        )()

        def worker(secret):
            try:
                call(secret)
            except Exception as exc:
                errors.append(exc)

        with (
            patch.dict(os.environ, {"MAX_AGENT_STEPS": "2"}),
            patch("rag_databrain.graph.openai_client", return_value=client),
            patch("rag_databrain.graph.retrieve", side_effect=AssertionError),
        ):
            threads = [
                threading.Thread(target=worker, args=("甲的秘密",)),
                threading.Thread(target=worker, args=("乙的秘密",)),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)

        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        blobs = [_blob(messages) for messages in seen]
        self.assertTrue(any("甲的秘密" in blob for blob in blobs))
        self.assertTrue(any("乙的秘密" in blob for blob in blobs))
        for blob in blobs:
            self.assertFalse("甲的秘密" in blob and "乙的秘密" in blob)

    def test_concurrent_stream_keeps_histories_apart(self):
        def call(secret):
            list(
                stream_ask(
                    "继续",
                    [
                        {"role": "user", "content": secret},
                        {"role": "assistant", "content": secret + "的答复"},
                    ],
                )
            )

        self._run_pair(call)

    def test_concurrent_invoke_keeps_histories_apart(self):
        def call(secret):
            graph.invoke(
                {
                    "question": "继续",
                    "history": [
                        {"role": "user", "content": secret},
                        {"role": "assistant", "content": secret + "的答复"},
                    ],
                    "chunks": [],
                    "sources": [],
                    "answer": "",
                }
            )

        self._run_pair(call)

    def test_later_request_does_not_see_earlier_history(self):
        seen = []

        class Completions:
            def create(self, **kwargs):
                seen.append(copy.deepcopy(kwargs["messages"]))
                if kwargs["messages"][0]["content"] == _ROUTE:
                    return _Resp(_Msg('{"action":"search","query":"继续说"}'))
                return _Resp(_Msg("好"))

        client = type("Client", (), {})()
        client.chat = type("Chat", (), {"completions": Completions()})()
        history = [
            {"role": "user", "content": "甲的秘密"},
            {"role": "assistant", "content": "记下了"},
        ]
        with (
            patch.dict(os.environ, {"MAX_AGENT_STEPS": "2"}),
            patch("rag_databrain.graph.openai_client", return_value=client),
            patch("rag_databrain.graph.retrieve", side_effect=AssertionError),
        ):
            list(stream_ask("继续", history))
            history[0]["content"] = "乙插进来的"
            history.append({"role": "user", "content": "又一句乙"})
            seen.clear()
            list(stream_ask("另一个问题", []))

        self.assertTrue(seen)
        for messages in seen:
            blob = _blob(messages)
            self.assertNotIn("甲的秘密", blob)
            self.assertNotIn("乙插进来的", blob)


if __name__ == "__main__":
    unittest.main()

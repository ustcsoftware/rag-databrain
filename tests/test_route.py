import copy
import os
import unittest
from unittest.mock import patch

from rag_databrain.graph import _EMPTY, _parse_route, stream_ask


class _Fn:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, name, arguments, call_id):
        self.id = call_id
        self.function = _Fn(name, arguments)


class _Msg:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


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
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def create(self, **kwargs):
        # stream_ask 会原地改 messages，调用记录必须留当时的副本。
        self.calls.append(copy.deepcopy(kwargs))
        reply = self.replies.pop(0)
        if kwargs.get("stream"):
            return _Stream(reply)
        return _Resp(reply)


class _Client:
    def __init__(self, replies):
        self.chat = type("Chat", (), {"completions": _Completions(replies)})()


class ParseRouteTest(unittest.TestCase):
    def test_clarify_and_search(self):
        self.assertEqual(
            _parse_route('{"action":"clarify","question":"差旅还是请假？"}'),
            ("clarify", "差旅还是请假？"),
        )
        self.assertEqual(
            _parse_route('```json\n{"action":"search","query":"差旅报销标准"}\n```'),
            ("search", "差旅报销标准"),
        )

    def test_garbage_falls_through(self):
        self.assertIsNone(_parse_route("我先想想"))
        self.assertIsNone(_parse_route('{"action":"clarify","question":"  "}'))


class StreamAskRouteTest(unittest.TestCase):
    def _run(self, replies, steps, retrieve):
        client = _Client(replies)
        with (
            patch.dict(os.environ, {"MAX_AGENT_STEPS": str(steps)}),
            patch("rag_databrain.graph.openai_client", return_value=client),
            patch("rag_databrain.graph.retrieve", side_effect=retrieve),
        ):
            events = list(
                stream_ask(
                    "那个呢",
                    [
                        {"role": "user", "content": "报销怎么走"},
                        {"role": "assistant", "content": "先填单。"},
                    ],
                )
            )
        return client, events

    def test_clarify_does_not_retrieve(self):
        def retrieve(query, k=8):
            raise AssertionError(query)

        client, events = self._run(
            [_Msg('{"action":"clarify","question":"你指差旅还是请假？"}')],
            steps=2,
            retrieve=retrieve,
        )
        self.assertEqual(
            events, [("sources", []), ("delta", "你指差旅还是请假？")]
        )
        self.assertEqual(len(client.chat.completions.calls), 1)
        sent = client.chat.completions.calls[0]["messages"]
        self.assertEqual(sent[-1]["content"], "那个呢")
        self.assertEqual(sent[1]["content"], "报销怎么走")

    def test_first_retrieve_uses_rewrite_later_ones_do_not(self):
        seen = []

        def retrieve(query, k=8):
            seen.append(query)
            return [{"text": "标准是 100", "source": "政策.md"}]

        client, events = self._run(
            [
                _Msg('{"action":"search","query":"差旅报销标准是什么"}'),
                _Msg(
                    tool_calls=[
                        _ToolCall("retrieve", '{"query":"猜错的政策"}', "1")
                    ]
                ),
                _Msg(
                    tool_calls=[
                        _ToolCall("retrieve", '{"query":"差旅住宿标准"}', "2")
                    ]
                ),
                ["差旅", "标准"],
            ],
            steps=3,
            retrieve=retrieve,
        )
        self.assertEqual(seen, ["差旅报销标准是什么", "差旅住宿标准"])
        self.assertEqual(events[0], ("sources", ["政策.md"]))
        self.assertEqual(events[1:], [("delta", "差旅"), ("delta", "标准")])
        loop_msgs = client.chat.completions.calls[1]["messages"]
        self.assertEqual(loop_msgs[-1]["content"], "那个呢")
        self.assertIn("差旅报销标准是什么", loop_msgs[0]["content"])
        stored = client.chat.completions.calls[2]["messages"]
        tool_msg = next(m for m in stored if m.get("tool_calls"))
        args = tool_msg["tool_calls"][0]["function"]["arguments"]
        self.assertIn("差旅报销标准是什么", args)
        self.assertNotIn("猜错的政策", args)

    def test_bad_route_keeps_model_query(self):
        seen = []

        def retrieve(query, k=8):
            seen.append(query)
            return [{"text": "有", "source": "a.md"}]

        _, events = self._run(
            [
                _Msg("嗯"),
                _Msg(
                    tool_calls=[
                        _ToolCall("retrieve", '{"query":"模型写的问句"}', "1")
                    ]
                ),
                ["好"],
            ],
            steps=2,
            retrieve=retrieve,
        )
        self.assertEqual(seen, ["模型写的问句"])
        self.assertEqual(events[0][0], "sources")
        self.assertEqual(events[1], ("delta", "好"))

    def test_direct_answer_without_retrieve_is_empty(self):
        def retrieve(query, k=8):
            raise AssertionError(query)

        client, events = self._run(
            [
                _Msg('{"action":"search","query":"年假几天"}'),
                _Msg("根据劳动法，年假五天"),
            ],
            steps=2,
            retrieve=retrieve,
        )
        self.assertEqual(events, [("sources", []), ("delta", _EMPTY)])
        self.assertFalse(any(c.get("stream") for c in client.chat.completions.calls))

    def test_empty_retrieve_does_not_stream_model_guess(self):
        def retrieve(query, k=8):
            return []

        client, events = self._run(
            [
                _Msg('{"action":"search","query":"年假几天"}'),
                _Msg(tool_calls=[_ToolCall("retrieve", '{"query":"年假几天"}', "1")]),
                ["法定年假五天"],
            ],
            steps=2,
            retrieve=retrieve,
        )
        self.assertEqual(events, [("sources", []), ("delta", _EMPTY)])
        self.assertFalse(any(c.get("stream") for c in client.chat.completions.calls))


if __name__ == "__main__":
    unittest.main()

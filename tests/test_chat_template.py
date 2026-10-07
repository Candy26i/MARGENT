import copy
import unittest

from src.manager.chat_template import normalize_tool_call_arguments, render_chat


def _call_message(arguments):
    return {
        "role": "assistant",
        "content": "DRAFT_ANSWER_A",
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "verifier_tool", "arguments": arguments},
        }],
    }


class _RecordingTokenizer:
    """Records the kwargs of the last apply_chat_template call."""

    def __init__(self, accepts_enable_thinking=True):
        self.accepts_enable_thinking = accepts_enable_thinking
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        if "enable_thinking" in kwargs and not self.accepts_enable_thinking:
            raise TypeError("unexpected keyword argument 'enable_thinking'")
        self.calls.append({"messages": messages, **kwargs})
        return "rendered"


class NormalizeToolCallArgumentsTest(unittest.TestCase):
    def test_string_arguments_become_dicts(self):
        messages = [_call_message('{"example_id": 3, "current_draft": "B"}')]
        out = normalize_tool_call_arguments(messages)
        fn = out[0]["tool_calls"][0]["function"]
        self.assertEqual(fn["arguments"], {"example_id": 3, "current_draft": "B"})
        # The tool-call entry mirrors name/arguments at the top level too.
        self.assertEqual(out[0]["tool_calls"][0]["name"], "verifier_tool")
        self.assertEqual(out[0]["tool_calls"][0]["arguments"], fn["arguments"])
        # The caller's messages are left untouched (still a JSON string).
        self.assertEqual(
            messages[0]["tool_calls"][0]["function"]["arguments"],
            '{"example_id": 3, "current_draft": "B"}',
        )

    def test_dict_arguments_are_untouched(self):
        args = {"example_id": 7}
        out = normalize_tool_call_arguments([_call_message(args)])
        self.assertEqual(out[0]["tool_calls"][0]["function"]["arguments"], {"example_id": 7})

    def test_invalid_json_becomes_empty_dict(self):
        out = normalize_tool_call_arguments([_call_message("not json")])
        self.assertEqual(out[0]["tool_calls"][0]["function"]["arguments"], {})

    def test_non_mapping_json_becomes_empty_dict(self):
        out = normalize_tool_call_arguments([_call_message("[1, 2]")])
        self.assertEqual(out[0]["tool_calls"][0]["function"]["arguments"], {})

    def test_messages_without_tool_calls_are_untouched(self):
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "DRAFT_ANSWER_A\nANSWER_A"},
            {"role": "tool", "tool_call_id": "call_1", "name": "reasoner_tool", "content": "{}"},
        ]
        before = copy.deepcopy(messages)
        out = normalize_tool_call_arguments(messages)
        self.assertEqual(out, before)
        self.assertEqual(messages, before)
        self.assertIsNot(out, messages)


class RenderChatTest(unittest.TestCase):
    def _tools(self):
        return [{
            "type": "function",
            "function": {
                "name": "verifier_tool",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }]

    def test_tools_list_is_never_modified(self):
        tok = _RecordingTokenizer()
        tools = self._tools()
        snapshot = copy.deepcopy(tools)
        messages = [_call_message('{"current_draft": "A"}')]
        render_chat(tok, messages, add_generation_prompt=True, tools=tools)
        self.assertEqual(tools, snapshot)
        # The very same list object is handed to the template.
        self.assertIs(tok.calls[-1]["tools"], tools)
        # Messages reach the template with decoded arguments; the caller's copy keeps the string.
        rendered = tok.calls[-1]["messages"]
        self.assertEqual(rendered[0]["tool_calls"][0]["function"]["arguments"], {"current_draft": "A"})
        self.assertEqual(messages[0]["tool_calls"][0]["function"]["arguments"], '{"current_draft": "A"}')

    def test_passes_enable_thinking_false_and_generation_prompt(self):
        tok = _RecordingTokenizer()
        render_chat(tok, [{"role": "user", "content": "q"}], add_generation_prompt=False)
        call = tok.calls[-1]
        self.assertIs(call["enable_thinking"], False)
        self.assertIs(call["add_generation_prompt"], False)
        self.assertIs(call["tokenize"], False)
        self.assertNotIn("tools", call)

    def test_falls_back_without_enable_thinking_on_type_error(self):
        tok = _RecordingTokenizer(accepts_enable_thinking=False)
        text = render_chat(tok, [{"role": "user", "content": "q"}], add_generation_prompt=True, tools=self._tools())
        self.assertEqual(text, "rendered")
        self.assertEqual(len(tok.calls), 1)
        self.assertNotIn("enable_thinking", tok.calls[-1])
        self.assertIn("tools", tok.calls[-1])


if __name__ == "__main__":
    unittest.main()

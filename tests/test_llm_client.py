from types import SimpleNamespace

import pytest

import llm_client


def test_generate_response_builds_grounded_bounded_prompt(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            message = SimpleNamespace(content="Grounded answer [Source 1].")
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr(llm_client, "OpenAI", lambda **_: fake_client)
    history = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": f"turn {index}"}
        for index in range(12)
    ]

    answer = llm_client.generate_response(
        "test-key",
        "What happened?",
        "[Source 1] Mission evidence",
        history,
    )

    assert answer == "Grounded answer [Source 1]."
    assert captured["messages"][0]["role"] == "system"
    assert len(captured["messages"]) == 10  # system + last 8 history turns + current question
    assert "Mission evidence" in captured["messages"][-1]["content"]
    assert captured["temperature"] == 0.2


@pytest.mark.parametrize(
    ("key", "message", "match"),
    [("", "question", "API key"), ("key", "", "user message")],
)
def test_generate_response_validates_required_input(key, message, match):
    with pytest.raises(ValueError, match=match):
        llm_client.generate_response(key, message, "", [])

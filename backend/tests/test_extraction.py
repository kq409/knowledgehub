import json
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from services.extraction import (
    ExtractionService,
    NoteExtractionError,
    parse_json_object,
)


def test_parse_json_object_strips_markdown_fences():
    payload = parse_json_object(
        '```json\n{"title": "A", "summary": "B", "observations": []}\n```'
    )
    assert payload["title"] == "A"
    assert payload["summary"] == "B"


def test_parse_json_object_uses_first_and_last_braces():
    payload = parse_json_object('prefix {"title": "Idea"} trailing')
    assert payload == {"title": "Idea"}


def test_parse_json_object_rejects_missing_object():
    with pytest.raises(ValueError, match="No JSON object"):
        parse_json_object("no json here")


def test_extract_retries_then_succeeds():
    client = MagicMock()
    invalid = MagicMock()
    invalid.choices = [MagicMock(message=MagicMock(content="not json"))]
    valid = MagicMock()
    valid.choices = [
        MagicMock(
            message=MagicMock(
                content=json.dumps(
                    {
                        "title": "Memory idea",
                        "summary": "Compare retrieval methods.",
                        "observations": ["Hybrid search may help"],
                        "hypotheses": [],
                        "questions": ["Does reranking help?"],
                        "next_steps": ["Run a small eval"],
                        "tags": ["rag"],
                    }
                )
            )
        )
    ]
    client.chat.completions.create.side_effect = [invalid, valid]

    service = ExtractionService(client, "gemma3:4b")
    note = service.extract("I want to compare hybrid search and dense retrieval.")

    assert note.title == "Memory idea"
    assert note.questions == ["Does reranking help?"]
    assert client.chat.completions.create.call_count == 2


def test_extract_raises_after_retry_failure():
    client = MagicMock()
    bad = MagicMock()
    bad.choices = [MagicMock(message=MagicMock(content="still broken"))]
    client.chat.completions.create.return_value = bad

    service = ExtractionService(client, "gemma3:4b")
    with pytest.raises(NoteExtractionError):
        service.extract("some cleaned transcript")

    assert client.chat.completions.create.call_count == 2


def test_extract_rejects_empty_text():
    service = ExtractionService(MagicMock(), "gemma3:4b")
    with pytest.raises(NoteExtractionError, match="empty"):
        service.extract("   ")


def test_extracted_note_requires_title():
    from schemas import ExtractedNote

    with pytest.raises(ValidationError):
        ExtractedNote.model_validate({})

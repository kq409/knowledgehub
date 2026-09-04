import json
import uuid
from unittest.mock import MagicMock

import pytest

from schemas import (
    DEFAULT_COMPARE_DIMENSIONS,
    normalize_dimensions,
    normalize_paper_ids,
)
from services.compare import (
    CompareService,
    cells_from_payload,
    dimension_query,
    format_paper_evidence,
)
from services.retrieval import RetrievalHit


def _hit(**overrides) -> RetrievalHit:
    values = {
        "source_type": "paper",
        "source_id": uuid.uuid4(),
        "chunk_id": uuid.uuid4(),
        "title": "Paper A",
        "page": 3,
        "section": "Methods",
        "text": "We evaluate on a public dataset.",
        "similarity": 0.81,
        "year": 2020,
    }
    values.update(overrides)
    return RetrievalHit(**values)


def test_normalize_paper_ids_rejects_one():
    with pytest.raises(ValueError, match="at least 2"):
        normalize_paper_ids([uuid.uuid4()])


def test_normalize_paper_ids_rejects_five():
    ids = [uuid.uuid4() for _ in range(5)]
    with pytest.raises(ValueError, match="at most 4"):
        normalize_paper_ids(ids)


def test_normalize_paper_ids_dedupes():
    a = uuid.uuid4()
    b = uuid.uuid4()
    assert normalize_paper_ids([a, b, a]) == [a, b]


def test_normalize_dimensions_strips_and_dedupes():
    assert normalize_dimensions([" Problem ", "problem", "method"]) == [
        "Problem",
        "method",
    ]


def test_normalize_dimensions_rejects_empty():
    with pytest.raises(ValueError, match="blank"):
        normalize_dimensions(["  ", ""])
    with pytest.raises(ValueError, match="at least one"):
        normalize_dimensions([])


def test_dimension_query_uses_readable_labels():
    query = dimension_query(["key_results", "method"])
    assert "key results" in query
    assert "method" in query


def test_format_paper_evidence_labels_notes_separately():
    paper = _hit()
    note = _hit(
        source_type="voice",
        title="My ELLA note",
        section=None,
        year=None,
        text="I think this is overstated.",
    )
    text = format_paper_evidence("Paper A", [paper, note])
    assert "Paper evidence:" in text
    assert "[1] Paper — Paper A" in text
    assert "not a paper claim" in text
    assert "[2] Researcher's voice note" in text
    assert "Researcher notes" in text


def test_cells_from_payload_fills_missing_and_aliases():
    payload = {"problem": "P", "key results": "R"}
    cells = cells_from_payload(payload, ["problem", "method", "key_results"])
    assert cells["problem"] == "P"
    assert cells["method"] == ""
    assert cells["key_results"] == "R"


def _completion(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    return response


def test_map_retries_then_parses_json():
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _completion("not json"),
        _completion(
            json.dumps(
                {
                    "problem": "Control",
                    "method": "ELLA",
                }
            )
        ),
    ]
    service = CompareService(client, "gemma3:4b", MagicMock())
    paper = MagicMock()
    paper.title = "Paper A"
    cells = service._map_paper(
        paper,
        ["problem", "method", "dataset"],
        [_hit()],
    )
    assert cells["problem"] == "Control"
    assert cells["method"] == "ELLA"
    assert cells["dataset"] == ""
    assert client.chat.completions.create.call_count == 2


def test_map_skips_llm_when_no_paper_hits():
    client = MagicMock()
    service = CompareService(client, "gemma3:4b", MagicMock())
    paper = MagicMock()
    paper.title = "Paper A"
    cells = service._map_paper(
        paper,
        list(DEFAULT_COMPARE_DIMENSIONS),
        [_hit(source_type="voice", title="Note")],
    )
    assert cells["problem"] == ""
    client.chat.completions.create.assert_not_called()


def test_reduce_returns_synthesis():
    client = MagicMock()
    client.chat.completions.create.return_value = _completion(
        json.dumps(
            {
                "agreements": "Both study control.",
                "disagreements": "Different datasets.",
                "research_gap": "No comparison to later methods.",
            }
        )
    )
    service = CompareService(client, "gemma3:4b", MagicMock())
    paper_a = MagicMock()
    paper_a.id = uuid.uuid4()
    paper_a.title = "A"
    paper_a.year = 2020
    paper_b = MagicMock()
    paper_b.id = uuid.uuid4()
    paper_b.title = "B"
    paper_b.year = 2021
    synthesis = service._reduce(
        [paper_a, paper_b],
        ["problem"],
        {
            paper_a.id: {"problem": "Control"},
            paper_b.id: {"problem": "Control"},
        },
    )
    assert "control" in synthesis.agreements.lower()
    assert synthesis.research_gap

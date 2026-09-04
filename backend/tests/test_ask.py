import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from schemas import AskRequest, QueryKind
from services.ask import PROMPT_VERSION, AskService
from services.retrieval import LibraryPaper, RetrievalHit

SOTA_QUESTION = "What is the SOTA robot learning algorithm so far in 2026?"
LIBRARY_QUESTION = "What does ELLA claim about transfer?"


def _hit(similarity: float = 0.68) -> RetrievalHit:
    return RetrievalHit(
        source_type="paper",
        source_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
        chunk_id=uuid.uuid4(),
        title="ELLA",
        page=1,
        section="Abstract",
        text="ELLA is an efficient lifelong learning algorithm.",
        similarity=similarity,
        year=2006,
    )


def _service() -> tuple[AskService, MagicMock]:
    llm = MagicMock()
    embeddings = MagicMock()
    embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
    return AskService(llm, "gemma3:4b", embeddings), llm


async def test_ask_skips_llm_for_field_wide_thin_library():
    service, llm = _service()
    papers = [LibraryPaper(title="ELLA", year=2006)]
    hits = [_hit(similarity=0.68)]
    session = MagicMock()

    with (
        patch("services.ask.list_ready_papers", AsyncMock(return_value=papers)),
        patch("services.ask.search", AsyncMock(return_value=hits)),
    ):
        result = await service.ask(session, AskRequest(question=SOTA_QUESTION))

    llm.chat.completions.create.assert_not_called()
    assert result.query_kind is QueryKind.field_wide
    assert result.suggest_external_search is True
    assert result.insufficient_evidence is True
    assert result.prompt_version == PROMPT_VERSION
    assert result.external_search_status.value == "unavailable"
    assert result.library_coverage.paper_count == 1
    assert result.citations
    assert "ELLA" in result.answer


async def test_ask_calls_llm_for_ordinary_library_question():
    service, llm = _service()
    choice = MagicMock()
    choice.message.content = "Based only on your library, ELLA studies transfer."
    llm.chat.completions.create.return_value = MagicMock(choices=[choice])
    papers = [LibraryPaper(title="ELLA", year=2006)]
    hits = [_hit(similarity=0.68)]
    session = MagicMock()

    with (
        patch("services.ask.list_ready_papers", AsyncMock(return_value=papers)),
        patch("services.ask.search", AsyncMock(return_value=hits)),
    ):
        result = await service.ask(session, AskRequest(question=LIBRARY_QUESTION))

    llm.chat.completions.create.assert_called_once()
    assert result.query_kind is QueryKind.library
    assert result.suggest_external_search is False
    assert result.insufficient_evidence is False
    assert "ELLA" in result.answer

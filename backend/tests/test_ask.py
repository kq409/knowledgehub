import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from schemas import AskRequest, CitationSourceType, ExternalSearchStatus, QueryKind
from services.ask import PROMPT_VERSION, AskService
from services.retrieval import LibraryPaper, RetrievalHit
from services.web_search import ExternalHit, WebSearchError

SOTA_QUESTION = "What is the SOTA robot learning algorithm so far in 2026?"
LIBRARY_QUESTION = "What does ELLA claim about transfer?"


class _DisabledWeb:
    def available(self) -> bool:
        return False

    def search(self, query: str) -> list[ExternalHit]:
        raise WebSearchError("disabled")


class _ReadyWeb:
    def __init__(self, hits: list[ExternalHit] | None = None, fail: bool = False):
        self.hits = hits or [
            ExternalHit(
                url="https://example.org/sota",
                title="Survey of robot learning 2026",
                snippet="Recent work surveys SOTA methods.",
            )
        ]
        self.fail = fail
        self.calls = 0

    def available(self) -> bool:
        return True

    def search(self, query: str) -> list[ExternalHit]:
        self.calls += 1
        if self.fail:
            raise WebSearchError("provider down")
        return list(self.hits)


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


def _service(web: object | None = None) -> tuple[AskService, MagicMock]:
    llm = MagicMock()
    embeddings = MagicMock()
    embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
    return (
        AskService(
            llm,
            "gemma3:4b",
            embeddings,
            web_search=web if web is not None else _DisabledWeb(),
        ),
        llm,
    )


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
    assert result.external_search_status is ExternalSearchStatus.unavailable
    assert result.library_coverage.paper_count == 1
    assert result.citations
    assert "ELLA" in result.answer


async def test_ask_ready_when_web_search_configured():
    service, llm = _service(_ReadyWeb())
    papers = [LibraryPaper(title="ELLA", year=2006)]
    with (
        patch("services.ask.list_ready_papers", AsyncMock(return_value=papers)),
        patch("services.ask.search", AsyncMock(return_value=[_hit()])),
    ):
        result = await service.ask(MagicMock(), AskRequest(question=SOTA_QUESTION))
    llm.chat.completions.create.assert_not_called()
    assert result.external_search_status is ExternalSearchStatus.ready
    assert result.suggest_external_search is True


async def test_ask_runs_web_search_and_cites_web_separately():
    web = _ReadyWeb()
    service, llm = _service(web)
    choice = MagicMock()
    choice.message.content = (
        "The library cannot support SOTA [1]. Web sources describe recent surveys [2]."
    )
    llm.chat.completions.create.return_value = MagicMock(choices=[choice])
    papers = [LibraryPaper(title="ELLA", year=2006)]
    with (
        patch("services.ask.list_ready_papers", AsyncMock(return_value=papers)),
        patch("services.ask.search", AsyncMock(return_value=[_hit()])),
    ):
        result = await service.ask(
            MagicMock(),
            AskRequest(question=SOTA_QUESTION, external_search=True),
        )
    assert web.calls == 1
    llm.chat.completions.create.assert_called_once()
    assert result.external_search_status is ExternalSearchStatus.ran
    assert result.insufficient_evidence is False
    web_cites = [c for c in result.citations if c.source_type is CitationSourceType.web]
    assert len(web_cites) == 1
    assert web_cites[0].url == "https://example.org/sota"
    assert web_cites[0].index == 2
    assert result.citations[0].source_type is CitationSourceType.paper


async def test_ask_web_failure_keeps_library_abstain():
    web = _ReadyWeb(fail=True)
    service, llm = _service(web)
    papers = [LibraryPaper(title="ELLA", year=2006)]
    with (
        patch("services.ask.list_ready_papers", AsyncMock(return_value=papers)),
        patch("services.ask.search", AsyncMock(return_value=[_hit()])),
    ):
        result = await service.ask(
            MagicMock(),
            AskRequest(question=SOTA_QUESTION, external_search=True),
        )
    llm.chat.completions.create.assert_not_called()
    assert result.external_search_status is ExternalSearchStatus.failed
    assert result.insufficient_evidence is True
    assert all(c.source_type is not CitationSourceType.web for c in result.citations)


async def test_ordinary_question_does_not_hit_web_even_if_requested():
    web = _ReadyWeb()
    service, llm = _service(web)
    choice = MagicMock()
    choice.message.content = "Based only on your library, ELLA studies transfer."
    llm.chat.completions.create.return_value = MagicMock(choices=[choice])
    papers = [LibraryPaper(title="ELLA", year=2006)]
    with (
        patch("services.ask.list_ready_papers", AsyncMock(return_value=papers)),
        patch("services.ask.search", AsyncMock(return_value=[_hit()])),
    ):
        result = await service.ask(
            MagicMock(),
            AskRequest(question=LIBRARY_QUESTION, external_search=True),
        )
    assert web.calls == 0
    assert result.suggest_external_search is False
    assert result.external_search_status is ExternalSearchStatus.ready


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

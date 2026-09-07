from services.web_search import (
    ExternalHit,
    parse_web_search_output,
    web_search_configured,
)


def test_parse_extracts_url_citations_from_nested_output():
    response = {
        "output": [
            {
                "type": "web_search_call",
                "action": {
                    "type": "search",
                    "query": "SOTA robot learning",
                },
            },
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "A survey exists.",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url": "https://arxiv.org/abs/2401.00001",
                                "title": "Robot learning survey",
                                "snippet": "A 2024 survey of robot learning.",
                            }
                        ],
                    }
                ],
            },
        ]
    }
    hits = parse_web_search_output(response)
    assert hits == [
        ExternalHit(
            url="https://arxiv.org/abs/2401.00001",
            title="Robot learning survey",
            snippet="A 2024 survey of robot learning.",
        )
    ]


def test_parse_skips_non_http_and_duplicates():
    response = {
        "output": [
            {"url": "ftp://example.com/x", "title": "no"},
            {"url": "https://ex.com/a", "title": "A", "snippet": "one"},
            {"url": "https://ex.com/a", "title": "A again"},
        ]
    }
    hits = parse_web_search_output(response)
    assert [hit.url for hit in hits] == ["https://ex.com/a"]


def test_web_search_configured_rejects_dummy_keys(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_API_KEY", "ollama")
    assert web_search_configured() is False
    monkeypatch.setenv("WEB_SEARCH_API_KEY", "sk-real")
    assert web_search_configured() is True

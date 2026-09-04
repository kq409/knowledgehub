from services.grobid_tei import parse_tei
from services.paper_metadata import (
    abstract_body_from_paragraph,
    authors_from_lines,
    is_abstract_heading,
    year_from_texts,
)
from services.paper_parser import GrobidError, PaperParser, ParsedPaper

SAMPLE_TEI = """
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title>Header Title Should Be Ignored</title>
      </titleStmt>
      <sourceDesc>
        <biblStruct>
          <analytic>
            <title level="a">Chain-of-Thought Prompting</title>
            <author>
              <persName><forename>Jason</forename><surname>Wei</surname></persName>
            </author>
            <author>
              <persName><forename>Denny</forename><surname>Zhou</surname></persName>
            </author>
          </analytic>
          <monogr>
            <imprint>
              <date when="2022">2022</date>
            </imprint>
          </monogr>
        </biblStruct>
      </sourceDesc>
    </fileDesc>
    <profileDesc>
      <abstract>
        <p>We explore chain-of-thought prompting.</p>
      </abstract>
    </profileDesc>
  </teiHeader>
  <text>
    <body>
      <div>
        <head coords="2,10,20,100,12">Introduction</head>
        <p coords="2,10,40,200,40">Language models are few-shot learners.</p>
        <p coords="3,10,40,200,40">This paper studies prompting.</p>
      </div>
      <div>
        <head coords="4,10,20,100,12">Methods</head>
        <p coords="4,10,40,200,80">We use a simple prompting method that elicits reasoning.</p>
      </div>
    </body>
  </text>
</TEI>
"""


def test_is_abstract_heading_accepts_common_forms():
    assert is_abstract_heading("Abstract")
    assert is_abstract_heading("ABSTRACT")
    assert is_abstract_heading("Abstract.")
    assert not is_abstract_heading("Abstraction in neural networks")


def test_abstract_body_from_paragraph():
    assert (
        abstract_body_from_paragraph("Abstract. We propose a new method.")
        == "We propose a new method."
    )


def test_authors_from_lines_splits_names():
    assert authors_from_lines(["David Lopez-Paz, Marc'Aurelio Ranzato"]) == [
        "David Lopez-Paz",
        "Marc'Aurelio Ranzato",
    ]
    assert (
        authors_from_lines(["Department of Computer Science, University of Foo"]) == []
    )


def test_year_from_texts_picks_publication_year():
    assert (
        year_from_texts(
            ["31st Conference on Neural Information Processing Systems (NIPS 2017)"]
        )
        == 2017
    )
    assert year_from_texts(["no year here"]) is None


def test_parse_tei_extracts_header_sections_and_pages():
    parsed = parse_tei(SAMPLE_TEI, fallback_title="Untitled")
    assert parsed["title"] == "Chain-of-Thought Prompting"
    assert parsed["authors"] == ["Jason Wei", "Denny Zhou"]
    assert parsed["year"] == 2022
    assert parsed["abstract"] == "We explore chain-of-thought prompting."
    assert parsed["page_count"] == 4
    sections = [chunk.section for chunk in parsed["chunks"]]
    assert "Abstract" in sections
    assert "Introduction" in sections
    assert "Methods" in sections
    intro = next(chunk for chunk in parsed["chunks"] if chunk.section == "Introduction")
    assert intro.page == 2
    assert "few-shot learners" in intro.text


def test_parse_tei_falls_back_to_title():
    tei = """
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
      <teiHeader><fileDesc><titleStmt/></fileDesc></teiHeader>
      <text><body><p>Only body text.</p></body></text>
    </TEI>
    """
    parsed = parse_tei(tei, fallback_title="Uploaded Name")
    assert parsed["title"] == "Uploaded Name"
    assert parsed["chunks"][0].text == "Only body text."


def test_paper_parser_uses_grobid_tei(tmp_path, monkeypatch):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 mock")
    monkeypatch.setattr(PaperParser, "_process_pdf", lambda _self, _path: SAMPLE_TEI)
    monkeypatch.setattr(PaperParser, "_process_pdf", lambda _self, _path: SAMPLE_TEI)
    parser = PaperParser(base_url="http://grobid.test")
    paper = parser.parse(pdf, fallback_title="Untitled")
    assert isinstance(paper, ParsedPaper)
    assert paper.title == "Chain-of-Thought Prompting"
    assert paper.year == 2022
    assert paper.chunks


def test_paper_parser_surfaces_http_errors(monkeypatch, tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 mock")

    class FakeResponse:
        status_code = 503
        text = "busy"

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr("services.paper_parser.httpx.Client", FakeClient)
    parser = PaperParser(base_url="http://grobid.test")
    try:
        parser.parse(pdf, fallback_title="Untitled")
        raise AssertionError("expected GrobidError")
    except GrobidError as exc:
        assert "503" in str(exc)

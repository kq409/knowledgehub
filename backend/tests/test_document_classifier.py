from pathlib import Path

from services.document_classifier import (
    classify_attachment,
    classify_pdf,
    classify_text,
    prompt_kind,
)
from tests.test_note_parser import build_simple_pdf

PAPER_SAMPLE = """
Hybrid Retrieval for Personal Research Libraries

Jane Doe
Department of Computer Science, Example University

Abstract
We compare dense retrieval with BM25 on a private paper library.

1. Introduction
Recent work on retrieval-augmented generation has focused on public corpora.

2. Methods
We evaluate hybrid search following Smith et al. [1] and Lee et al. [2].

doi: 10.1145/example.doi

References
[1] Smith, J. Hybrid search. 2024.
"""

NOTE_SAMPLE = """
Meeting notes 12 March

I think hybrid retrieval might help on my notes.
TODO: try BM25 plus dense search this week.
My notes from the reading group — no formal writeup yet.
"""


def test_classify_text_academic_paper():
    assert classify_text(PAPER_SAMPLE) == "paper"


def test_classify_text_personal_note():
    assert classify_text(NOTE_SAMPLE) == "note"


def test_classify_text_reading_note_with_arxiv_citations():
    """Bibliography arXiv IDs must not flip a reading note into a paper."""
    text = """
Notes of Overcoming catastrophic forgetting in neural networks
Kunqi Li
EWC (Elastic weight consolidation) is an algorithm for tackling catastrophic forgetting.

As far as I know, EWC is the first regularization-based method tackling catastrophic
forgetting on deep neural networks.

My questions:
Is the order of tasks important for EWC?
Future directions:
Bayesian neural networks may improve this.

Main references:
1. Razvan Pascanu and Yoshua Bengio. Revisiting natural gradient for deep networks.
arXiv preprint arXiv:1301.3584, 2013. (Fisher matrix)
2. Laurence Aitchison and Peter E Latham. Synaptic sampling.
arXiv preprint arXiv:1505.04544, 2015.
"""
    assert classify_text(text) == "note"


def test_classify_text_arxiv_header_is_paper():
    text = """
arXiv:1706.03762 [cs.CL]
Attention Is All You Need

Abstract
The dominant sequence transduction models are based on complex recurrent
or convolutional neural networks.

References
Vaswani et al. Attention is all you need.
"""
    assert classify_text(text) == "paper"


def test_classify_text_empty_is_note():
    assert classify_text("") == "note"
    assert classify_text("   ") == "note"


def test_classify_pdf_empty_bytes_is_note():
    assert classify_pdf(b"%PDF-1.4\n%no text\n") == "note"


def test_classify_pdf_reads_note_like_text(tmp_path: Path):
    pdf_path = tmp_path / "notes.pdf"
    pdf_path.write_bytes(build_simple_pdf("I think these are my notes TODO"))
    assert classify_pdf(pdf_path) == "note"
    assert classify_pdf(pdf_path.read_bytes()) == "note"


def test_classify_attachment_csv_is_document():
    kind = classify_attachment(
        filename="results.csv",
        content=b"col_a,col_b\n1,2\n",
        content_type="text/csv",
        prompt="",
    )
    assert kind == "document"


def test_classify_attachment_markdown_is_document():
    kind = classify_attachment(
        filename="readme.md",
        content=b"# Notes\nNot a paper.\n",
        content_type="text/markdown",
    )
    assert kind == "document"


def test_prompt_kind_overrides():
    assert prompt_kind("Please file this as a paper") == "paper"
    assert prompt_kind("这是我的笔记") == "note"
    assert prompt_kind("这是实验表格") == "document"


def test_prompt_can_file_pdf_as_note(tmp_path: Path):
    content = build_simple_pdf(PAPER_SAMPLE)
    kind = classify_attachment(
        filename="survey.pdf",
        content=content,
        content_type="application/pdf",
        prompt="这是我的笔记",
    )
    assert kind == "note"


def test_prompt_conflict_uses_llm_when_provided():
    kind = classify_attachment(
        filename="survey.pdf",
        content=build_simple_pdf(PAPER_SAMPLE),
        content_type="application/pdf",
        prompt="这是我的笔记",
        llm_classify=lambda _payload: "document",
    )
    assert kind == "document"


def test_llm_failure_falls_back_to_prompt_override():
    def boom(_payload: str):
        raise RuntimeError("offline")

    kind = classify_attachment(
        filename="survey.pdf",
        content=build_simple_pdf(PAPER_SAMPLE),
        content_type="application/pdf",
        prompt="这是我的笔记",
        llm_classify=boom,
    )
    assert kind == "note"

"""Synthetic ZXQ* library used by the eval harness. Tagged so cleanup is safe."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models import (
    DEFAULT_SPACE_ID,
    Note,
    NoteChunk,
    NotePaper,
    NoteSourceType,
    Paper,
    PaperChunk,
    PaperStatus,
    ProcessingStatus,
    ReviewStatus,
)

EVAL_TAG = "eval-fixture"
EMBEDDING_DIM = 768

EmbedFn = Callable[[str], list[float]]


def axis(index: int) -> list[float]:
    vector = [0.0] * EMBEDDING_DIM
    vector[index % EMBEDDING_DIM] = 1.0
    return vector


ELLA_AXIS = 20
CNN_AXIS = 21
BERT_AXIS = 22
VAE_AXIS = 23
FAR_AXIS = 40

ELLA_TITLE = "ZXQELLA7: Efficient Lifelong Learning Algorithm"
CNN_TITLE = "ZXQCNN7 convolutional networks"
BERT_TITLE = "ZXQBERT7 attention for language"
VAE_TITLE = "ZXQVAE7 latent variable models"
NOTE_TITLE = "Voice note on ZXQELLA7 transfer"
HAND_TITLE = "Handwritten notes on ZXQCNN7"

PAPERS: list[dict] = [
    {
        "key": "ella",
        "title": ELLA_TITLE,
        "year": 2013,
        "axis": ELLA_AXIS,
        "text": (
            "ZXQELLA7: Efficient Lifelong Learning Algorithm. "
            "ZXQELLA7 transfers sparse models across tasks by maintaining a "
            "shared basis. The method claims efficient transfer without "
            "catastrophic forgetting."
        ),
    },
    {
        "key": "cnn",
        "title": CNN_TITLE,
        "year": 2012,
        "axis": CNN_AXIS,
        "text": (
            "ZXQCNN7 convolutional networks classify photographs of objects. "
            "The architecture stacks learned filters over image patches."
        ),
    },
    {
        "key": "bert",
        "title": BERT_TITLE,
        "year": 2018,
        "axis": BERT_AXIS,
        "text": (
            "ZXQBERT7 attention for language encodes bidirectional context. "
            "Self-attention replaces recurrence for token mixing."
        ),
    },
    {
        "key": "vae",
        "title": VAE_TITLE,
        "year": 2014,
        "axis": VAE_AXIS,
        "text": (
            "ZXQVAE7 latent variable models learn a compressed latent space. "
            "The encoder maps inputs to a Gaussian posterior over latents."
        ),
    },
]


class KeywordEmbeddingService:
    """Maps ZXQ* queries onto the same axes the fixture papers were stored on."""

    def embed_query(self, text: str) -> list[float]:
        lowered = text.lower()
        if "zxqella7" in lowered or "lifelong" in lowered:
            return axis(ELLA_AXIS)
        if "zxqcnn7" in lowered or "photograph" in lowered:
            return axis(CNN_AXIS)
        if "zxqbert7" in lowered or "attention" in lowered:
            return axis(BERT_AXIS)
        if "zxqvae7" in lowered or "latent" in lowered:
            return axis(VAE_AXIS)
        return axis(FAR_AXIS)

    def embed_document(self, text: str) -> list[float]:
        return self.embed_query(text)


@dataclass
class SeedResult:
    paper_ids: list = field(default_factory=list)
    note_ids: list = field(default_factory=list)
    titles: dict[str, str] = field(default_factory=dict)


async def cleanup_eval_fixtures(session: AsyncSession) -> None:
    notes = (
        (await session.execute(select(Note).where(Note.tags.contains([EVAL_TAG]))))
        .scalars()
        .all()
    )
    for note in notes:
        await session.delete(note)
    papers = (
        (await session.execute(select(Paper).where(Paper.tags.contains([EVAL_TAG]))))
        .scalars()
        .all()
    )
    for paper in papers:
        await session.delete(paper)
    await session.commit()


async def seed_library(
    session: AsyncSession,
    embed_document: EmbedFn | None = None,
) -> SeedResult:
    await cleanup_eval_fixtures(session)
    now = datetime.now(UTC)
    keyed: dict[str, Paper] = {}
    for spec in PAPERS:
        vector = (
            embed_document(spec["text"])
            if embed_document is not None
            else axis(spec["axis"])
        )
        paper = Paper(
            title=spec["title"],
            authors=["Eval Fixture"],
            year=spec["year"],
            abstract=spec["text"][:400],
            tags=[EVAL_TAG],
            original_filename=f"{spec['key']}.pdf",
            original_file=f"/tmp/eval-{spec['key']}.pdf",
            processing_status=PaperStatus.ready.value,
            created_at=now,
            updated_at=now,
            space_id=DEFAULT_SPACE_ID,
        )
        session.add(paper)
        await session.flush()
        session.add(
            PaperChunk(
                paper_id=paper.id,
                chunk_index=0,
                text=spec["text"],
                page=1,
                section="Abstract",
                embedding=vector,
            )
        )
        keyed[spec["key"]] = paper

    ella_vec = (
        embed_document(NOTE_TITLE + " ZXQELLA7 transfers sparse models")
        if embed_document is not None
        else axis(ELLA_AXIS)
    )
    voice = Note(
        source_type=NoteSourceType.voice.value,
        title=NOTE_TITLE,
        summary="Hypothesis that ZXQELLA7 transfer helps robot tasks.",
        hypotheses=["ZXQELLA7 transfer will work on my robot tasks."],
        cleaned_transcript="I think ZXQELLA7 transfer will work on my robot tasks.",
        tags=[EVAL_TAG],
        review_status=ReviewStatus.accepted.value,
        processing_status=ProcessingStatus.ready.value,
        created_at=now,
        updated_at=now,
        space_id=DEFAULT_SPACE_ID,
    )
    session.add(voice)
    await session.flush()
    session.add(
        NoteChunk(
            note_id=voice.id,
            chunk_index=0,
            text=(
                "Voice note on ZXQELLA7 transfer. "
                "I think ZXQELLA7 transfer will work on my robot tasks."
            ),
            section="Voice Note",
            embedding=ella_vec,
        )
    )
    session.add(
        NotePaper(note_id=voice.id, paper_id=keyed["ella"].id, source="researcher")
    )

    cnn_vec = (
        embed_document("Handwritten notes on ZXQCNN7 classify photographs")
        if embed_document is not None
        else axis(CNN_AXIS)
    )
    hand = Note(
        source_type=NoteSourceType.handwritten.value,
        title=HAND_TITLE,
        summary="Page of notes about ZXQCNN7 photographs.",
        extracted_text="ZXQCNN7 convolutional networks classify photographs.",
        tags=[EVAL_TAG],
        original_filename="cnn-notes.pdf",
        original_file="/tmp/eval-cnn-notes.pdf",
        review_status=ReviewStatus.accepted.value,
        processing_status=ProcessingStatus.ready.value,
        created_at=now,
        updated_at=now,
        space_id=DEFAULT_SPACE_ID,
    )
    session.add(hand)
    await session.flush()
    session.add(
        NoteChunk(
            note_id=hand.id,
            chunk_index=0,
            text="ZXQCNN7 convolutional networks classify photographs of objects.",
            page=1,
            section="Page 1",
            embedding=cnn_vec,
        )
    )
    session.add(
        NotePaper(note_id=hand.id, paper_id=keyed["cnn"].id, source="researcher")
    )
    await session.commit()
    return SeedResult(
        paper_ids=[paper.id for paper in keyed.values()],
        note_ids=[voice.id, hand.id],
        titles={key: paper.title for key, paper in keyed.items()},
    )


async def fixture_paper_ids(session: AsyncSession) -> list:
    result = await session.execute(
        select(Paper.id).where(
            Paper.tags.contains([EVAL_TAG]),
            Paper.processing_status == PaperStatus.ready.value,
        )
    )
    return [row[0] for row in result.all()]

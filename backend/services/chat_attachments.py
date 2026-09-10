from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from routers.documents import schedule_document_processing
from routers.notes import schedule_note_processing
from routers.papers import schedule_paper_processing
from schemas import ChatFiledAttachment
from services.document_classifier import (
    AttachmentKind,
    attachment_preview,
    classify_attachment,
)
from services.library_ingest import (
    create_pending_document,
    create_pending_handwritten_note,
    create_pending_paper,
    validate_library_file,
    validate_pdf_bytes,
)


@dataclass
class FiledAttachment:
    kind: AttachmentKind
    attachment: ChatFiledAttachment
    preview: str


def _llm_classify(app: FastAPI):
    client = getattr(app.state, "agent", None)
    if client is None:
        return None
    llm_client = getattr(client, "llm_client", None)
    llm_model = getattr(client, "llm_model", None)
    if llm_client is None or not llm_model:
        return None

    def classify(payload: str) -> AttachmentKind:
        from services.llm_chat import complete_chat

        result = complete_chat(
            llm_client,
            {
                "model": llm_model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Classify the attached file for a personal research library. "
                            "Reply with exactly one word: paper, note, or document. "
                            "paper = published-style academic PDF. "
                            "note = the researcher's own notes. "
                            "document = other files (markdown, csv, docx, generic text)."
                        ),
                    },
                    {"role": "user", "content": payload},
                ],
                "temperature": 0,
                "max_tokens": 16,
                "stream": False,
            },
        )
        token = (result.text or "").strip().lower().split()[0] if result.text else ""
        token = token.strip(".,:;\"'")
        if token in ("paper", "note", "document"):
            return token  # type: ignore[return-value]
        raise ValueError(f"unexpected classifier reply: {result.text!r}")

    return classify


async def file_chat_attachments(
    app: FastAPI,
    session: AsyncSession,
    files: list[UploadFile],
    prompt: str,
    *,
    space_id=None,
) -> list[FiledAttachment]:
    if not files:
        return []
    from services.demo import require_uploads

    require_uploads()
    llm_classify = _llm_classify(app)
    filed: list[FiledAttachment] = []
    for upload in files:
        content = await upload.read()
        filename = upload.filename or "attachment"
        content_type = upload.content_type
        try:
            filename = validate_library_file(
                filename, content_type, content, fallback_name=filename
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        preview = await asyncio.to_thread(
            attachment_preview, filename, content, content_type
        )
        kind = await asyncio.to_thread(
            classify_attachment,
            filename=filename,
            content=content,
            content_type=content_type,
            prompt=prompt,
            llm_classify=llm_classify,
        )
        if kind == "paper":
            try:
                filename = validate_pdf_bytes(
                    filename, content_type, content, fallback_name=filename
                )
            except ValueError:
                kind = "document"
            else:
                paper = await create_pending_paper(
                    session, content, filename, space_id=space_id
                )
                schedule_paper_processing(app, paper.id)
                attachment = ChatFiledAttachment(
                    kind="paper",
                    id=paper.id,
                    title=paper.title,
                    filename=filename,
                    status=paper.processing_status,
                    preview=preview,
                )
                filed.append(
                    FiledAttachment(kind=kind, attachment=attachment, preview=preview)
                )
                continue
        if kind == "note":
            try:
                filename = validate_pdf_bytes(
                    filename, content_type, content, fallback_name=filename
                )
            except ValueError:
                kind = "document"
            else:
                note = await create_pending_handwritten_note(
                    session, content, filename, space_id=space_id
                )
                schedule_note_processing(app, note.id)
                attachment = ChatFiledAttachment(
                    kind="note",
                    id=note.id,
                    title=note.title,
                    filename=filename,
                    status=note.processing_status,
                    preview=preview,
                )
                filed.append(
                    FiledAttachment(kind=kind, attachment=attachment, preview=preview)
                )
                continue
        document = await create_pending_document(
            session, content, filename, mime_type=content_type, space_id=space_id
        )
        schedule_document_processing(app, document.id)
        attachment = ChatFiledAttachment(
            kind="document",
            id=document.id,
            title=document.title,
            filename=filename,
            status=document.processing_status,
            preview=preview,
        )
        filed.append(FiledAttachment(kind=kind, attachment=attachment, preview=preview))
    return filed


def format_attachment_context(items: list[FiledAttachment]) -> str:
    if not items:
        return ""
    blocks: list[str] = [
        "The researcher attached file(s) this turn. They are being filed in the "
        "library (processing may still be running). Use the previews below to "
        "answer now; search_library / read_* once they are ready.",
    ]
    for item in items:
        preview = item.preview.strip() or "(no extractable preview)"
        att = item.attachment
        blocks.append(
            f"- {att.kind} id={att.id} title={att.title!r} file={att.filename}\n"
            f"Preview:\n{preview[:4000]}"
        )
    return "\n\n".join(blocks)


def default_attachment_question(filenames: list[str]) -> str:
    if len(filenames) == 1:
        name = Path(filenames[0]).name
        return f"Please file and use the attached file ({name})."
    return f"Please file and use the {len(filenames)} attached files."

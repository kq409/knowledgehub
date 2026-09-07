import json
import re
from pathlib import Path

from openai import OpenAI
from pydantic import ValidationError

from schemas import ExtractedNote

PROMPT_FILE = Path(__file__).resolve().parent.parent / "extract_note_prompt.txt"
EXTRACT_PROMPT = PROMPT_FILE.read_text().strip()
PROMPT_VERSION = "voice-note-extract-v1"
EXTRACT_MAX_TOKENS = 1500


class NoteExtractionError(Exception):
    """Raised when the LLM does not return a valid structured note."""


def parse_json_object(text: str) -> dict:
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
    if fence:
        cleaned = fence.group(1).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output")

    parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("JSON payload is not an object")
    return parsed


class ExtractionService:
    def __init__(self, llm_client: OpenAI, llm_model: str):
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.prompt_version = PROMPT_VERSION

    def _complete(self, messages: list[dict[str, str]]) -> str:
        from services.llm_chat import effort_from_env, message_text, with_chat_extras

        effort = effort_from_env(
            "EXTRACT_REASONING_EFFORT", "LLM_REASONING_EFFORT", default="none"
        )
        response = self.llm_client.chat.completions.create(
            **with_chat_extras(
                {
                    "model": self.llm_model,
                    "messages": messages,
                    "temperature": 0.2,
                    "max_tokens": EXTRACT_MAX_TOKENS,
                    "stream": False,
                },
                effort=effort,
            )
        )
        return message_text(response.choices[0].message)

    def extract(self, cleaned_text: str, raw_text: str | None = None) -> ExtractedNote:
        if not cleaned_text.strip():
            raise NoteExtractionError("Cleaned text is empty")

        user_content = cleaned_text
        if raw_text and raw_text.strip() and raw_text.strip() != cleaned_text.strip():
            user_content = (
                f"Cleaned transcript:\n{cleaned_text}\n\n"
                f"Raw transcript (for context only):\n{raw_text}"
            )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": EXTRACT_PROMPT},
            {"role": "user", "content": user_content},
        ]

        last_error: Exception | None = None
        for attempt in range(2):
            output = self._complete(messages)
            try:
                payload = parse_json_object(output)
                return ExtractedNote.model_validate(payload)
            except (json.JSONDecodeError, ValueError, ValidationError) as exc:
                last_error = exc
                messages.append({"role": "assistant", "content": output})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous reply was not valid JSON matching the schema. "
                            f"Error: {exc}. Return ONLY the JSON object."
                        ),
                    }
                )
                print(f"⚠️  Note extraction attempt {attempt + 1} failed: {exc}")

        raise NoteExtractionError(
            f"Could not extract a structured note: {last_error}"
        ) from last_error

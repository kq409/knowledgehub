#!/usr/bin/env python3
"""
Uses OpenAI API format, compatible with Ollama, OpenAI, LM Studio, and other providers.
Configuration is loaded from .env file.
"""

import threading
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from openai import OpenAI

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

# Edit system_prompt.txt to change how the LLM cleans transcriptions
PROMPT_FILE = Path(__file__).parent / "system_prompt.txt"
SYSTEM_PROMPT = PROMPT_FILE.read_text().strip()


class TranscriptionService:
    """Uses OpenAI-compatible API, works with any provider (Ollama, OpenAI, LM Studio, etc.)."""

    def __init__(
        self,
        whisper_model: str,
        llm_base_url: str,
        llm_api_key: str,
        llm_model: str,
        *,
        llm_client: OpenAI | None = None,
        load_whisper: bool = True,
    ):
        self._whisper_model_name = whisper_model
        self._whisper: WhisperModel | None = None
        self._whisper_lock = threading.Lock()

        if llm_client is not None:
            self.llm_client = llm_client
        else:
            print(f"🔄 Connecting to LLM at {llm_base_url}...")
            self.llm_client = OpenAI(base_url=llm_base_url, api_key=llm_api_key)
            try:
                self.llm_client.models.list()
                print("✅ Connected to LLM API!")
            except Exception as e:
                print(f"⚠️  Warning: Could not connect to LLM: {e}")
                print(f"   Make sure your LLM server is running at {llm_base_url}")
        self.llm_model = llm_model

        if load_whisper:
            self.ensure_whisper_loaded()

    def ensure_whisper_loaded(self) -> None:
        if self._whisper is not None:
            return
        with self._whisper_lock:
            if self._whisper is not None:
                return
            print(f"🔄 Loading Whisper model '{self._whisper_model_name}'...")
            from faster_whisper import WhisperModel

            self._whisper = WhisperModel(
                self._whisper_model_name,
                device="auto",  # Auto-detect: Metal (Mac), CUDA (NVIDIA), or CPU
                compute_type="int8",
            )
            print(f"✅ Whisper model '{self._whisper_model_name}' loaded!")

    def transcribe(self, audio_file):
        print("🔄 Transcribing...")
        self.ensure_whisper_loaded()
        assert self._whisper is not None

        segments, info = self._whisper.transcribe(
            audio_file, beam_size=5, language="en", condition_on_previous_text=False
        )

        text = " ".join([segment.text for segment in segments]).strip()
        print(f"📝 Raw: {text}")
        return text

    def get_default_system_prompt(self):
        return SYSTEM_PROMPT

    def open_clean_stream(self, text, system_prompt=None):
        """Start the LLM request without consuming tokens, so HTTP can stream them live."""
        if not text:
            return None

        prompt_to_use = system_prompt if system_prompt else SYSTEM_PROMPT
        print("🤖 Cleaning with LLM...")
        return self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": prompt_to_use},
                {"role": "user", "content": text},
            ],
            temperature=0.3,
            max_tokens=200,
            stream=True,
        )

    def iter_clean_tokens(self, stream) -> Iterator[str]:
        if stream is None:
            return

        collected: list[str] = []
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                collected.append(delta)
                yield delta

        cleaned = "".join(collected).strip()
        print(f"✨ Cleaned: {cleaned}")

    def clean_with_llm_stream(self, text, system_prompt=None) -> Iterator[str]:
        yield from self.iter_clean_tokens(self.open_clean_stream(text, system_prompt))

    def clean_with_llm(self, text, system_prompt=None):
        return "".join(self.clean_with_llm_stream(text, system_prompt)).strip()

    def transcribe_file(self, audio_file_path: str, use_llm: bool = True) -> dict:
        raw_text = self.transcribe(audio_file_path)

        result = {"raw_text": raw_text}

        if use_llm and raw_text:
            cleaned_text = self.clean_with_llm(raw_text)
            result["cleaned_text"] = cleaned_text
        else:
            result["cleaned_text"] = raw_text

        return result

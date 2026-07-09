from __future__ import annotations

from typing import Any

from openai import OpenAI

from app.config import settings


class OpenRouterClient:
    """Small wrapper around OpenRouter's OpenAI-compatible API."""

    def __init__(self) -> None:
        if not settings.openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY belum diisi. "
                "Buat file .env dari .env.example, lalu isi API key kamu."
            )

        self.client = OpenAI(
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
            timeout=settings.openrouter_timeout_seconds,
            default_headers={
                "HTTP-Referer": settings.site_url,
                "X-Title": settings.site_name,
            },
        )

    def generate_answer(
        self,
        question: str,
        selected_context: str,
        model: str | None = None,
        temperature: float = 0.2,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        default_system_prompt = """
Anda adalah asisten RAG untuk menjawab pertanyaan berdasarkan dokumen PDF.
Jawab hanya berdasarkan konteks yang diberikan.
Jika jawaban tidak ada di konteks, katakan bahwa informasi tidak ditemukan di dokumen.
Gunakan bahasa Indonesia yang jelas, ringkas, dan tegas.
Sebutkan sumber yang digunakan, misalnya Sumber 1 atau Sumber 2.
""".strip()

        user_prompt = f"""
KONTEKS DOKUMEN:
{selected_context}

PERTANYAAN:
{question}

JAWABAN:
""".strip()

        kwargs: dict[str, Any] = {
            "model": model or settings.openrouter_model,
            "messages": [
                {"role": "system", "content": system_prompt or default_system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        # JSON mode optional. Tidak semua model OpenRouter mendukung ini, jadi caller
        # boleh mematikan lewat OPENROUTER_JSON_MODE=false.
        if response_format is not None:
            kwargs["response_format"] = response_format
        elif settings.openrouter_json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        answer = response.choices[0].message.content or ""
        return answer.strip()

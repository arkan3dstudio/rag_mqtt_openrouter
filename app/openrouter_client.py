from __future__ import annotations

from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - depends on deployment environment
    OpenAI = None

from app.config import settings


AGRICULTURE_AGENT_POLICY = """
Anda adalah AI agent pertanian/agronomi.
Ruang lingkup jawaban hanya: budidaya tanaman, tanah, pH, EC, NPK, pupuk, irigasi, hama/penyakit tanaman, fase tumbuh, panen, sensor tanah, dan dokumen RAG pertanian.
Jika pertanyaan di luar pertanian atau objeknya bukan tanaman/lahan/sensor pertanian, jangan menjawab substansinya. Katakan singkat bahwa topik di luar cakupan agent pertanian.
Jangan membuat jawaban dari pengetahuan umum bila konteks dokumen tidak mendukung. Gunakan konteks yang diberikan sebagai dasar utama.
Gunakan bahasa Indonesia yang natural, singkat, dan mudah dipahami.
Jika caller meminta output JSON, tetap balas JSON valid sesuai schema yang diminta caller.
""".strip()


def _with_agriculture_policy(system_prompt: str) -> str:
    system_prompt = str(system_prompt or "").strip()
    if not system_prompt:
        return AGRICULTURE_AGENT_POLICY
    return f"{AGRICULTURE_AGENT_POLICY}\n\nATURAN TUGAS KHUSUS:\n{system_prompt}"


class OpenRouterClient:
    """Small wrapper around OpenRouter's OpenAI-compatible API."""

    def __init__(self) -> None:
        if OpenAI is None:
            raise ValueError(
                "Package openai belum terpasang. Jalankan: pip install openai. "
                "Sistem tetap dapat memakai fallback rule engine bila error ini tertangkap di MQTT processor."
            )

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
        enforce_agriculture_scope: bool = True,
    ) -> str:
        default_system_prompt = """
Anda adalah asisten RAG pertanian untuk menjawab pertanyaan berdasarkan dokumen PDF.
Jawab hanya berdasarkan konteks yang diberikan.
Jika jawaban tidak ada di konteks, katakan bahwa informasi tidak ditemukan di dokumen.
Gunakan bahasa Indonesia yang jelas, ringkas, dan tegas.
Sebutkan sumber yang digunakan, misalnya Sumber 1 atau Sumber 2.
""".strip()

        active_system_prompt = system_prompt or default_system_prompt
        if enforce_agriculture_scope:
            active_system_prompt = _with_agriculture_policy(active_system_prompt)

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
                {"role": "system", "content": active_system_prompt},
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

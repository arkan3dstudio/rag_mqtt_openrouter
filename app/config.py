from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    return float(value)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables.

    Settings ini tetap kompatibel dengan versi lama, tetapi menambahkan parameter
    penting untuk MQTT processor, debounce LLM, metadata dokumen, dan retrieval.
    """

    project_root: Path = ROOT_DIR
    pdf_dir: Path = ROOT_DIR / "data" / "pdfs"
    markdown_dir: Path = ROOT_DIR / "storage" / "markdown"
    index_dir: Path = ROOT_DIR / "storage" / "index"
    index_file: Path = ROOT_DIR / "storage" / "index" / "tfidf_index.joblib"
    index_manifest_file: Path = ROOT_DIR / "storage" / "index" / "tfidf_index_manifest.json"
    document_manifest_file: Path = ROOT_DIR / "data" / "document_manifest.json"

    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_base_url: str = os.getenv(
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1",
    )
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "openrouter/free")
    openrouter_timeout_seconds: int = _env_int("OPENROUTER_TIMEOUT_SECONDS", 40)

    site_url: str = os.getenv("SITE_URL", "http://localhost:8501")
    site_name: str = os.getenv("SITE_NAME", "RAG PDF OpenRouter")

    # RAG / retrieval
    chunk_size: int = _env_int("CHUNK_SIZE", 900)
    chunk_overlap: int = _env_int("CHUNK_OVERLAP", 150)
    top_k: int = _env_int("TOP_K", 5)
    min_score: float = _env_float("MIN_SCORE", 0.05)
    tfidf_max_features: int = _env_int("TFIDF_MAX_FEATURES", 100_000)
    tfidf_ngram_max: int = _env_int("TFIDF_NGRAM_MAX", 2)
    enable_query_expansion: bool = _env_bool("ENABLE_QUERY_EXPANSION", True)
    auto_rebuild_index: bool = _env_bool("AUTO_REBUILD_INDEX", True)

    # MQTT processor reads these with getattr(settings, ...). Keep names stable.
    rag_output_mode: str = os.getenv("RAG_OUTPUT_MODE", "hybrid")
    rag_max_answer_tokens: int = _env_int("RAG_MAX_ANSWER_TOKENS", 1200)
    rag_temperature: float = _env_float("RAG_TEMPERATURE", 0.2)
    rag_language: str = os.getenv("RAG_LANGUAGE", "id")
    rag_answer_style: str = os.getenv("RAG_ANSWER_STYLE", "detail_terstruktur")
    llm_timeout_seconds: int = _env_int("LLM_TIMEOUT_SECONDS", 45)
    llm_min_interval_seconds: int = _env_int("LLM_MIN_INTERVAL_SECONDS", 300)
    rag_confidence_reference_score: float = _env_float("RAG_CONFIDENCE_REFERENCE_SCORE", 0.65)

    # Strict mode: result MQTT hanya dikirim bila retrieval RAG relevan DAN LLM
    # mengembalikan JSON valid. Jika gagal, worker hanya publish feedback failed.
    rag_require_llm_success: bool = _env_bool("RAG_REQUIRE_LLM_SUCCESS", True)
    rag_require_relevant_sources: bool = _env_bool("RAG_REQUIRE_RELEVANT_SOURCES", True)
    rag_fallback_on_llm_timeout: bool = _env_bool("RAG_FALLBACK_ON_LLM_TIMEOUT", True)
    chat_max_sensor_age_seconds: int = _env_int("CHAT_MAX_SENSOR_AGE_SECONDS", 1800)

    # Prompt / response control
    openrouter_json_mode: bool = _env_bool("OPENROUTER_JSON_MODE", False)

    # Field validation defaults. Use these when calibration/soil-test evidence is
    # handled outside MQTT payloads, for example directly on the sensor device.
    assume_device_sensor_calibrated: bool = _env_bool("ASSUME_DEVICE_SENSOR_CALIBRATED", False)
    assume_soil_test_validated: bool = _env_bool("ASSUME_SOIL_TEST_VALIDATED", False)
    field_validation_note: str = os.getenv("FIELD_VALIDATION_NOTE", "")


settings = Settings()


def ensure_directories() -> None:
    """Create required project folders if they do not exist."""
    settings.pdf_dir.mkdir(parents=True, exist_ok=True)
    settings.markdown_dir.mkdir(parents=True, exist_ok=True)
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    settings.document_manifest_file.parent.mkdir(parents=True, exist_ok=True)

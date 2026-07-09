from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import math
import re
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.chunker import Chunk
from app.config import settings


@dataclass
class RetrievedChunk:
    score: float
    source: str
    chunk_id: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None
    rerank_score: float | None = None


AGRONOMY_SYNONYMS: dict[str, list[str]] = {
    "nitrogen": ["nitrogen", "unsur n", "hara n", "urea", "pupuk n", "defisiensi n", "kekurangan nitrogen", "ammonium", "nitrat", "pertumbuhan vegetatif"],
    "fosfor": ["fosfor", "phosphorus", "unsur p", "hara p", "sp-36", "sp36", "tsp", "pupuk p", "fosfat", "phosphate", "perakaran", "pupuk dasar"],
    "kalium": ["kalium", "potassium", "unsur k", "hara k", "kcl", "pupuk k", "pembungaan", "pembuahan", "ketahanan tanaman"],
    "ph": ["ph", "pH", "kemasaman", "tanah asam", "tanah basa", "dolomit", "pengapuran", "kapur pertanian", "ketersediaan hara"],
    "ec": ["ec", "electrical conductivity", "konduktivitas", "salinitas", "garam terlarut", "daya hantar listrik", "drainase", "kualitas air"],
    "kelembapan_tanah": ["humidity", "soil moisture", "kelembapan tanah", "kadar air tanah", "air tanah", "irigasi", "pengairan", "zona akar", "drainase", "genangan"],
    "padi": ["padi", "oryza sativa", "rice", "sawah"],
    "jagung": ["jagung", "zea mays", "maize", "corn"],
    "cabai": ["cabai", "cabe", "capsicum", "chili", "chilli"],
    "bawang": ["bawang", "bawang merah", "shallot", "allium"],
    "tomat": ["tomat", "tomato", "lycopersicum"],
    "vegetatif": ["vegetatif", "pertumbuhan daun", "pertumbuhan batang", "anakan"],
    "pembungaan": ["pembungaan", "bunga", "flowering"],
    "pembuahan": ["pembuahan", "buah", "bulir", "umbi", "polong"],
    "uji_tanah": ["uji tanah", "analisis tanah", "soil test", "laboratorium tanah", "validasi lapang"],
    "pemupukan_berimbang": ["pemupukan berimbang", "rekomendasi pupuk", "dosis pupuk", "pupuk majemuk", "pupuk organik", "bahan organik"],
    "faktor_pembatas": ["faktor pembatas", "limiting factor", "cek gejala", "serapan hara", "akar", "stres tanaman"],
}

DOC_TYPE_WEIGHT = {
    "sop_manual": 0.18,
    "petunjuk_teknis": 0.17,
    "manual_book": 0.16,
    "modul_pelatihan": 0.13,
    "ebook_manual_resmi": 0.13,
    "paper_riset_asli": 0.06,
    "paper": 0.06,
    "jurnal": 0.05,
    "unknown": 0.0,
}


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {_stringify(val)}" for key, val in value.items())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_stringify(item) for item in value)
    return str(value)


def normalize_agronomy_text(text: str) -> str:
    """Normalize common agronomy terms without removing the original text.

    Prinsipnya additive: term asli tetap ada, lalu sinonim utama ditambahkan agar
    TF-IDF lebih mampu menangkap istilah N/P/K, pupuk, pH, EC, dan soil moisture.
    """
    if not text:
        return ""

    lowered = text.lower()
    additions: list[str] = []

    for canonical, synonyms in AGRONOMY_SYNONYMS.items():
        for synonym in synonyms:
            pattern = re.escape(synonym.lower())
            if re.search(rf"(?<!\w){pattern}(?!\w)", lowered):
                additions.append(canonical)
                # Tambah beberapa sinonim penting, tetapi batasi agar dokumen tidak terlalu panjang.
                additions.extend(synonyms[:4])
                break

    normalized = text
    if additions:
        normalized += "\nAGRONOMY_TERMS: " + " ".join(dict.fromkeys(additions))
    return normalized


def expand_query(question: str) -> str:
    if not settings.enable_query_expansion:
        return question
    return normalize_agronomy_text(question)


def _chunk_from_payload(payload: dict[str, Any]) -> Chunk:
    # Backward compatible untuk index lama yang hanya punya source/chunk_id/text.
    return Chunk(
        source=payload.get("source", ""),
        chunk_id=int(payload.get("chunk_id", payload.get("id", 0)) or 0),
        text=payload.get("text", ""),
        metadata=dict(payload.get("metadata") or {}),
        page_start=payload.get("page_start"),
        page_end=payload.get("page_end"),
        section=payload.get("section"),
    )


def _normalize_filter_value(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def _metadata_matches(chunk: Chunk, filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True

    metadata = chunk.metadata or {}
    crop_filter = _normalize_filter_value(filters.get("crop") or filters.get("komoditas"))
    stage_filter = _normalize_filter_value(filters.get("growth_stage") or filters.get("fase"))

    crop_meta = _normalize_filter_value(metadata.get("crop") or metadata.get("komoditas"))
    crop_general = crop_meta in {"all", "general", "umum", "semua", "multi", "unknown"}
    if crop_filter and crop_meta and not crop_general and crop_filter not in crop_meta and crop_meta not in crop_filter:
        return False

    stage_meta = _normalize_filter_value(metadata.get("growth_stage") or metadata.get("fase"))
    stage_general = stage_meta in {"all", "general", "umum", "semua", "multi", "unknown"}
    if stage_filter and stage_meta and not stage_general and stage_filter not in stage_meta and stage_meta not in stage_filter:
        return False

    return True


def _topic_hits(chunk: Chunk, filters: dict[str, Any] | None) -> int:
    if not filters:
        return 0
    topics = [str(t).lower() for t in filters.get("topics", []) if str(t).strip()]
    if not topics:
        return 0
    haystack = _stringify([chunk.text, chunk.metadata]).lower().replace("_", " ")
    return sum(1 for topic in topics if topic.replace("_", " ") in haystack or topic in haystack)


def _rerank_bonus(chunk: Chunk, filters: dict[str, Any] | None) -> float:
    metadata = chunk.metadata or {}
    bonus = 0.0

    if filters:
        crop_filter = _normalize_filter_value(filters.get("crop") or filters.get("komoditas"))
        stage_filter = _normalize_filter_value(filters.get("growth_stage") or filters.get("fase"))
        crop_meta = _normalize_filter_value(metadata.get("crop") or metadata.get("komoditas"))
        stage_meta = _normalize_filter_value(metadata.get("growth_stage") or metadata.get("fase"))

        if crop_filter:
            if crop_meta and (crop_filter in crop_meta or crop_meta in crop_filter):
                bonus += 0.18
            elif not crop_meta and crop_filter.replace("_", " ") in chunk.text.lower():
                bonus += 0.08

        if stage_filter:
            if stage_meta and (stage_filter in stage_meta or stage_meta in stage_filter):
                bonus += 0.06
            elif not stage_meta and stage_filter.replace("_", " ") in chunk.text.lower():
                bonus += 0.03

        bonus += min(_topic_hits(chunk, filters) * 0.015, 0.09)

        preferred_doc_types = {str(x).lower() for x in filters.get("preferred_doc_types", [])}
        issue_parameters = [str(x).lower() for x in filters.get("issue_parameters", [])]
        if preferred_doc_types:
            doc_type_preview = str(metadata.get("doc_type") or metadata.get("jenis_dokumen") or "").lower()
            if doc_type_preview in preferred_doc_types:
                bonus += 0.06
        if issue_parameters:
            haystack = _stringify([chunk.text, metadata]).lower().replace("_", " ")
            bonus += min(sum(1 for item in issue_parameters if item and item in haystack) * 0.03, 0.09)

    authority = str(metadata.get("authority") or "").upper()
    if authority == "A":
        bonus += 0.08
    elif authority == "B":
        bonus += 0.04

    doc_type = str(metadata.get("doc_type") or "unknown").lower()
    source_type = str(metadata.get("source_type") or "unknown").lower()
    bonus += DOC_TYPE_WEIGHT.get(doc_type, 0.0)

    if source_type in {"ebook_manual_resmi", "manual", "sop", "petunjuk_teknis"}:
        bonus += 0.08
    elif source_type in {"paper", "paper_riset_asli", "jurnal", "journal"}:
        bonus += 0.025

    return bonus


class TfidfRetriever:
    """Domain-aware TF-IDF retriever.

    Tetap deploy-friendly dan kompatibel dengan API lama, tetapi sekarang:
    - memakai n-gram untuk frasa agronomi;
    - melakukan query expansion N/P/K/pH/EC/soil moisture;
    - menyimpan metadata chunk;
    - menerima filters/metadata_filters/filter dari caller MQTT processor;
    - melakukan rerank ringan berdasarkan crop, fase, topic, doc_type, authority.
    """

    def __init__(self, max_features: int | None = None, ngram_max: int | None = None):
        self.max_features = int(max_features or settings.tfidf_max_features)
        self.ngram_max = int(ngram_max or settings.tfidf_ngram_max)
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            max_features=self.max_features,
            ngram_range=(1, max(1, self.ngram_max)),
            sublinear_tf=True,
            max_df=1.0,
            token_pattern=r"(?u)\b\w[\w\-\.]+\b",
        )
        self.matrix = None
        self.chunks: list[Chunk] = []

    def fit(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("chunks kosong. Tidak bisa membuat retriever.")

        self.chunks = chunks
        chunk_texts = [normalize_agronomy_text(chunk.text + "\n" + _stringify(chunk.metadata)) for chunk in chunks]
        self.matrix = self.vectorizer.fit_transform(chunk_texts)

    def search(
        self,
        question: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        metadata_filters: dict[str, Any] | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        if self.matrix is None or not self.chunks:
            raise RuntimeError("Retriever belum dibuat. Jalankan build_index terlebih dahulu.")

        question = str(question or "").strip()
        if not question:
            return []

        active_filters = filters or metadata_filters or filter or {}
        question_vector = self.vectorizer.transform([expand_query(question)])
        scores = cosine_similarity(question_vector, self.matrix).flatten()

        # Ambil kandidat lebih banyak dulu, lalu soft-filter + rerank.
        candidate_count = min(len(self.chunks), max(top_k * 12, 80))
        top_indices = scores.argsort()[::-1][:candidate_count]

        filtered_indices = [idx for idx in top_indices if _metadata_matches(self.chunks[int(idx)], active_filters)]
        if not filtered_indices:
            filtered_indices = list(top_indices)

        results: list[RetrievedChunk] = []
        for raw_index in filtered_indices:
            index = int(raw_index)
            chunk = self.chunks[index]
            base_score = float(scores[index]) if math.isfinite(float(scores[index])) else 0.0
            rerank_score = base_score + _rerank_bonus(chunk, active_filters)
            results.append(
                RetrievedChunk(
                    score=base_score,
                    source=chunk.source,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    metadata=dict(chunk.metadata or {}),
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section=chunk.section,
                    rerank_score=round(rerank_score, 6),
                )
            )

        results.sort(key=lambda item: item.rerank_score if item.rerank_score is not None else item.score, reverse=True)
        return results[:top_k]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": "1.2.0-agronomy",
            "max_features": self.max_features,
            "ngram_max": self.ngram_max,
            "vectorizer": self.vectorizer,
            "matrix": self.matrix,
            "chunks": [asdict(chunk) for chunk in self.chunks],
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: Path) -> "TfidfRetriever":
        if not path.exists():
            raise FileNotFoundError(f"Index tidak ditemukan: {path}")

        payload = joblib.load(path)
        retriever = cls(
            max_features=payload.get("max_features", settings.tfidf_max_features),
            ngram_max=payload.get("ngram_max", settings.tfidf_ngram_max),
        )
        retriever.vectorizer = payload["vectorizer"]
        retriever.matrix = payload["matrix"]
        retriever.chunks = [_chunk_from_payload(chunk) for chunk in payload.get("chunks", [])]
        return retriever

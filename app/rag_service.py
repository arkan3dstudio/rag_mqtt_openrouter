from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from app.chunker import build_chunks
from app.config import ensure_directories, settings
from app.document_manifest import DocumentManifest
from app.openrouter_client import OpenRouterClient
from app.pdf_loader import list_pdf_files, load_documents
from app.retriever import RetrievedChunk, TfidfRetriever


class RAGService:
    """Main service for building index and asking questions.

    Backward-compatible dengan versi lama, tetapi sekarang index punya manifest
    sehingga otomatis rebuild saat PDF/config/metadata berubah.
    """

    def __init__(self) -> None:
        ensure_directories()
        self.retriever: TfidfRetriever | None = None

    # ==========================================================
    # Index manifest / freshness
    # ==========================================================

    def _pdf_fingerprints(self) -> list[dict[str, Any]]:
        files = list_pdf_files()
        fingerprints: list[dict[str, Any]] = []
        for path in files:
            stat = path.stat()
            try:
                rel_path = str(path.relative_to(settings.project_root))
            except ValueError:
                rel_path = str(path)
            fingerprints.append({
                "path": rel_path.replace("\\", "/"),
                "mtime": stat.st_mtime,
                "size": stat.st_size,
            })
        return fingerprints

    def _current_manifest(self) -> dict[str, Any]:
        doc_manifest = DocumentManifest().fingerprint()
        return {
            "version": "1.1.0",
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "tfidf_max_features": settings.tfidf_max_features,
            "tfidf_ngram_max": settings.tfidf_ngram_max,
            "document_manifest": doc_manifest,
            "pdfs": self._pdf_fingerprints(),
        }

    def _saved_manifest(self) -> dict[str, Any] | None:
        if not settings.index_manifest_file.exists():
            return None
        try:
            return json.loads(settings.index_manifest_file.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _index_is_fresh(self) -> bool:
        if not settings.index_file.exists():
            return False
        if not settings.auto_rebuild_index:
            return True
        return self._saved_manifest() == self._current_manifest()

    def _write_index_manifest(self) -> None:
        settings.index_manifest_file.parent.mkdir(parents=True, exist_ok=True)
        settings.index_manifest_file.write_text(
            json.dumps(self._current_manifest(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ==========================================================
    # Build/load/ask
    # ==========================================================

    def build_index(self, force_extract: bool = False) -> dict:
        documents = load_documents(force_extract=force_extract)
        chunks = build_chunks(
            documents,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )

        retriever = TfidfRetriever(
            max_features=settings.tfidf_max_features,
            ngram_max=settings.tfidf_ngram_max,
        )
        retriever.fit(chunks)
        retriever.save(settings.index_file)

        self.retriever = retriever
        self._write_index_manifest()

        return {
            "total_documents": len(documents),
            "total_chunks": len(chunks),
            "index_file": str(settings.index_file.relative_to(settings.project_root)),
            "index_manifest_file": str(settings.index_manifest_file.relative_to(settings.project_root)),
        }

    def load_or_build_index(self) -> None:
        if self._index_is_fresh():
            self.retriever = TfidfRetriever.load(settings.index_file)
        else:
            self.build_index(force_extract=False)

    def _format_context(self, retrieved: list[RetrievedChunk]) -> str:
        blocks: list[str] = []
        for index, item in enumerate(retrieved):
            metadata = item.metadata or {}
            title = metadata.get("document_title") or item.source
            doc_type = metadata.get("doc_type") or "unknown"
            authority = metadata.get("authority") or "C"
            crop = metadata.get("crop") or "unknown"
            stage = metadata.get("growth_stage") or "unknown"
            page_info = ""
            if item.page_start is not None:
                if item.page_end and item.page_end != item.page_start:
                    page_info = f" | Halaman {item.page_start}-{item.page_end}"
                else:
                    page_info = f" | Halaman {item.page_start}"
            section_info = f" | Bagian {item.section}" if item.section else ""
            rerank_info = f" | Rerank {item.rerank_score:.4f}" if item.rerank_score is not None else ""

            header = (
                f"[Sumber {index + 1}: {title} | File {item.source} | "
                f"Chunk {item.chunk_id}{page_info}{section_info} | "
                f"Score {item.score:.4f}{rerank_info} | "
                f"DocType {doc_type} | Authority {authority} | Crop {crop} | Stage {stage}]"
            )
            blocks.append(f"{header}\n{item.text}")
        return "\n\n".join(blocks)

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        min_score: float | None = None,
        model: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> dict:
        if self.retriever is None:
            self.load_or_build_index()

        assert self.retriever is not None

        top_k = top_k or settings.top_k
        min_score = settings.min_score if min_score is None else min_score

        retrieved = self.retriever.search(question, top_k=top_k, filters=filters)

        if not retrieved or retrieved[0].score < min_score:
            return {
                "answer": "Pertanyaan tidak ditemukan atau tidak relevan dengan dokumen PDF.",
                "sources": [asdict(item) for item in retrieved],
                "mode": "Mode Generatif / RAG OpenRouter",
                "retrieval": {
                    "top_score": retrieved[0].score if retrieved else None,
                    "min_score": min_score,
                    "filters": filters or {},
                    "warning": "Top score di bawah threshold MIN_SCORE.",
                },
            }

        selected_context = self._format_context(retrieved)
        client = OpenRouterClient()

        try:
            answer = client.generate_answer(
                question=question,
                selected_context=selected_context,
                model=model or settings.openrouter_model,
            )
        except Exception as error:
            answer = f"Gagal memanggil OpenRouter: {error}"

        if not answer.strip():
            answer = "Jawaban tidak ditemukan secara jelas di dokumen."

        return {
            "answer": answer,
            "sources": [asdict(item) for item in retrieved],
            "mode": "Mode Generatif / RAG OpenRouter",
            "retrieval": {
                "top_score": retrieved[0].score if retrieved else None,
                "top_rerank_score": retrieved[0].rerank_score if retrieved else None,
                "min_score": min_score,
                "filters": filters or {},
            },
        }

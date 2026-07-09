from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from app.pdf_loader import Document, DocumentPage


@dataclass
class Chunk:
    source: str
    chunk_id: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None


def _clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split long text while trying to keep paragraph boundaries."""
    text = _clean_text(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        target_end = min(start + chunk_size, len(text))
        end = target_end

        # Prefer paragraph or sentence boundary near target end.
        if target_end < len(text):
            boundary_candidates = [
                text.rfind("\n\n", start, target_end),
                text.rfind(". ", start, target_end),
                text.rfind("; ", start, target_end),
            ]
            boundary = max(boundary_candidates)
            if boundary > start + int(chunk_size * 0.55):
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break
        start = max(end - overlap, start + 1)

    return chunks


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    """Backward-compatible text splitter.

    Versi ini tetap menerima string biasa, tetapi lebih menjaga batas paragraf
    dibanding splitter karakter murni.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size harus lebih besar dari 0.")
    if overlap < 0:
        raise ValueError("overlap tidak boleh negatif.")
    if overlap >= chunk_size:
        raise ValueError("overlap harus lebih kecil dari chunk_size.")
    return _split_long_text(text, chunk_size=chunk_size, overlap=overlap)


def _detect_section(text: str, fallback: str | None = None) -> str | None:
    for line in text.splitlines():
        cleaned = line.strip().strip("#").strip()
        if not cleaned:
            continue
        if len(cleaned) <= 120 and (
            cleaned.isupper()
            or re.match(r"^(bab|bagian|[0-9]+\.|[A-Z]\.)\s+", cleaned, flags=re.IGNORECASE)
            or any(term in cleaned.lower() for term in ["pemupukan", "budidaya", "hara", "ph", "npk", "pengairan", "penyakit"])
        ):
            return cleaned
        break
    return fallback


def _chunk_page(
    document: Document,
    page: DocumentPage,
    next_chunk_id: int,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    text = page.text.strip()
    if not text:
        return []

    # Prefix ringan agar setiap chunk membawa konteks sumber/halaman.
    title = str(document.metadata.get("document_title") or document.source)
    page_prefix = f"Dokumen: {title}\nSumber: {document.source}\nHalaman: {page.page_number}\n"
    section = _detect_section(text)

    parts = _split_long_text(text, chunk_size=max(200, chunk_size - len(page_prefix)), overlap=overlap)
    chunks: list[Chunk] = []
    for part in parts:
        chunk_text_value = _clean_text(page_prefix + (f"Bagian: {section}\n" if section else "") + part)
        metadata = dict(document.metadata)
        metadata.update({
            "source": document.source,
            "markdown_path": document.markdown_path,
            "page_start": page.page_number,
            "page_end": page.page_number,
            "section": section,
        })
        chunks.append(
            Chunk(
                source=document.source,
                chunk_id=next_chunk_id + len(chunks),
                text=chunk_text_value,
                metadata=metadata,
                page_start=page.page_number,
                page_end=page.page_number,
                section=section,
            )
        )
    return chunks


def build_chunks(
    documents: list[Document],
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[Chunk]:
    """Create metadata-rich chunks for all loaded documents."""
    if chunk_size <= 0:
        raise ValueError("chunk_size harus lebih besar dari 0.")
    if overlap < 0:
        raise ValueError("overlap tidak boleh negatif.")
    if overlap >= chunk_size:
        raise ValueError("overlap harus lebih kecil dari chunk_size.")

    all_chunks: list[Chunk] = []

    for document in documents:
        if document.pages:
            for page in document.pages:
                page_chunks = _chunk_page(
                    document=document,
                    page=page,
                    next_chunk_id=len(all_chunks),
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
                all_chunks.extend(page_chunks)
        else:
            for chunk_text_value in chunk_text(document.text, chunk_size=chunk_size, overlap=overlap):
                metadata = dict(document.metadata)
                metadata.update({"source": document.source, "markdown_path": document.markdown_path})
                all_chunks.append(
                    Chunk(
                        source=document.source,
                        chunk_id=len(all_chunks),
                        text=chunk_text_value,
                        metadata=metadata,
                        page_start=None,
                        page_end=None,
                        section=_detect_section(chunk_text_value),
                    )
                )

    if not all_chunks:
        raise ValueError("Tidak ada chunk yang berhasil dibuat dari dokumen.")

    return all_chunks

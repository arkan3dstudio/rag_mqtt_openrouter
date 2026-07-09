from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import re
from typing import Any

import fitz  # PyMuPDF

from app.config import settings
from app.document_manifest import DocumentManifest, normalize_document_metadata


@dataclass
class DocumentPage:
    page_number: int
    text: str


@dataclass
class Document:
    source: str
    text: str
    markdown_path: str
    pages: list[DocumentPage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def list_pdf_files(pdf_dir: Path | None = None) -> list[Path]:
    """Return all PDF files stored inside the project PDF folder."""
    target_dir = pdf_dir or settings.pdf_dir
    return sorted(target_dir.rglob("*.pdf"))


def _safe_cache_name(pdf_path: Path) -> str:
    """Create a stable markdown filename without collisions across subfolders."""
    try:
        rel = str(pdf_path.relative_to(settings.project_root))
    except ValueError:
        rel = str(pdf_path)
    digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:10]
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", pdf_path.stem).strip("_") or "document"
    return f"{safe_stem}_{digest}.md"


def _markdown_cache_path(pdf_path: Path) -> Path:
    return settings.markdown_dir / _safe_cache_name(pdf_path)


def _extract_pages(pdf_path: Path) -> list[DocumentPage]:
    pages: list[DocumentPage] = []
    with fitz.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if text:
                # Normalisasi whitespace ringan agar retrieval lebih stabil, tanpa merusak isi.
                text = re.sub(r"[ \t]+", " ", text)
                text = re.sub(r"\n{3,}", "\n\n", text).strip()
                pages.append(DocumentPage(page_number=page_number, text=text))
    return pages


def extract_pdf_to_markdown(pdf_path: Path, force: bool = False) -> Path:
    """Extract a PDF to markdown cache with explicit page markers.

    Catatan: PyMuPDF cukup untuk PDF teks biasa. Untuk PDF tabel kompleks,
    file ini tetap menyimpan marker halaman agar chunk bisa dilacak ke sumber.
    """
    settings.markdown_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = _markdown_cache_path(pdf_path)

    if (
        markdown_path.exists()
        and not force
        and markdown_path.stat().st_mtime >= pdf_path.stat().st_mtime
    ):
        return markdown_path

    pages = _extract_pages(pdf_path)
    page_blocks = [f"## Halaman {page.page_number}\n\n{page.text}" for page in pages]
    markdown_text = f"# {pdf_path.name}\n\n" + "\n\n---\n\n".join(page_blocks)
    markdown_path.write_text(markdown_text.strip() + "\n", encoding="utf-8")
    return markdown_path


def _load_pages_from_markdown(markdown_path: Path) -> list[DocumentPage]:
    text = markdown_path.read_text(encoding="utf-8")
    matches = list(re.finditer(r"^##\s+Halaman\s+(\d+)\s*$", text, flags=re.MULTILINE | re.IGNORECASE))
    if not matches:
        stripped = text.strip()
        return [DocumentPage(page_number=1, text=stripped)] if stripped else []

    pages: list[DocumentPage] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        page_number = int(match.group(1))
        body = text[start:end]
        body = re.sub(r"\n?---\n?", "\n", body).strip()
        if body:
            pages.append(DocumentPage(page_number=page_number, text=body))
    return pages


def load_documents(force_extract: bool = False) -> list[Document]:
    """Load all PDFs from data/pdfs and return extracted markdown documents."""
    pdf_files = list_pdf_files()

    if not pdf_files:
        raise FileNotFoundError(
            f"Tidak ada PDF di folder: {settings.pdf_dir}. "
            "Masukkan file .pdf ke folder data/pdfs terlebih dahulu."
        )

    manifest = DocumentManifest()
    documents: list[Document] = []

    for pdf_path in pdf_files:
        markdown_path = extract_pdf_to_markdown(pdf_path, force=force_extract)
        text = markdown_path.read_text(encoding="utf-8").strip()
        pages = _load_pages_from_markdown(markdown_path)

        if not text:
            continue

        try:
            source = str(pdf_path.relative_to(settings.project_root))
        except ValueError:
            source = str(pdf_path)

        metadata = manifest.metadata_for(pdf_path, project_root=settings.project_root)
        metadata = normalize_document_metadata({
            **metadata,
            "source": source,
            "document_title": metadata.get("document_title") or pdf_path.stem.replace("_", " "),
        })

        documents.append(
            Document(
                source=source,
                text=text,
                markdown_path=str(markdown_path.relative_to(settings.project_root)),
                pages=pages,
                metadata=metadata,
            )
        )

    if not documents:
        raise ValueError("PDF ditemukan, tetapi tidak ada teks yang berhasil diekstrak.")

    return documents

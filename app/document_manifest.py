from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.config import settings


MANUAL_SOURCE_KEYWORDS = {
    "manual",
    "sop",
    "standard_operational_procedure",
    "petunjuk",
    "teknis",
    "juknis",
    "teknologi_budidaya",
    "budidaya",
    "gap",
    "pht",
    "modul",
    "buku",
    "ebook",
    "pedoman",
    "panduan",
}

RESEARCH_SOURCE_KEYWORDS = {
    "penelitian",
    "paper",
    "riset",
    "jurnal",
    "journal",
    "prosiding",
    "skripsi",
    "thesis",
    "tesis",
}

CROP_SOURCE_KEYWORDS = {
    "padi": {"padi", "oryza", "rice"},
    "jagung": {"jagung", "zea", "maize", "corn"},
    "cabai": {"cabai", "cabe", "capsicum", "chili", "chilli"},
    "cabai_merah": {"cabai_merah", "cabe_merah", "cabai-besar", "capsicum_annuum"},
    "cabai_rawit": {"cabai_rawit", "cabe_rawit", "capsicum_frutescens"},
    "tomat": {"tomat", "tomato", "lycopersicum"},
    "bawang": {"bawang", "onion", "allium"},
    "bawang_merah": {"bawang_merah", "shallot", "allium_ascalonicum", "allium_cepa"},
    "kedelai": {"kedelai", "soybean", "glycine"},
    "kentang": {"kentang", "potato", "tuberosum"},
    "terong": {"terong", "terung", "eggplant", "melongena"},
    "timun": {"timun", "mentimun", "ketimun", "cucumber", "cucumis"},
}


def _norm_path(value: str | Path) -> str:
    return str(value).replace("\\", "/").strip().lstrip("./")


def _norm_key(value: str | Path) -> str:
    text = _norm_path(value).lower()
    text = text.replace("-", "_").replace(" ", "_")
    return re.sub(r"_+", "_", text)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, (tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()] if str(value).strip() else []


class DocumentManifest:
    """Read optional document metadata from data/document_manifest.json.

    File ini tidak wajib. Jika tidak ada, metadata akan diinfer dari nama file.
    Format yang didukung:
    {
      "documents": [
        {"file": "data/pdfs/juknis_padi.pdf", "crop": "padi", ...}
      ]
    }
    """

    def __init__(self, manifest_file: Path | None = None) -> None:
        self.manifest_file = manifest_file or settings.document_manifest_file
        self._by_path: dict[str, dict[str, Any]] = {}
        self._by_name: dict[str, dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        if not self.manifest_file.exists():
            return

        try:
            raw = json.loads(self.manifest_file.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Gagal membaca document manifest: {self.manifest_file}: {exc}") from exc

        items = raw.get("documents", raw if isinstance(raw, list) else [])
        if not isinstance(items, list):
            raise ValueError("document_manifest.json harus berisi list atau object dengan key 'documents'.")

        for item in items:
            if not isinstance(item, dict):
                continue
            file_value = item.get("file") or item.get("path") or item.get("source")
            if not file_value:
                continue
            metadata = dict(item)
            metadata.pop("file", None)
            metadata.pop("path", None)
            metadata.pop("source", None)
            metadata = normalize_document_metadata(metadata)

            path_key = _norm_key(file_value)
            name_key = _norm_key(Path(str(file_value)).name)
            stem_key = _norm_key(Path(str(file_value)).stem)
            self._by_path[path_key] = metadata
            self._by_name[name_key] = metadata
            self._by_name[stem_key] = metadata

    def metadata_for(self, pdf_path: Path, project_root: Path | None = None) -> dict[str, Any]:
        project_root = project_root or settings.project_root
        candidates: list[str] = []
        try:
            candidates.append(_norm_key(pdf_path.relative_to(project_root)))
        except ValueError:
            pass
        candidates.append(_norm_key(pdf_path))
        candidates.append(_norm_key(pdf_path.name))
        candidates.append(_norm_key(pdf_path.stem))

        for key in candidates:
            if key in self._by_path:
                return dict(self._by_path[key])
            if key in self._by_name:
                return dict(self._by_name[key])

        return infer_metadata_from_path(pdf_path)

    def fingerprint(self) -> dict[str, Any] | None:
        if not self.manifest_file.exists():
            return None
        stat = self.manifest_file.stat()
        return {
            "path": _norm_path(self.manifest_file.relative_to(settings.project_root)),
            "mtime": stat.st_mtime,
            "size": stat.st_size,
        }


def normalize_document_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    result = dict(metadata or {})

    if "komoditas" in result and "crop" not in result:
        result["crop"] = result.pop("komoditas")
    if "fase" in result and "growth_stage" not in result:
        result["growth_stage"] = result.pop("fase")
    if "jenis_dokumen" in result and "doc_type" not in result:
        result["doc_type"] = result.pop("jenis_dokumen")
    if "tingkat_kepercayaan" in result and "authority" not in result:
        result["authority"] = result.pop("tingkat_kepercayaan")

    if "crop" in result and result["crop"] is not None:
        result["crop"] = str(result["crop"]).strip().lower().replace(" ", "_").replace("-", "_")
    if "growth_stage" in result and result["growth_stage"] is not None:
        result["growth_stage"] = str(result["growth_stage"]).strip().lower().replace(" ", "_").replace("-", "_")
    if "doc_type" in result and result["doc_type"] is not None:
        result["doc_type"] = str(result["doc_type"]).strip().lower().replace(" ", "_").replace("-", "_")
    if "source_type" in result and result["source_type"] is not None:
        result["source_type"] = str(result["source_type"]).strip().lower().replace(" ", "_").replace("-", "_")
    if "authority" in result and result["authority"] is not None:
        result["authority"] = str(result["authority"]).strip().upper()

    result["topics"] = _as_list(result.get("topics") or result.get("topic"))
    result.pop("topic", None)

    if not result.get("doc_type"):
        result["doc_type"] = "unknown"
    if not result.get("source_type"):
        result["source_type"] = "unknown"
    if not result.get("authority"):
        result["authority"] = "C"

    return result


def infer_metadata_from_path(path: Path | str) -> dict[str, Any]:
    text = _norm_key(path)
    metadata: dict[str, Any] = {}

    for crop, keywords in CROP_SOURCE_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            metadata["crop"] = crop
            break

    is_manual = any(keyword in text for keyword in MANUAL_SOURCE_KEYWORDS)
    is_research = any(keyword in text for keyword in RESEARCH_SOURCE_KEYWORDS)

    if is_manual:
        if "sop" in text or "standard_operational_procedure" in text:
            metadata["doc_type"] = "sop_manual"
        elif "petunjuk" in text or "teknis" in text or "juknis" in text:
            metadata["doc_type"] = "petunjuk_teknis"
        elif "modul" in text:
            metadata["doc_type"] = "modul_pelatihan"
        else:
            metadata["doc_type"] = "manual_book"
        metadata["source_type"] = "ebook_manual_resmi"
        metadata["authority"] = "A"
    elif is_research:
        metadata["doc_type"] = "paper_riset_asli"
        metadata["source_type"] = "paper_riset_asli"
        metadata["authority"] = "B"
    else:
        metadata["doc_type"] = "unknown"
        metadata["source_type"] = "unknown"
        metadata["authority"] = "C"

    filename = Path(str(path)).name
    if filename:
        metadata["document_title"] = re.sub(r"\.[A-Za-z0-9]+$", "", filename).replace("_", " ").strip()

    metadata.setdefault("topics", [])
    return normalize_document_metadata(metadata)

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import re
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Iterable, Optional
from uuid import uuid4

from app.config import settings
from app.openrouter_client import OpenRouterClient
from app.rag_service import RAGService
from mqtt_config import (
    DEFAULT_CROP,
    DEFAULT_GROWTH_STAGE,
    RAG_ANSWER_STYLE,
    RAG_LANGUAGE,
    RAG_MAX_ANSWER_TOKENS,
    RAG_TEMPERATURE,
)

logger = logging.getLogger("mqtt_processor")
WIB = timezone(timedelta(hours=7))

_rag_service: Optional[RAGService] = None
_rag_lock = asyncio.Lock()

# Debounce state in-memory. Untuk deployment multi-worker, pindahkan ke Redis/cache bersama.
_last_llm_state: dict[str, dict[str, Any]] = {}
_last_llm_lock = asyncio.Lock()

# Cache ringkas RAG/LLM per device + fingerprint status rule engine.
# Tujuan: ketika debounce aktif, response tidak kembali kosong/template;
# sistem memakai rag_answer dan sources terakhir selama status rule engine sama.
_rag_answer_cache: dict[str, dict[str, Any]] = {}
_rag_answer_cache_lock = asyncio.Lock()

# ==========================================================
# Payload locked final MVP:
# {
#   "id":"SS8IN12462",
#   "t":23.9, "h":45.0, "ec":1362, "ph":7.5,
#   "n":68, "p":96, "k":215, "f":748,
#   "lat":3.60645, "lon":98.71109,
#   "crop":"padi",
#   "growth_stage":"vegetatif"
# }
#
# Sistem internal tetap memakai nama panjang agar schema response
# dan prompt RAG mudah dibaca.
# ==========================================================

FIELD_ALIASES = {
    "device_id": ("id", "device_id"),
    "temperature": ("t", "temperature"),
    "humidity": ("h", "humidity"),
    "ec": ("ec",),
    "ph": ("ph",),
    "nitrogen": ("n", "nitrogen"),
    "phosphorus": ("p", "phosphorus"),
    "potassium": ("k", "potassium"),
    "fertility": ("f", "fertility"),
    "latitude": ("lat", "latitude"),
    "longitude": ("lon", "longitude"),
}

REQUIRED_CANONICAL_SENSOR_FIELDS = [
    "temperature",
    "humidity",
    "ec",
    "ph",
    "nitrogen",
    "phosphorus",
    "potassium",
    "fertility",
]

SENSOR_UNITS = {
    "temperature": "C",
    "humidity": "%",
    "ec": "uS/cm",
    "ph": "pH",
    "nitrogen": "mg/kg",
    "phosphorus": "mg/kg",
    "potassium": "mg/kg",
    "fertility": "index",
    "latitude": "degree",
    "longitude": "degree",
}

PRIORITY_WEIGHT = {"low": 1, "medium": 2, "high": 3}

ALLOWED_CROPS = {
    "padi",
    "jagung",
    "cabai",          # kategori umum; gunakan cabai_merah/cabai_rawit bila ingin lebih spesifik
    "cabai_merah",
    "cabai_rawit",
    "tomat",
    "bawang",         # kategori umum; default agronomi mengikuti bawang_merah
    "bawang_merah",
    "kedelai",
    "kentang",
    "terong",
    "timun",          # nama internal untuk mentimun/ketimun
}

CROP_ALIASES = {
    "padi_sawah": "padi",
    "oryza_sativa": "padi",
    "jagung_manis": "jagung",
    "zea_mays": "jagung",
    "cabai": "cabai",
    "cabe": "cabai",
    "cabe_merah": "cabai_merah",
    "cabai_merah": "cabai_merah",
    "cabai_besar": "cabai_merah",
    "cabai_rawit": "cabai_rawit",
    "cabe_rawit": "cabai_rawit",
    "tomat": "tomat",
    "bawang": "bawang",
    "bawang_merah": "bawang_merah",
    "kedelai": "kedelai",
    "kentang": "kentang",
    "terong": "terong",
    "terung": "terong",
    "timun": "timun",
    "ketimun": "timun",
    "mentimun": "timun",
}

ALLOWED_GROWTH_STAGES = {
    "awal_tanam",
    "vegetatif",
    "pembungaan",
    "pembuahan",
    "pematangan",
}

# Nilai ini tetap rule awal. Threshold produksi harus dikalibrasi dengan:
# 1) vendor sensor 8-in-1,
# 2) hasil uji tanah lokal,
# 3) komoditas,
# 4) fase tanaman,
# 5) satuan aktual NPK sensor.
# Struktur dibuat per crop supaya mudah diubah tanpa mengubah logic.
GLOBAL_RULE_DEFAULTS: dict[str, Any] = {
    "ph_optimal": (5.8, 7.2),
    "ec": {"very_low": 200, "low": 800, "medium": 1500, "high": 3000},
    "nitrogen": {"low": 40, "adequate": 80, "high": 150},
    "phosphorus": {"low": 15, "adequate": 40, "high": 80},
    "potassium": {"low": 100, "adequate": 200, "high": 300},
    "fertility": {"low": 300, "medium": 600, "good": 900},
}

CROP_STAGE_RULES: dict[str, dict[str, Any]] = {
    "padi": {
        "ph_optimal": (5.5, 7.0),
        "ec": {"very_low": 200, "low": 800, "medium": 1800, "high": 3500},
        "nitrogen": {"low": 40, "adequate": 85, "high": 160},
        "phosphorus": {"low": 15, "adequate": 45, "high": 85},
        "potassium": {"low": 90, "adequate": 190, "high": 300},
    },
    "jagung": {
        "ph_optimal": (5.8, 7.2),
        "ec": {"very_low": 200, "low": 900, "medium": 1800, "high": 3500},
        "nitrogen": {"low": 45, "adequate": 95, "high": 170},
        "phosphorus": {"low": 15, "adequate": 45, "high": 85},
        "potassium": {"low": 100, "adequate": 220, "high": 330},
    },
    "cabai": {
        "ph_optimal": (6.0, 7.0),
        "ec": {"very_low": 250, "low": 1000, "medium": 2000, "high": 3500},
        "nitrogen": {"low": 40, "adequate": 90, "high": 160},
        "phosphorus": {"low": 15, "adequate": 45, "high": 85},
        "potassium": {"low": 120, "adequate": 240, "high": 360},
    },
    "cabai_merah": {
        "ph_optimal": (6.0, 7.0),
        "ec": {"very_low": 250, "low": 1000, "medium": 2000, "high": 3500},
        "nitrogen": {"low": 40, "adequate": 90, "high": 160},
        "phosphorus": {"low": 15, "adequate": 45, "high": 85},
        "potassium": {"low": 120, "adequate": 240, "high": 360},
    },
    "cabai_rawit": {
        "ph_optimal": (6.0, 7.0),
        "ec": {"very_low": 250, "low": 1000, "medium": 2000, "high": 3500},
        "nitrogen": {"low": 40, "adequate": 90, "high": 160},
        "phosphorus": {"low": 15, "adequate": 45, "high": 85},
        "potassium": {"low": 120, "adequate": 240, "high": 360},
    },
    "tomat": {
        "ph_optimal": (6.0, 7.0),
        "ec": {"very_low": 250, "low": 1200, "medium": 2500, "high": 4000},
        "nitrogen": {"low": 40, "adequate": 90, "high": 160},
        "phosphorus": {"low": 15, "adequate": 45, "high": 85},
        "potassium": {"low": 120, "adequate": 250, "high": 380},
    },
    "bawang": {
        "ph_optimal": (6.0, 7.0),
        "ec": {"very_low": 200, "low": 900, "medium": 1800, "high": 3200},
        "nitrogen": {"low": 35, "adequate": 80, "high": 150},
        "phosphorus": {"low": 15, "adequate": 45, "high": 85},
        "potassium": {"low": 110, "adequate": 230, "high": 350},
    },
    "bawang_merah": {
        "ph_optimal": (6.0, 7.0),
        "ec": {"very_low": 200, "low": 900, "medium": 1800, "high": 3200},
        "nitrogen": {"low": 35, "adequate": 80, "high": 150},
        "phosphorus": {"low": 15, "adequate": 45, "high": 85},
        "potassium": {"low": 110, "adequate": 230, "high": 350},
    },
    "kedelai": {
        "ph_optimal": (5.8, 7.0),
        "ec": {"very_low": 200, "low": 900, "medium": 1800, "high": 3200},
        "nitrogen": {"low": 30, "adequate": 70, "high": 130},
        "phosphorus": {"low": 15, "adequate": 45, "high": 85},
        "potassium": {"low": 90, "adequate": 200, "high": 320},
    },
    "kentang": {
        "ph_optimal": (5.5, 6.5),
        "ec": {"very_low": 250, "low": 1000, "medium": 2000, "high": 3500},
        "nitrogen": {"low": 45, "adequate": 95, "high": 170},
        "phosphorus": {"low": 20, "adequate": 55, "high": 95},
        "potassium": {"low": 130, "adequate": 260, "high": 390},
    },
    "terong": {
        "ph_optimal": (5.8, 7.0),
        "ec": {"very_low": 250, "low": 1100, "medium": 2200, "high": 3800},
        "nitrogen": {"low": 40, "adequate": 90, "high": 160},
        "phosphorus": {"low": 15, "adequate": 45, "high": 85},
        "potassium": {"low": 120, "adequate": 250, "high": 380},
    },
    "timun": {
        "ph_optimal": (5.8, 7.0),
        "ec": {"very_low": 250, "low": 1100, "medium": 2200, "high": 3800},
        "nitrogen": {"low": 40, "adequate": 90, "high": 160},
        "phosphorus": {"low": 15, "adequate": 45, "high": 85},
        "potassium": {"low": 120, "adequate": 250, "high": 380},
    },
}

# Faktor kecil berbasis fase. Ini bukan rekomendasi dosis; hanya tuning klasifikasi awal.
STAGE_NUTRIENT_MULTIPLIERS: dict[str, dict[str, float]] = {
    "awal_tanam": {"nitrogen": 1.00, "phosphorus": 1.00, "potassium": 1.00},
    "vegetatif": {"nitrogen": 1.08, "phosphorus": 1.00, "potassium": 1.00},
    "pembungaan": {"nitrogen": 0.95, "phosphorus": 1.05, "potassium": 1.08},
    "pembuahan": {"nitrogen": 0.90, "phosphorus": 1.05, "potassium": 1.12},
    "pematangan": {"nitrogen": 0.85, "phosphorus": 1.00, "potassium": 1.05},
}

CROP_TERMS = {
    "padi": "padi sawah Oryza sativa tanaman padi pemupukan padi",
    "jagung": "jagung Zea mays pemupukan jagung",
    "cabai": "cabai Capsicum annuum Capsicum frutescens pemupukan cabai merah cabai rawit",
    "cabai_merah": "cabai merah Capsicum annuum pemupukan cabai merah",
    "cabai_rawit": "cabai rawit Capsicum frutescens pemupukan cabai rawit",
    "tomat": "tomat Solanum lycopersicum pemupukan tomat",
    "bawang": "bawang merah Allium cepa Allium ascalonicum pemupukan bawang",
    "bawang_merah": "bawang merah Allium cepa Allium ascalonicum pemupukan bawang merah",
    "kedelai": "kedelai Glycine max pemupukan kedelai",
    "kentang": "kentang Solanum tuberosum pemupukan kentang pembumbunan umbi",
    "terong": "terong terung Solanum melongena pemupukan terong",
    "timun": "timun mentimun ketimun Cucumis sativus pemupukan mentimun irigasi",
}

STAGE_TERMS = {
    "awal_tanam": "awal tanam persiapan lahan pemupukan dasar",
    "vegetatif": "fase vegetatif pertumbuhan daun batang anakan",
    "pembungaan": "fase pembungaan pembentukan bunga",
    "pembuahan": "fase pembuahan pembentukan buah bulir umbi polong",
    "pematangan": "fase pematangan menjelang panen",
}

ISSUE_TERMS = {
    ("phosphorus", "tinggi"): "fosfor tinggi pupuk P SP-36 TSP pemupukan fosfat berlebih",
    ("phosphorus", "sangat_tinggi"): "fosfor sangat tinggi hindari pupuk P SP-36 TSP fosfat berlebih",
    ("potassium", "tinggi"): "kalium tinggi pupuk K KCl berlebih keseimbangan hara",
    ("potassium", "sangat_tinggi"): "kalium sangat tinggi hindari KCl pemupukan K berlebih",
    ("nitrogen", "rendah"): "nitrogen rendah urea pemupukan N bertahap fase vegetatif",
    ("nitrogen", "tinggi"): "nitrogen tinggi urea berlebih pertumbuhan vegetatif berlebihan",
    ("ph", "sangat_asam"): "pH tanah sangat asam pengapuran dolomit toksisitas aluminium",
    ("ph", "asam"): "pH tanah asam pengapuran dolomit ketersediaan hara",
    ("ph", "agak_basa"): "pH agak basa unsur mikro ketersediaan hara tanah basa",
    ("ph", "basa_kuat"): "pH tanah basa kuat unsur mikro ketersediaan hara",
    ("ec", "tinggi"): "EC tinggi salinitas garam terlarut drainase pencucian tanah",
    ("ec", "sangat_tinggi"): "EC sangat tinggi salinitas garam terlarut cek kualitas air drainase",
}


# Preferensi dokumen untuk jawaban praktis.
# Manual/SOP diprioritaskan untuk rekomendasi budidaya ke petani;
# paper tetap dipakai sebagai evidence pendukung, bukan konteks utama bila manual tersedia.
PRACTICAL_DOC_TYPE_WEIGHT = {
    "sop_manual": 0.18,
    "petunjuk_teknis": 0.17,
    "manual_book": 0.16,
    "modul_pelatihan": 0.13,
    "ebook_manual_resmi": 0.13,
    "paper_riset_asli": 0.06,
    "paper": 0.06,
    "jurnal": 0.05,
    "unknown": 0.00,
}

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

DEFAULT_RAG_CONFIDENCE_REFERENCE_SCORE = 0.65

SIGNIFICANT_STATUSES = {
    "rendah",
    "tinggi",
    "sangat_tinggi",
    "sangat_asam",
    "asam",
    "basa_kuat",
    "perlu_validasi",
}

DEFAULT_LLM_TIMEOUT_SECONDS = 20
DEFAULT_LLM_MIN_INTERVAL_SECONDS = 300


# ==========================================================
# Agronomic decision layer
# ==========================================================
# Layer ini sengaja deterministik. Tujuannya agar output tidak hanya berupa
# ringkasan LLM, tetapi benar-benar memuat diagnosis agronomi: faktor pembatas,
# arah intervensi, dan rencana monitoring. Nilai ambang tetap perlu dikalibrasi
# dengan vendor sensor 8-in-1 dan hasil uji tanah lokal.

AGRONOMIC_STATUS_SEVERITY: dict[str, int] = {
    "normal": 0,
    "optimal_awal": 0,
    "baik": 0,
    "cukup": 0,
    "sedang": 0,
    "cukup_rendah": 1,
    "cukup_lembap": 0,
    "agak_asam": 1,
    "agak_basa": 2,
    "rendah": 2,
    "asam": 2,
    "tinggi": 2,
    "sangat_asam": 3,
    "basa_kuat": 3,
    "sangat_tinggi": 3,
    "sangat_lembap": 2,
    "perlu_validasi": 3,
}

STAGE_AGRONOMIC_FOCUS: dict[str, dict[str, list[str]]] = {
    "awal_tanam": {
        "focus": ["validasi pH", "kesiapan lahan", "ketersediaan P awal", "bahan organik"],
        "avoid": ["pemupukan tinggi sebelum data tanah valid", "aplikasi pupuk saat tanah terlalu basah"],
    },
    "vegetatif": {
        "focus": ["pertumbuhan akar dan daun", "nitrogen bertahap", "kelembapan tanah stabil", "pH mendukung serapan hara"],
        "avoid": ["kelebihan nitrogen", "penambahan P/K saat sudah tinggi", "salinitas akibat pemupukan rapat"],
    },
    "pembungaan": {
        "focus": ["keseimbangan N-P-K", "dukungan K untuk pembungaan", "air stabil", "hindari stres salinitas"],
        "avoid": ["N berlebih yang mendorong vegetatif", "kekeringan atau genangan", "pemupukan tanpa cek EC"],
    },
    "pembuahan": {
        "focus": ["kalium proporsional", "air cukup", "serapan hara stabil", "hindari pH ekstrem"],
        "avoid": ["penambahan N berlebih", "K berlebih saat sensor sudah tinggi", "drainase buruk"],
    },
    "pematangan": {
        "focus": ["stabilitas kelembapan", "hindari pemupukan agresif", "monitor stres tanaman"],
        "avoid": ["pupuk N tinggi menjelang panen", "akumulasi garam", "intervensi korektif tanpa validasi"],
    },
}

PARAMETER_ACTION_DIRECTIONS: dict[str, dict[str, str]] = {
    "temperature": {
        "perlu_validasi": "ulang pembacaan suhu setelah sensor stabil; jangan jadikan suhu ekstrem sebagai dasar pemupukan",
        "rendah": "pantau ulang pada waktu berbeda karena aktivitas akar dan mikroba dapat melambat",
        "tinggi": "pastikan ketersediaan air dan kurangi stres panas",
        "sangat_tinggi": "validasi sensor dan mitigasi stres panas/air",
    },
    "humidity": {
        "rendah": "cek kebutuhan pengairan di zona akar dan ulang pembacaan soil moisture",
        "sangat_lembap": "cek genangan, struktur tanah, dan drainase sebelum pemupukan",
    },
    "ec": {
        "rendah": "korelasikan dengan NPK; jangan langsung menaikkan semua pupuk tanpa melihat unsur pembatas",
        "tinggi": "tunda pemupukan pekat, cek kualitas air, dan pantau drainase",
        "sangat_tinggi": "prioritaskan validasi EC, kualitas air, dan risiko salinitas sebelum rekomendasi pupuk",
    },
    "ph": {
        "sangat_asam": "evaluasi pengapuran/dolomit berbasis uji tanah dan target pH komoditas",
        "asam": "rencanakan koreksi pH bertahap agar serapan NPK lebih efektif",
        "agak_basa": "hindari perlakuan yang menaikkan pH; pantau potensi hambatan unsur mikro",
        "basa_kuat": "validasi pH dan susun koreksi berbasis rekomendasi lokal sebelum pemupukan lanjutan",
    },
    "nitrogen": {
        "rendah": "prioritaskan N bertahap sesuai fase dan gejala tanaman, bukan aplikasi sekaligus",
        "tinggi": "hindari tambahan N agar tanaman tidak terlalu vegetatif",
        "sangat_tinggi": "tahan pupuk N dan validasi dengan gejala lapang/riwayat pemupukan",
    },
    "phosphorus": {
        "rendah": "evaluasi P terutama pada awal tanam dan perakaran, dengan mempertimbangkan pH",
        "tinggi": "batasi pupuk P seperti SP-36/TSP sampai ada dasar kebutuhan baru",
        "sangat_tinggi": "hindari pupuk P sementara; validasi kemungkinan akumulasi P",
    },
    "potassium": {
        "rendah": "evaluasi K terutama pada pembungaan/pembuahan dan kondisi stres air",
        "tinggi": "batasi pupuk K/KCl dan jaga keseimbangan hara lain",
        "sangat_tinggi": "hindari tambahan K; validasi potensi antagonisme hara dan akumulasi garam",
    },
    "fertility": {
        "rendah": "evaluasi bahan organik, pH, EC, dan NPK secara terpadu",
        "sangat_tinggi": "pastikan tidak terjadi akumulasi hara/garam akibat pemupukan berlebih",
    },
}


def _agronomic_severity(status: Any) -> int:
    return AGRONOMIC_STATUS_SEVERITY.get(str(status or "").strip().lower(), 1)


def _agronomic_effect(parameter: str, item: Dict[str, Any], crop: str, growth_stage: str) -> str:
    status = str(item.get("status") or "").strip().lower()
    value = item.get("value")
    unit = item.get("unit") or ""
    prefix = f"{parameter}={value} {unit}".strip()

    effects = {
        ("ph", "asam"): f"{prefix}: pH asam dapat menghambat perkembangan akar dan efisiensi serapan hara pada {crop}.",
        ("ph", "sangat_asam"): f"{prefix}: pH sangat asam berisiko menekan ketersediaan hara dan meningkatkan toksisitas unsur tertentu.",
        ("ph", "agak_basa"): f"{prefix}: pH agak basa dapat menurunkan ketersediaan beberapa unsur mikro.",
        ("ph", "basa_kuat"): f"{prefix}: pH basa kuat berisiko mengganggu ketersediaan hara dan perlu validasi lapang.",
        ("ec", "tinggi"): f"{prefix}: EC tinggi menunjukkan akumulasi ion/garam terlarut yang dapat menekan serapan air dan hara.",
        ("ec", "sangat_tinggi"): f"{prefix}: EC sangat tinggi perlu diperlakukan sebagai risiko salinitas sampai terbukti sebaliknya.",
        ("humidity", "rendah"): f"{prefix}: kelembapan tanah rendah dapat menghambat serapan hara di zona akar.",
        ("humidity", "sangat_lembap"): f"{prefix}: kelembapan tanah sangat tinggi dapat memperbesar risiko drainase buruk dan gangguan akar.",
        ("nitrogen", "rendah"): f"{prefix}: N rendah dapat membatasi pertumbuhan vegetatif, terutama pada fase {growth_stage}.",
        ("nitrogen", "tinggi"): f"{prefix}: N tinggi dapat mendorong vegetatif berlebih dan menurunkan efisiensi pemupukan.",
        ("nitrogen", "sangat_tinggi"): f"{prefix}: N sangat tinggi perlu divalidasi karena berisiko menyebabkan ketidakseimbangan pertumbuhan.",
        ("phosphorus", "rendah"): f"{prefix}: P rendah dapat mengganggu perakaran dan energi pertumbuhan awal.",
        ("phosphorus", "tinggi"): f"{prefix}: P tinggi menunjukkan pupuk fosfat perlu dibatasi sampai ada dasar kebutuhan baru.",
        ("phosphorus", "sangat_tinggi"): f"{prefix}: P sangat tinggi mengarah pada risiko akumulasi fosfat dan pemborosan input.",
        ("potassium", "rendah"): f"{prefix}: K rendah dapat membatasi regulasi air dan pembentukan hasil pada fase generatif.",
        ("potassium", "tinggi"): f"{prefix}: K tinggi membuat tambahan K/KCl tidak disarankan tanpa validasi.",
        ("potassium", "sangat_tinggi"): f"{prefix}: K sangat tinggi dapat berkontribusi pada ketidakseimbangan hara dan salinitas.",
        ("fertility", "rendah"): f"{prefix}: indeks kesuburan rendah perlu dibaca bersama pH, EC, bahan organik, dan NPK.",
        ("fertility", "sangat_tinggi"): f"{prefix}: indeks kesuburan sangat tinggi perlu dicek agar tidak mencerminkan akumulasi hara/garam.",
        ("temperature", "perlu_validasi"): f"{prefix}: suhu tidak wajar untuk lahan tropis sehingga perlu validasi sensor sebelum tindakan agronomi.",
    }
    return effects.get((parameter, status), f"{prefix}: status {status or 'tidak diketahui'} perlu dibaca bersama komoditas, fase, dan kondisi lapang.")


def _action_direction(parameter: str, item: Dict[str, Any]) -> str:
    status = str(item.get("status") or "").strip().lower()
    return PARAMETER_ACTION_DIRECTIONS.get(parameter, {}).get(
        status,
        str(item.get("recommendation") or "Pantau ulang parameter ini bersama kondisi lapang."),
    )


def build_agronomic_diagnosis(rag_request: Dict[str, Any], rule_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Diagnosis deterministik agar sistem lebih fokus pada agronomi, bukan sekadar template RAG."""
    crop = rag_request.get("crop_context", {}).get("crop", "unknown")
    growth_stage = rag_request.get("crop_context", {}).get("growth_stage", "unknown")
    pa = rule_analysis.get("parameter_analysis", {})
    npk = rule_analysis.get("npk_analysis", {})
    stage_focus = STAGE_AGRONOMIC_FOCUS.get(growth_stage, {"focus": [], "avoid": []})

    limiting_factors: list[Dict[str, Any]] = []
    for parameter, item in pa.items():
        status = item.get("status")
        priority = item.get("priority", "low")
        severity = _agronomic_severity(status)
        if priority in {"medium", "high"} or severity >= 2:
            limiting_factors.append({
                "parameter": parameter,
                "value": item.get("value"),
                "unit": item.get("unit"),
                "status": status,
                "priority": priority,
                "severity_score": severity,
                "agronomic_effect": _agronomic_effect(parameter, item, crop, growth_stage),
                "action_direction": _action_direction(parameter, item),
            })

    limiting_factors.sort(
        key=lambda x: (PRIORITY_WEIGHT.get(str(x.get("priority")), 1), int(x.get("severity_score", 0))),
        reverse=True,
    )

    nutrient_direction: list[str] = []
    n_status = npk.get("nitrogen_status")
    p_status = npk.get("phosphorus_status")
    k_status = npk.get("potassium_status")
    high_status = {"tinggi", "sangat_tinggi"}

    if n_status == "rendah":
        nutrient_direction.append("N menjadi kandidat faktor pembatas; koreksi sebaiknya bertahap dan disesuaikan fase.")
    elif n_status in high_status:
        nutrient_direction.append("N sudah tinggi; hindari tambahan urea/pupuk N tanpa validasi gejala lapang.")

    if p_status == "rendah":
        nutrient_direction.append("P perlu dievaluasi bersama pH karena efektivitas P sangat dipengaruhi kemasaman tanah.")
    elif p_status in high_status:
        nutrient_direction.append("P sudah tinggi; batasi SP-36/TSP/pupuk fosfat sampai ada rekomendasi baru.")

    if k_status == "rendah":
        nutrient_direction.append("K perlu diprioritaskan terutama bila tanaman memasuki fase pembungaan/pembuahan.")
    elif k_status in high_status:
        nutrient_direction.append("K sudah tinggi; hindari tambahan KCl/pupuk K berlebih untuk mencegah ketidakseimbangan hara.")

    if not nutrient_direction:
        nutrient_direction.append("NPK relatif tidak menunjukkan faktor pembatas besar; fokus pada pemupukan berimbang dan monitoring berkala.")

    immediate_actions = [item["action_direction"] for item in limiting_factors[:4]]
    if not immediate_actions:
        immediate_actions = ["Lanjutkan monitoring sensor dan cocokkan dengan gejala tanaman sebelum pemupukan berikutnya."]

    monitoring_plan = [
        "Ulangi pembacaan pada titik yang sama setelah sensor stabil.",
        "Bandingkan hasil sensor dengan kondisi visual tanaman dan riwayat pemupukan terakhir.",
        "Gunakan uji tanah/lab atau rekomendasi penyuluh sebagai validasi sebelum dosis pupuk spesifik.",
    ]

    limiting_factor_text = (
        ", ".join(
            f"{item.get('parameter')}={item.get('status')}"
            for item in limiting_factors[:4]
        )
        if limiting_factors
        else "tidak ada faktor pembatas medium/high dari rule engine"
    )

    focus_text = ", ".join(stage_focus.get("focus", [])[:4]) or "monitoring hara dan kondisi tanah"
    diagnosis_summary = (
        f"Fokus agronomi {crop} fase {growth_stage}: {focus_text}. "
        f"Faktor pembatas utama: {limiting_factor_text}."
    )

    return {
        "diagnosis_summary": diagnosis_summary,
        "stage_focus": stage_focus.get("focus", []),
        "stage_avoid": stage_focus.get("avoid", []),
        "limiting_factors": limiting_factors,
        "nutrient_strategy": {
            "npk_balance_status": npk.get("balance_status"),
            "direction": nutrient_direction,
        },
        "immediate_actions": _unique_strings(immediate_actions),
        "monitoring_plan": monitoring_plan,
        "dose_policy": "Tidak memberikan dosis pupuk pasti sebelum satuan sensor, luas lahan, umur tanaman, varietas, riwayat pemupukan, dan validasi uji tanah tersedia.",
    }

# ==========================================================
# Basic utilities
# ==========================================================


def now_wib_iso() -> str:
    return datetime.now(WIB).isoformat(timespec="seconds")


def safe_json_loads(payload_bytes: bytes) -> Optional[Dict[str, Any]]:
    """Hanya menerima JSON object/dict."""
    try:
        obj = json.loads(payload_bytes.decode("utf-8", errors="strict"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def dumps_compact(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def normalize_crop_name(value: Any, default: str | None = None) -> str:
    """Normalisasi nama komoditas dari UI/payload ke nama internal backend."""
    raw = str(value or default or "").strip().lower()
    raw = raw.replace("-", "_").replace(" ", "_")
    raw = re.sub(r"_+", "_", raw).strip("_")
    return CROP_ALIASES.get(raw, raw)


def normalize_humidity_type(payload: Dict[str, Any] | None = None, default: str = "soil") -> str:
    """Compatibility helper.

    Untuk sistem ini field `humidity` dari sensor 8-in-1 sudah dipastikan sebagai
    soil moisture / kelembapan tanah. Fungsi ini tetap dipertahankan agar kode lama
    yang memanggilnya tidak error, tetapi hasilnya selalu `soil`.
    Nilai ini hanya dipakai internal dan tidak ditampilkan di JSON response.
    """
    return "soil"


def _first_value(payload: Dict[str, Any], aliases: tuple[str, ...], default: Any = None) -> Any:
    for alias in aliases:
        if alias in payload:
            return payload.get(alias)
    return default


def _has_any_alias(payload: Dict[str, Any], canonical_key: str) -> bool:
    return any(alias in payload for alias in FIELD_ALIASES[canonical_key])


def _is_finite_number(value: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _to_float_alias(payload: Dict[str, Any], canonical_key: str) -> float:
    aliases = FIELD_ALIASES[canonical_key]
    for alias in aliases:
        if alias in payload:
            try:
                value = float(payload[alias])
            except Exception as exc:
                raise ValueError(
                    f"Field {alias} untuk {canonical_key} harus berupa angka. "
                    f"Nilai diterima: {payload.get(alias)!r}"
                ) from exc
            if not _is_finite_number(value):
                raise ValueError(f"Field {alias} untuk {canonical_key} harus angka finite, bukan NaN/inf.")
            return value
    raise ValueError(f"Field wajib tidak ada: {'/'.join(aliases)}")


def _to_optional_float_alias(payload: Dict[str, Any], canonical_key: str) -> float | None:
    aliases = FIELD_ALIASES[canonical_key]
    value = _first_value(payload, aliases, default=None)
    if value is None or value == "":
        return None
    try:
        converted = float(value)
    except Exception as exc:
        raise ValueError(
            f"Field {'/'.join(aliases)} untuk {canonical_key} harus berupa angka atau null. "
            f"Nilai diterima: {value!r}"
        ) from exc
    if not _is_finite_number(converted):
        raise ValueError(f"Field {'/'.join(aliases)} untuk {canonical_key} harus angka finite, bukan NaN/inf.")
    return converted


def _setting_value(name: str, default: Any) -> Any:
    return getattr(settings, name, default)


# ==========================================================
# Telemetry validation and normalization
# ==========================================================


def validate_telemetry(payload: Dict[str, Any]) -> list[str]:
    """Validasi payload dan nilai dasar sensor.

    Perubahan utama:
    - required field dicek berdasarkan canonical field + alias pendek/panjang;
    - nilai tidak fisik ditolak dengan ValueError;
    - nilai mencurigakan tetapi masih mungkin diberi warning;
    - crop/growth_stage tetap fleksibel tetapi diberi warning bila di luar dropdown.
    """
    warnings: list[str] = []

    missing = [
        field
        for field in REQUIRED_CANONICAL_SENSOR_FIELDS
        if not _has_any_alias(payload, field)
    ]
    if "crop" not in payload and "tanaman" not in payload:
        missing.append("crop")
    if "growth_stage" not in payload and "fase" not in payload:
        missing.append("growth_stage")

    if missing:
        raise ValueError(f"Field telemetry belum lengkap: {', '.join(missing)}")

    lat = _to_optional_float_alias(payload, "latitude")
    lon = _to_optional_float_alias(payload, "longitude")
    if lat is None or lon is None:
        warnings.append("Latitude/longitude belum tersedia. Analisis tetap diproses tanpa konteks lokasi presisi.")
    else:
        if not (-90 <= lat <= 90):
            raise ValueError("lat/latitude tidak valid. Rentang valid -90 sampai 90.")
        if not (-180 <= lon <= 180):
            raise ValueError("lon/longitude tidak valid. Rentang valid -180 sampai 180.")

    temperature = _to_float_alias(payload, "temperature")
    if not (-20 <= temperature <= 70):
        raise ValueError("Nilai suhu tidak valid untuk sensor lapang. Rentang diterima -20 sampai 70 C.")
    if temperature < 10 or temperature > 45:
        warnings.append("Nilai suhu sangat tidak umum untuk lahan tropis. Cek stabilitas dan kalibrasi sensor suhu.")

    humidity = _to_float_alias(payload, "humidity")
    if not (0 <= humidity <= 100):
        raise ValueError("Nilai humidity tidak valid. Rentang valid 0 sampai 100%.")

    ph = _to_float_alias(payload, "ph")
    if not (0 <= ph <= 14):
        raise ValueError("Nilai pH tidak valid. Rentang fisik pH adalah 0 sampai 14.")

    ec = _to_float_alias(payload, "ec")
    if ec < 0:
        raise ValueError("Nilai EC tidak valid karena negatif.")
    if ec > 20000:
        warnings.append("Nilai EC sangat tinggi. Cek satuan sensor, kualitas probe, dan kalibrasi.")

    for field in ["nitrogen", "phosphorus", "potassium", "fertility"]:
        value = _to_float_alias(payload, field)
        if value < 0:
            raise ValueError(f"Nilai {field} tidak valid karena negatif.")
        if field != "fertility" and value > 5000:
            warnings.append(f"Nilai {field} sangat tinggi. Cek satuan sensor dan kalibrasi NPK.")
        if field == "fertility" and value > 5000:
            warnings.append("Nilai fertility sangat tinggi. Cek skala index dari vendor sensor.")

    crop = normalize_crop_name(payload.get("crop") or payload.get("tanaman"))
    growth_stage = str(payload.get("growth_stage") or payload.get("fase") or "").strip().lower()

    if not crop:
        raise ValueError("Field wajib tidak ada: crop")
    if not growth_stage:
        raise ValueError("Field wajib tidak ada: growth_stage")

    if crop not in ALLOWED_CROPS:
        warnings.append(
            f"Nilai crop={crop!r} belum ada di daftar dropdown resmi. "
            f"Gunakan salah satu: {', '.join(sorted(ALLOWED_CROPS))}."
        )
    if growth_stage not in ALLOWED_GROWTH_STAGES:
        warnings.append(
            f"Nilai growth_stage={growth_stage!r} belum ada di daftar fase resmi. "
            f"Gunakan salah satu: {', '.join(sorted(ALLOWED_GROWTH_STAGES))}."
        )


    payload_device_id = _first_value(payload, FIELD_ALIASES["device_id"])
    if payload_device_id in (None, ""):
        warnings.append("Field id kosong/tidak ada. Device ID diambil dari topic MQTT.")

    if payload.get("is_calibrated", True) is False:
        warnings.append("Sensor ditandai belum terkalibrasi. Rekomendasi perlu dianggap sebagai indikasi awal.")

    return warnings


def normalize_sensor_payload(payload: Dict[str, Any], topic_device_id: str) -> Dict[str, Any]:
    """Normalisasi payload ringkas device menjadi snapshot sensor standar."""
    payload_device_id = _first_value(payload, FIELD_ALIASES["device_id"])
    device_id = str(payload_device_id or topic_device_id).strip() or topic_device_id

    # Topic device_id tetap sumber utama agar routing MQTT aman.
    if payload_device_id and str(payload_device_id) != topic_device_id:
        logger.warning(
            "device_id payload berbeda dengan topic. payload=%s topic=%s. Menggunakan topic.",
            payload_device_id,
            topic_device_id,
        )
        device_id = topic_device_id

    return {
        "device_id": device_id,
        "temperature": _to_float_alias(payload, "temperature"),
        "humidity": _to_float_alias(payload, "humidity"),
        "ec": _to_float_alias(payload, "ec"),
        "ph": _to_float_alias(payload, "ph"),
        "nitrogen": _to_float_alias(payload, "nitrogen"),
        "phosphorus": _to_float_alias(payload, "phosphorus"),
        "potassium": _to_float_alias(payload, "potassium"),
        "fertility": _to_float_alias(payload, "fertility"),
        "latitude": _to_optional_float_alias(payload, "latitude"),
        "longitude": _to_optional_float_alias(payload, "longitude"),
    }


# ==========================================================
# Rule resolver per crop/stage
# ==========================================================


def _deep_merge_rule(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = json.loads(json.dumps(base))
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


def resolve_crop_stage_rule(crop: str, growth_stage: str) -> dict[str, Any]:
    crop_rule = CROP_STAGE_RULES.get(crop, {})
    rule = _deep_merge_rule(GLOBAL_RULE_DEFAULTS, crop_rule)

    multipliers = STAGE_NUTRIENT_MULTIPLIERS.get(growth_stage, {})
    for nutrient in ["nitrogen", "phosphorus", "potassium"]:
        multiplier = float(multipliers.get(nutrient, 1.0))
        if nutrient in rule and isinstance(rule[nutrient], dict):
            rule[nutrient] = {
                key: round(float(value) * multiplier, 2)
                for key, value in rule[nutrient].items()
            }
    return rule


# ==========================================================
# Rule engine agronomi
# ==========================================================


def _analysis_item(
    value: float,
    unit: str,
    status: str,
    interpretation: str,
    recommendation: str,
    priority: str = "low",
) -> Dict[str, Any]:
    return {
        "value": value,
        "unit": unit,
        "status": status,
        "priority": priority,
        "interpretation": interpretation,
        "recommendation": recommendation,
    }


def classify_temperature(value: float, crop: str | None = None, growth_stage: str | None = None) -> Dict[str, Any]:
    if value < 15:
        return _analysis_item(
            value,
            "C",
            "perlu_validasi",
            "Nilai suhu sangat rendah untuk kondisi lahan tropis umum; ada kemungkinan sensor belum stabil, titik ukur terlalu dingin, atau kalibrasi perlu dicek.",
            "Ulangi pembacaan setelah sensor stabil dan cek kalibrasi sensor suhu.",
            "high",
        )
    if value < 20:
        return _analysis_item(value, "C", "rendah", "Suhu relatif rendah sehingga aktivitas mikroba dan pertumbuhan tanaman dapat melambat.", "Pantau ulang pada waktu berbeda dan sesuaikan dengan kondisi lokasi.", "medium")
    if value <= 35:
        return _analysis_item(value, "C", "normal", "Suhu berada pada rentang umum yang masih mendukung pertumbuhan banyak tanaman tropis.", "Pertahankan pemantauan berkala.", "low")
    if value <= 40:
        return _analysis_item(value, "C", "tinggi", "Suhu cukup tinggi dan dapat meningkatkan stres tanaman serta penguapan.", "Perhatikan ketersediaan air dan lakukan pemantauan lebih sering.", "medium")
    return _analysis_item(value, "C", "sangat_tinggi", "Suhu sangat tinggi dan berisiko menyebabkan stres tanaman.", "Cek ulang sensor dan lakukan tindakan mitigasi panas/air bila kondisi lapangan sesuai.", "high")


def classify_humidity(value: float, humidity_type: str = "soil") -> Dict[str, Any]:
    """Klasifikasi humidity sebagai soil moisture / kelembapan tanah.

    Parameter `humidity_type` dipertahankan untuk backward compatibility, tetapi
    sistem sekarang menganggap humidity selalu berasal dari sensor kelembapan tanah.
    """
    if value < 30:
        return _analysis_item(
            value,
            "%",
            "rendah",
            "Kelembapan tanah rendah sehingga ketersediaan air di sekitar perakaran perlu diperhatikan.",
            "Cek kondisi tanah di sekitar titik sensor dan pertimbangkan kebutuhan pengairan sesuai kondisi lapang.",
            "medium",
        )
    if value < 60:
        return _analysis_item(
            value,
            "%",
            "sedang",
            "Kelembapan tanah berada pada tingkat sedang.",
            "Pertahankan pemantauan berkala dan sesuaikan pengairan dengan kondisi tanaman.",
            "low",
        )
    if value <= 80:
        return _analysis_item(
            value,
            "%",
            "cukup_lembap",
            "Kelembapan tanah cukup lembap dan umumnya masih aman untuk banyak kondisi budidaya.",
            "Pantau agar tanah tidak terlalu basah, terutama pada tanaman rentan penyakit akar atau kondisi drainase buruk.",
            "low",
        )
    return _analysis_item(
        value,
        "%",
        "sangat_lembap",
        "Kelembapan tanah sangat tinggi sehingga kondisi genangan, drainase, dan risiko penyakit terkait kelembapan perlu diperhatikan.",
        "Cek kondisi tanah, genangan, dan drainase di sekitar titik sensor; pantau gejala penyakit atau pertumbuhan yang tidak normal.",
        "medium",
    )


def classify_ec(value: float, rule: dict[str, Any]) -> Dict[str, Any]:
    thresholds = rule.get("ec", GLOBAL_RULE_DEFAULTS["ec"])
    if value < thresholds["very_low"]:
        return _analysis_item(value, "uS/cm", "rendah", "EC rendah mengindikasikan ion/nutrisi terlarut relatif rendah.", "Korelasikan dengan status NPK dan kondisi tanaman sebelum pemupukan.", "medium")
    if value < thresholds["low"]:
        return _analysis_item(value, "uS/cm", "cukup_rendah", "EC cukup rendah sampai sedang; risiko salinitas umumnya rendah.", "Pantau bersama pH dan NPK.", "low")
    if value < thresholds["medium"]:
        return _analysis_item(value, "uS/cm", "sedang", "EC berada pada tingkat sedang; kandungan ion terlarut terdeteksi cukup.", "Pantau agar tidak meningkat berlebihan setelah pemupukan.", "low")
    if value < thresholds["high"]:
        return _analysis_item(value, "uS/cm", "tinggi", "EC tinggi dapat mengindikasikan konsentrasi garam/nutrisi terlarut yang perlu diawasi.", "Hindari pemupukan berlebihan dan cek kondisi air/drainase.", "medium")
    return _analysis_item(value, "uS/cm", "sangat_tinggi", "EC sangat tinggi dan berpotensi menekan serapan air/hara pada tanaman sensitif.", "Cek ulang sensor, kualitas air, dan lakukan evaluasi salinitas.", "high")


def classify_ph(value: float, crop: str, growth_stage: str, rule: dict[str, Any]) -> Dict[str, Any]:
    optimal_low, optimal_high = rule.get("ph_optimal", GLOBAL_RULE_DEFAULTS["ph_optimal"])

    if value < 4.5:
        return _analysis_item(value, "pH", "sangat_asam", "Tanah sangat asam; ketersediaan hara dan toksisitas unsur tertentu dapat menjadi masalah.", "Perlu evaluasi pengapuran dan validasi ulang pH.", "high")
    if value < optimal_low - 0.5:
        return _analysis_item(value, "pH", "asam", f"pH berada di bawah rentang awal yang ditargetkan untuk {crop} fase {growth_stage}.", "Pertimbangkan perbaikan pH sesuai komoditas, fase tanaman, dan rekomendasi lokal.", "medium")
    if value < optimal_low:
        return _analysis_item(value, "pH", "agak_asam", f"pH sedikit di bawah rentang target awal {optimal_low}-{optimal_high} untuk {crop}.", "Pantau pH berkala dan sesuaikan dengan kebutuhan tanaman.", "low")
    if value <= optimal_high:
        return _analysis_item(value, "pH", "optimal_awal", f"pH berada dalam rentang target awal {optimal_low}-{optimal_high} untuk {crop} berdasarkan rule internal.", "Pertahankan pemantauan berkala.", "low")
    if value <= optimal_high + 0.8:
        return _analysis_item(value, "pH", "agak_basa", f"pH berada di atas rentang target awal {optimal_low}-{optimal_high} untuk {crop}.", "Hindari perlakuan yang semakin menaikkan pH. Sesuaikan pemupukan dengan komoditas dan hasil uji tanah.", "medium")
    return _analysis_item(value, "pH", "basa_kuat", "pH sangat basa dan dapat mengganggu ketersediaan beberapa hara.", "Cek ulang sensor dan lakukan strategi koreksi pH berbasis rekomendasi lokal.", "high")


def _classify_nutrient(value: float, unit: str, nutrient: str, thresholds: dict[str, float], interpretation_name: str, low_reco: str, high_reco: str) -> Dict[str, Any]:
    if value < thresholds["low"]:
        return _analysis_item(value, unit, "rendah", f"{interpretation_name} relatif rendah berdasarkan rule awal sensor.", low_reco, "medium")
    if value < thresholds["adequate"]:
        return _analysis_item(value, unit, "cukup", f"{interpretation_name} berada pada kisaran cukup berdasarkan rule awal sensor.", "Pertahankan pemantauan dan hindari pemupukan berlebihan.", "low")
    if value < thresholds["high"]:
        return _analysis_item(value, unit, "tinggi", f"{interpretation_name} relatif tinggi berdasarkan rule awal sensor.", high_reco, "medium")
    return _analysis_item(value, unit, "sangat_tinggi", f"{interpretation_name} sangat tinggi berdasarkan rule awal sensor.", high_reco, "high")


def classify_nitrogen(value: float, rule: dict[str, Any]) -> Dict[str, Any]:
    return _classify_nutrient(
        value,
        "mg/kg",
        "nitrogen",
        rule.get("nitrogen", GLOBAL_RULE_DEFAULTS["nitrogen"]),
        "Nitrogen",
        "Pertimbangkan pemupukan N bertahap sesuai fase tanaman dan rekomendasi dokumen.",
        "Kurangi risiko kelebihan N, terutama jika tanaman terlalu vegetatif. Hindari tambahan N tanpa dasar kebutuhan tanaman.",
    )


def classify_phosphorus(value: float, rule: dict[str, Any]) -> Dict[str, Any]:
    return _classify_nutrient(
        value,
        "mg/kg",
        "phosphorus",
        rule.get("phosphorus", GLOBAL_RULE_DEFAULTS["phosphorus"]),
        "Fosfor",
        "Pertimbangkan sumber P sesuai fase tanaman dan kondisi pH.",
        "Batasi atau hindari penambahan pupuk P kecuali ada rekomendasi spesifik dari uji tanah/lapang.",
    )


def classify_potassium(value: float, rule: dict[str, Any]) -> Dict[str, Any]:
    return _classify_nutrient(
        value,
        "mg/kg",
        "potassium",
        rule.get("potassium", GLOBAL_RULE_DEFAULTS["potassium"]),
        "Kalium",
        "Pertimbangkan pupuk K sesuai komoditas dan fase generatif.",
        "Jangan menambah K berlebihan tanpa indikasi kebutuhan tanaman.",
    )


def classify_fertility(value: float, rule: dict[str, Any]) -> Dict[str, Any]:
    thresholds = rule.get("fertility", GLOBAL_RULE_DEFAULTS["fertility"])
    if value < thresholds["low"]:
        return _analysis_item(value, "index", "rendah", "Indeks kesuburan rendah berdasarkan skala awal sistem.", "Perlu evaluasi bahan organik, pH, EC, dan NPK secara terpadu.", "high")
    if value < thresholds["medium"]:
        return _analysis_item(value, "index", "sedang", "Indeks kesuburan sedang berdasarkan skala awal sistem.", "Lakukan pemupukan berimbang dan pemantauan berkala.", "medium")
    if value < thresholds["good"]:
        return _analysis_item(value, "index", "baik", "Indeks kesuburan berada pada kategori baik berdasarkan skala awal sistem.", "Pertahankan pengelolaan tanah dan hindari pemupukan berlebihan.", "low")
    return _analysis_item(value, "index", "sangat_tinggi", "Indeks kesuburan sangat tinggi berdasarkan skala awal sistem.", "Pastikan tidak terjadi akumulasi garam/hara berlebih.", "medium")


def build_parameter_analysis(sensor: Dict[str, Any], crop: str, growth_stage: str, humidity_type: str = "unknown") -> Dict[str, Dict[str, Any]]:
    rule = resolve_crop_stage_rule(crop, growth_stage)
    return {
        "temperature": classify_temperature(sensor["temperature"], crop, growth_stage),
        "humidity": classify_humidity(sensor["humidity"], humidity_type=humidity_type),
        "ec": classify_ec(sensor["ec"], rule),
        "ph": classify_ph(sensor["ph"], crop, growth_stage, rule),
        "nitrogen": classify_nitrogen(sensor["nitrogen"], rule),
        "phosphorus": classify_phosphorus(sensor["phosphorus"], rule),
        "potassium": classify_potassium(sensor["potassium"], rule),
        "fertility": classify_fertility(sensor["fertility"], rule),
    }


def _max_priority(items: list[Dict[str, Any]]) -> str:
    if not items:
        return "low"
    return max((item.get("priority", "low") for item in items), key=lambda p: PRIORITY_WEIGHT.get(p, 1))


def build_npk_analysis(parameter_analysis: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    n = parameter_analysis["nitrogen"]
    p = parameter_analysis["phosphorus"]
    k = parameter_analysis["potassium"]

    statuses = [n["status"], p["status"], k["status"]]
    high_statuses = {"tinggi", "sangat_tinggi"}

    if any(s in high_statuses for s in statuses) and any(s == "rendah" for s in statuses):
        balance_status = "tidak_seimbang"
    elif any(s in high_statuses for s in statuses):
        balance_status = "cenderung_tinggi"
    elif any(s == "rendah" for s in statuses):
        balance_status = "cenderung_rendah"
    else:
        balance_status = "relatif_seimbang"

    recommendations = []
    if n["status"] == "rendah":
        recommendations.append("prioritaskan pemenuhan nitrogen secara bertahap")
    if n["status"] in high_statuses:
        recommendations.append("hindari tambahan nitrogen tanpa dasar kebutuhan tanaman")
    if p["status"] in high_statuses:
        recommendations.append("batasi penambahan pupuk fosfor")
    if k["status"] in high_statuses:
        recommendations.append("batasi penambahan pupuk kalium")
    if not recommendations:
        recommendations.append("gunakan pemupukan berimbang sesuai komoditas dan fase tanaman")

    return {
        "summary": f"Status N={n['status']}, P={p['status']}, K={k['status']} dengan keseimbangan {balance_status}.",
        "nitrogen_status": n["status"],
        "phosphorus_status": p["status"],
        "potassium_status": k["status"],
        "balance_status": balance_status,
        "interpretation": "Klasifikasi NPK dibuat oleh rule engine agar tidak bergantung pada tebakan LLM.",
        "recommendation": "; ".join(recommendations) + ".",
    }


def build_soil_condition(parameter_analysis: Dict[str, Dict[str, Any]], warnings: list[str]) -> Dict[str, Any]:
    priority_items = [item for item in parameter_analysis.values() if item.get("priority") in {"medium", "high"}]
    priority_level = _max_priority(priority_items)

    if priority_level == "high":
        overall_status = "perlu_validasi"
    elif priority_level == "medium":
        overall_status = "perlu_perhatian"
    else:
        overall_status = "baik"

    issue_labels = []
    for key, item in parameter_analysis.items():
        if item.get("priority") in {"medium", "high"}:
            issue_labels.append(f"{key}: {item.get('status')}")

    main_issue = ", ".join(issue_labels[:4]) if issue_labels else "Tidak ada isu utama dari rule awal."
    short_summary = (
        f"Analisis rule engine menunjukkan status keseluruhan {overall_status}. "
        f"Isu utama: {main_issue}."
    )
    if warnings:
        short_summary += f" Catatan kualitas data: {'; '.join(warnings[:2])}."

    return {
        "overall_status": overall_status,
        "priority_level": priority_level,
        "main_issue": main_issue,
        "short_summary": short_summary,
    }


def build_recommendation(
    parameter_analysis: Dict[str, Dict[str, Any]],
    npk_analysis: Dict[str, Any],
    crop: str,
    growth_stage: str,
    humidity_type: str = "unknown",
) -> Dict[str, Any]:
    actions: list[Dict[str, str]] = []

    temp_status = parameter_analysis["temperature"]["status"]
    if temp_status == "perlu_validasi":
        actions.append({
            "action": "Validasi ulang sensor suhu",
            "reason": "Suhu terbaca sangat rendah sehingga dapat membuat rekomendasi agronomi bias.",
        })

    if parameter_analysis["ph"]["status"] in {"agak_basa", "basa_kuat"}:
        actions.append({
            "action": "Perhatikan pengelolaan pH tanah",
            "reason": "pH cenderung basa dapat memengaruhi ketersediaan beberapa unsur hara.",
        })
    elif parameter_analysis["ph"]["status"] in {"asam", "sangat_asam"}:
        actions.append({
            "action": "Evaluasi perbaikan pH tanah",
            "reason": "pH asam dapat memengaruhi ketersediaan hara dan perkembangan akar.",
        })

    if parameter_analysis["ec"]["status"] in {"tinggi", "sangat_tinggi"}:
        actions.append({
            "action": "Pantau EC dan hindari pemupukan berlebihan",
            "reason": "EC tinggi dapat menunjukkan akumulasi ion/garam terlarut.",
        })

    humidity_status = parameter_analysis["humidity"]["status"]
    if humidity_status in {"rendah", "sangat_lembap"}:
        actions.append({
            "action": "Cek kelembapan tanah dan drainase",
            "reason": (
                f"Soil moisture terbaca {humidity_status}. Cek kondisi tanah, genangan, drainase, "
                "dan kebutuhan pengairan di sekitar titik sensor sebelum tindakan lapang."
            ),
        })

    if parameter_analysis["nitrogen"]["status"] == "rendah":
        actions.append({
            "action": "Evaluasi kebutuhan nitrogen bertahap",
            "reason": "Nitrogen rendah dapat membatasi pertumbuhan vegetatif.",
        })
    elif parameter_analysis["nitrogen"]["status"] in {"tinggi", "sangat_tinggi"}:
        actions.append({
            "action": "Hindari tambahan nitrogen berlebihan",
            "reason": "Nitrogen tinggi dapat mendorong pertumbuhan vegetatif berlebihan dan ketidakseimbangan hara.",
        })

    if parameter_analysis["phosphorus"]["status"] in {"tinggi", "sangat_tinggi"}:
        actions.append({
            "action": "Batasi pupuk fosfor",
            "reason": "Fosfor sudah terbaca tinggi berdasarkan rule awal sensor.",
        })

    if parameter_analysis["potassium"]["status"] in {"tinggi", "sangat_tinggi"}:
        actions.append({
            "action": "Batasi pupuk kalium",
            "reason": "Kalium sudah terbaca tinggi berdasarkan rule awal sensor.",
        })

    if not actions:
        actions.append({
            "action": "Lakukan pemantauan berkala",
            "reason": "Tidak ada isu kritis dari rule awal, tetapi rekomendasi akhir tetap perlu mengikuti fase tanaman dan dokumen budidaya.",
        })

    priority = _max_priority([item for item in parameter_analysis.values()])
    main_recommendation = (
        f"Untuk tanaman {crop} fase {growth_stage}, gunakan hasil rule engine sebagai dasar awal: "
        f"{npk_analysis['recommendation']} Rekomendasi dosis spesifik sebaiknya diberikan hanya jika satuan sensor, luas lahan, varietas, umur tanaman, dan fase tanaman sudah terkonfirmasi."
    )

    return {
        "main_recommendation": main_recommendation,
        "priority": priority,
        "actions": actions,
    }


def build_risk_assessment(parameter_analysis: Dict[str, Dict[str, Any]], humidity_type: str = "unknown") -> Dict[str, Any]:
    risks: list[Dict[str, str]] = []

    if parameter_analysis["temperature"]["status"] == "perlu_validasi":
        risks.append({
            "risk": "Data suhu tidak wajar",
            "impact": "Rekomendasi dapat bias jika sensor belum stabil atau salah kalibrasi.",
        })
    if parameter_analysis["ph"]["status"] in {"agak_basa", "basa_kuat"}:
        risks.append({
            "risk": "pH cenderung basa",
            "impact": "Beberapa unsur hara mikro dapat kurang tersedia bagi tanaman.",
        })
    if parameter_analysis["ph"]["status"] in {"asam", "sangat_asam"}:
        risks.append({
            "risk": "pH cenderung asam",
            "impact": "Ketersediaan hara dan perkembangan akar dapat terganggu pada sebagian komoditas.",
        })
    if parameter_analysis["ec"]["status"] in {"tinggi", "sangat_tinggi"}:
        risks.append({
            "risk": "EC tinggi",
            "impact": "Potensi akumulasi garam/nutrisi terlarut perlu dipantau.",
        })
    if parameter_analysis["humidity"]["status"] in {"rendah", "sangat_lembap"}:
        risks.append({
            "risk": "Kelembapan tanah perlu perhatian",
            "impact": "Soil moisture yang terlalu rendah atau terlalu tinggi dapat mengganggu perakaran, efisiensi pemupukan, dan kondisi kesehatan tanaman.",
        })
    if parameter_analysis["nitrogen"]["status"] in {"tinggi", "sangat_tinggi"}:
        risks.append({
            "risk": "Nitrogen berlebih",
            "impact": "Tanaman dapat terlalu vegetatif dan pemupukan menjadi tidak efisien.",
        })
    if parameter_analysis["phosphorus"]["status"] in {"tinggi", "sangat_tinggi"}:
        risks.append({
            "risk": "Fosfor berlebih",
            "impact": "Penambahan P yang tidak perlu dapat mengganggu keseimbangan hara.",
        })
    if parameter_analysis["potassium"]["status"] in {"tinggi", "sangat_tinggi"}:
        risks.append({
            "risk": "Kalium berlebih",
            "impact": "Pemupukan K berlebihan dapat menimbulkan ketidakseimbangan hara.",
        })

    if not risks:
        risks.append({
            "risk": "Risiko rendah berdasarkan rule awal",
            "impact": "Tetap diperlukan pemantauan berkala dan validasi lapang.",
        })

    if any(item.get("priority") == "high" for item in parameter_analysis.values()):
        risk_level = "high"
    elif any(item.get("priority") == "medium" for item in parameter_analysis.values()):
        risk_level = "medium"
    else:
        risk_level = "low"

    return {"risk_level": risk_level, "risks": risks}


def build_rule_based_analysis(rag_request: Dict[str, Any], warnings: list[str]) -> Dict[str, Any]:
    sensor = rag_request["input_sensor"]
    crop = rag_request["crop_context"]["crop"]
    growth_stage = rag_request["crop_context"]["growth_stage"]
    humidity_type = "soil"

    parameter_analysis = build_parameter_analysis(sensor, crop, growth_stage, humidity_type=humidity_type)
    npk_analysis = build_npk_analysis(parameter_analysis)
    soil_condition = build_soil_condition(parameter_analysis, warnings)
    recommendation = build_recommendation(parameter_analysis, npk_analysis, crop, growth_stage, humidity_type=humidity_type)
    risk_assessment = build_risk_assessment(parameter_analysis, humidity_type=humidity_type)

    base_analysis = {
        "soil_condition": soil_condition,
        "parameter_analysis": parameter_analysis,
        "npk_analysis": npk_analysis,
        "recommendation": recommendation,
        "risk_assessment": risk_assessment,
    }
    base_analysis["agronomic_diagnosis"] = build_agronomic_diagnosis(rag_request, base_analysis)
    return base_analysis


# ==========================================================
# RAG request, retrieval query, prompt LLM
# ==========================================================


def build_soil_rag_request(
    device_id: str,
    payload: Dict[str, Any],
    warnings: list[str] | None = None,
) -> Dict[str, Any]:
    timestamp = payload.get("timestamp") or now_wib_iso()
    request_id = str(payload.get("request_id") or f"req-{int(time.time())}-{uuid4().hex[:8]}")
    message_id = str(payload.get("message_id") or f"msg-{int(time.time())}-{uuid4().hex[:8]}")

    sensor = normalize_sensor_payload(payload, topic_device_id=device_id)

    crop = normalize_crop_name(payload.get("crop") or payload.get("tanaman"), default=DEFAULT_CROP)
    growth_stage = str(payload.get("growth_stage") or payload.get("fase") or DEFAULT_GROWTH_STAGE).strip().lower()

    return {
        "version": "1.6",
        "request_id": request_id,
        "message_id": message_id,
        "type": "soil_rag_request",
        "timestamp": timestamp,
        "device": {
            "device_id": device_id,
            "device_type": str(payload.get("device_type") or "soil_sensor_8in1"),
            "source": "mqtt",
        },
        "location": {
            "latitude": sensor["latitude"],
            "longitude": sensor["longitude"],
            "region": str(payload.get("region") or "Indonesia"),
        },
        "crop_context": {
            "crop": crop,
            "growth_stage": growth_stage,
            "soil_type": payload.get("soil_type"),
            "planting_date": payload.get("planting_date"),
            "variety": payload.get("variety"),
            "area_m2": payload.get("area_m2"),
        },
        "input_sensor": {
            "temperature": sensor["temperature"],
            "humidity": sensor["humidity"],
            "ec": sensor["ec"],
            "ph": sensor["ph"],
            "nitrogen": sensor["nitrogen"],
            "phosphorus": sensor["phosphorus"],
            "potassium": sensor["potassium"],
            "fertility": sensor["fertility"],
            "latitude": sensor["latitude"],
            "longitude": sensor["longitude"],
        },
        "sensor_units": SENSOR_UNITS,
        "data_quality": {
            "status": "valid" if not warnings else "valid_with_warning",
            "is_calibrated": payload.get("is_calibrated", True),
            "missing_fields": [],
            "warnings": warnings or [],
        },
        "rag_query": {
            "language": RAG_LANGUAGE,
            "answer_style": RAG_ANSWER_STYLE,
        },
        "rag_options": {
            "top_k": settings.top_k,
            "min_score": settings.min_score,
            "max_answer_tokens": RAG_MAX_ANSWER_TOKENS,
        },
    }


def build_retrieval_query(rag_request: Dict[str, Any], rule_analysis: Dict[str, Any]) -> str:
    """Backward-compatible single query. Multi-query utama ada di build_retrieval_queries()."""
    return build_retrieval_queries(rag_request, rule_analysis)[0]


def build_retrieval_queries(rag_request: Dict[str, Any], rule_analysis: Dict[str, Any]) -> list[str]:
    crop = rag_request["crop_context"]["crop"]
    growth_stage = rag_request["crop_context"]["growth_stage"]
    pa = rule_analysis["parameter_analysis"]
    diagnosis = rule_analysis.get("agronomic_diagnosis", {})

    crop_terms = CROP_TERMS.get(crop, crop.replace("_", " "))
    stage_terms = STAGE_TERMS.get(growth_stage, growth_stage.replace("_", " "))
    stage_focus = " ".join(diagnosis.get("stage_focus", [])[:4])

    issue_terms: list[str] = []
    for issue in diagnosis.get("limiting_factors", [])[:4]:
        param = str(issue.get("parameter") or "")
        status = str(issue.get("status") or "")
        issue_terms.append(f"{param} {status}")
        issue_query = ISSUE_TERMS.get((param, status))
        if issue_query:
            issue_terms.append(issue_query)

    # Query pertama sengaja paling diagnostik: komoditas + fase + isu utama.
    # Ini membuat retrieval mengambil bagian dokumen yang relevan dengan masalah lapang,
    # bukan hanya halaman umum budidaya.
    queries = [
        f"{crop_terms} {stage_terms} {' '.join(issue_terms)} rekomendasi agronomi pemupukan berimbang soil fertility",
        f"{crop_terms} {stage_terms} {stage_focus} SOP petunjuk teknis manual teknologi budidaya pemupukan tanah",
        f"{crop_terms} {stage_terms} kebutuhan hara N P K pH EC kelembapan tanah drainase",
        f"{crop_terms} {stage_terms} faktor pembatas hara tanah serapan akar validasi uji tanah",
    ]

    for param in ["ph", "ec", "humidity", "nitrogen", "phosphorus", "potassium", "fertility"]:
        status = pa[param]["status"]
        issue_query = ISSUE_TERMS.get((param, status))
        if issue_query:
            queries.append(f"{crop_terms} {stage_terms} {issue_query} rekomendasi lapang")

    # Dedup dan batasi latency MQTT.
    deduped: list[str] = []
    for query in queries:
        query = re.sub(r"\s+", " ", query).strip()
        if query and query not in deduped:
            deduped.append(query)
    return deduped[:6]


def build_metadata_filters(rag_request: Dict[str, Any], rule_analysis: Dict[str, Any]) -> Dict[str, Any]:
    crop = rag_request["crop_context"]["crop"]
    growth_stage = rag_request["crop_context"]["growth_stage"]
    pa = rule_analysis["parameter_analysis"]
    diagnosis = rule_analysis.get("agronomic_diagnosis", {})

    topics = ["budidaya", "pemupukan", "NPK", "pH", "EC", "kelembapan tanah", "soil fertility", "uji tanah"]
    topics.extend(diagnosis.get("stage_focus", []))
    for issue in diagnosis.get("limiting_factors", [])[:5]:
        topics.append(str(issue.get("parameter", "")))
        topics.append(str(issue.get("status", "")))
        topics.append(str(issue.get("action_direction", "")))

    for key, item in pa.items():
        if item.get("priority") in {"medium", "high"}:
            topics.append(key)
            topics.append(str(item.get("status", "")))

    return {
        "crop": crop,
        "growth_stage": growth_stage,
        "topics": sorted(set(t for t in topics if str(t).strip())),
        "preferred_authority": ["A", "B"],
        "preferred_doc_types": ["sop_manual", "petunjuk_teknis", "manual_book", "modul_pelatihan", "ebook_manual_resmi"],
        "issue_parameters": [str(x.get("parameter")) for x in diagnosis.get("limiting_factors", [])[:5] if x.get("parameter")],
    }


def build_llm_question(rag_request: Dict[str, Any], rule_analysis: Dict[str, Any]) -> str:
    """
    LLM tidak diminta membuat seluruh response utama.
    Python/rule engine sudah membuat parameter_analysis, npk_analysis, risk, dan recommendation.
    LLM hanya membuat ringkasan berbasis dokumen dan wajib membahas isu medium/high.
    """
    pa = rule_analysis.get("parameter_analysis", {})
    required_issues: list[Dict[str, Any]] = []
    for key, item in pa.items():
        if item.get("priority") in {"medium", "high"}:
            required_issues.append({
                "parameter": key,
                "value": item.get("value"),
                "unit": item.get("unit"),
                "status": item.get("status"),
                "priority": item.get("priority"),
                "interpretation": item.get("interpretation"),
                "recommendation": item.get("recommendation"),
            })

    data_quality_warnings = rag_request.get("data_quality", {}).get("warnings", [])

    payload = {
        "tugas": "Buat penjelasan agronomi teknikal ringkas berbasis manual/SOP RAG dan hasil rule engine.",
        "aturan": [
            "Balas hanya JSON valid tanpa markdown.",
            "Jangan membuat sources; sources akan diisi server.",
            "Prioritaskan manual book/SOP/petunjuk teknis sebagai dasar rekomendasi praktis. Paper penelitian hanya sebagai evidence pendukung.",
            "Gunakan gaya teknikal untuk agronom/admin/penyuluh, tetapi tetap jelas dan tidak bertele-tele.",
            "Jangan mengubah angka sensor.",
            "Jangan mengubah status rule_engine; gunakan sebagai dasar.",
            "Jangan memberi dosis pupuk pasti jika luas lahan, satuan sensor, umur tanaman, varietas, dan fase tanaman belum lengkap.",
            "Wajib bahas semua isu pada wajib_bahas.isu_medium_high di human_readable_answer.",
            "Parameter humidity pada input_sensor adalah soil moisture / kelembapan tanah.",
            "Jangan menambahkan field tambahan untuk jenis kelembapan di output JSON.",
            "Boleh menyebut humidity sebagai kelembapan tanah atau soil moisture.",
            "Sebutkan parameter yang sudah baik/cukup secara ringkas agar jawaban tidak hanya fokus pada masalah.",
            "Jika konteks dokumen tidak cukup relevan, nyatakan keterbatasan referensi dan gunakan rule engine sebagai analisis awal.",
            "Utamakan diagnosis faktor pembatas, arah tindakan, dan monitoring; jangan membuat kalimat generik tanpa kaitan ke crop, fase, dan parameter bermasalah.",
            "Jika P/K/N sudah tinggi, jangan menyarankan tambahan pupuk unsur tersebut kecuali hanya sebagai validasi/monitoring.",
        ],
        "schema_output": {
            "rag_answer": {
                "human_readable_answer": "string ringkas 5-8 kalimat",
                "reference_based_notes": ["string"]
            },
            "recommendation_notes": ["string"]
        },
        "wajib_bahas": {
            "isu_medium_high": required_issues,
            "data_quality_warnings": data_quality_warnings,
            "soil_condition": rule_analysis.get("soil_condition", {}),
            "npk_analysis": rule_analysis.get("npk_analysis", {}),
            "agronomic_diagnosis": rule_analysis.get("agronomic_diagnosis", {}),
        },
        "request_summary": {
            "request_id": rag_request["request_id"],
            "crop_context": rag_request["crop_context"],
            "input_sensor": rag_request["input_sensor"],
            "data_quality": rag_request["data_quality"],
            "rule_engine_result": rule_analysis,
            "output_focus": "diagnosis agronomi, faktor pembatas, tindakan aman, monitoring, dan keterbatasan dosis",
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def soil_json_system_prompt() -> str:
    return """
Anda adalah asisten RAG agronomi.
Tugas Anda hanya memberi penjelasan teknikal berbasis dokumen RAG, terutama manual/SOP/petunjuk teknis, dan hasil rule engine.
Balas hanya JSON valid, tanpa markdown, tanpa ```json.
Jangan membuat field sources.
Jangan membuat ulang seluruh schema soil_rag_response.
Jangan mengubah angka sensor dan jangan mengubah klasifikasi rule engine.
Prioritaskan manual book/SOP/petunjuk teknis untuk rekomendasi praktis; paper penelitian hanya evidence pendukung.
Wajib membahas semua isu medium/high yang diberikan di wajib_bahas.isu_medium_high.
Parameter humidity adalah soil moisture / kelembapan tanah. Jangan menambahkan field jenis kelembapan di jawaban.
Sebutkan parameter yang sudah normal/cukup secara singkat agar jawaban seimbang.
Fokuskan jawaban pada diagnosis faktor pembatas, strategi hara, dan monitoring lapang.
Jangan menyarankan penambahan pupuk pada unsur yang statusnya tinggi/sangat_tinggi.
Jika konteks dokumen tidak cukup relevan, tulis keterbatasan referensi secara singkat.
Output wajib persis memiliki struktur:
{
  "rag_answer": {
    "human_readable_answer": "...",
    "reference_based_notes": ["..."]
  },
  "recommendation_notes": ["..."]
}
""".strip()



def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE)
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(cleaned[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def validate_llm_payload(obj: Dict[str, Any]) -> tuple[bool, str | None]:
    rag_answer = obj.get("rag_answer")
    if not isinstance(rag_answer, dict):
        return False, "Field rag_answer harus object."

    human_answer = rag_answer.get("human_readable_answer")
    if not isinstance(human_answer, str) or not human_answer.strip():
        return False, "Field rag_answer.human_readable_answer harus string non-kosong."

    reference_notes = rag_answer.get("reference_based_notes")
    if not isinstance(reference_notes, list):
        return False, "Field rag_answer.reference_based_notes harus list."

    for idx, item in enumerate(reference_notes):
        if not isinstance(item, str):
            return False, f"Item reference_based_notes[{idx}] harus string."

    recommendation_notes = obj.get("recommendation_notes")
    if not isinstance(recommendation_notes, list):
        return False, "Field recommendation_notes harus list."

    for idx, item in enumerate(recommendation_notes):
        if not isinstance(item, str):
            return False, f"Item recommendation_notes[{idx}] harus string."

    return True, None


def build_rule_based_human_answer(rag_request: Dict[str, Any], rule_analysis: Dict[str, Any]) -> str:
    crop = rag_request["crop_context"]["crop"]
    growth_stage = rag_request["crop_context"]["growth_stage"]
    pa = rule_analysis["parameter_analysis"]
    soil = rule_analysis["soil_condition"]
    npk = rule_analysis["npk_analysis"]
    diagnosis = rule_analysis.get("agronomic_diagnosis", {})
    limiting = diagnosis.get("limiting_factors", [])[:3]

    if limiting:
        limiting_text = "; ".join(
            f"{item.get('parameter')} {item.get('status')} ({item.get('value')} {item.get('unit')})"
            for item in limiting
        )
        action_text = "; ".join(diagnosis.get("immediate_actions", [])[:3])
    else:
        limiting_text = "tidak ada faktor pembatas medium/high dari rule engine"
        action_text = "lanjutkan monitoring berkala dan cocokkan data sensor dengan kondisi visual tanaman"

    normal_parts = []
    for key in ["temperature", "humidity", "ec", "ph", "nitrogen", "phosphorus", "potassium", "fertility"]:
        status = pa.get(key, {}).get("status")
        if _agronomic_severity(status) <= 1:
            normal_parts.append(f"{key} {status}")
    normal_text = ", ".join(normal_parts[:4]) if normal_parts else "belum ada parameter yang benar-benar bebas catatan"

    return (
        f"Analisis agronomi awal untuk tanaman {crop} fase {growth_stage} menunjukkan status keseluruhan {soil['overall_status']}. "
        f"Faktor pembatas utama adalah {limiting_text}. "
        f"Ringkasan NPK: N {pa['nitrogen']['status']}, P {pa['phosphorus']['status']}, K {pa['potassium']['status']}; {npk['recommendation']} "
        f"Parameter yang relatif aman/cukup: {normal_text}. "
        f"Arah tindakan aman: {action_text}. "
        "Dosis pupuk spesifik belum diberikan karena harus menunggu validasi satuan sensor, luas lahan, umur tanaman, varietas, riwayat pemupukan, dan/atau uji tanah."
    )


# ==========================================================
# Ringkasan petani: bahasa sederhana untuk UI
# ==========================================================


def _farmer_crop_label(crop: str) -> str:
    labels = {
        "padi": "Padi",
        "jagung": "Jagung",
        "cabai": "Cabai",
        "cabai_merah": "Cabai Merah",
        "cabai_rawit": "Cabai Rawit",
        "tomat": "Tomat",
        "bawang": "Bawang",
        "bawang_merah": "Bawang Merah",
        "kedelai": "Kedelai",
        "kentang": "Kentang",
        "terong": "Terong",
        "timun": "Timun/Mentimun",
    }
    return labels.get(crop, crop.replace("_", " ").title())


def _farmer_stage_label(stage: str) -> str:
    labels = {
        "awal_tanam": "Awal Tanam",
        "vegetatif": "Pertumbuhan Daun/Batang",
        "pembungaan": "Pembungaan",
        "pembuahan": "Pembentukan Buah/Bulir/Umbi",
        "pematangan": "Pematangan / Menjelang Panen",
    }
    return labels.get(stage, stage.replace("_", " ").title())


def build_ui_status(rule_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Status tunggal untuk UI agar frontend tidak perlu menebak dari banyak field."""
    soil = rule_analysis.get("soil_condition", {})
    risk = rule_analysis.get("risk_assessment", {})
    overall_status = str(soil.get("overall_status") or "baik")
    priority = str(soil.get("priority_level") or "low")
    risk_level = str(risk.get("risk_level") or priority)

    if priority == "high" or risk_level == "high" or overall_status == "perlu_validasi":
        code = "perlu_validasi"
        label = "Perlu validasi"
        severity = "high"
    elif priority == "medium" or risk_level == "medium" or overall_status == "perlu_perhatian":
        code = "perlu_perhatian"
        label = "Perlu perhatian"
        severity = "medium"
    else:
        code = "baik"
        label = "Baik"
        severity = "low"

    return {
        "code": code,
        "label": label,
        "severity": severity,
        "reason": soil.get("main_issue") or "Tidak ada isu utama dari rule awal.",
    }


def build_farmer_summary(rag_request: Dict[str, Any], rule_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Membuat ringkasan praktis yang konsisten dengan soil_condition dan ramah petani."""
    crop = rag_request["crop_context"]["crop"]
    stage = rag_request["crop_context"]["growth_stage"]
    pa = rule_analysis["parameter_analysis"]
    npk = rule_analysis["npk_analysis"]
    ui_status = build_ui_status(rule_analysis)

    ph_status = pa["ph"]["status"]
    ec_status = pa["ec"]["status"]
    humidity_status = pa["humidity"]["status"]
    n_status = npk["nitrogen_status"]
    p_status = npk["phosphorus_status"]
    k_status = npk["potassium_status"]

    problems: list[str] = []
    simple_actions: list[str] = []

    if humidity_status in {"sangat_lembap", "tinggi"}:
        problems.append("kelembapan tanah terlalu tinggi")
        simple_actions.append("Cek genangan, drainase, dan kondisi tanah di sekitar titik sensor.")
    elif humidity_status == "rendah":
        problems.append("kelembapan tanah rendah")
        simple_actions.append("Pantau kebutuhan air tanaman dan ulangi pembacaan sensor.")

    if ph_status in {"agak_basa", "basa_kuat"}:
        problems.append("pH tanah cenderung basa")
        simple_actions.append("Pantau ulang pH tanah sebelum pemupukan berikutnya.")
    elif ph_status in {"asam", "sangat_asam"}:
        problems.append("pH tanah asam")
        simple_actions.append("Pertimbangkan perbaikan pH sesuai arahan penyuluh atau hasil uji tanah.")

    if ec_status in {"tinggi", "sangat_tinggi"}:
        problems.append("EC tanah tinggi")
        simple_actions.append("Hindari pemupukan berlebihan dan cek drainase/kualitas air.")

    if n_status == "rendah":
        problems.append("nitrogen rendah")
        simple_actions.append("Evaluasi kebutuhan pupuk nitrogen secara bertahap sesuai kondisi tanaman.")
    elif n_status in {"tinggi", "sangat_tinggi"}:
        problems.append("nitrogen tinggi")
        simple_actions.append("Hindari tambahan pupuk nitrogen/urea untuk sementara kecuali ada dasar kebutuhan lapang.")
    elif n_status == "cukup":
        simple_actions.append("Gunakan pupuk nitrogen secukupnya, jangan berlebihan.")

    if p_status in {"tinggi", "sangat_tinggi"}:
        problems.append("fosfor sudah tinggi")
        simple_actions.append("Jangan tambah pupuk fosfor seperti SP-36/TSP untuk sementara.")
    elif p_status == "rendah":
        problems.append("fosfor rendah")
        simple_actions.append("Pertimbangkan pupuk fosfor sesuai fase tanaman dan anjuran setempat.")

    if k_status in {"tinggi", "sangat_tinggi"}:
        problems.append("kalium sudah tinggi")
        simple_actions.append("Kurangi atau hindari pupuk KCl berlebihan.")
    elif k_status == "rendah":
        problems.append("kalium rendah")
        simple_actions.append("Pertimbangkan pupuk kalium sesuai kebutuhan tanaman.")

    simple_actions.append("Cek ulang data tanah sebelum pemupukan berikutnya.")
    simple_actions.append("Validasi hasil sensor dengan kondisi lapang atau uji tanah bila tersedia.")
    simple_actions.append("Tambahkan kompos atau bahan organik bila tersedia.")

    unique_actions: list[str] = []
    for action in simple_actions:
        if action not in unique_actions:
            unique_actions.append(action)

    if ui_status["code"] == "baik":
        summary = f"Kondisi tanah untuk {_farmer_crop_label(crop)} fase {_farmer_stage_label(stage)} secara umum masih cukup baik. Tetap lakukan pemantauan berkala."
        main_advice = "Lanjutkan perawatan tanaman dan hindari pemupukan berlebihan."
    else:
        problem_text = ", ".join(problems[:5]) if problems else rule_analysis.get("soil_condition", {}).get("main_issue", "ada parameter yang perlu diperhatikan")
        summary = f"Untuk {_farmer_crop_label(crop)} fase {_farmer_stage_label(stage)}, kondisi perlu perhatian: {problem_text}."
        main_advice = "Tunda pemupukan tambahan yang berisiko berlebihan, ulangi pembacaan sensor, dan sesuaikan tindakan dengan kondisi lapang."

    return {
        "status": ui_status["label"],
        "ui_status": ui_status["code"],
        "severity": ui_status["severity"],
        "crop_label": _farmer_crop_label(crop),
        "growth_stage_label": _farmer_stage_label(stage),
        "summary": summary,
        "main_advice": main_advice,
        "simple_actions": unique_actions[:6],
        "farmer_note": "Rekomendasi ini adalah panduan awal. Untuk dosis pasti, sesuaikan dengan kondisi lapang dan arahan penyuluh.",
    }


# ==========================================================
# Technical analysis: jawaban teknikal berbasis manual/SOP
# ==========================================================


def _response_output_mode() -> str:
    """Mode output untuk UI.

    - farmer: hanya ringkasan petani + rag_answer
    - technical: jawaban teknikal + rag_answer
    - hybrid: farmer_summary + technical_analysis + rag_answer

    Default hybrid agar backend tetap cocok untuk UI petani dan dashboard teknis.
    """
    raw = str(_setting_value("rag_output_mode", "hybrid")).strip().lower()
    return raw if raw in {"farmer", "technical", "hybrid"} else "hybrid"


def _is_manual_source(source: Dict[str, Any]) -> bool:
    doc_type = str(source.get("doc_type") or "").strip().lower()
    source_type = str(source.get("source_type") or "").strip().lower()
    used_for = str(source.get("used_for") or "").strip().lower()
    document = str(source.get("document") or "").strip().lower()

    if source_type == "ebook_manual_resmi":
        return True
    if doc_type in {"sop_manual", "petunjuk_teknis", "manual_book", "modul_pelatihan", "ebook_manual_resmi"}:
        return True
    if "manual/sop" in used_for:
        return True
    return any(keyword in document for keyword in MANUAL_SOURCE_KEYWORDS)


def _source_summary_for_technical(source: Dict[str, Any]) -> Dict[str, Any]:
    metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
    return {
        "source_id": source.get("source_id"),
        "document": source.get("document"),
        "document_title": metadata.get("document_title") or source.get("document"),
        "doc_type": source.get("doc_type") or metadata.get("doc_type"),
        "source_type": source.get("source_type") or metadata.get("source_type"),
        "authority": source.get("authority") or metadata.get("authority"),
        "chunk_id": source.get("chunk_id"),
        "score": source.get("score"),
        "rerank_score": source.get("rerank_score"),
        "used_for": source.get("used_for"),
    }


def _parameter_technical_interpretation(
    name: str,
    item: Dict[str, Any],
    rag_request: Dict[str, Any],
) -> str:
    value = item.get("value")
    unit = item.get("unit")
    status = item.get("status")
    priority = item.get("priority")
    label = {
        "temperature": "Suhu",
        "humidity": "Kelembapan",
        "ec": "EC",
        "ph": "pH",
        "nitrogen": "Nitrogen",
        "phosphorus": "Fosfor",
        "potassium": "Kalium",
        "fertility": "Indeks kesuburan",
    }.get(name, name)

    if name == "humidity":
        return (
            f"Kelembapan tanah (soil moisture) {value} {unit} berstatus {status} dengan prioritas {priority}. "
            "Gunakan nilai ini bersama observasi genangan, drainase, dan kondisi tanah di sekitar titik sensor."
        )

    return f"{label} {value} {unit} berstatus {status} dengan prioritas {priority}."


def _technical_validation_requirements(rag_request: Dict[str, Any], rule_analysis: Dict[str, Any]) -> list[str]:
    crop_context = rag_request.get("crop_context", {})
    sensor = rag_request.get("input_sensor", {})
    requirements: list[str] = []

    if not crop_context.get("area_m2"):
        requirements.append("Tambahkan area_m2 bila sistem akan menghitung kebutuhan pupuk berbasis luas.")
    if not crop_context.get("variety"):
        requirements.append("Tambahkan variety/varietas agar interpretasi fase dan vigor tanaman lebih presisi.")
    if not crop_context.get("planting_date"):
        requirements.append("Tambahkan planting_date atau umur tanaman untuk menyesuaikan rekomendasi dengan fase aktual.")
    if not rag_request.get("data_quality", {}).get("is_calibrated", True):
        requirements.append("Kalibrasi sensor sebelum memakai rekomendasi sebagai dasar keputusan lapang.")

    pa = rule_analysis.get("parameter_analysis", {})
    if any(item.get("priority") in {"medium", "high"} for item in pa.values()):
        requirements.append("Validasi parameter prioritas medium/high dengan observasi lapang atau uji tanah bila tersedia.")

    unique: list[str] = []
    for item in requirements:
        if item not in unique:
            unique.append(item)
    return unique



def _unique_strings(items: Iterable[str]) -> list[str]:
    unique: list[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in unique:
            unique.append(text)
    return unique


def _action_specific_validation_requirements(
    action: Dict[str, Any],
    rag_request: Dict[str, Any],
    rule_analysis: Dict[str, Any],
) -> list[str]:
    """Validasi teknis yang spesifik untuk setiap action.

    Versi sebelumnya mengulang semua validation_requirements pada tiap action.
    Fungsi ini membuat daftar validasi lebih presisi berdasarkan jenis action:
    humidity, nitrogen, fosfor, kalium, pH, EC, suhu, atau fallback umum.
    """
    sensor = rag_request.get("input_sensor", {})
    crop_context = rag_request.get("crop_context", {})
    data_quality = rag_request.get("data_quality", {})
    action_text = f"{action.get('action', '')} {action.get('reason', '')}".lower()
    requirements: list[str] = []

    def add_common_sensor_quality() -> None:
        if data_quality.get("is_calibrated", True) is False:
            requirements.append("Kalibrasi sensor sebelum action dipakai sebagai dasar keputusan lapang.")
        requirements.append("Ulangi pembacaan sensor pada titik yang sama untuk memastikan nilai stabil.")

    if any(term in action_text for term in ["humidity", "kelembapan", "pengairan", "drainase", "sirkulasi"]):
        requirements.extend([
            "Cek genangan, drainase petakan, dan kondisi tanah di sekitar titik sensor.",
            "Bandingkan pembacaan soil moisture dengan observasi visual kelembapan tanah.",
            "Pastikan titik sensor mewakili zona perakaran tanaman dan tidak berada di area anomali.",
            "Jangan membuat keputusan pengairan/drainase hanya dari satu pembacaan; cek tren atau ulangi pengukuran.",
        ])
        add_common_sensor_quality()
        return _unique_strings(requirements)

    if any(term in action_text for term in ["nitrogen", "urea", " pupuk n", " n/"]):
        requirements.extend([
            "Cek riwayat pemupukan N/urea terakhir, termasuk dosis dan tanggal aplikasi.",
            "Amati warna daun, vigor tanaman, dan jumlah anakan/pertumbuhan vegetatif.",
            "Konfirmasi umur tanaman atau planting_date agar fase vegetatif sesuai kondisi aktual.",
            "Validasi satuan N sensor 8-in-1 atau bandingkan dengan uji tanah/alat bantu lapang bila tersedia.",
        ])
        if not crop_context.get("area_m2"):
            requirements.append("Tambahkan area_m2 sebelum menghitung kebutuhan pupuk berbasis luas.")
        if not crop_context.get("variety"):
            requirements.append("Tambahkan variety/varietas untuk menilai vigor dan respons pemupukan lebih presisi.")
        add_common_sensor_quality()
        return _unique_strings(requirements)

    if any(term in action_text for term in ["fosfor", "phosphorus", "sp-36", "tsp", "pupuk p"]):
        requirements.extend([
            "Cek riwayat aplikasi pupuk P seperti SP-36/TSP dan tanggal aplikasinya.",
            "Konfirmasi pH tanah karena ketersediaan fosfor sangat dipengaruhi pH.",
            "Validasi satuan P sensor atau bandingkan dengan uji tanah bila tersedia.",
        ])
        if not crop_context.get("area_m2"):
            requirements.append("Tambahkan area_m2 sebelum menghitung kebutuhan pupuk P berbasis luas.")
        add_common_sensor_quality()
        return _unique_strings(requirements)

    if any(term in action_text for term in ["kalium", "potassium", "kcl", "pupuk k"]):
        requirements.extend([
            "Cek riwayat aplikasi pupuk K/KCl dan tanggal aplikasinya.",
            "Amati gejala ketidakseimbangan hara pada daun dan pertumbuhan tanaman.",
            "Validasi satuan K sensor atau bandingkan dengan uji tanah bila tersedia.",
        ])
        if not crop_context.get("area_m2"):
            requirements.append("Tambahkan area_m2 sebelum menghitung kebutuhan pupuk K berbasis luas.")
        add_common_sensor_quality()
        return _unique_strings(requirements)

    if "ph" in action_text or "pengapuran" in action_text or "dolomit" in action_text:
        requirements.extend([
            "Ulangi pengukuran pH pada beberapa titik lahan untuk memastikan nilai representatif.",
            "Cek riwayat pengapuran/dolomit dan bahan amelioran lain.",
            "Gunakan hasil uji tanah lokal sebelum tindakan koreksi pH skala besar.",
        ])
        add_common_sensor_quality()
        return _unique_strings(requirements)

    if "ec" in action_text or "salinitas" in action_text or "garam" in action_text:
        requirements.extend([
            "Ulangi pembacaan EC setelah pemupukan atau pengairan agar nilai lebih stabil.",
            "Cek kualitas air irigasi dan kondisi drainase lahan.",
            "Bandingkan EC dengan gejala stres garam atau pertumbuhan tanaman di lapang.",
        ])
        add_common_sensor_quality()
        return _unique_strings(requirements)

    if "suhu" in action_text or "temperature" in action_text:
        requirements.extend([
            "Pastikan sensor suhu sudah stabil dan tidak terkena panas langsung yang tidak representatif.",
            "Ulangi pembacaan pada waktu berbeda untuk membedakan anomali sensor dan kondisi mikroklimat.",
        ])
        add_common_sensor_quality()
        return _unique_strings(requirements)

    # Fallback: tetap spesifik pada kualitas data dan konteks umum, tidak mengulang seluruh daftar global.
    requirements.extend([
        "Validasi action dengan observasi lapang pada parameter yang diprioritaskan rule engine.",
        "Lengkapi data konteks yang langsung memengaruhi action sebelum membuat keputusan operasional.",
    ])
    add_common_sensor_quality()
    return _unique_strings(requirements)


def build_technical_analysis(
    rag_request: Dict[str, Any],
    rule_analysis: Dict[str, Any],
    sources: list[Dict[str, Any]],
) -> Dict[str, Any]:
    """Membuat jawaban teknikal deterministik berbasis rule engine + manual/SOP.

    Field ini ditujukan untuk dashboard agronom/admin/penyuluh. LLM tidak
    menentukan status sensor di sini; LLM hanya boleh mengisi rag_answer.
    """
    crop = rag_request["crop_context"]["crop"]
    stage = rag_request["crop_context"]["growth_stage"]
    sensor = rag_request["input_sensor"]
    pa = rule_analysis["parameter_analysis"]
    ui_status = build_ui_status(rule_analysis)
    agronomic_diagnosis = rule_analysis.get("agronomic_diagnosis", {})

    manual_sources = [src for src in sources if _is_manual_source(src)]
    research_sources = [src for src in sources if not _is_manual_source(src)]
    primary_sources = manual_sources if manual_sources else sources

    sensor_interpretation = {
        key: {
            "value": item.get("value"),
            "unit": item.get("unit"),
            "status": item.get("status"),
            "priority": item.get("priority"),
            "technical_summary": _parameter_technical_interpretation(key, item, rag_request),
            "rule_interpretation": item.get("interpretation"),
            "rule_recommendation": item.get("recommendation"),
        }
        for key, item in pa.items()
    }

    manual_guidance: list[str] = []
    if manual_sources:
        manual_guidance.append("Sumber manual/SOP tersedia dan diprioritaskan sebagai dasar rekomendasi teknis praktis.")
    else:
        manual_guidance.append("Sumber manual/SOP belum ditemukan pada hasil retrieval; rekomendasi teknis memakai rule engine dan sumber pendukung yang tersedia.")

    manual_guidance.extend([
        "Gunakan manual budidaya sebagai rujukan SOP untuk urutan tindakan lapang, bukan untuk menebak ulang angka sensor.",
        "Dosis pupuk spesifik belum dihitung karena membutuhkan luas lahan, umur tanaman, varietas, riwayat pemupukan, dan validasi satuan sensor.",
        "Paper penelitian dipakai sebagai evidence pendukung; manual/SOP tetap menjadi prioritas untuk rekomendasi praktis.",
    ])

    technical_recommendations: list[Dict[str, Any]] = []
    for action in rule_analysis.get("recommendation", {}).get("actions", []):
        technical_recommendations.append({
            "priority": rule_analysis.get("recommendation", {}).get("priority", ui_status["severity"]),
            "action": action.get("action"),
            "reason": action.get("reason"),
            "basis": "rule_engine + manual_book_priority + agronomic_diagnosis",
            "agronomic_context": agronomic_diagnosis.get("diagnosis_summary"),
            "requires_validation": _action_specific_validation_requirements(action, rag_request, rule_analysis),
        })

    if not technical_recommendations:
        technical_recommendations.append({
            "priority": "low",
            "action": "Lakukan pemantauan berkala",
            "reason": "Tidak ada isu prioritas medium/high dari rule engine.",
            "basis": "rule_engine",
            "requires_validation": _technical_validation_requirements(rag_request, rule_analysis),
        })

    limitations = [
        "Threshold NPK, EC, pH, dan fertility masih berbasis rule internal awal dan perlu kalibrasi dengan sensor/vendor serta uji tanah lokal.",
        "Technical_analysis tidak menggantikan rekomendasi resmi penyuluh atau hasil laboratorium tanah.",
    ]

    return {
        "basis": "manual_book_priority",
        "answer_style": "technical",
        "crop_stage_context": f"{_farmer_crop_label(crop)} fase {_farmer_stage_label(stage)}",
        "ui_status": ui_status,
        "source_policy": {
            "primary": "manual/SOP/petunjuk teknis resmi",
            "secondary": "paper penelitian sebagai evidence pendukung",
            "decision_owner": "rule_engine deterministik; LLM hanya merangkum",
        },
        "evidence_mix": {
            "manual_sources": len(manual_sources),
            "research_sources": len(research_sources),
            "total_sources": len(sources),
            "manual_priority_active": bool(manual_sources),
        },
        "primary_manual_sources": [_source_summary_for_technical(src) for src in primary_sources[:3]],
        "sensor_interpretation": sensor_interpretation,
        "agronomic_diagnosis": agronomic_diagnosis,
        "npk_analysis": rule_analysis.get("npk_analysis", {}),
        "manual_book_guidance": manual_guidance,
        "technical_recommendations": technical_recommendations,
        "risk_assessment": rule_analysis.get("risk_assessment", {}),
        "validation_requirements": _technical_validation_requirements(rag_request, rule_analysis),
        "limitations": limitations,
    }


# ==========================================================
# Retrieval helpers: multi-query, metadata filter, rerank
# ==========================================================


def _item_to_dict(item: Any) -> Dict[str, Any]:
    if isinstance(item, dict):
        return dict(item)

    result: Dict[str, Any] = {}
    if hasattr(item, "__dict__"):
        try:
            result.update(dict(item.__dict__))
        except Exception:
            pass

    # Fallback untuk object ringan/proxy yang menyimpan value sebagai property/class attr.
    for attr in ["source", "document", "file", "path", "chunk_id", "id", "score", "text", "content", "metadata", "title", "document_title"]:
        if attr not in result and hasattr(item, attr):
            try:
                result[attr] = getattr(item, attr)
            except Exception:
                pass
    return result


def _get_score(item: Any) -> float:
    data = _item_to_dict(item)
    try:
        return float(data.get("score", 0.0))
    except Exception:
        return 0.0


def _infer_metadata_from_source(data: Dict[str, Any]) -> Dict[str, Any]:
    source_text = _stringify([
        data.get("source"),
        data.get("document"),
        data.get("file"),
        data.get("path"),
        data.get("title"),
        data.get("document_title"),
    ]).lower()
    source_norm = source_text.replace("\\", "/").replace("-", "_").replace(" ", "_")

    inferred: Dict[str, Any] = {}

    for crop, keywords in CROP_SOURCE_KEYWORDS.items():
        if any(keyword in source_norm for keyword in keywords):
            inferred["crop"] = crop
            break

    is_manual = any(keyword in source_norm for keyword in MANUAL_SOURCE_KEYWORDS)
    is_research = any(keyword in source_norm for keyword in RESEARCH_SOURCE_KEYWORDS)

    if is_manual:
        if "sop" in source_norm or "standard_operational_procedure" in source_norm:
            doc_type = "sop_manual"
        elif "petunjuk" in source_norm or "teknis" in source_norm or "juknis" in source_norm:
            doc_type = "petunjuk_teknis"
        elif "modul" in source_norm:
            doc_type = "modul_pelatihan"
        else:
            doc_type = "manual_book"
        inferred["doc_type"] = doc_type
        inferred["source_type"] = "ebook_manual_resmi"
        inferred["authority"] = "A"
    elif is_research:
        inferred["doc_type"] = "paper_riset_asli"
        inferred["source_type"] = "paper_riset_asli"
        inferred["authority"] = "B"
    else:
        inferred["doc_type"] = "unknown"
        inferred["source_type"] = "unknown"
        inferred["authority"] = "C"

    # Ambil nama file sebagai fallback title.
    raw_source = str(data.get("source") or data.get("document") or "")
    if raw_source:
        filename = raw_source.replace("\\", "/").rsplit("/", 1)[-1]
        inferred["document_title"] = re.sub(r"\.[A-Za-z0-9]+$", "", filename).replace("_", " ").strip()

    inferred.setdefault("topics", [])
    return inferred


def _get_metadata(item: Any) -> Dict[str, Any]:
    data = _item_to_dict(item)
    metadata = data.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    metadata = dict(metadata)

    # Banyak retriever menyimpan metadata sebagai field datar.
    for key in [
        "crop",
        "komoditas",
        "growth_stage",
        "fase",
        "topic",
        "topics",
        "source_type",
        "doc_type",
        "jenis_dokumen",
        "authority",
        "tingkat_kepercayaan",
        "year",
        "document_title",
        "publisher",
        "url",
    ]:
        if key in data and key not in metadata:
            metadata[key] = data[key]

    inferred = _infer_metadata_from_source(data)
    for key, value in inferred.items():
        if key not in metadata or metadata.get(key) in (None, "", [], {}):
            metadata[key] = value

    if "crop" not in metadata and "komoditas" in metadata:
        metadata["crop"] = metadata["komoditas"]
    if "growth_stage" not in metadata and "fase" in metadata:
        metadata["growth_stage"] = metadata["fase"]
    if "doc_type" not in metadata and "jenis_dokumen" in metadata:
        metadata["doc_type"] = metadata["jenis_dokumen"]
    if "authority" not in metadata and "tingkat_kepercayaan" in metadata:
        metadata["authority"] = metadata["tingkat_kepercayaan"]

    return metadata


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join(_stringify(v) for v in value)
    if isinstance(value, dict):
        return " ".join(f"{k} {_stringify(v)}" for k, v in value.items())
    return str(value)


def _candidate_text(item: Any) -> str:
    data = _item_to_dict(item)
    meta = _get_metadata(item)
    parts = [
        data.get("source"),
        data.get("document"),
        data.get("chunk_id"),
        data.get("text"),
        data.get("content"),
        meta,
    ]
    return _stringify(parts).lower()


def _metadata_matches(item: Any, metadata_filters: Dict[str, Any]) -> bool:
    """Soft filter.

    Jika metadata crop tidak tersedia, item tidak dibuang agar kompatibel dengan index lama.
    Jika metadata crop tersedia dan berbeda, item dibuang.
    """
    metadata = _get_metadata(item)
    crop_filter = str(metadata_filters.get("crop") or "").lower()
    stage_filter = str(metadata_filters.get("growth_stage") or "").lower()

    crop_meta = _stringify(metadata.get("crop") or metadata.get("komoditas")).lower()
    crop_general = crop_meta in {"all", "general", "umum", "semua", "multi", "unknown"}
    if crop_meta and crop_filter and not crop_general and crop_filter not in crop_meta and crop_meta not in crop_filter:
        return False

    stage_meta = _stringify(metadata.get("growth_stage") or metadata.get("fase")).lower()
    stage_general = stage_meta in {"all", "general", "umum", "semua", "multi", "unknown"}
    if stage_meta and stage_filter and not stage_general and stage_filter not in stage_meta and stage_meta not in stage_filter:
        return False

    return True


def _computed_rerank_score(item: Any, metadata_filters: Dict[str, Any]) -> float:
    score = _get_score(item)
    text = _candidate_text(item)
    metadata = _get_metadata(item)

    crop = str(metadata_filters.get("crop") or "").lower()
    growth_stage = str(metadata_filters.get("growth_stage") or "").lower()
    topics = [str(t).lower() for t in metadata_filters.get("topics", [])]
    preferred_doc_types = {str(t).lower() for t in metadata_filters.get("preferred_doc_types", [])}
    issue_parameters = {str(t).lower() for t in metadata_filters.get("issue_parameters", [])}

    metadata_crop = _stringify(metadata.get("crop") or metadata.get("komoditas")).lower()
    if crop and metadata_crop:
        if crop == metadata_crop or crop in metadata_crop:
            score += 0.18
        else:
            score -= 0.25
    elif crop and (crop.replace("_", " ") in text or crop in text):
        score += 0.10

    metadata_stage = _stringify(metadata.get("growth_stage") or metadata.get("fase")).lower()
    if growth_stage and metadata_stage:
        if growth_stage == metadata_stage or growth_stage in metadata_stage:
            score += 0.06
    elif growth_stage and (growth_stage.replace("_", " ") in text or growth_stage in text):
        score += 0.03

    topic_hits = sum(1 for t in topics if t and t.lower() in text)
    issue_hits = sum(1 for t in issue_parameters if t and t.lower() in text)
    score += min(topic_hits * 0.015, 0.09)
    score += min(issue_hits * 0.03, 0.09)

    authority = str(metadata.get("authority") or metadata.get("tingkat_kepercayaan") or "").upper()
    doc_type = str(metadata.get("doc_type") or metadata.get("jenis_dokumen") or "unknown").lower()
    source_type = str(metadata.get("source_type") or "unknown").lower()

    if authority == "A":
        score += 0.08
    elif authority == "B":
        score += 0.04

    # Prioritas practical-answer: manual/SOP/juknis lebih tinggi dari paper.
    score += PRACTICAL_DOC_TYPE_WEIGHT.get(doc_type, 0.0)
    if preferred_doc_types and doc_type in preferred_doc_types:
        score += 0.06
    if source_type in {"ebook_manual_resmi", "manual", "sop", "petunjuk_teknis"}:
        score += 0.08
    elif source_type in {"paper", "paper_riset_asli", "jurnal", "journal"}:
        score += 0.025

    return score


def dedupe_retrieved(results: list[Any]) -> list[Any]:
    seen: set[str] = set()
    unique: list[Any] = []

    for item in results:
        data = _item_to_dict(item)
        key = str(data.get("chunk_id") or data.get("id") or f"{data.get('source')}::{data.get('text', '')[:80]}")
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)

    return unique


def rerank_retrieved(results: list[Any], metadata_filters: Dict[str, Any]) -> list[Any]:
    filtered = [item for item in results if _metadata_matches(item, metadata_filters)]
    if not filtered:
        filtered = results
    return sorted(filtered, key=lambda item: _computed_rerank_score(item, metadata_filters), reverse=True)


def _retriever_accepts_kwarg(retriever: Any, kwarg_name: str) -> bool:
    try:
        signature = inspect.signature(retriever.search)
    except Exception:
        return False
    return kwarg_name in signature.parameters


def _search_retriever_once(retriever: Any, query: str, top_k: int, metadata_filters: Dict[str, Any]) -> list[Any]:
    """Cari dengan filter bila retriever mendukung, fallback bila tidak."""
    try:
        if _retriever_accepts_kwarg(retriever, "filters"):
            return list(retriever.search(query, top_k=top_k, filters=metadata_filters))
        if _retriever_accepts_kwarg(retriever, "metadata_filters"):
            return list(retriever.search(query, top_k=top_k, metadata_filters=metadata_filters))
        if _retriever_accepts_kwarg(retriever, "filter"):
            return list(retriever.search(query, top_k=top_k, filter=metadata_filters))
    except TypeError:
        pass
    except Exception:
        logger.exception("Retriever search with metadata filter failed. Falling back to plain search.")

    return list(retriever.search(query, top_k=top_k))


def retrieve_with_multi_query(service: RAGService, rag_request: Dict[str, Any], rule_analysis: Dict[str, Any]) -> tuple[list[Any], list[str], Dict[str, Any]]:
    if service.retriever is None:
        raise RuntimeError("Retriever belum siap. Jalankan build_index.py atau pastikan PDF tersedia di data/pdfs.")

    top_k = int(_setting_value("top_k", 5))
    queries = build_retrieval_queries(rag_request, rule_analysis)
    metadata_filters = build_metadata_filters(rag_request, rule_analysis)

    all_results: list[Any] = []
    per_query_top_k = max(top_k, 4)
    for query in queries:
        try:
            all_results.extend(_search_retriever_once(service.retriever, query, top_k=per_query_top_k, metadata_filters=metadata_filters))
        except Exception:
            logger.exception("Retriever search failed for query=%s", query)

    deduped = dedupe_retrieved(all_results)
    reranked = rerank_retrieved(deduped, metadata_filters)
    return reranked[:top_k], queries, metadata_filters


def _normalize_sources_from_items(items: list[Any], metadata_filters: Optional[Dict[str, Any]] = None) -> list[Dict[str, Any]]:
    normalized = []
    metadata_filters = metadata_filters or {}
    for idx, item in enumerate(items, start=1):
        data = _item_to_dict(item)
        metadata = _get_metadata(item)
        doc_type = str(metadata.get("doc_type") or metadata.get("jenis_dokumen") or "unknown")
        source_type = str(metadata.get("source_type") or "unknown")
        used_for = "konteks RAG"
        if source_type == "ebook_manual_resmi" or doc_type in {"sop_manual", "petunjuk_teknis", "manual_book", "modul_pelatihan"}:
            used_for = "manual/SOP utama untuk rekomendasi praktis"
        elif source_type == "paper_riset_asli" or doc_type in {"paper_riset_asli", "paper", "jurnal"}:
            used_for = "evidence pendukung dari paper penelitian"

        normalized.append(
            {
                "source_id": idx,
                "document": data.get("source") or data.get("document") or metadata.get("document_title") or "",
                "chunk_id": data.get("chunk_id") or data.get("id"),
                "score": round(_get_score(item), 4),
                "rerank_score": round(_computed_rerank_score(item, metadata_filters), 4),
                "metadata": metadata,
                "source_type": source_type,
                "doc_type": doc_type,
                "authority": metadata.get("authority"),
                "used_for": used_for,
            }
        )
    return normalized


def _normalize_sources(sources: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    """Backward-compatible normalizer untuk caller lama."""
    normalized = []
    for idx, item in enumerate(sources, start=1):
        normalized.append(
            {
                "source_id": idx,
                "document": item.get("source", ""),
                "chunk_id": item.get("chunk_id"),
                "score": round(float(item.get("score", 0.0)), 4),
                "metadata": item.get("metadata", {}),
                "used_for": "konteks RAG",
            }
        )
    return normalized


# ==========================================================
# LLM debounce and confidence
# ==========================================================


def _analysis_fingerprint(rule_analysis: Dict[str, Any]) -> str:
    """Fingerprint status rule engine untuk debounce LLM.

    Versi ini tidak hanya memakai status/prioritas, tetapi juga severity_score.
    Alasannya: perubahan P/K/N dari "tinggi" ke "sangat ekstrem" bisa tetap berada
    dalam label status yang sama, padahal secara agronomi perlu ringkasan baru.
    """
    pa = rule_analysis.get("parameter_analysis", {})
    status_pairs = {}
    for key, value in pa.items():
        status = value.get("status")
        numeric_value = value.get("value")
        # bucket kasar agar perubahan kecil sensor tidak memicu LLM terus-menerus,
        # tetapi perubahan besar tetap terdeteksi.
        try:
            value_bucket = round(float(numeric_value) / 10.0) * 10
        except Exception:
            value_bucket = None
        status_pairs[key] = {
            "status": status,
            "priority": value.get("priority"),
            "severity": _agronomic_severity(status),
            "value_bucket": value_bucket,
        }
    return json.dumps(status_pairs, sort_keys=True, ensure_ascii=False)


def _rounded_sensor_value(value: Any, digits: int = 2) -> Any:
    try:
        if value is None or value == "":
            return None
        number = float(value)
        if not math.isfinite(number):
            return None
        return round(number, digits)
    except Exception:
        return value


def _sensor_value_fingerprint(rag_request: Dict[str, Any]) -> str:
    """Fingerprint angka sensor aktual untuk memastikan cache tidak membawa angka lama.

    Jika status rule engine sama tetapi angka sensor berubah, cache RAG tetap boleh
    dipakai untuk sources/reference notes, tetapi human_readable_answer harus dibuat
    ulang dari rule engine terbaru.
    """
    sensor = rag_request.get("input_sensor") or {}
    context = rag_request.get("crop_context") or {}
    payload = {
        "crop": context.get("crop"),
        "growth_stage": context.get("growth_stage"),
        "sensor": {key: _rounded_sensor_value(sensor.get(key), digits=2) for key in SENSOR_CACHE_KEYS},
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _rag_cache_key(device_id: str, rule_analysis: Dict[str, Any]) -> str:
    # Key tetap status-based agar cache bisa ditemukan ketika angka sensor berubah kecil.
    # Saat cache diterapkan, _sensor_value_fingerprint menentukan apakah full answer aman dipakai.
    return f"{device_id}:{_analysis_fingerprint(rule_analysis)}"


async def get_cached_rag_bundle(device_id: str, rule_analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    key = _rag_cache_key(device_id, rule_analysis)
    async with _rag_answer_cache_lock:
        cached = _rag_answer_cache.get(key)
        return dict(cached) if isinstance(cached, dict) else None


async def set_cached_rag_bundle(
    device_id: str,
    rule_analysis: Dict[str, Any],
    *,
    rag_request: Dict[str, Any],
    llm_payload: Optional[Dict[str, Any]],
    raw_answer: str,
    sources: list[Dict[str, Any]],
    retrieval_queries: list[str],
    metadata_filters: Dict[str, Any],
    rag_relevance_ok: bool,
    top_score: float | None,
) -> None:
    """Simpan hasil RAG/LLM terakhir untuk status rule engine yang sama.

    Cache menyimpan sensor_fingerprint. Jika telemetry berikutnya punya status sama
    tetapi angka sensor berbeda, sistem tidak akan memakai cached human answer.
    Yang dipakai hanya notes/sources agar tidak muncul angka sensor lama di UI.
    """
    if not llm_payload:
        return

    key = _rag_cache_key(device_id, rule_analysis)
    async with _rag_answer_cache_lock:
        _rag_answer_cache[key] = {
            "cached_at": now_wib_iso(),
            "sensor_fingerprint": _sensor_value_fingerprint(rag_request),
            "llm_payload": llm_payload,
            "raw_answer": raw_answer or "",
            "sources": sources or [],
            "retrieval_queries": retrieval_queries or [],
            "metadata_filters": metadata_filters or {},
            "rag_relevance_ok": bool(rag_relevance_ok),
            "top_score": top_score,
        }

        # Batasi cache agar tidak membesar bila banyak device/status.
        max_cache = int(_setting_value("rag_answer_cache_max_items", 500))
        if len(_rag_answer_cache) > max_cache:
            oldest_key = next(iter(_rag_answer_cache.keys()))
            _rag_answer_cache.pop(oldest_key, None)


def _list_of_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _filter_notes_for_current_rule(notes: list[str], rule_analysis: Dict[str, Any]) -> list[str]:
    """Hilangkan notes cache yang bertentangan dengan status rule terbaru.

    Contoh: pH saat ini agak basa, maka catatan/rekomendasi pengapuran dari
    konteks pH asam tidak boleh tampil sebagai arahan utama.
    """
    pa = rule_analysis.get("parameter_analysis", {})
    ph_status = str((pa.get("ph") or {}).get("status") or "")
    filtered: list[str] = []

    high_ph = ph_status in {"agak_basa", "basa_kuat"}
    for note in notes:
        text = str(note).strip()
        if not text:
            continue
        low = text.lower()
        if high_ph and any(term in low for term in ["pengapuran", "kapur", "dolomit", "menaikkan ph", "menaikkan pH".lower()]):
            # Skip notes yang relevan untuk pH asam, karena sensor sekarang pH tinggi/basa.
            continue
        filtered.append(text)

    if high_ph:
        reminder = (
            "Karena pH saat ini berada di atas rentang target awal, hindari pengapuran/dolomit "
            "atau perlakuan lain yang dapat menaikkan pH sebelum validasi ulang."
        )
        if reminder not in filtered:
            filtered.append(reminder)

    # Hindari notes terlalu panjang untuk Kodular.
    return filtered[:8]


def _safe_cached_llm_payload(
    cached_llm_payload: Optional[Dict[str, Any]],
    rag_request: Dict[str, Any],
    rule_analysis: Dict[str, Any],
    *,
    sensor_same: bool,
) -> Optional[Dict[str, Any]]:
    """Bangun payload cache yang aman terhadap perubahan angka sensor.

    - Jika sensor_same=True: cached human answer boleh dipakai penuh.
    - Jika sensor_same=False: human answer dibuat ulang dari rule engine terbaru,
      notes/sources dari cache tetap boleh dipakai setelah difilter.
    """
    if not isinstance(cached_llm_payload, dict):
        return None

    try:
        payload = json.loads(json.dumps(cached_llm_payload, ensure_ascii=False))
    except Exception:
        payload = dict(cached_llm_payload)

    rag_answer = payload.get("rag_answer") if isinstance(payload.get("rag_answer"), dict) else {}
    reference_notes = _list_of_strings(rag_answer.get("reference_based_notes"))
    recommendation_notes = _list_of_strings(payload.get("recommendation_notes"))

    reference_notes = _filter_notes_for_current_rule(reference_notes, rule_analysis)
    recommendation_notes = _filter_notes_for_current_rule(recommendation_notes, rule_analysis)

    if sensor_same:
        human_answer = str(rag_answer.get("human_readable_answer") or "").strip()
        if not human_answer:
            human_answer = build_rule_based_human_answer(rag_request, rule_analysis)
    else:
        # Jangan pakai cached human answer karena berpotensi memuat angka telemetry lama.
        human_answer = build_rule_based_human_answer(rag_request, rule_analysis)

    safe_payload = {
        "rag_answer": {
            "human_readable_answer": sanitize_humidity_wording(human_answer, rag_request),
            "reference_based_notes": reference_notes,
        },
        "recommendation_notes": recommendation_notes,
    }
    return sanitize_llm_payload_for_humidity(safe_payload, rag_request)


def sanitize_llm_payload_for_current_rule(
    llm_payload: Optional[Dict[str, Any]],
    rag_request: Dict[str, Any],
    rule_analysis: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Sanitizer tambahan untuk output LLM fresh maupun cache.

    Fokus: cegah rekomendasi yang bertentangan dengan rule engine terbaru,
    terutama kasus pH agak basa tetapi notes menyarankan kapur/dolomit.
    """
    if not isinstance(llm_payload, dict):
        return llm_payload

    try:
        payload = json.loads(json.dumps(llm_payload, ensure_ascii=False))
    except Exception:
        payload = dict(llm_payload)

    rag_answer = payload.get("rag_answer") if isinstance(payload.get("rag_answer"), dict) else {}
    if rag_answer:
        rag_answer["reference_based_notes"] = _filter_notes_for_current_rule(
            _list_of_strings(rag_answer.get("reference_based_notes")),
            rule_analysis,
        )
        human_answer = rag_answer.get("human_readable_answer")
        if isinstance(human_answer, str):
            rag_answer["human_readable_answer"] = sanitize_humidity_wording(human_answer, rag_request)
        payload["rag_answer"] = rag_answer

    payload["recommendation_notes"] = _filter_notes_for_current_rule(
        _list_of_strings(payload.get("recommendation_notes")),
        rule_analysis,
    )
    return sanitize_llm_payload_for_humidity(payload, rag_request)


async def apply_cached_rag_bundle(
    device_id: str,
    rule_analysis: Dict[str, Any],
    *,
    rag_request: Dict[str, Any],
    normalized_sources: list[Dict[str, Any]],
    retrieval_queries: list[str],
    metadata_filters: Dict[str, Any],
    retrieval_warning: Optional[str],
    llm_skip_reason: Optional[str],
) -> tuple[list[Dict[str, Any]], list[str], Dict[str, Any], Optional[str], Optional[Dict[str, Any]], str, bool, str]:
    cached = await get_cached_rag_bundle(device_id, rule_analysis)
    if not cached:
        warning = retrieval_warning or "LLM dilewati/gagal dan belum ada cache RAG untuk status sensor saat ini."
        return normalized_sources, retrieval_queries, metadata_filters, warning, None, "", False, "none"

    cached_sources = cached.get("sources") if isinstance(cached.get("sources"), list) else []
    cached_queries = cached.get("retrieval_queries") if isinstance(cached.get("retrieval_queries"), list) else []
    cached_filters = cached.get("metadata_filters") if isinstance(cached.get("metadata_filters"), dict) else {}
    cached_llm_payload = cached.get("llm_payload") if isinstance(cached.get("llm_payload"), dict) else None
    cached_raw_answer = str(cached.get("raw_answer") or "")
    cached_at = cached.get("cached_at")

    current_sensor_fp = _sensor_value_fingerprint(rag_request)
    cached_sensor_fp = str(cached.get("sensor_fingerprint") or "")
    sensor_same = bool(cached_sensor_fp and cached_sensor_fp == current_sensor_fp)

    safe_payload = _safe_cached_llm_payload(
        cached_llm_payload,
        rag_request,
        rule_analysis,
        sensor_same=sensor_same,
    )
    cache_mode = "full_cache_same_sensor" if sensor_same else "notes_only_sensor_changed"

    warning_parts = []
    if retrieval_warning:
        warning_parts.append(str(retrieval_warning))
    if sensor_same:
        warning_parts.append(
            f"Menggunakan cache RAG penuh karena LLM tidak dipanggil/bermasalah dan angka sensor sama. "
            f"cached_at={cached_at}; reason={llm_skip_reason or 'not_available'}"
        )
    else:
        warning_parts.append(
            f"Menggunakan cache RAG terbatas karena LLM tidak dipanggil/bermasalah, tetapi angka sensor terbaru berubah. "
            f"Human answer dibuat ulang dari rule engine terbaru; cache hanya dipakai untuk notes/sources. "
            f"cached_at={cached_at}; reason={llm_skip_reason or 'not_available'}"
        )

    return (
        normalized_sources or cached_sources,
        retrieval_queries or cached_queries,
        metadata_filters or cached_filters,
        " ".join(warning_parts),
        safe_payload,
        cached_raw_answer if sensor_same else "",
        True,
        cache_mode,
    )


def _has_significant_issue(rule_analysis: Dict[str, Any]) -> bool:
    pa = rule_analysis.get("parameter_analysis", {})
    for item in pa.values():
        if item.get("priority") in {"medium", "high"}:
            return True
        if item.get("status") in SIGNIFICANT_STATUSES:
            return True
    return False


async def should_call_rag_llm(
    device_id: str,
    rule_analysis: Dict[str, Any],
    min_interval_seconds: int | None = None,
) -> tuple[bool, str]:
    min_interval = int(min_interval_seconds or _setting_value("llm_min_interval_seconds", DEFAULT_LLM_MIN_INTERVAL_SECONDS))
    now_ts = time.time()
    fingerprint = _analysis_fingerprint(rule_analysis)

    async with _last_llm_lock:
        previous = _last_llm_state.get(device_id)
        if previous is None:
            return True, "first_analysis_for_device"

        last_at = float(previous.get("last_at", 0.0))
        last_fingerprint = str(previous.get("fingerprint", ""))

        if fingerprint != last_fingerprint:
            return True, "rule_status_changed"
        if now_ts - last_at >= min_interval:
            return True, "min_interval_elapsed"
        if _has_significant_issue(rule_analysis) and now_ts - last_at >= max(300, min_interval // 3):
            return True, "significant_issue_refresh"

        remaining = max(0, int(min_interval - (now_ts - last_at)))
        return False, f"debounced_same_status_remaining_seconds={remaining}"


async def mark_llm_called(device_id: str, rule_analysis: Dict[str, Any]) -> None:
    async with _last_llm_lock:
        _last_llm_state[device_id] = {
            "last_at": time.time(),
            "fingerprint": _analysis_fingerprint(rule_analysis),
        }


def build_confidence(
    rag_request: Dict[str, Any],
    retrieved: list[Any],
    llm_ok: bool,
    rag_relevance_ok: bool,
    llm_called: bool,
    cache_used: bool = False,
    cache_mode: str | None = None,
) -> Dict[str, float]:
    sensor_quality = 1.0
    warnings = rag_request["data_quality"].get("warnings", [])
    sensor_quality -= min(len(warnings) * 0.08, 0.4)

    if not rag_request["data_quality"].get("is_calibrated", True):
        sensor_quality -= 0.25

    sensor_quality = max(0.0, min(sensor_quality, 1.0))

    top_score = _get_score(retrieved[0]) if retrieved else 0.0
    reference_score = float(_setting_value("rag_confidence_reference_score", DEFAULT_RAG_CONFIDENCE_REFERENCE_SCORE))
    reference_score = max(reference_score, 1e-6)
    rag_score = min(top_score / reference_score, 1.0)

    if not rag_relevance_ok:
        rag_score = min(rag_score, 0.35)

    # Kurangi confidence kalau metadata hasil retrieval masih miskin.
    if retrieved:
        top_metadata = _get_metadata(retrieved[0])
        if not top_metadata.get("crop"):
            rag_score *= 0.85
        if str(top_metadata.get("doc_type") or "unknown") == "unknown":
            rag_score *= 0.90

    if cache_used:
        # Cache tidak setara dengan LLM fresh. Jika angka sensor berubah, cache hanya
        # dipakai untuk notes/sources sehingga confidence LLM harus diturunkan.
        if cache_mode == "full_cache_same_sensor":
            llm_score = 0.75 if llm_ok else 0.55
        elif cache_mode == "notes_only_sensor_changed":
            llm_score = 0.55 if llm_ok else 0.45
        else:
            llm_score = 0.60 if llm_ok else 0.45
    else:
        llm_score = 1.0 if llm_ok else 0.7 if not llm_called else 0.45

    # Rule engine deterministik, tetapi tetap tergantung kualitas sensor.
    rule_engine_score = 0.9 * sensor_quality
    overall = (sensor_quality * 0.35) + (rule_engine_score * 0.25) + (rag_score * 0.25) + (llm_score * 0.15)

    return {
        "sensor_quality": round(sensor_quality, 2),
        "rule_engine": round(rule_engine_score, 2),
        "rag_retrieval": round(max(0.0, min(rag_score, 1.0)), 2),
        "llm_response": round(max(0.0, min(llm_score, 1.0)), 2),
        "overall": round(max(0.0, min(overall, 1.0)), 2),
    }


# ==========================================================
# LLM safety sanitizer
# ==========================================================


def sanitize_humidity_wording(text: str, rag_request: Dict[str, Any]) -> str:
    """Rapikan istilah humidity agar konsisten sebagai soil moisture.

    Output JSON tidak lagi memakai field tambahan untuk jenis kelembapan. Semua narasi humidity harus
    ditulis sebagai kelembapan tanah / soil moisture.
    """
    if not isinstance(text, str) or not text.strip():
        return text

    sanitized = text.strip()
    replacements = {
        "jenis kelembapan": "kelembapan tanah",
        "Jenis kelembapan": "Kelembapan tanah",
        "kelembapan udara atau tanah": "kelembapan tanah",
        "kelembapan tanah atau udara": "kelembapan tanah",
        "sensor kelembapan tanah atau kelembapan udara": "sensor kelembapan tanah",
        "humidity_type": "soil moisture",
        "nilai kelembapan atau udara": "kelembapan tanah",
        "berasal dari nilai kelembapan atau udara": "berasal dari sensor kelembapan tanah",
        "air humidity": "soil moisture",
        "kelembapan udara": "kelembapan tanah",
    }
    for source, target in replacements.items():
        sanitized = sanitized.replace(source, target)

    awkward_patterns = [
        r"(?i)kelembapan\s+belum\s+jelas\s+apakah\s+berasal\s+dari\s+nilai\s+kelembapan\s+atau\s+udara",
        r"(?i)nilai\s+kelembapan\s+belum\s+jelas\s+apakah\s+berasal\s+dari\s+sensor\s+kelembapan\s+tanah\s+atau\s+kelembapan\s+udara",
    ]
    for pattern in awkward_patterns:
        sanitized = re.sub(pattern, "Nilai kelembapan tanah berasal dari sensor soil moisture", sanitized)

    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    sanitized = re.sub(r"\s+([,.!?])", r"\1", sanitized)
    sanitized = re.sub(r"\.{2,}", ".", sanitized)
    return sanitized


# Backward-compatible name: sanitizer tetap dipakai pada payload LLM.
def sanitize_llm_payload_for_humidity(llm_payload: Optional[Dict[str, Any]], rag_request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(llm_payload, dict):
        return llm_payload

    rag_answer = llm_payload.get("rag_answer")
    if isinstance(rag_answer, dict):
        human_answer = rag_answer.get("human_readable_answer")
        if isinstance(human_answer, str):
            rag_answer["human_readable_answer"] = sanitize_humidity_wording(human_answer, rag_request)

    notes = llm_payload.get("recommendation_notes")
    if isinstance(notes, list):
        llm_payload["recommendation_notes"] = [
            sanitize_humidity_wording(str(note), rag_request)
            for note in notes
            if str(note).strip()
        ]

    return llm_payload


# ==========================================================
# Response builder
# ==========================================================


def build_response(
    rag_request: Dict[str, Any],
    rule_analysis: Dict[str, Any],
    sources: list[Dict[str, Any]],
    llm_payload: Optional[Dict[str, Any]],
    raw_answer: str,
    llm_error: Optional[str],
    confidence: Optional[Dict[str, float]] = None,
    retrieval_info: Optional[Dict[str, Any]] = None,
    llm_called: bool = True,
    llm_skip_reason: Optional[str] = None,
) -> Dict[str, Any]:
    llm_ok = llm_payload is not None and not llm_error
    if llm_error:
        status = "partial_success"
    else:
        status = "success"

    human_answer = build_rule_based_human_answer(rag_request, rule_analysis)
    human_answer = sanitize_humidity_wording(human_answer, rag_request)
    reference_based_notes: list[str] = []
    recommendation_notes: list[str] = []

    llm_payload = sanitize_llm_payload_for_humidity(llm_payload, rag_request)

    if llm_payload:
        rag_answer_obj = llm_payload.get("rag_answer")
        if isinstance(rag_answer_obj, dict):
            llm_text = rag_answer_obj.get("human_readable_answer")
            if isinstance(llm_text, str) and llm_text.strip():
                human_answer = llm_text.strip()
            notes = rag_answer_obj.get("reference_based_notes")
            if isinstance(notes, list):
                reference_based_notes = [str(x) for x in notes if str(x).strip()]

        notes = llm_payload.get("recommendation_notes")
        if isinstance(notes, list):
            recommendation_notes = [str(x) for x in notes if str(x).strip()]

    ui_status = build_ui_status(rule_analysis)
    answer_mode = _response_output_mode()
    technical_analysis = build_technical_analysis(rag_request, rule_analysis, sources)

    response: Dict[str, Any] = {
        "version": "1.6",
        "answer_mode": answer_mode,
        "request_id": rag_request["request_id"],
        "message_id": f"res-{int(time.time())}-{uuid4().hex[:8]}",
        "type": "soil_rag_response",
        "timestamp": now_wib_iso(),
        "device": {"device_id": rag_request["device"]["device_id"]},
        "status": status,
        "input_snapshot": rag_request["input_sensor"],
        "sensor_units": rag_request["sensor_units"],
        "data_quality": rag_request["data_quality"],
        "crop_context": rag_request["crop_context"],
        "ui_status": ui_status,
        "soil_condition": rule_analysis["soil_condition"],
        "parameter_analysis": rule_analysis["parameter_analysis"],
        "npk_analysis": rule_analysis["npk_analysis"],
        "recommendation": rule_analysis["recommendation"],
        "risk_assessment": rule_analysis["risk_assessment"],
        "agronomic_diagnosis": rule_analysis.get("agronomic_diagnosis", {}),
        "farmer_summary": build_farmer_summary(rag_request, rule_analysis),
        "technical_analysis": technical_analysis,
        "rag_answer": {
            "human_readable_answer": human_answer,
            "reference_based_notes": reference_based_notes,
            "recommendation_notes": recommendation_notes,
        },
        "sources": sources,
        "retrieval": retrieval_info or {},
        "confidence": confidence or {},
        "model_usage": {
            "model": settings.openrouter_model,
            "answer_mode": answer_mode,
            "top_k": settings.top_k,
            "min_score": settings.min_score,
            "llm_called": llm_called,
            "llm_skip_reason": llm_skip_reason,
            "llm_json_valid": llm_ok,
        },
    }

    if llm_error:
        response["error"] = {
            "code": "LLM_OR_RAG_GENERATION_FAILED",
            "message": llm_error,
            "raw_answer_preview": raw_answer[:500] if raw_answer else "",
        }

    return response



def _join_text(value: Any, sep: str = "\n") -> str:
    """Ubah list/string menjadi text datar agar mudah dibaca Kodular."""
    if value is None:
        return ""
    if isinstance(value, list):
        return sep.join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()

def _capital_first(value: Any) -> Any:
    """Ubah teks menjadi kapital di huruf pertama saja.

    Contoh:
    padi -> Padi
    jagung -> Jagung
    awal_tanam -> Awal_tanam
    pembuahan -> Pembuahan
    """
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return text

    return text[:1].upper() + text[1:].lower()


def build_kodular_response(full: Dict[str, Any]) -> Dict[str, Any]:
    """Response final maksimal 2 tingkat/nested untuk Kodular.

    Aturan struktur:
    - root -> group -> primitive/string/number/bool/null
    - tidak ada object di dalam object level-2
    - tidak ada list of object
    - list penting diubah menjadi string dengan pemisah newline
    """
    pa = full.get("parameter_analysis") or {}
    farmer = full.get("farmer_summary") or {}
    rag = full.get("rag_answer") or {}
    recommendation = full.get("recommendation") or {}
    risk = full.get("risk_assessment") or {}
    soil = full.get("soil_condition") or {}
    npk = full.get("npk_analysis") or {}
    ui_status = full.get("ui_status") or {}
    confidence = full.get("confidence") or {}
    retrieval = full.get("retrieval") or {}
    sources = full.get("sources") or {}
    data_quality = full.get("data_quality") or {}
    agronomy = full.get("agronomic_diagnosis") or {}

    analysis: Dict[str, Any] = {}

    for key in [
        "temperature",
        "humidity",
        "ec",
        "ph",
        "nitrogen",
        "phosphorus",
        "potassium",
        "fertility",
    ]:
        item = pa.get(key) or {}

        analysis[f"{key}_value"] = item.get("value")
        analysis[f"{key}_unit"] = item.get("unit")
        analysis[f"{key}_status"] = item.get("status")
        analysis[f"{key}_priority"] = item.get("priority")
        analysis[f"{key}_interpretation"] = item.get("interpretation")
        analysis[f"{key}_recommendation"] = item.get("recommendation")

    action_lines: list[str] = []
    for item in recommendation.get("actions") or []:
        if isinstance(item, dict):
            action = str(item.get("action") or "").strip()
            reason = str(item.get("reason") or "").strip()
            if action or reason:
                action_lines.append(f"{action} - {reason}".strip(" -"))
        else:
            action_lines.append(str(item))

    risk_lines: list[str] = []
    for item in risk.get("risks") or []:
        if isinstance(item, dict):
            risk_name = str(item.get("risk") or "").strip()
            impact = str(item.get("impact") or "").strip()
            if risk_name or impact:
                risk_lines.append(f"{risk_name} - {impact}".strip(" -"))
        else:
            risk_lines.append(str(item))

    limiting_lines: list[str] = []
    for item in agronomy.get("limiting_factors") or []:
        if isinstance(item, dict):
            parameter = str(item.get("parameter") or "").strip()
            status = str(item.get("status") or "").strip()
            effect = str(item.get("agronomic_effect") or "").strip()
            action = str(item.get("action_direction") or "").strip()
            text = f"{parameter} {status} - {effect} - {action}".strip(" -")
            if text:
                limiting_lines.append(text)
        else:
            limiting_lines.append(str(item))

    source_docs: list[str] = []
    if isinstance(sources, list):
        for item in sources:
            if isinstance(item, dict):
                document = str(item.get("document") or "").strip()
                if document:
                    source_docs.append(document)
            else:
                source_docs.append(str(item))

    return {
        "meta": {
            "version": full.get("version"),
            "answer_mode": full.get("answer_mode"),
            "request_id": full.get("request_id"),
            "message_id": full.get("message_id"),
            "type": full.get("type"),
            "timestamp": full.get("timestamp"),
            "device_id": (full.get("device") or {}).get("device_id"),
            "status": full.get("status"),
        },

        "sensor": full.get("input_snapshot") or {},

        "context": {
        "crop": _capital_first((full.get("crop_context") or {}).get("crop")),
        "growth_stage": _capital_first((full.get("crop_context") or {}).get("growth_stage")),
        "soil_type": (full.get("crop_context") or {}).get("soil_type"),
        "planting_date": (full.get("crop_context") or {}).get("planting_date"),
        "variety": (full.get("crop_context") or {}).get("variety"),
        "area_m2": (full.get("crop_context") or {}).get("area_m2"),
    },

        "data_quality": {
            "status": data_quality.get("status"),
            "is_calibrated": data_quality.get("is_calibrated"),
            "missing_fields_text": _join_text(data_quality.get("missing_fields")),
            "warnings_text": _join_text(data_quality.get("warnings")),
        },

        "status": {
            "ui_status": ui_status.get("code") or farmer.get("ui_status"),
            "status_label": ui_status.get("label") or farmer.get("status"),
            "severity": ui_status.get("severity") or farmer.get("severity"),
            "overall_status": soil.get("overall_status"),
            "priority_level": soil.get("priority_level"),
            "main_issue": soil.get("main_issue"),
            "short_summary": soil.get("short_summary"),
        },

        "analysis": analysis,

        "npk": {
            "summary": npk.get("summary"),
            "nitrogen_status": npk.get("nitrogen_status"),
            "phosphorus_status": npk.get("phosphorus_status"),
            "potassium_status": npk.get("potassium_status"),
            "balance_status": npk.get("balance_status"),
            "interpretation": npk.get("interpretation"),
            "recommendation": npk.get("recommendation"),
        },

        "recommendation": {
            "main": recommendation.get("main_recommendation"),
            "priority": recommendation.get("priority"),
            "actions_text": _join_text(action_lines),
            "farmer_status": farmer.get("status"),
            "farmer_summary": farmer.get("summary"),
            "farmer_advice": farmer.get("main_advice"),
            "simple_actions_text": _join_text(farmer.get("simple_actions")),
            "farmer_note": farmer.get("farmer_note"),
        },

        "risk": {
            "risk_level": risk.get("risk_level"),
            "risks_text": _join_text(risk_lines),
        },

        "agronomy": {
            "diagnosis_summary": agronomy.get("diagnosis_summary"),
            "stage_focus_text": _join_text(agronomy.get("stage_focus")),
            "stage_avoid_text": _join_text(agronomy.get("stage_avoid")),
            "limiting_factors_text": _join_text(limiting_lines),
            "nutrient_strategy_text": _join_text((agronomy.get("nutrient_strategy") or {}).get("direction")),
            "immediate_actions_text": _join_text(agronomy.get("immediate_actions")),
            "monitoring_plan_text": _join_text(agronomy.get("monitoring_plan")),
            "dose_policy": agronomy.get("dose_policy"),
        },

        "rag": {
            "answer": rag.get("human_readable_answer"),
            "reference_notes_text": _join_text(rag.get("reference_based_notes")),
            "recommendation_notes_text": _join_text(rag.get("recommendation_notes")),
        },

        "source": {
            "documents_text": _join_text(source_docs, sep=" | "),
            "top_score": retrieval.get("top_score"),
            "min_score": retrieval.get("min_score"),
            "rag_relevance_ok": retrieval.get("rag_relevance_ok"),
            "warning": retrieval.get("warning"),
        },

        "confidence": {
            "sensor_quality": confidence.get("sensor_quality"),
            "rule_engine": confidence.get("rule_engine"),
            "rag_retrieval": confidence.get("rag_retrieval"),
            "llm_response": confidence.get("llm_response"),
            "overall": confidence.get("overall"),
        },
    }


async def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is not None:
        return _rag_service

    async with _rag_lock:
        if _rag_service is None:
            service = RAGService()
            await asyncio.to_thread(service.load_or_build_index)
            _rag_service = service
            logger.info("RAG index loaded/built for MQTT processor.")
    return _rag_service


async def process_telemetry(device_id: str, topic: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Process one MQTT telemetry payload.

    Improvement penting dibanding versi awal:
    - retrieval tetap dijalankan walaupun LLM terkena debounce;
    - rag_answer/sources dapat memakai cache terakhir untuk fingerprint rule engine yang sama;
    - response Kodular tidak kosong/template hanya karena debounce;
    - retrieval_info menjelaskan apakah jawaban berasal dari live LLM, cache, atau fallback rule engine.
    """
    warnings = validate_telemetry(payload)
    rag_request = build_soil_rag_request(device_id=device_id, payload=payload, warnings=warnings)
    rule_analysis = build_rule_based_analysis(rag_request, warnings=warnings)

    service = await get_rag_service()
    if service.retriever is None:
        raise RuntimeError("Retriever belum siap. Jalankan build_index.py atau pastikan PDF tersedia di data/pdfs.")

    retrieved: list[Any] = []
    normalized_sources: list[Dict[str, Any]] = []
    retrieval_queries: list[str] = []
    metadata_filters: Dict[str, Any] = {}
    retrieval_warning: Optional[str] = None
    rag_relevance_ok = False
    cache_used = False
    cache_mode = "none"

    llm_called = False
    llm_skip_reason: Optional[str] = None
    raw_answer = ""
    llm_payload: Optional[Dict[str, Any]] = None
    llm_error: Optional[str] = None

    should_call, reason = await should_call_rag_llm(device_id, rule_analysis)
    llm_skip_reason = None if should_call else reason

    # Retrieval sekarang tetap dicoba lebih awal. Ini cepat dibanding LLM dan
    # membuat sources/retrieval_info tetap terisi pada kasus debounce.
    try:
        retrieved, retrieval_queries, metadata_filters = retrieve_with_multi_query(service, rag_request, rule_analysis)
        normalized_sources = _normalize_sources_from_items(retrieved, metadata_filters=metadata_filters)

        top_score = _computed_rerank_score(retrieved[0], metadata_filters) if retrieved else None
        min_score = float(_setting_value("min_score", 0.0))
        rag_relevance_ok = bool(retrieved and top_score is not None and top_score >= min_score)

        if not rag_relevance_ok:
            retrieval_warning = (
                f"Relevansi RAG rendah. top_score={top_score}, min_score={min_score}. "
                "Konteks dokumen tidak cukup kuat untuk dijadikan dasar LLM."
            )
            logger.warning("Low RAG relevance device_id=%s %s", device_id, retrieval_warning)
            # Jangan tampilkan source low-score sebagai sumber rekomendasi.
            normalized_sources = []
    except Exception as exc:
        retrieval_warning = f"Retrieval RAG gagal: {exc}"
        logger.exception("Retriever failed device_id=%s", device_id)

    if retrieval_warning:
        rag_request["data_quality"].setdefault("warnings", []).append(retrieval_warning)
        if rag_request["data_quality"].get("status") == "valid":
            rag_request["data_quality"]["status"] = "valid_with_warning"

    retrieved_for_confidence = retrieved

    if should_call:
        llm_called = True
        try:
            if rag_relevance_ok:
                selected_context = service._format_context(retrieved)
            else:
                selected_context = (
                    "Tidak ada konteks dokumen yang cukup relevan. "
                    "Gunakan hanya rule engine dan nyatakan keterbatasan referensi."
                )

            llm_question = build_llm_question(rag_request, rule_analysis)
            client = OpenRouterClient()
            timeout_seconds = int(_setting_value("llm_timeout_seconds", DEFAULT_LLM_TIMEOUT_SECONDS))

            try:
                raw_answer = await asyncio.wait_for(
                    asyncio.to_thread(
                        client.generate_answer,
                        question=llm_question,
                        selected_context=selected_context,
                        model=settings.openrouter_model,
                        temperature=RAG_TEMPERATURE,
                        system_prompt=soil_json_system_prompt(),
                        max_tokens=RAG_MAX_ANSWER_TOKENS,
                    ),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                llm_error = f"LLM timeout setelah {timeout_seconds} detik. Response utama dibuat oleh rule engine/cache."
                logger.warning("OpenRouter timeout device_id=%s timeout=%s", device_id, timeout_seconds)
            else:
                extracted = _extract_json_object(raw_answer)
                if extracted is None:
                    llm_error = "Model tidak mengembalikan JSON valid. Response utama tetap dibuat oleh rule engine/cache."
                    logger.warning("OpenRouter returned non-JSON answer. Using cache/rule-engine response.")
                else:
                    valid_schema, schema_error = validate_llm_payload(extracted)
                    if not valid_schema:
                        llm_error = f"JSON LLM tidak sesuai schema: {schema_error}. Response utama tetap dibuat oleh rule engine/cache."
                        logger.warning("OpenRouter returned invalid schema: %s", schema_error)
                    else:
                        llm_payload = sanitize_llm_payload_for_current_rule(
                            extracted,
                            rag_request,
                            rule_analysis,
                        )
                        await mark_llm_called(device_id, rule_analysis)
                        await set_cached_rag_bundle(
                            device_id,
                            rule_analysis,
                            rag_request=rag_request,
                            llm_payload=llm_payload,
                            raw_answer=raw_answer,
                            sources=normalized_sources,
                            retrieval_queries=retrieval_queries,
                            metadata_filters=metadata_filters,
                            rag_relevance_ok=rag_relevance_ok,
                            top_score=_computed_rerank_score(retrieved[0], metadata_filters) if retrieved else None,
                        )

        except Exception as exc:
            llm_error = f"Gagal memanggil/parse OpenRouter atau retrieval RAG: {exc}"
            logger.exception("OpenRouter/RAG generation failed device_id=%s", device_id)
    else:
        retrieval_warning = (
            (retrieval_warning + " " if retrieval_warning else "")
            + f"LLM dilewati karena debounce; retrieval tetap diproses. reason={reason}"
        )

    # Jika LLM tidak dipanggil, timeout, non-JSON, atau invalid schema, pakai cache
    # untuk fingerprint yang sama agar rag_answer tidak kosong/template.
    if llm_payload is None:
        (
            normalized_sources,
            retrieval_queries,
            metadata_filters,
            retrieval_warning,
            cached_payload,
            cached_raw_answer,
            cache_used,
            cache_mode,
        ) = await apply_cached_rag_bundle(
            device_id,
            rule_analysis,
            rag_request=rag_request,
            normalized_sources=normalized_sources,
            retrieval_queries=retrieval_queries,
            metadata_filters=metadata_filters,
            retrieval_warning=retrieval_warning,
            llm_skip_reason=llm_skip_reason or reason,
        )
        if cached_payload is not None:
            llm_payload = cached_payload
            raw_answer = cached_raw_answer
            # Cache valid dianggap cukup untuk menghindari partial_success yang tidak perlu
            # pada kasus debounce. Jika error aktual terjadi saat LLM dipanggil, tetap catat
            # sebagai warning retrieval/model_usage, bukan error utama UI.
            if not should_call:
                llm_error = None

    retrieval_info = {
        "queries": retrieval_queries,
        "metadata_filters": metadata_filters,
        "top_score": _computed_rerank_score(retrieved[0], metadata_filters) if retrieved else None,
        "base_top_score": _get_score(retrieved[0]) if retrieved else None,
        "min_score": float(_setting_value("min_score", 0.0)),
        "confidence_reference_score": float(_setting_value("rag_confidence_reference_score", DEFAULT_RAG_CONFIDENCE_REFERENCE_SCORE)),
        "rag_relevance_ok": rag_relevance_ok,
        "warning": retrieval_warning,
        "cache_used": cache_used,
        "cache_mode": cache_mode,
    }

    confidence = build_confidence(
        rag_request=rag_request,
        retrieved=retrieved_for_confidence,
        llm_ok=llm_payload is not None and llm_error is None,
        rag_relevance_ok=rag_relevance_ok,
        llm_called=llm_called,
        cache_used=cache_used,
        cache_mode=cache_mode,
    )

    full_response = build_response(
        rag_request=rag_request,
        rule_analysis=rule_analysis,
        sources=normalized_sources,
        llm_payload=llm_payload,
        raw_answer=raw_answer,
        llm_error=llm_error,
        confidence=confidence,
        retrieval_info=retrieval_info,
        llm_called=llm_called,
        llm_skip_reason=llm_skip_reason,
    )

    kodular_response = build_kodular_response(full_response)
    # Info cache ditambahkan di output 2-level agar Kodular/debug bisa melihat sumber jawaban.
    kodular_response.setdefault("source", {})["cache_used"] = cache_used
    kodular_response.setdefault("source", {})["cache_mode"] = cache_mode
    kodular_response.setdefault("source", {})["retrieval_queries_text"] = _join_text(retrieval_queries, sep=" | ")
    return kodular_response


# ==========================================================
# Feedback response
# ==========================================================


def make_feedback(
    device_id: str,
    level: str,
    status: str,
    message: str,
    request_id: str | None = None,
    code: str | None = None,
    stage: str | None = None,
    progress: int | None = None,
    detail: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "version": "1.6",
        "message_id": f"fb-{int(time.time())}-{uuid4().hex[:8]}",
        "request_id": request_id,
        "device_id": device_id,
        "timestamp": now_wib_iso(),
        "type": "rag_feedback",
        "level": level,
        "status": status,
        "message": message,
    }
    if stage:
        data["stage"] = stage
    if progress is not None:
        data["progress"] = max(0, min(100, int(progress)))
    if detail:
        data["detail"] = detail
    if code:
        if level == "error":
            data["error"] = {"code": code, "message": message}
        elif level == "warning":
            data["warning"] = {"code": code, "message": message}
    return data


# ==========================================================
# Offline evaluation helpers
# Jalankan manual dari shell/dev notebook untuk uji rule engine tanpa MQTT/LLM:
#   python -m app.mqtt_processor
# atau panggil run_offline_rule_engine_evaluation()
# ==========================================================

OFFLINE_EVAL_CASES: list[Dict[str, Any]] = [
    {
        "name": "padi_vegetatif_p_tinggi_k_tinggi",
        "payload": {
            "id": "SS8IN12462",
            "t": 23.9,
            "h": 45.0,
            "ec": 1362,
            "ph": 7.5,
            "n": 68,
            "p": 96,
            "k": 215,
            "f": 748,
            "lat": 3.60645,
            "lon": 98.71109,
            "crop": "padi",
            "growth_stage": "vegetatif",
        },
        "expected": {
            "ph": {"status_in": ["agak_basa", "optimal_awal"]},
            "phosphorus": {"status_in": ["sangat_tinggi", "tinggi"]},
            "potassium": {"status_in": ["tinggi", "cukup"]},
        },
    },
    {
        "name": "cabai_merah_ph_asam_n_rendah",
        "payload": {
            "id": "TEST-CABAI-1",
            "t": 28.0,
            "h": 55.0,
            "ec": 700,
            "ph": 5.2,
            "n": 20,
            "p": 25,
            "k": 160,
            "f": 420,
            "crop": "cabai_merah",
            "growth_stage": "vegetatif",
        },
        "expected": {
            "ph": {"status_in": ["asam", "agak_asam"]},
            "nitrogen": {"status_in": ["rendah"]},
        },
    },
]


def evaluate_rule_engine_payload(payload: Dict[str, Any], device_id: str = "offline-device") -> Dict[str, Any]:
    warnings = validate_telemetry(payload)
    request = build_soil_rag_request(device_id=device_id, payload=payload, warnings=warnings)
    analysis = build_rule_based_analysis(request, warnings=warnings)
    return {
        "warnings": warnings,
        "request": request,
        "analysis": analysis,
        "farmer_summary": build_farmer_summary(request, analysis),
    }


def run_offline_rule_engine_evaluation() -> Dict[str, Any]:
    results: list[Dict[str, Any]] = []
    passed = 0

    for case in OFFLINE_EVAL_CASES:
        name = case["name"]
        try:
            result = evaluate_rule_engine_payload(case["payload"], device_id=str(case["payload"].get("id", "offline-device")))
            pa = result["analysis"]["parameter_analysis"]
            checks: list[Dict[str, Any]] = []
            ok = True

            for param, expectation in case.get("expected", {}).items():
                actual_status = pa[param]["status"]
                allowed = expectation.get("status_in", [])
                check_ok = actual_status in allowed
                checks.append({
                    "param": param,
                    "actual_status": actual_status,
                    "allowed_statuses": allowed,
                    "passed": check_ok,
                })
                if not check_ok:
                    ok = False

            if ok:
                passed += 1

            results.append({
                "name": name,
                "passed": ok,
                "checks": checks,
                "farmer_summary": result["farmer_summary"],
            })
        except Exception as exc:
            results.append({
                "name": name,
                "passed": False,
                "error": str(exc),
            })

    return {
        "passed": passed,
        "total": len(OFFLINE_EVAL_CASES),
        "results": results,
    }


if __name__ == "__main__":
    print(json.dumps(run_offline_rule_engine_evaluation(), ensure_ascii=False, indent=2))

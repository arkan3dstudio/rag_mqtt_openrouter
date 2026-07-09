from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.rag_service import RAGService
from mqtt_processor import (
    _is_agriculture_chat_question,
    build_confidence,
    build_soil_health_score,
    evaluate_rule_engine_payload,
    run_offline_rule_engine_evaluation,
)


@dataclass(frozen=True)
class RetrievalCase:
    name: str
    query: str
    filters: dict[str, Any]
    expected_source_contains: tuple[str, ...]
    min_score: float = 0.05


RETRIEVAL_CASES = [
    RetrievalCase(
        name="padi_npk",
        query="rekomendasi pemupukan padi nitrogen fosfor kalium sawah",
        filters={
            "crop": "padi",
            "topics": ["pemupukan", "NPK"],
            "preferred_doc_types": ["petunjuk_teknis", "sop_manual", "manual_book"],
        },
        expected_source_contains=("padi",),
    ),
    RetrievalCase(
        name="cabai_ph_asam",
        query="cabai merah pH tanah asam pengapuran pemupukan NPK",
        filters={
            "crop": "cabai_merah",
            "topics": ["cabai", "pH", "pemupukan"],
            "preferred_doc_types": ["manual_book", "petunjuk_teknis"],
        },
        expected_source_contains=("Cab_mer", "cabai"),
    ),
    RetrievalCase(
        name="jagung_budidaya",
        query="budidaya jagung pengairan pemupukan nitrogen kalium",
        filters={
            "crop": "jagung",
            "topics": ["jagung", "budidaya", "pemupukan"],
            "preferred_doc_types": ["manual_book", "petunjuk_teknis"],
        },
        expected_source_contains=("Jagung", "jagung"),
    ),
]


MQTT_PAYLOAD_CASES = [
    {
        "name": "padi_ph_agak_asam_n_rendah",
        "payload": {
            "id": "offline-padi-001",
            "t": 28.0,
            "h": 58.0,
            "ec": 1100.0,
            "ph": 5.4,
            "n": 30.0,
            "p": 42.0,
            "k": 145.0,
            "f": 520.0,
            "crop": "padi",
            "growth_stage": "vegetatif",
        },
        "expected": {"ph": {"agak_asam", "asam"}, "nitrogen": {"rendah"}},
    },
    {
        "name": "cabai_merah_ec_tinggi_k_tinggi",
        "payload": {
            "id": "offline-cabai-001",
            "t": 30.0,
            "h": 48.0,
            "ec": 3800.0,
            "ph": 6.4,
            "n": 95.0,
            "p": 50.0,
            "k": 390.0,
            "f": 760.0,
            "crop": "cabai_merah",
            "growth_stage": "pembungaan",
        },
        "expected": {"ec": {"tinggi", "sangat_tinggi"}, "potassium": {"tinggi", "sangat_tinggi"}},
    },
    {
        "name": "jagung_ph_basa_p_tinggi",
        "payload": {
            "id": "offline-jagung-001",
            "t": 29.0,
            "h": 52.0,
            "ec": 1600.0,
            "ph": 7.6,
            "n": 100.0,
            "p": 92.0,
            "k": 240.0,
            "f": 810.0,
            "crop": "jagung",
            "growth_stage": "vegetatif",
        },
        "expected": {"ph": {"agak_basa", "basa_kuat"}, "phosphorus": {"tinggi", "sangat_tinggi"}},
    },
]


CHAT_SCOPE_CASES = [
    ("kapan mulai menanam padi?", True),
    ("cara budidaya kakao", True),
    ("monitor tanaman tembakau", True),
    ("menanam komputer", False),
    ("siapa presiden indonesia?", False),
]


def _check_rule_case(case: dict[str, Any]) -> tuple[bool, str]:
    result = evaluate_rule_engine_payload(case["payload"], device_id=case["payload"]["id"])
    parameter_analysis = result["analysis"]["parameter_analysis"]
    failures: list[str] = []

    for parameter, allowed_statuses in case["expected"].items():
        actual = parameter_analysis[parameter]["status"]
        if actual not in allowed_statuses:
            failures.append(f"{parameter}: expected one of {sorted(allowed_statuses)}, got {actual}")

    if failures:
        return False, "; ".join(failures)
    return True, "ok"


def _check_field_validation_guardrail() -> tuple[bool, str]:
    payload = {
        "id": "offline-validation-001",
        "t": 33.9,
        "h": 20.9,
        "ec": 655.0,
        "ph": 6.1,
        "n": 29.0,
        "p": 33.0,
        "k": 147.0,
        "f": 421.0,
        "crop": "padi",
        "growth_stage": "awal_tanam",
    }
    result = evaluate_rule_engine_payload(payload, device_id=payload["id"])
    validation = result["request"]["data_quality"]["validation_status"]
    confidence = build_confidence(
        result["request"],
        retrieved=[],
        llm_ok=True,
        rag_relevance_ok=True,
        llm_called=True,
    )

    if validation["status"] not in {"needs_field_validation", "partially_validated"}:
        return False, f"unexpected validation status: {validation['status']}"
    expected_cap = 0.85 if validation["status"] == "partially_validated" else 0.78
    if confidence["overall"] > expected_cap:
        return False, f"confidence cap failed: {confidence['overall']}"
    return True, f"{validation['status']} overall={confidence['overall']}"


def _check_chat_scope_cases() -> tuple[int, int]:
    passed = 0
    for question, expected in CHAT_SCOPE_CASES:
        actual = _is_agriculture_chat_question(question)
        ok = actual is expected
        passed += int(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {question!r}: {actual}")
    return passed, len(CHAT_SCOPE_CASES)


def _check_soil_score_case() -> tuple[bool, str]:
    good = evaluate_rule_engine_payload(
        {
            "id": "score-good",
            "t": 28.3,
            "h": 55.8,
            "ec": 956,
            "ph": 6.3,
            "n": 74,
            "p": 35,
            "k": 177,
            "f": 689,
            "crop": "padi",
            "growth_stage": "vegetatif",
        },
        device_id="score-good",
    )
    poor = evaluate_rule_engine_payload(
        {
            "id": "score-poor",
            "t": 34,
            "h": 20,
            "ec": 4100,
            "ph": 4.8,
            "n": 18,
            "p": 95,
            "k": 360,
            "f": 260,
            "crop": "padi",
            "growth_stage": "vegetatif",
        },
        device_id="score-poor",
    )
    good_score = build_soil_health_score(good["analysis"])
    poor_score = build_soil_health_score(poor["analysis"])
    if good_score <= poor_score:
        return False, f"expected good_score > poor_score, got {good_score} <= {poor_score}"
    return True, f"good={good_score} poor={poor_score}"


def _check_retrieval_case(service: RAGService, case: RetrievalCase) -> tuple[bool, str]:
    assert service.retriever is not None
    results = service.retriever.search(case.query, top_k=3, filters=case.filters)
    if not results:
        return False, "no results"

    best = results[0]
    score = best.rerank_score if best.rerank_score is not None else best.score
    source_text = f"{best.source} {best.metadata.get('document_title', '')}".lower()
    expected_match = any(token.lower() in source_text for token in case.expected_source_contains)

    if score < case.min_score:
        return False, f"score too low: {score:.4f} < {case.min_score:.4f}"
    if not expected_match:
        return False, f"unexpected source: {best.source}"
    return True, f"{best.source} score={score:.4f}"


def main() -> None:
    print("Rule engine built-in cases")
    built_in = run_offline_rule_engine_evaluation()
    print(f"  passed {built_in['passed']} / {built_in['total']}")

    print("\nRule engine MQTT payload cases")
    rule_passed = 0
    for case in MQTT_PAYLOAD_CASES:
        ok, detail = _check_rule_case(case)
        rule_passed += int(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {case['name']}: {detail}")

    print("\nField validation guardrail")
    validation_ok, validation_detail = _check_field_validation_guardrail()
    print(f"  [{'PASS' if validation_ok else 'FAIL'}] validation_required_without_evidence: {validation_detail}")

    print("\nChat scope guardrail")
    chat_passed, chat_total = _check_chat_scope_cases()

    print("\nSoil health score")
    score_ok, score_detail = _check_soil_score_case()
    print(f"  [{'PASS' if score_ok else 'FAIL'}] score_separates_good_and_poor: {score_detail}")

    print("\nRetrieval cases")
    service = RAGService()
    service.load_or_build_index()
    retrieval_passed = 0
    for case in RETRIEVAL_CASES:
        ok, detail = _check_retrieval_case(service, case)
        retrieval_passed += int(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {case.name}: {detail}")

    total_passed = built_in["passed"] + rule_passed + retrieval_passed + int(validation_ok) + chat_passed + int(score_ok)
    total = built_in["total"] + len(MQTT_PAYLOAD_CASES) + len(RETRIEVAL_CASES) + 1 + chat_total + 1
    print(f"\nTotal: {total_passed} / {total} passed")

    if total_passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

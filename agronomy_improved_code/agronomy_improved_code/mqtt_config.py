from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()


def _get_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def _get_str(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


# =========================
# MQTT Broker Configuration
# =========================
BROKER = (os.getenv("MQTT_BROKER") or os.getenv("BROKER", "localhost")).strip()
PORT = _get_int("MQTT_PORT", _get_int("PORT", 1883))

USERNAME = os.getenv("MQTT_USERNAME", "").strip()
PASSWORD = os.getenv("MQTT_PASSWORD", "").strip()

CLIENT_ID = (os.getenv("MQTT_CLIENT_ID") or os.getenv("CLIENT_ID", "rag-mqtt-server")).strip()
KEEPALIVE = _get_int("MQTT_KEEPALIVE", _get_int("KEEPALIVE", 60))

# =========================
# MQTT Topic Configuration
# =========================
# Dua format didukung:
# 1) legacy/compact, cocok dengan device Anda saat ini:
#    pkmunimed/{device_id}/SmartSoilsense-BIMA
#    pkmunimed/{device_id}/SmartSoilsense-BIMA-result
#    pkmunimed/{device_id}/SmartSoilsense-BIMA-feedback
#
# 2) path/clean, lebih rapi untuk pengembangan berikutnya:
#    pkmunimed/{device_id}/SmartSoilsense-BIMA/telemetry
#    pkmunimed/{device_id}/SmartSoilsense-BIMA/result
#    pkmunimed/{device_id}/SmartSoilsense-BIMA/feedback
#
# Set MQTT_TOPIC_STYLE=path bila firmware/device sudah mengikuti format path.
MQTT_TOPIC_STYLE = _get_str("MQTT_TOPIC_STYLE", "legacy").lower()
MQTT_TOPIC_NAMESPACE = _get_str("MQTT_TOPIC_NAMESPACE", "pkmunimed")
MQTT_PROJECT_SLUG = _get_str("MQTT_PROJECT_SLUG", "SmartSoilsense-BIMA")

if MQTT_TOPIC_STYLE == "path":
    DEFAULT_TOPIC_TELEMETRY = f"{MQTT_TOPIC_NAMESPACE}/+/{MQTT_PROJECT_SLUG}/telemetry"
    DEFAULT_TOPIC_RESPONSE = f"{MQTT_TOPIC_NAMESPACE}/{{device_id}}/{MQTT_PROJECT_SLUG}/result"
    DEFAULT_TOPIC_FEEDBACK = f"{MQTT_TOPIC_NAMESPACE}/{{device_id}}/{MQTT_PROJECT_SLUG}/feedback"
else:
    DEFAULT_TOPIC_TELEMETRY = f"{MQTT_TOPIC_NAMESPACE}/+/{MQTT_PROJECT_SLUG}"
    DEFAULT_TOPIC_RESPONSE = f"{MQTT_TOPIC_NAMESPACE}/{{device_id}}/{MQTT_PROJECT_SLUG}-result"
    DEFAULT_TOPIC_FEEDBACK = f"{MQTT_TOPIC_NAMESPACE}/{{device_id}}/{MQTT_PROJECT_SLUG}-feedback"

TOPIC_TELEMETRY = os.getenv("TOPIC_TELEMETRY", DEFAULT_TOPIC_TELEMETRY).strip()
TOPIC_RESPONSE = os.getenv("TOPIC_RESPONSE", DEFAULT_TOPIC_RESPONSE).strip()
TOPIC_FEEDBACK = os.getenv("TOPIC_FEEDBACK", DEFAULT_TOPIC_FEEDBACK).strip()

# =========================
# MQTT Runtime Configuration
# =========================
# QoS 1 lebih aman untuk telemetry/result/feedback karena Kodular menunggu status.
QOS = _get_int("QOS", _get_int("MQTT_QOS", 1))
PUBLISH_RETAIN = _get_bool("PUBLISH_RETAIN", False)
MQTT_RECONNECT_DELAY_MIN = _get_int("MQTT_RECONNECT_DELAY_MIN", 1)
MQTT_RECONNECT_DELAY_MAX = _get_int("MQTT_RECONNECT_DELAY_MAX", 30)

# =========================
# Worker / Queue Configuration
# =========================
MAX_CONCURRENCY = _get_int("MAX_CONCURRENCY", 5)
WORKER_COUNT = _get_int("WORKER_COUNT", 3)
QUEUE_MAXSIZE = _get_int("QUEUE_MAXSIZE", 1000)

# =========================
# Crop defaults for enriched RAG request
# =========================
DEFAULT_CROP = os.getenv("DEFAULT_CROP", "cabai")
DEFAULT_GROWTH_STAGE = os.getenv("DEFAULT_GROWTH_STAGE", "vegetatif")
DEFAULT_FIELD_ID = os.getenv("DEFAULT_FIELD_ID", "lahan_1")
DEFAULT_BLOCK_ID = os.getenv("DEFAULT_BLOCK_ID", "blok_a")

# =========================
# RAG output behaviour
# =========================
RAG_MAX_ANSWER_TOKENS = _get_int("RAG_MAX_ANSWER_TOKENS", 1200)
RAG_TEMPERATURE = _get_float("RAG_TEMPERATURE", 0.2)
RAG_LANGUAGE = os.getenv("RAG_LANGUAGE", "id")
RAG_ANSWER_STYLE = os.getenv("RAG_ANSWER_STYLE", "detail_terstruktur")

# =========================
# Optional latest JSON file output for dashboard/debugging
# =========================
WEB_OUTPUT_ENABLED = _get_bool("WEB_OUTPUT_ENABLED", True)
WEB_OUTPUT_DIR = os.getenv("WEB_OUTPUT_DIR", "web_data")
WEB_OUTPUT_BASENAME = os.getenv("WEB_OUTPUT_BASENAME", "latest")
WEB_OUTPUT_PER_DEVICE = _get_bool("WEB_OUTPUT_PER_DEVICE", True)

if __name__ == "__main__":
    print("BROKER:", repr(BROKER))
    print("PORT:", PORT)
    print("USERNAME:", repr(USERNAME))
    print("PASSWORD_LEN:", len(PASSWORD))
    print("CLIENT_ID:", repr(CLIENT_ID))
    print("MQTT_TOPIC_STYLE:", repr(MQTT_TOPIC_STYLE))
    print("TOPIC_TELEMETRY:", repr(TOPIC_TELEMETRY))
    print("TOPIC_RESPONSE:", repr(TOPIC_RESPONSE))
    print("TOPIC_FEEDBACK:", repr(TOPIC_FEEDBACK))

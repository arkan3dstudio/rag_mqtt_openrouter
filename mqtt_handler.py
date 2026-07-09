from __future__ import annotations

import asyncio
from collections import defaultdict, deque
import hashlib
import logging
import os
import re
import time
from typing import Any, Dict, Tuple

import paho.mqtt.client as mqtt

from mqtt_config import (
    TOPIC_TELEMETRY,
    TOPIC_RESPONSE,
    TOPIC_FEEDBACK,
    TOPIC_CHAT,
    TOPIC_CHAT_RESPONSE,
    QOS,
    MAX_CONCURRENCY,
    CLIENT_ID,
    QUEUE_MAXSIZE,
    PUBLISH_RETAIN,
    MQTT_RECONNECT_DELAY_MIN,
    MQTT_RECONNECT_DELAY_MAX,
    MQTT_TLS_CA_CERT,
    MQTT_TLS_ENABLED,
    MQTT_TLS_INSECURE,
    MQTT_MAX_MESSAGES_PER_MINUTE,
    MQTT_CHAT_MAX_MESSAGES_PER_MINUTE,
    MQTT_CHAT_MAX_QUESTION_CHARS,
    MQTT_DUPLICATE_WINDOW_SECONDS,
    MQTT_MAX_PAYLOAD_BYTES,
    MQTT_PROCESSING_TIMEOUT_SECONDS,
    MQTT_QUEUE_LOG_INTERVAL_SECONDS,
    MQTT_QUEUE_WARN_SIZE,
    MQTT_RATE_LIMIT_SCOPE,
    WEB_OUTPUT_ENABLED,
    WEB_OUTPUT_DIR,
    WEB_OUTPUT_BASENAME,
    WEB_OUTPUT_PER_DEVICE,
)
from mqtt_processor import (
    dumps_compact,
    make_feedback,
    process_chat_question,
    process_telemetry,
    record_request_event,
    safe_json_loads,
    update_pipeline_status,
)

MessageItem = Tuple[str, str, str, Dict[str, Any]]  # (message_type, device_id, topic, payload_dict)


class MqttApp:
    def __init__(self):
        self.queue: asyncio.Queue[MessageItem] = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
        self.client = mqtt.Client(client_id=CLIENT_ID)
        self.worker_tasks = []
        self.rate_limit_window_seconds = 60.0
        self.rate_limit_hits: dict[str, deque[float]] = defaultdict(deque)
        self.last_queue_health_log = 0.0
        self.pending_telemetry_devices: set[str] = set()
        self.recent_telemetry_payloads: dict[str, float] = {}

        self.client.reconnect_delay_set(
            min_delay=MQTT_RECONNECT_DELAY_MIN,
            max_delay=MQTT_RECONNECT_DELAY_MAX,
        )

    def _topic_response(self, device_id: str) -> str:
        return TOPIC_RESPONSE.format(device_id=device_id)

    def _topic_feedback(self, device_id: str) -> str:
        return TOPIC_FEEDBACK.format(device_id=device_id)

    def _topic_chat_response(self, device_id: str) -> str:
        return TOPIC_CHAT_RESPONSE.format(device_id=device_id)

    def _publish_json(self, topic: str, payload: Dict[str, Any]) -> None:
        output = dumps_compact(payload)
        result = self.client.publish(topic, output, qos=QOS, retain=PUBLISH_RETAIN)

        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            logging.info("Published MQTT topic=%s", topic)
        else:
            logging.error("Publish failed rc=%s topic=%s", result.rc, topic)

    def _rate_limit_key(self, device_id: str, topic: str, message_type: str) -> str:
        scope = str(MQTT_RATE_LIMIT_SCOPE or "device").strip().lower()
        if scope == "topic":
            return f"{message_type}:topic:{str(topic or '').strip()}"
        if scope == "device_topic":
            return f"{message_type}:device_topic:{device_id}:{str(topic or '').strip()}"
        return f"{message_type}:device:{device_id}"

    def _rate_limit_for_type(self, message_type: str) -> int:
        if message_type == "chat":
            return int(MQTT_CHAT_MAX_MESSAGES_PER_MINUTE or 0)
        return int(MQTT_MAX_MESSAGES_PER_MINUTE or 0)

    def _is_rate_limited(self, device_id: str, topic: str, message_type: str = "telemetry") -> tuple[bool, int, int]:
        limit = self._rate_limit_for_type(message_type)
        if limit <= 0:
            return False, 0, limit

        now = time.monotonic()
        key = self._rate_limit_key(device_id, topic, message_type)
        hits = self.rate_limit_hits[key]
        window_start = now - self.rate_limit_window_seconds

        while hits and hits[0] <= window_start:
            hits.popleft()

        if len(hits) >= limit:
            return True, len(hits), limit

        hits.append(now)
        return False, len(hits), limit

    def _log_queue_health(self, reason: str) -> None:
        queue_size = self.queue.qsize()
        now = time.monotonic()
        warn_size = int(MQTT_QUEUE_WARN_SIZE or 0)
        interval = max(int(MQTT_QUEUE_LOG_INTERVAL_SECONDS or 30), 1)

        if warn_size > 0 and queue_size >= warn_size:
            logging.warning(
                "MQTT queue high reason=%s queue_size=%s maxsize=%s warn_size=%s",
                reason,
                queue_size,
                QUEUE_MAXSIZE,
                warn_size,
            )
            self.last_queue_health_log = now
            return

        if now - self.last_queue_health_log >= interval:
            logging.info(
                "MQTT queue health reason=%s queue_size=%s maxsize=%s workers=%s concurrency=%s",
                reason,
                queue_size,
                QUEUE_MAXSIZE,
                len(self.worker_tasks),
                MAX_CONCURRENCY,
            )
            self.last_queue_health_log = now

    def _is_duplicate_telemetry(self, device_id: str, payload_bytes: bytes) -> bool:
        window = int(MQTT_DUPLICATE_WINDOW_SECONDS or 0)
        if window <= 0:
            return False

        now = time.monotonic()
        cutoff = now - window
        stale_keys = [key for key, seen_at in self.recent_telemetry_payloads.items() if seen_at <= cutoff]
        for key in stale_keys:
            self.recent_telemetry_payloads.pop(key, None)

        digest = hashlib.sha256(payload_bytes).hexdigest()
        key = f"{device_id}:{digest}"
        if key in self.recent_telemetry_payloads:
            return True
        self.recent_telemetry_payloads[key] = now
        return False

    def _save_latest_json(self, device_id: str, output: Dict[str, Any]) -> None:
        if not WEB_OUTPUT_ENABLED:
            return

        try:
            os.makedirs(WEB_OUTPUT_DIR, exist_ok=True)

            if WEB_OUTPUT_PER_DEVICE:
                filename = f"{WEB_OUTPUT_BASENAME}_{device_id}.json"
            else:
                filename = f"{WEB_OUTPUT_BASENAME}.json"

            final_path = os.path.join(WEB_OUTPUT_DIR, filename)
            tmp_path = final_path + ".tmp"

            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(dumps_compact(output))

            os.replace(tmp_path, final_path)
            logging.info("Saved web output: %s", final_path)

        except Exception as e:
            logging.exception("Failed saving web output device_id=%s err=%s", device_id, e)

    def setup(self, username: str, password: str, loop: asyncio.AbstractEventLoop):
        username = str(username or "").strip()
        password = str(password or "").strip()

        if username or password:
            self.client.username_pw_set(username, password)
            logging.info(
                "MQTT auth enabled username=%r password_len=%s",
                username,
                len(password),
            )
        else:
            logging.warning("MQTT auth disabled because username/password is empty.")

        if MQTT_TLS_ENABLED:
            self.client.tls_set(ca_certs=MQTT_TLS_CA_CERT or None)
            self.client.tls_insecure_set(bool(MQTT_TLS_INSECURE))
            logging.info(
                "MQTT TLS enabled ca_cert=%s insecure=%s",
                MQTT_TLS_CA_CERT or "system-default",
                MQTT_TLS_INSECURE,
            )

        self.client.user_data_set({"loop": loop})
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_message = self.on_message
        self.client.on_subscribe = self.on_subscribe

    # ======================================================
    # MQTT callbacks
    # ======================================================

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logging.info("MQTT Connected (rc=%s). Subscribing topics=%s, %s", rc, TOPIC_TELEMETRY, TOPIC_CHAT)

            result, mid = client.subscribe([(TOPIC_TELEMETRY, QOS), (TOPIC_CHAT, QOS)])

            if result == mqtt.MQTT_ERR_SUCCESS:
                logging.info("Subscribe request sent telemetry=%s chat=%s mid=%s qos=%s", TOPIC_TELEMETRY, TOPIC_CHAT, mid, QOS)
            else:
                logging.error("Subscribe failed result=%s topics=%s,%s", result, TOPIC_TELEMETRY, TOPIC_CHAT)
        else:
            reason = {
                1: "Unacceptable protocol version",
                2: "Identifier rejected",
                3: "Server unavailable",
                4: "Bad username or password",
                5: "Not authorized",
            }.get(rc, "Unknown error")

            logging.error("MQTT Connection failed rc=%s reason=%s", rc, reason)

    def on_subscribe(self, client, userdata, mid, granted_qos):
        logging.info("MQTT Subscribe confirmed mid=%s granted_qos=%s", mid, granted_qos)

    def on_disconnect(self, client, userdata, rc):
        if rc == 0:
            logging.info("MQTT Disconnected gracefully (rc=%s)", rc)
        else:
            logging.warning("MQTT Disconnected unexpectedly (rc=%s)", rc)

    def _extract_device_id_from_topic_pattern(self, topic: str, topic_pattern: str) -> str | None:
        """
        Ambil device_id dari topic MQTT.

        Support format baru:
            TOPIC_TELEMETRY = pkmunimed/+/SmartSoilsense-BIMA-telemetry
            Real topic      = pkmunimed/SS8IN12458/SmartSoilsense-BIMA-telemetry
            device_id       = SS8IN12458

        Tetap support format lama:
            agri/v1/SS8IN12458/rag/telemetry
        """
        topic = str(topic or "").strip().strip("/")
        topic_pattern = str(topic_pattern or "").strip().strip("/")

        if not topic:
            return None

        # Cara utama: ikuti posisi wildcard '+' dari TOPIC_TELEMETRY.
        try:
            if topic_pattern and mqtt.topic_matches_sub(topic_pattern, topic):
                pattern_parts = topic_pattern.split("/")
                topic_parts = topic.split("/")

                # Cocok untuk pattern biasa dengan '+'
                if len(pattern_parts) == len(topic_parts):
                    for index, pattern_part in enumerate(pattern_parts):
                        if pattern_part == "+":
                            device_id = topic_parts[index].strip()
                            if device_id:
                                return device_id

                # Kalau tidak ada '+', tidak bisa ambil device_id dari pattern.
                logging.warning(
                    "Topic matched pattern but no usable '+' wildcard found. pattern=%s topic=%s",
                    topic_pattern,
                    topic,
                )

        except Exception:
            logging.exception(
                "Failed matching topic pattern. pattern=%s topic=%s",
                topic_pattern,
                topic,
            )

        parts = topic.split("/")

        # Fallback format path/clean:
        # pkmunimed/SS8IN12458/SmartSoilsense-BIMA/telemetry
        # pkmunimed/SS8IN12458/SmartSoilsense-BIMA/result
        # pkmunimed/SS8IN12458/SmartSoilsense-BIMA/feedback
        if len(parts) == 4:
            if (
                parts[0] == "pkmunimed"
                and parts[2] == "SmartSoilsense-BIMA"
                and parts[3] in {"telemetry", "result", "response", "feedback"}
            ):
                device_id = parts[1].strip()
                return device_id or None

        # Fallback format legacy/compact eksplisit:
        # pkmunimed/SS8IN12458/SmartSoilsense-BIMA-telemetry
        # pkmunimed/SS8IN12458/SmartSoilsense-BIMA-result
        # pkmunimed/SS8IN12458/SmartSoilsense-BIMA-feedback
        if len(parts) == 3:
            if parts[0] == "pkmunimed" and (
                parts[2] == "SmartSoilsense-BIMA-telemetry"
            ):
                device_id = parts[1].strip()
                return device_id or None

        # Fallback format lama:
        # agri/v1/SS8IN12458/rag/telemetry
        if len(parts) == 5:
            if (
                parts[0] == "agri"
                and parts[1] == "v1"
                and parts[3] == "rag"
                and parts[4] == "telemetry"
            ):
                device_id = parts[2].strip()
                return device_id or None

        return None

    def _extract_device_id_from_topic(self, topic: str) -> str | None:
        return self._extract_device_id_from_topic_pattern(topic, TOPIC_TELEMETRY)

    def _extract_chat_device_id_from_topic(self, topic: str) -> str | None:
        device_id = self._extract_device_id_from_topic_pattern(topic, TOPIC_CHAT)
        if device_id:
            return device_id

        topic = str(topic or "").strip().strip("/")
        parts = topic.split("/")
        if len(parts) == 3 and parts[0] == "pkmunimed" and parts[2] == "SmartSoilsense-BIMA-chat":
            return parts[1].strip() or None
        if len(parts) == 4 and parts[0] == "pkmunimed" and parts[2] == "SmartSoilsense-BIMA" and parts[3] == "chat":
            return parts[1].strip() or None
        return None

    def _extract_device_id_from_payload(self, payload: Dict[str, Any]) -> str | None:
        if not isinstance(payload, dict):
            return None

        device_id = (
            payload.get("id")
            or payload.get("device_id")
            or payload.get("deviceId")
            or payload.get("DEVICE_ID")
            or payload.get("device")
        )

        if isinstance(device_id, dict):
            device_id = device_id.get("device_id") or device_id.get("id")

        device_id = str(device_id or "").strip()
        return device_id or None

    def _validated_device_id(self, value: str | None) -> str | None:
        device_id = str(value or "").strip()
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", device_id):
            return device_id
        return None

    def on_message(self, client, userdata, msg):
        logging.info(
            "MQTT message received topic=%s payload_size=%s",
            msg.topic,
            len(msg.payload or b""),
        )

        is_chat = mqtt.topic_matches_sub(TOPIC_CHAT, msg.topic)
        message_type = "chat" if is_chat else "telemetry"
        device_id = self._extract_chat_device_id_from_topic(msg.topic) if is_chat else self._extract_device_id_from_topic(msg.topic)

        payload_size = len(msg.payload or b"")
        if MQTT_MAX_PAYLOAD_BYTES > 0 and payload_size > MQTT_MAX_PAYLOAD_BYTES:
            safe_device_id = self._validated_device_id(device_id) or "unknown"
            logging.warning(
                "MQTT payload too large type=%s device_id=%s size=%s limit=%s",
                message_type,
                safe_device_id,
                payload_size,
                MQTT_MAX_PAYLOAD_BYTES,
            )
            self._publish_json(
                self._topic_chat_response(safe_device_id) if is_chat else self._topic_feedback(safe_device_id),
                make_feedback(
                    device_id=safe_device_id,
                    level="error",
                    status="failed",
                    code="PAYLOAD_TOO_LARGE",
                    message=f"Payload melebihi batas {MQTT_MAX_PAYLOAD_BYTES} byte.",
                ),
            )
            return

        payload = safe_json_loads(msg.payload)
        if payload is None:
            logging.error("Invalid JSON from topic=%s payload=%r", msg.topic, msg.payload[:200])

            feedback_device_id = device_id or "unknown"
            feedback_topic = self._topic_chat_response(feedback_device_id) if is_chat else self._topic_feedback(feedback_device_id)

            self._publish_json(
                feedback_topic,
                make_feedback(
                    device_id=feedback_device_id,
                    level="error",
                    status="failed",
                    code="INVALID_JSON",
                    message="Payload MQTT bukan JSON object yang valid.",
                ),
            )
            return

        # Fallback: kalau device_id gagal dari topic, ambil dari payload.
        if not device_id:
            device_id = self._extract_device_id_from_payload(payload)

        raw_device_id = device_id
        device_id = self._validated_device_id(device_id)
        if not device_id:
            logging.warning(
                "Invalid or unsafe device_id. topic=%s pattern=%s raw_device_id=%r payload_keys=%s",
                msg.topic,
                TOPIC_CHAT if is_chat else TOPIC_TELEMETRY,
                raw_device_id,
                list(payload.keys()) if isinstance(payload, dict) else None,
            )
            return

        if is_chat:
            question = str(
                payload.get("question")
                or payload.get("message")
                or payload.get("text")
                or payload.get("prompt")
                or payload.get("query")
                or ""
            ).strip()
            if MQTT_CHAT_MAX_QUESTION_CHARS > 0 and len(question) > MQTT_CHAT_MAX_QUESTION_CHARS:
                self._publish_json(
                    self._topic_chat_response(device_id),
                    make_feedback(
                        device_id=device_id,
                        level="error",
                        status="failed",
                        request_id=payload.get("request_id"),
                        code="QUESTION_TOO_LONG",
                        message=f"Pertanyaan melebihi batas {MQTT_CHAT_MAX_QUESTION_CHARS} karakter.",
                    ),
                )
                return

            if device_id in self.pending_telemetry_devices:
                logging.info(
                    "Chat blocked because telemetry analysis is still pending device_id=%s",
                    device_id,
                )
                update_pipeline_status(
                    device_id=device_id,
                    request_id=payload.get("request_id"),
                    stage="chat_waiting_analysis",
                    status="waiting_analysis",
                    message="Chat ditahan karena analysis device masih berjalan.",
                    progress=0,
                )
                self._publish_json(
                    self._topic_chat_response(device_id),
                    make_feedback(
                        device_id=device_id,
                        level="warning",
                        status="waiting_analysis",
                        request_id=payload.get("request_id"),
                        code="ANALYSIS_NOT_READY",
                        message=(
                            "Analisis sensor untuk device ini belum selesai. "
                            "Tunggu feedback success/result analysis terlebih dahulu, lalu kirim pertanyaan chat lagi."
                        ),
                        stage="analysis_guard",
                        progress=0,
                    ),
                )
                return

        if message_type == "telemetry" and self._is_duplicate_telemetry(device_id, bytes(msg.payload or b"")):
            logging.warning("Duplicate telemetry ignored device_id=%s topic=%s", device_id, msg.topic)
            self._publish_json(
                self._topic_feedback(device_id),
                make_feedback(
                    device_id=device_id,
                    level="info",
                    status="duplicate_ignored",
                    request_id=payload.get("request_id") if isinstance(payload, dict) else None,
                    code="DUPLICATE_IGNORED",
                    message="Telemetry identik sudah diterima beberapa detik sebelumnya dan tidak diproses ulang.",
                    stage="deduplication",
                    progress=0,
                ),
            )
            return

        limited, current_count, active_limit = self._is_rate_limited(device_id, msg.topic, message_type=message_type)
        if limited:
            logging.warning(
                "Rate limit exceeded. Dropping %s device_id=%s topic=%s count=%s limit=%s scope=%s",
                message_type,
                device_id,
                msg.topic,
                current_count,
                active_limit,
                MQTT_RATE_LIMIT_SCOPE,
            )
            self._publish_json(
                self._topic_chat_response(device_id) if message_type == "chat" else self._topic_feedback(device_id),
                make_feedback(
                    device_id=device_id,
                    level="warning",
                    status="rate_limited",
                    request_id=payload.get("request_id") if isinstance(payload, dict) else None,
                    code="RATE_LIMITED",
                    message=(
                        "Telemetry terlalu sering. Pesan ini tidak diproses agar queue RAG/LLM tidak penuh. "
                        f"Batas saat ini {MQTT_MAX_MESSAGES_PER_MINUTE} pesan per menit per {MQTT_RATE_LIMIT_SCOPE}."
                        if message_type != "chat"
                        else f"Batas chat saat ini {MQTT_CHAT_MAX_MESSAGES_PER_MINUTE} pesan per menit per {MQTT_RATE_LIMIT_SCOPE}."
                    ),
                    stage="rate_limit",
                    progress=0,
                    detail={
                        "limit_per_minute": active_limit,
                        "scope": MQTT_RATE_LIMIT_SCOPE,
                        "current_count": current_count,
                    },
                ),
            )
            return

        logging.info("MQTT %s accepted device_id=%s topic=%s", message_type, device_id, msg.topic)

        loop = userdata.get("loop") if isinstance(userdata, dict) else None
        if loop is None:
            logging.error("Event loop not found in userdata.")
            return

        def _enqueue():
            try:
                if message_type == "telemetry" and device_id in self.pending_telemetry_devices:
                    logging.warning(
                        "Telemetry skipped because device is already queued/processing device_id=%s",
                        device_id,
                    )
                    update_pipeline_status(
                        device_id=device_id,
                        request_id=payload.get("request_id") if isinstance(payload, dict) else None,
                        stage="queue_guard",
                        status="busy",
                        message="Telemetry sebelumnya masih diproses.",
                        progress=0,
                    )
                    self._publish_json(
                        self._topic_feedback(device_id),
                        make_feedback(
                            device_id=device_id,
                            level="warning",
                            status="busy",
                            request_id=payload.get("request_id") if isinstance(payload, dict) else None,
                            code="DEVICE_BUSY",
                            message="Telemetry sebelumnya dari device ini masih diproses. Kirim ulang setelah result/feedback selesai.",
                            stage="queue_guard",
                            progress=0,
                        ),
                    )
                    return

                if message_type == "telemetry":
                    self.pending_telemetry_devices.add(device_id)
                    update_pipeline_status(
                        device_id=device_id,
                        request_id=payload.get("request_id") if isinstance(payload, dict) else None,
                        stage="telemetry_queued",
                        status="queued",
                        message="Telemetry masuk antrean worker.",
                        progress=5,
                    )
                self.queue.put_nowait((message_type, device_id, msg.topic, payload))
                record_request_event(
                    device_id=device_id,
                    request_id=payload.get("request_id") if isinstance(payload, dict) else None,
                    event=f"{message_type}_queued",
                    status="queued",
                    stage="queue",
                    detail={"topic": msg.topic, "queue_size": self.queue.qsize()},
                )
                logging.info(
                    "MQTT %s queued device_id=%s queue_size=%s",
                    message_type,
                    device_id,
                    self.queue.qsize(),
                )
                self._log_queue_health(reason=f"{message_type}_enqueued")

            except asyncio.QueueFull:
                if message_type == "telemetry":
                    self.pending_telemetry_devices.discard(device_id)
                update_pipeline_status(
                    device_id=device_id,
                    request_id=payload.get("request_id") if isinstance(payload, dict) else None,
                    stage="queue",
                    status="failed",
                    message="Queue server penuh.",
                    progress=0,
                )
                logging.error(
                    "Queue full. Dropping message topic=%s device_id=%s",
                    msg.topic,
                    device_id,
                )

                self._publish_json(
                    self._topic_chat_response(device_id) if message_type == "chat" else self._topic_feedback(device_id),
                    make_feedback(
                        device_id=device_id,
                        level="error",
                        status="failed",
                        code="QUEUE_FULL",
                        message="Queue server penuh. Telemetry tidak diproses.",
                    ),
                )

            except Exception as e:
                if message_type == "telemetry":
                    self.pending_telemetry_devices.discard(device_id)
                logging.exception("Failed to enqueue message: %s", e)

        try:
            loop.call_soon_threadsafe(_enqueue)
        except Exception as e:
            logging.exception("Failed scheduling enqueue: %s", e)

    # ======================================================
    # Async workers
    # ======================================================

    async def worker(self):
        while True:
            message_type, device_id, topic, payload = await self.queue.get()
            request_id = payload.get("request_id") if isinstance(payload, dict) else None

            try:
                async with self.semaphore:
                    if message_type == "chat":
                        self._publish_json(
                            self._topic_feedback(device_id),
                            make_feedback(
                                device_id=device_id,
                                level="info",
                                status="processing",
                                request_id=request_id,
                                message="Pertanyaan chat diterima. Server menyiapkan jawaban berbasis sensor terakhir dan dokumen RAG.",
                                stage="chat_accepted",
                                progress=20,
                            ),
                        )

                        timeout_seconds = int(MQTT_PROCESSING_TIMEOUT_SECONDS or 0)
                        logging.info(
                            "Chat processing started device_id=%s topic=%s timeout=%ss",
                            device_id,
                            topic,
                            timeout_seconds if timeout_seconds > 0 else "disabled",
                        )
                        if timeout_seconds > 0:
                            output = await asyncio.wait_for(
                                process_chat_question(device_id, topic, payload),
                                timeout=timeout_seconds,
                            )
                        else:
                            output = await process_chat_question(device_id, topic, payload)

                        self._publish_json(self._topic_chat_response(device_id), output)
                        self._publish_json(
                            self._topic_feedback(device_id),
                            make_feedback(
                                device_id=device_id,
                                level="info",
                                status="success",
                                request_id=output.get("request_id") or request_id,
                                message="Jawaban chat dikirim ke topic chat-result.",
                                stage="chat_done",
                                progress=100,
                            ),
                        )
                        logging.info(
                            "Worker selesai memproses chat device_id=%s. Response dikirim ke topic=%s",
                            device_id,
                            self._topic_chat_response(device_id),
                        )
                        continue

                    self._publish_json(
                        self._topic_feedback(device_id),
                        make_feedback(
                            device_id=device_id,
                            level="info",
                            status="processing",
                            request_id=request_id,
                            message="Telemetry diterima. Server menunggu retrieval RAG dan LLM selesai sebelum mengirim result.",
                            stage="accepted",
                            progress=20,
                        ),
                    )

                    timeout_seconds = int(MQTT_PROCESSING_TIMEOUT_SECONDS or 0)
                    logging.info(
                        "Worker processing started device_id=%s topic=%s timeout=%ss",
                        device_id,
                        topic,
                        timeout_seconds if timeout_seconds > 0 else "disabled",
                    )

                    if timeout_seconds > 0:
                        output = await asyncio.wait_for(
                            process_telemetry(device_id, topic, payload),
                            timeout=timeout_seconds,
                        )
                    else:
                        output = await process_telemetry(device_id, topic, payload)

                    logging.info("Worker processing finished device_id=%s topic=%s", device_id, topic)

                    self._publish_json(self._topic_response(device_id), output)
                    self._save_latest_json(device_id, output)

                    output_request_id = (
                        output.get("request_id")
                        or (output.get("meta") or {}).get("request_id")
                        or request_id
                    )
                    fallback_used = bool((output.get("source") or {}).get("fallback_used"))
                    completion_message = (
                        "Result fallback rule engine dikirim karena layanan LLM tidak selesai, sementara retrieval RAG tetap relevan."
                        if fallback_used
                        else "Retrieval RAG dan LLM berhasil. Result dikirim ke topic response."
                    )

                    self._publish_json(
                        self._topic_feedback(device_id),
                        make_feedback(
                            device_id=device_id,
                            level="info",
                            status="success",
                            request_id=output_request_id,
                            message=completion_message,
                            stage="done",
                            progress=100,
                        ),
                    )

                    logging.info(
                        "Worker selesai memproses device_id=%s. Response dikirim ke topic=%s",
                        device_id,
                        self._topic_response(device_id),
                    )

            except asyncio.CancelledError:
                raise

            except asyncio.TimeoutError:
                process_label = "chat" if message_type == "chat" else "telemetry"
                message = (
                    f"Pemrosesan {process_label} melewati batas {MQTT_PROCESSING_TIMEOUT_SECONDS} detik. "
                    "Result tidak dikirim untuk mencegah worker menggantung."
                )
                logging.warning(
                    "Worker processing timeout device_id=%s topic=%s timeout=%s",
                    device_id,
                    topic,
                    MQTT_PROCESSING_TIMEOUT_SECONDS,
                )
                update_pipeline_status(
                    device_id=device_id,
                    request_id=request_id,
                    stage=f"{process_label}_timeout",
                    status="failed",
                    message=message,
                    progress=0,
                )

                self._publish_json(
                    self._topic_chat_response(device_id) if message_type == "chat" else self._topic_feedback(device_id),
                    make_feedback(
                        device_id=device_id,
                        level="error",
                        status="failed",
                        request_id=request_id,
                        code="PROCESSING_TIMEOUT",
                        message=message,
                    ),
                )

                logging.info(
                    "Worker gagal memproses device_id=%s. Menunggu telemetry berikutnya...",
                    device_id,
                )

            except Exception as e:
                error_text = str(e)
                expected_block = (
                    "Result tidak dikirim" in error_text
                    or "LLM timeout" in error_text
                    or "RAG retrieval belum cukup relevan" in error_text
                    or "JSON LLM tidak sesuai schema" in error_text
                    or "Model tidak mengembalikan JSON valid" in error_text
                )
                if expected_block:
                    logging.warning(
                        "Worker blocked result device_id=%s topic=%s reason=%s",
                        device_id,
                        topic,
                        error_text,
                    )
                else:
                    logging.exception("Error in worker device_id=%s topic=%s: %s", device_id, topic, e)
                update_pipeline_status(
                    device_id=device_id,
                    request_id=request_id,
                    stage=f"{message_type}_error",
                    status="failed",
                    message=error_text,
                    progress=0,
                )

                self._publish_json(
                    self._topic_chat_response(device_id) if message_type == "chat" else self._topic_feedback(device_id),
                    make_feedback(
                        device_id=device_id,
                        level="error",
                        status="failed",
                        request_id=request_id,
                        code="PROCESSING_BLOCKED" if expected_block else "PROCESSING_ERROR",
                        message=error_text,
                    ),
                )

                logging.info(
                    "Worker gagal memproses device_id=%s. Menunggu telemetry berikutnya...",
                    device_id,
                )

            finally:
                self.queue.task_done()
                if message_type == "telemetry":
                    self.pending_telemetry_devices.discard(device_id)
                self._log_queue_health(reason=f"{message_type}_done")

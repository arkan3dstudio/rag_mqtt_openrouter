from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, Tuple

import paho.mqtt.client as mqtt

from mqtt_config import (
    TOPIC_TELEMETRY,
    TOPIC_RESPONSE,
    TOPIC_FEEDBACK,
    QOS,
    MAX_CONCURRENCY,
    CLIENT_ID,
    QUEUE_MAXSIZE,
    PUBLISH_RETAIN,
    MQTT_RECONNECT_DELAY_MIN,
    MQTT_RECONNECT_DELAY_MAX,
    WEB_OUTPUT_ENABLED,
    WEB_OUTPUT_DIR,
    WEB_OUTPUT_BASENAME,
    WEB_OUTPUT_PER_DEVICE,
)
from mqtt_processor import dumps_compact, make_feedback, process_telemetry, safe_json_loads

MessageItem = Tuple[str, str, Dict[str, Any]]  # (device_id, topic, payload_dict)


class MqttApp:
    def __init__(self):
        self.queue: asyncio.Queue[MessageItem] = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
        self.client = mqtt.Client(client_id=CLIENT_ID)
        self.worker_tasks = []

        self.client.reconnect_delay_set(
            min_delay=MQTT_RECONNECT_DELAY_MIN,
            max_delay=MQTT_RECONNECT_DELAY_MAX,
        )

    def _topic_response(self, device_id: str) -> str:
        return TOPIC_RESPONSE.format(device_id=device_id)

    def _topic_feedback(self, device_id: str) -> str:
        return TOPIC_FEEDBACK.format(device_id=device_id)

    def _publish_json(self, topic: str, payload: Dict[str, Any]) -> None:
        output = dumps_compact(payload)
        result = self.client.publish(topic, output, qos=QOS, retain=PUBLISH_RETAIN)

        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            logging.info("Published MQTT topic=%s", topic)
        else:
            logging.error("Publish failed rc=%s topic=%s", result.rc, topic)

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
            logging.info("MQTT Connected (rc=%s). Subscribing topic=%s", rc, TOPIC_TELEMETRY)

            result, mid = client.subscribe(TOPIC_TELEMETRY, qos=QOS)

            if result == mqtt.MQTT_ERR_SUCCESS:
                logging.info("Subscribe request sent topic=%s mid=%s qos=%s", TOPIC_TELEMETRY, mid, QOS)
            else:
                logging.error("Subscribe failed result=%s topic=%s", result, TOPIC_TELEMETRY)
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

    def _extract_device_id_from_topic(self, topic: str) -> str | None:
        """
        Ambil device_id dari topic MQTT.

        Support format baru:
            TOPIC_TELEMETRY = pkmunimed/+/SmartSoilsense-BIMA
            Real topic      = pkmunimed/SS8IN12458/SmartSoilsense-BIMA
            device_id       = SS8IN12458

        Tetap support format lama:
            agri/v1/SS8IN12458/rag/telemetry
        """
        topic = str(topic or "").strip().strip("/")
        topic_pattern = str(TOPIC_TELEMETRY or "").strip().strip("/")

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
        # pkmunimed/SS8IN12458/SmartSoilsense-BIMA
        # pkmunimed/SS8IN12458/SmartSoilsense-BIMA-result
        # pkmunimed/SS8IN12458/SmartSoilsense-BIMA-feedback
        if len(parts) == 3:
            if parts[0] == "pkmunimed" and (
                parts[2] == "SmartSoilsense-BIMA"
                or parts[2].startswith("SmartSoilsense-BIMA-")
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

    def on_message(self, client, userdata, msg):
        logging.info(
            "MQTT message received topic=%s payload_size=%s",
            msg.topic,
            len(msg.payload or b""),
        )

        device_id = self._extract_device_id_from_topic(msg.topic)

        payload = safe_json_loads(msg.payload)
        if payload is None:
            logging.error("Invalid JSON from topic=%s payload=%r", msg.topic, msg.payload[:200])

            feedback_device_id = device_id or "unknown"

            self._publish_json(
                self._topic_feedback(feedback_device_id),
                make_feedback(
                    device_id=feedback_device_id,
                    level="error",
                    status="failed",
                    code="INVALID_JSON",
                    message="Payload telemetry bukan JSON object yang valid.",
                ),
            )
            return

        # Fallback: kalau device_id gagal dari topic, ambil dari payload.
        if not device_id:
            device_id = self._extract_device_id_from_payload(payload)

        if not device_id:
            logging.warning(
                "Invalid telemetry topic format and payload has no device_id. topic=%s pattern=%s payload_keys=%s",
                msg.topic,
                TOPIC_TELEMETRY,
                list(payload.keys()) if isinstance(payload, dict) else None,
            )
            return

        logging.info("Telemetry accepted device_id=%s topic=%s", device_id, msg.topic)

        loop = userdata.get("loop") if isinstance(userdata, dict) else None
        if loop is None:
            logging.error("Event loop not found in userdata.")
            return

        def _enqueue():
            try:
                self.queue.put_nowait((device_id, msg.topic, payload))
                logging.info(
                    "Telemetry queued device_id=%s queue_size=%s",
                    device_id,
                    self.queue.qsize(),
                )

            except asyncio.QueueFull:
                logging.error(
                    "Queue full. Dropping message topic=%s device_id=%s",
                    msg.topic,
                    device_id,
                )

                self._publish_json(
                    self._topic_feedback(device_id),
                    make_feedback(
                        device_id=device_id,
                        level="error",
                        status="failed",
                        code="QUEUE_FULL",
                        message="Queue server penuh. Telemetry tidak diproses.",
                    ),
                )

            except Exception as e:
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
            device_id, topic, payload = await self.queue.get()
            request_id = payload.get("request_id") if isinstance(payload, dict) else None

            try:
                async with self.semaphore:
                    self._publish_json(
                        self._topic_feedback(device_id),
                        make_feedback(
                            device_id=device_id,
                            level="info",
                            status="processing",
                            request_id=request_id,
                            message="Telemetry diterima dan sedang diproses oleh RAG.",
                            stage="accepted",
                            progress=20,
                        ),
                    )

                    output = await process_telemetry(device_id, topic, payload)

                    self._publish_json(self._topic_response(device_id), output)
                    self._save_latest_json(device_id, output)

                    output_request_id = (
                        output.get("request_id")
                        or (output.get("meta") or {}).get("request_id")
                        or request_id
                    )

                    self._publish_json(
                        self._topic_feedback(device_id),
                        make_feedback(
                            device_id=device_id,
                            level="info",
                            status="success",
                            request_id=output_request_id,
                            message="Analisis RAG berhasil dibuat dan dikirim ke topic response.",
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

            except Exception as e:
                logging.exception("Error in worker device_id=%s topic=%s: %s", device_id, topic, e)

                self._publish_json(
                    self._topic_feedback(device_id),
                    make_feedback(
                        device_id=device_id,
                        level="error",
                        status="failed",
                        request_id=request_id,
                        code="PROCESSING_ERROR",
                        message=str(e),
                    ),
                )

                logging.info(
                    "Worker gagal memproses device_id=%s. Menunggu telemetry berikutnya...",
                    device_id,
                )

            finally:
                self.queue.task_done()
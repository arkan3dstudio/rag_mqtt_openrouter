from __future__ import annotations

import asyncio
import logging
import signal

from logging_setup import setup_logging
from mqtt_config import BROKER, PORT, USERNAME, PASSWORD, KEEPALIVE, WORKER_COUNT
from mqtt_handler import MqttApp


async def main():
    setup_logging()
    logger = logging.getLogger("mqtt_main")

    app = MqttApp()
    loop = asyncio.get_running_loop()
    app.setup(USERNAME, PASSWORD, loop)

    stop_event = asyncio.Event()

    def _request_shutdown():
        logger.info("Shutdown signal received.")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_shutdown)
        except NotImplementedError:
            # Windows fallback
            pass

    logger.info("Connecting to MQTT broker %s:%s", BROKER, PORT)
    app.client.connect(BROKER, PORT, KEEPALIVE)
    app.client.loop_start()

    app.worker_tasks = [
        asyncio.create_task(app.worker(), name=f"worker-{i+1}")
        for i in range(WORKER_COUNT)
    ]

    try:
        await stop_event.wait()
    finally:
        logger.info("Stopping MQTT RAG application...")
        for task in app.worker_tasks:
            task.cancel()
        await asyncio.gather(*app.worker_tasks, return_exceptions=True)
        app.client.loop_stop()
        app.client.disconnect()
        logger.info("MQTT RAG application stopped.")


if __name__ == "__main__":
    asyncio.run(main())

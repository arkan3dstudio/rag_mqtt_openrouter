# Deploy Docker SULTANI RAG MQTT

Panduan ini menjalankan service MQTT worker:

```text
telemetry MQTT -> rule engine -> RAG -> OpenRouter -> result/feedback/chat-result
```

## 1. Siapkan environment

Salin template:

```bash
copy .env.docker.example .env
```

Lalu isi minimal:

```text
OPENROUTER_API_KEY=...
MQTT_BROKER=...
MQTT_PORT=...
MQTT_USERNAME=...
MQTT_PASSWORD=...
```

Catatan: jangan commit `.env` karena berisi secret.

## 2. Pastikan data tersedia

Folder berikut dipakai sebagai volume:

```text
data/pdfs/                  # PDF referensi RAG
data/document_manifest.json # metadata dokumen
storage/                    # markdown dan index
web_data/                   # latest result, pipeline status, request audit
```

`data/pdfs` dan `data/document_manifest.json` dimount read-only di container.

## 3. Build image

```bash
docker compose -f docker-compose.mqtt.yml build
```

Image yang dibuat:

```text
sultani-rag-mqtt:latest
```

## 4. Bangun index RAG

Jalankan sekali sebelum worker MQTT dinaikkan, terutama pada server baru:

```bash
docker compose -f docker-compose.mqtt.yml run --rm rag_mqtt python build_index.py
```

Jika index sudah ada di `storage/index`, langkah ini bisa dilewati.

## 5. Jalankan MQTT worker

```bash
docker compose -f docker-compose.mqtt.yml up -d
```

Lihat log:

```bash
docker compose -f docker-compose.mqtt.yml logs -f rag_mqtt
```

Stop service:

```bash
docker compose -f docker-compose.mqtt.yml down
```

## 6. Cek evaluasi offline di container

```bash
docker compose -f docker-compose.mqtt.yml run --rm rag_mqtt python offline_eval.py
```

Ekspektasi saat ini:

```text
Total: 15 / 15 passed
```

## 7. File output runtime

Saat worker berjalan, folder `web_data` akan berisi:

```text
latest_{device_id}.json       # result terakhir per device
pipeline_{device_id}.json     # status pipeline per device
request_audit.jsonl           # audit event JSON lines
```

Ini berguna untuk WebView/debug:

```text
telemetry_received -> analyzing -> rag_retrieval -> llm_generation -> result_success -> chat_ready
```

## 8. Update image setelah perubahan kode

```bash
docker compose -f docker-compose.mqtt.yml build
docker compose -f docker-compose.mqtt.yml up -d
```

Jika PDF/manifest berubah, rebuild index:

```bash
docker compose -f docker-compose.mqtt.yml run --rm rag_mqtt python build_index.py
docker compose -f docker-compose.mqtt.yml restart rag_mqtt
```

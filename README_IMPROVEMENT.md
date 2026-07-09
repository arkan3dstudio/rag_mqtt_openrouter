# Agronomy Improved Code - Stable Patch

Patch ini memperbaiki versi agronomy improved sebelumnya dengan fokus pada stabilitas MQTT result dan kualitas output agronomi.

## Perubahan utama

1. `meta.status` sekarang tetap `success` jika rule engine dan agronomic diagnosis berhasil, walaupun LLM/OpenRouter gagal, timeout, atau dilewati debounce.
2. Status/model error LLM dipindahkan menjadi `source.model_warning` dan `source.fallback_used`, bukan lagi membuat result utama menjadi `partial_success`.
3. `source.documents_text` sudah dideduplicate agar tidak menampilkan PDF yang sama berulang di Kodular.
4. Confidence `rag_retrieval` sekarang memakai rerank score yang sama dengan `source.top_score`, sehingga tidak rendah ketika dokumen sebenarnya relevan.
5. `openrouter_client.py` tidak lagi membuat seluruh aplikasi gagal import jika package `openai` belum terpasang. Error OpenRouter akan ditangkap oleh processor dan sistem fallback ke rule engine.
6. Log OpenRouter fallback dibuat warning singkat, bukan stack trace panjang berulang.

## Cara pakai

Salin seluruh isi folder ini ke root project Anda, atau minimal ganti file berikut:

- `mqtt_processor.py`
- `app/openrouter_client.py`
- `app/retriever.py` bila Anda belum memakai retriever dari versi improved sebelumnya.

Lalu jalankan:

```bash
python -m compileall .
python build_index.py
python mqtt_main.py
```

## Catatan

Output ini tetap memberikan rekomendasi agronomi awal, bukan dosis pupuk final. Dosis spesifik tetap memerlukan validasi satuan sensor, luas lahan, umur tanaman, varietas, riwayat pemupukan, dan/atau uji tanah.

# Agronomy-focused RAG MQTT improvements

Perubahan utama:

1. `mqtt_processor.py`
   - Menambahkan `agronomic_diagnosis` deterministik: faktor pembatas, efek agronomi, arah tindakan, strategi NPK, monitoring plan, dan kebijakan dosis.
   - Query RAG sekarang dibangun dari komoditas + fase + isu prioritas, bukan hanya query umum budidaya.
   - Metadata filter menambahkan `preferred_doc_types` dan `issue_parameters` agar manual/SOP/petunjuk teknis lebih dominan.
   - Prompt LLM dipertegas agar tidak menyarankan tambahan pupuk untuk unsur yang sudah tinggi/sangat tinggi.
   - Fallback rule-based answer dibuat lebih agronomis dan tidak terlalu template.
   - Output Kodular mendapatkan group baru `agronomy` pada depth 2.
   - Debounce fingerprint sekarang mempertimbangkan severity dan bucket nilai sensor, sehingga perubahan besar dalam status yang sama tetap bisa memicu refresh LLM.
   - Bug sanitizer punctuation diperbaiki: `test , ok .` menjadi `test, ok.`.

2. `app/retriever.py`
   - Sinonim agronomi diperluas untuk N/P/K, pH, EC, kelembapan tanah, uji tanah, pemupukan berimbang, dan faktor pembatas.
   - Soft filter crop/stage sekarang menerima metadata umum seperti `all`, `general`, `umum`, `semua`, `multi`, dan `unknown`.
   - Rerank memberi bonus untuk preferred manual/SOP/petunjuk teknis dan issue parameters.
   - Kandidat retrieval diperbesar dari `top_k*8/30` menjadi `top_k*12/80` agar dokumen isu spesifik lebih mungkin masuk.

Catatan penggunaan:
- Copy isi folder ini ke struktur proyek Anda.
- Jalankan ulang index: `python build_index.py` atau `python main.py --rebuild`.
- Pastikan package project tetap memiliki struktur `app/` seperti di folder ini.
- Jika OpenRouter digunakan, pastikan dependency `openai` tersedia dan `.env` berisi `OPENROUTER_API_KEY`.

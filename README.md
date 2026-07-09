# RAG PDF OpenRouter — Versi Modular VS Code

Project ini adalah versi modular dari notebook RAG kamu.

Yang dipertahankan:
- PDF dibaca dari folder project.
- Dokumen diekstrak menjadi teks/markdown.
- Teks dipecah menjadi chunk.
- Retrieval menggunakan TF-IDF seperti notebook awal.
- Jawaban dibuat dengan RAG OpenRouter.

Yang dihapus:
- Model QA lokal/manual.
- Google Colab.
- Google Drive.
- Cell notebook interaktif.
- Path `/content/...`.

## Struktur Folder

```text
rag_openrouter_modular/
├── app/
│   ├── config.py
│   ├── pdf_loader.py
│   ├── chunker.py
│   ├── retriever.py
│   ├── openrouter_client.py
│   └── rag_service.py
├── data/
│   └── pdfs/
│       └── taruh_file_pdf_di_sini.pdf
├── storage/
│   ├── markdown/
│   └── index/
├── build_index.py
├── main.py
├── web_app.py
├── requirements.txt
├── .env.example
├── Dockerfile
└── README.md
```

## Cara Menjalankan di VS Code

### 1. Buka folder project

Buka folder ini di VS Code.

### 2. Masukkan PDF

Masukkan file PDF ke folder:

```text
data/pdfs/
```

Contoh:

```text
data/pdfs/jurnal_penelitian.pdf
```

### 3. Buat virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Mac/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 4. Install dependency

```bash
pip install -r requirements.txt
```

### 5. Buat file `.env`

Salin `.env.example` menjadi `.env`, lalu isi:

```text
OPENROUTER_API_KEY=api_key_kamu
OPENROUTER_MODEL=openrouter/free
```

### 6. Bangun index

```bash
python build_index.py
```

### 7. Jalankan mode terminal

```bash
python main.py
```

Atau rebuild index langsung saat menjalankan:

```bash
python main.py --rebuild
```

### 8. Jalankan mode web Streamlit

```bash
streamlit run web_app.py
```

## Cara Kerja Sistem

```text
PDF di data/pdfs
        ↓
PyMuPDF mengekstrak teks
        ↓
Markdown disimpan ke storage/markdown
        ↓
Teks dipecah menjadi chunk
        ↓
TF-IDF membuat index retrieval
        ↓
Pertanyaan mencari chunk relevan
        ↓
Konteks dikirim ke OpenRouter
        ↓
Jawaban dikembalikan beserta sumber
```

## Catatan Penting

Default extractor memakai PyMuPDF agar mudah dijalankan di VS Code dan deploy. Pada notebook awal kamu memakai MinerU. Kalau nanti ingin kembali memakai MinerU untuk PDF kompleks, cukup ubah fungsi `extract_pdf_to_markdown()` di:

```text
app/pdf_loader.py
```

Bagian RAG lain tidak perlu diubah.

## Deploy dengan Docker

Build image:

```bash
docker build -t rag-openrouter .
```

Run:

```bash
docker run -p 8501:8501 --env-file .env rag-openrouter
```

Buka:

```text
http://localhost:8501
```

Untuk service MQTT worker SULTANI, gunakan panduan:

```text
README_DOCKER.md
```

## Tips Akurasi

- Untuk dokumen pendek, `TOP_K=3` biasanya cukup.
- Untuk dokumen panjang, gunakan `TOP_K=5` sampai `8`.
- Jika sistem terlalu mudah menjawab pertanyaan tidak relevan, naikkan `MIN_SCORE`, misalnya `0.08`.
- Jika sistem sering menolak padahal pertanyaan relevan, turunkan `MIN_SCORE`, misalnya `0.03`.

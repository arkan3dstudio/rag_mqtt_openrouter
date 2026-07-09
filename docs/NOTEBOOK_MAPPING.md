# Mapping dari Notebook ke Project Modular

Notebook awal:
- Upload PDF dari Colab
- MinerU ke output markdown
- Baca markdown
- Chunking
- TF-IDF retrieval
- QA lokal
- OpenRouter RAG
- Loop interaktif

Project modular:
- `app/pdf_loader.py`: pengganti upload PDF + ekstraksi markdown
- `app/chunker.py`: pengganti fungsi `chunk_text`
- `app/retriever.py`: pengganti `TfidfVectorizer` dan `retrieve_chunks`
- `app/openrouter_client.py`: pengganti setup `OpenAI(base_url=...)`
- `app/rag_service.py`: pengganti `ask_openrouter_rag`
- `main.py`: pengganti loop interaktif terminal
- `web_app.py`: versi web Streamlit
- `build_index.py`: proses rebuild index

Bagian yang dihapus:
- `LocalQAPipeline`
- `ask_local_qa`
- model `bert-squad-trained`
- pilihan mode QA lokal
- Google Drive mount
- dependency `transformers`, `torch`, `datasets`

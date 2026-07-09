import streamlit as st

from app.config import settings
from app.rag_service import RAGService


st.set_page_config(
    page_title="RAG PDF OpenRouter",
    page_icon="📄",
    layout="wide",
)

st.title("📄 RAG PDF OpenRouter")
st.caption("Versi modular dari notebook: PDF folder → ekstraksi teks → TF-IDF retrieval → OpenRouter RAG.")

service = RAGService()

with st.sidebar:
    st.header("Pengaturan")
    top_k = st.slider("Top K sumber", min_value=1, max_value=10, value=settings.top_k)
    min_score = st.slider(
        "Minimum similarity score",
        min_value=0.0,
        max_value=0.5,
        value=float(settings.min_score),
        step=0.01,
    )
    model = st.text_input("Model OpenRouter", value=settings.openrouter_model)

    st.divider()
    st.write("Folder PDF:")
    st.code(str(settings.pdf_dir), language="text")

    uploaded_files = st.file_uploader(
        "Tambahkan PDF ke folder project",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        settings.pdf_dir.mkdir(parents=True, exist_ok=True)
        for uploaded_file in uploaded_files:
            target_path = settings.pdf_dir / uploaded_file.name
            target_path.write_bytes(uploaded_file.getbuffer())
        st.success("PDF berhasil disimpan. Klik 'Bangun ulang index'.")

    if st.button("Bangun ulang index", use_container_width=True):
        try:
            with st.spinner("Mengekstrak PDF dan membuat index..."):
                result = service.build_index(force_extract=True)
            st.success(
                f"Index berhasil dibuat: {result['total_documents']} dokumen, "
                f"{result['total_chunks']} chunk."
            )
        except Exception as error:
            st.error(str(error))

try:
    service.load_or_build_index()
except Exception as error:
    st.warning(str(error))
    st.stop()

question = st.chat_input("Tulis pertanyaan tentang isi PDF...")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Mengambil konteks dan memanggil OpenRouter..."):
            result = service.ask(
                question=question,
                top_k=top_k,
                min_score=min_score,
                model=model,
            )

        st.markdown(result["answer"])

        with st.expander("Lihat sumber yang digunakan"):
            for index, source in enumerate(result["sources"], start=1):
                st.markdown(
                    f"**Sumber {index}** — `{source['source']}` | "
                    f"chunk `{source['chunk_id']}` | score `{source['score']:.4f}`"
                )
                st.text(source["text"][:1500])

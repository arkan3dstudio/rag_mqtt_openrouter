from app.rag_service import RAGService


def main() -> None:
    service = RAGService()
    result = service.build_index(force_extract=True)

    print("Index berhasil dibuat.")
    print(f"Total dokumen : {result['total_documents']}")
    print(f"Total chunk   : {result['total_chunks']}")
    print(f"Index file    : {result['index_file']}")


if __name__ == "__main__":
    main()

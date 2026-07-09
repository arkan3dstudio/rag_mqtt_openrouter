import argparse

from app.config import settings
from app.rag_service import RAGService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLI RAG PDF OpenRouter")
    parser.add_argument("--rebuild", action="store_true", help="Bangun ulang index dari PDF.")
    parser.add_argument("--top-k", type=int, default=settings.top_k, help="Jumlah chunk sumber.")
    parser.add_argument("--min-score", type=float, default=settings.min_score, help="Threshold relevansi.")
    parser.add_argument("--model", type=str, default=settings.openrouter_model, help="Model OpenRouter.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = RAGService()

    if args.rebuild:
        result = service.build_index(force_extract=True)
        print("Index berhasil dibuat ulang.")
        print(f"Total dokumen : {result['total_documents']}")
        print(f"Total chunk   : {result['total_chunks']}")
    else:
        service.load_or_build_index()

    print("\nSistem RAG OpenRouter siap.")
    print("Ketik pertanyaan lalu tekan Enter.")
    print("Ketik exit / quit / keluar untuk berhenti.\n")

    while True:
        question = input("Anda: ").strip()

        if question.lower() in {"exit", "quit", "keluar"}:
            print("Selesai.")
            break

        if not question:
            continue

        result = service.ask(
            question=question,
            top_k=args.top_k,
            min_score=args.min_score,
            model=args.model,
        )

        print(f"\nMode: {result['mode']}")
        print("\nJawaban:")
        print(result["answer"])

        print("\nSumber:")
        for index, source in enumerate(result["sources"], start=1):
            print(
                f"{index}. {source['source']} | "
                f"chunk {source['chunk_id']} | "
                f"score {source['score']:.4f}"
            )

        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()

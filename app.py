from rag_pipeline import setup_pipeline, answer_question

def main():
    print("=" * 60)
    print("  NOTES Q&A BOT — RAG se powered")
    print("=" * 60)
    vectordb = setup_pipeline()
    print("\nBot ready hai! Apne notes se kuch bhi poocho.")
    print("Exit karne ke liye 'quit' likho.\n")

    while True:
        question = input("Tumhara sawaal: ").strip()
        if question.lower() in ["quit", "exit", "q"]:
            print("Bye!")
            break
        if not question:
            continue
        print("\nSoch raha hoon...\n")
        result = answer_question(vectordb, question)
        print(f"Jawab:\n{result['answer']}\n")
        print("Sources:")
        for i, src in enumerate(result["sources"], 1):
            print(f"  [{i}] {src['content_preview']}")
        print("-" * 60 + "\n")

if __name__ == "__main__":
    main()
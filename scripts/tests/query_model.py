from rag_core import run_rag

if __name__ == "__main__":
    print("RAG CLI. exit=выход, reset=сброс\n")

    history = []

    while True:
        q = input(">>> ").strip()

        if q.lower() in ("exit", "выход", "quit"):
            break

        if q.lower() in ("reset", "сброс"):
            history = []
            print("История очищена.\n")
            continue

        answer, history = run_rag(q, history)

        print("\n===== ОТВЕТ =====")
        print(answer)
        print()

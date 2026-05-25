import sys
from langchain_core.messages import HumanMessage
from src import config
from src.indexing.indexer import build_index
from src.graph.graph import build_rag_graph


def cmd_index():
    print("=== Building Vector Index ===")
    build_index(force_recreate=True)
    print("Done!")


def cmd_ask():
    graph = build_rag_graph()
    messages = []

    status = "enabled" if config.LLM_THINKING_ENABLED else "disabled"
    print("=== Legal RAG Q&A ===")
    print(f"Thinking: {status} | Enter 'quit' to exit, 'clear' to clear history.\n")

    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if question.lower() == "clear":
            messages = []
            print("History cleared.")
            continue
        if not question:
            continue

        messages.append(HumanMessage(content=question))

        result = graph.invoke({
            "messages": messages,
            "current_question": question,
            "expanded_question": "",
            "rewritten_question": "",
            "intent": "",
            "intent_reason": "",
            "retrieved_docs": [],
            "answer": "",
            "references": [],
            "next_questions": [],
            "error": "",
            "blocked": False,
        })

        if result.get("blocked"):
            messages.pop()
            print(f"\nAI: {result['answer']}\n")
            continue

        print(f"\nAI: {result['answer']}\n")

        if result.get("references"):
            print("--- 参考来源 ---")
            for ref in result["references"]:
                print(f"  📖 {ref['source']} (相关度: {ref['score']})")
                print(f"     {ref['preview']}")
            print()

        if result.get("next_questions"):
            print("--- 您可能还想问 ---")
            for i, q in enumerate(result["next_questions"], 1):
                print(f"  {i}. {q}")
            print()

        messages = result["messages"]


def print_usage():
    print("Usage:")
    print("  uv run python -m src.main index")
    print("  uv run python -m src.main ask [--no-think]")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "index":
        cmd_index()
    elif cmd == "ask":
        if "--no-think" in sys.argv:
            config.LLM_THINKING_ENABLED = False
        cmd_ask()
    else:
        print(f"Unknown command: {cmd}")
        print_usage()

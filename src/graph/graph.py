from langgraph.graph import StateGraph, START, END
from src.graph.state import RAGState
from src.graph.nodes import (
    intent_recognition_node,
    reject_non_legal_node,
    context_expansion_node,
    question_rewriting_node,
    retrieve_node,
    generate_answer_node,
    generate_references_node,
    generate_next_questions_node,
    weather_answer_node,
    router,
)


def build_rag_graph():
    builder = StateGraph(RAGState)

    builder.add_node("context_expansion", context_expansion_node)
    builder.add_node("intent_recognition", intent_recognition_node)
    builder.add_node("reject_non_legal", reject_non_legal_node)
    builder.add_node("weather_answer", weather_answer_node)
    builder.add_node("question_rewriting", question_rewriting_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate_answer", generate_answer_node)
    builder.add_node("generate_references", generate_references_node)
    builder.add_node("generate_next_questions", generate_next_questions_node)

    builder.add_edge(START, "context_expansion")
    builder.add_edge("context_expansion", "intent_recognition")

    builder.add_conditional_edges(
        "intent_recognition",
        router,
        {
            "reject_non_legal": "reject_non_legal",
            "weather_answer": "weather_answer",
            "question_rewriting": "question_rewriting",
        },
    )
    builder.add_edge("reject_non_legal", END)
    builder.add_edge("weather_answer", END)
    builder.add_edge("question_rewriting", "retrieve")
    builder.add_edge("retrieve", "generate_answer")
    builder.add_edge("generate_answer", "generate_references")
    builder.add_edge("generate_references", "generate_next_questions")
    builder.add_edge("generate_next_questions", END)

    return builder.compile()

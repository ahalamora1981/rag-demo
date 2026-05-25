from typing import TypedDict, List, Optional, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class RAGState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    current_question: str
    expanded_question: str
    rewritten_question: str
    intent: str
    intent_reason: str
    retrieved_docs: List[dict]
    answer: str
    references: List[dict]
    next_questions: List[str]
    error: str
    blocked: bool

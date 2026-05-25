import json
import urllib.request
import urllib.parse
from datetime import date
from typing import Literal, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.outputs import ChatResult
from langchain_core.callbacks import CallbackManagerForLLMRun
from src.graph.state import RAGState
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from tavily import TavilyClient
from src import config
from src.embeddings import create_embeddings
from src.indexing.indexer import get_vectorstore
from src.timer import timer

TODAY = date.today().strftime("%Y年%m月%d日")


class DeepSeekChatOpenAI(ChatOpenAI):
    """ChatOpenAI subclass that supports DeepSeek thinking mode via extra_body."""

    thinking_type: Optional[str] = None  # "enabled" | "disabled"

    def _generate(
        self,
        messages: list,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> ChatResult:
        self._ensure_sync_client_available()
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        extra = {}
        if self.thinking_type:
            extra["extra_body"] = {"thinking": {"type": self.thinking_type}}
        raw_response = self.client.with_raw_response.create(**payload, **extra)
        return self._create_chat_result(raw_response.parse())


def _get_llm(reasoning=False, disable_thinking=False):
    kwargs = {
        "model": config.LLM_MODEL,
        "base_url": config.LLM_BASE_URL,
        "api_key": config.LLM_API_KEY,
        "temperature": 0.1,
    }
    if reasoning:
        if config.LLM_THINKING_ENABLED:
            kwargs["reasoning_effort"] = config.LLM_REASONING_EFFORT
            kwargs["thinking_type"] = "enabled"
        else:
            kwargs["thinking_type"] = "disabled"
    elif disable_thinking:
        kwargs["thinking_type"] = "disabled"
    return DeepSeekChatOpenAI(**kwargs)


@tool
@timer("get_weather")
def get_weather(location: str = "") -> str:
    """获取当前天气信息。如果不指定location，则基于IP自动定位。"""
    try:
        loc = urllib.parse.quote(location.strip()) if location else ""
        url = f"https://wttr.in/{loc}?format=j1"
        req = urllib.request.Request(url, headers={"User-Agent": "curl"})
        data = json.loads(urllib.request.urlopen(req, timeout=10).read().decode())
        current = data["current_condition"][0]
        area = data["nearest_area"][0]["areaName"][0]["value"]
        weather_desc = current["weatherDesc"][0]["value"]
        temp_c = current["temp_C"]
        feels = current["FeelsLikeC"]
        humidity = current["humidity"]
        wind = f"{current['winddir16Point']} {current['windspeedKmph']}km/h"
        return (
            f"位置：{area}\n"
            f"天气：{weather_desc}\n"
            f"温度：{temp_c}°C（体感 {feels}°C）\n"
            f"湿度：{humidity}%\n"
            f"风向风速：{wind}"
        )
    except Exception as e:
        return f"获取天气失败: {e}"


@tool
@timer("search_web")
def search_web(query: str) -> str:
    """搜索互联网获取实时信息。适用于新闻、百科、时事、人物、科技等需要联网查询的问题。
    返回结果的标题和内容摘要。"""
    try:
        client = TavilyClient(api_key=config.TAVILY_API_KEY)
        result = client.search(query, search_depth="basic", max_results=5)
        lines = []
        for r in result.get("results", []):
            lines.append(f"- {r['title']}\n  {r['content']}\n  来源: {r['url']}")
        return "\n\n".join(lines) if lines else "未找到相关结果"
    except Exception as e:
        return f"搜索失败: {e}"


@timer("intent_recognition")
def intent_recognition_node(state: RAGState) -> dict:
    llm = _get_llm()
    question = state.get("expanded_question") or state["current_question"]

    prompt = f"""你是一个法律问答系统的意图识别器。
判断用户问题的类型：

legal - 劳动法、劳动合同法、民法典、消费者权益保护法、合同纠纷、婚姻家庭、继承、侵权、物权、债权、劳动争议、消费维权等
chat  - 天气、笑话、日常闲聊等其他非法律问题

请以JSON格式回答，不要其他内容：
{{
    "intent": "legal/chat",
    "reason": "简短判断理由"
}}

用户问题：{question}"""

    resp = llm.invoke([HumanMessage(content=prompt)])
    try:
        result = json.loads(resp.content.strip().removeprefix("```json").removesuffix("```").strip())
    except json.JSONDecodeError:
        result = {"intent": "legal", "reason": "解析失败，默认按法律问题处理"}

    return {
        "intent": result.get("intent", "legal"),
        "intent_reason": result.get("reason", ""),
    }


@timer("chat")
def chat_node(state: RAGState) -> dict:
    llm = _get_llm(disable_thinking=True)
    question = state.get("expanded_question") or state["current_question"]

    system = SystemMessage(content=(
        f"今天是{TODAY}。\n"
        "你是一个友好的中文助手。你可以使用工具获取实时信息：\n"
        "- get_weather: 查询天气（用户不指定城市时 location 留空）\n"
        "- search_web: 搜索互联网获取新闻、百科、实时数据"
    ))
    llm_with_tools = llm.bind_tools([get_weather, search_web])
    msgs = [system, HumanMessage(content=question)]

    for _ in range(5):
        resp = llm_with_tools.invoke(msgs)
        if not resp.tool_calls:
            break
        msgs.append(resp)
        for tc in resp.tool_calls:
            name = tc["name"]
            if name == "get_weather":
                result = get_weather.invoke(tc["args"])
            elif name == "search_web":
                result = search_web.invoke(tc["args"])
            else:
                result = f"未知工具: {name}"
            msgs.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    return {"answer": resp.content.strip()}


@timer("context_expansion")
def context_expansion_node(state: RAGState) -> dict:
    if len(state["messages"]) <= 1:
        return {"expanded_question": state["current_question"]}

    llm = _get_llm()
    history_text = "\n".join(
        f"{'用户' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
        for m in state["messages"][-4:-1]
    )

    prompt = f"""下面是对话历史，以及用户的最新问题。
请结合历史上下文，将最新问题改写为一个独立、完整的单轮问题，
确保不依赖上下文也能理解。

对话历史：
{history_text}

最新问题：{state["current_question"]}

请只输出改写后的问题，不要其他内容。"""

    resp = llm.invoke([HumanMessage(content=prompt)])
    return {"expanded_question": resp.content.strip()}


@timer("question_rewriting")
def question_rewriting_node(state: RAGState) -> dict:
    llm = _get_llm()
    question = state.get("expanded_question") or state["current_question"]

    prompt = f"""你是一个法律检索优化专家。
请将用户的问题改写成更适合向量检索的形式，要求：
1. 提取关键法律概念和关键词
2. 去除指代和上下文依赖
3. 保持法律术语的准确性
4. 输出简洁的检索语句

原问题：{question}

请只输出改写后的检索语句，不要其他内容。"""

    resp = llm.invoke([HumanMessage(content=prompt)])
    return {"rewritten_question": resp.content.strip()}


@timer("retrieve")
def retrieve_node(state: RAGState) -> dict:
    try:
        vectorstore = get_vectorstore()
    except Exception as e:
        return {"error": f"Failed to connect to Chroma: {e}", "retrieved_docs": []}

    query = state.get("rewritten_question") or state.get("expanded_question") or state["current_question"]

    results = vectorstore.similarity_search_with_score(query, k=config.RETRIEVAL_K)

    docs = []
    for doc, score in results:
        docs.append({
            "content": doc.page_content,
            "law_name": doc.metadata.get("law_name", ""),
            "article_range": doc.metadata.get("article_range", ""),
            "source_url": doc.metadata.get("source_url", ""),
            "score": round(score, 4),
        })

    return {"retrieved_docs": docs}


@timer("generate_answer")
def generate_answer_node(state: RAGState) -> dict:
    llm = _get_llm(reasoning=True)
    docs = state.get("retrieved_docs", [])
    question = state.get("expanded_question") or state["current_question"]

    if not docs:
        return {"answer": "抱歉，在法律法规库中没有找到相关内容。请尝试换一种方式提问。"}

    context_parts = []
    for i, d in enumerate(docs, 1):
        ctx = f"[{i}] 来源：《{d['law_name']}》"
        if d["article_range"]:
            ctx += f" {d['article_range']}"
        ctx += f"\n{d['content']}"
        context_parts.append(ctx)

    context = "\n\n".join(context_parts)

    prompt = f"""你是一个专业法律顾问。今天是{TODAY}。请基于以下法律法规内容回答用户问题。

要求：
1. 回答应当准确引用法律条文
2. 如果信息不足，明确告知用户
3. 用通俗易懂的中文解释
4. 标注引用来源（如：根据《劳动法》第X条）

参考法律条文：
{context}

用户问题：{question}

请给出专业回答："""

    resp = llm.invoke([HumanMessage(content=prompt)])
    return {"answer": resp.content.strip()}


@timer("generate_references")
def generate_references_node(state: RAGState) -> dict:
    docs = state.get("retrieved_docs", [])
    references = []

    for d in docs:
        law_name = d.get("law_name", "未知")
        article = d.get("article_range", "")
        preview = d.get("content", "")[:config.REFERENCE_PREVIEW_CHARS].replace("\n", "")
        source = f"《{law_name}》"
        if article:
            source += f" {article}"
        references.append({
            "source": source,
            "url": d.get("source_url", ""),
            "preview": f"{preview}...",
            "score": d.get("score", 0),
        })

    return {"references": references}


@timer("generate_next_questions")
def generate_next_questions_node(state: RAGState) -> dict:
    llm = _get_llm()
    question = state.get("expanded_question") or state["current_question"]
    answer = state.get("answer", "")
    docs = state.get("retrieved_docs", [])

    law_names = list(set(d.get("law_name", "") for d in docs))
    law_context = "、".join(law_names) if law_names else "相关法律"

    prompt = f"""基于以下问答内容，生成3个用户可能继续追问的法律相关问题。
每个问题应当简短、独立、有实际参考价值。

用户问题：{question}
回答：{answer[:300]}
涉及法律：{law_context}

请以JSON数组格式输出，例如：
["问题1", "问题2", "问题3"]

只输出JSON数组，不要其他内容。"""

    resp = llm.invoke([HumanMessage(content=prompt)])
    try:
        questions = json.loads(resp.content.strip().removeprefix("```json").removesuffix("```").strip())
        if not isinstance(questions, list):
            questions = []
    except (json.JSONDecodeError, ValueError):
        questions = []

    return {"next_questions": questions[:3]}


def router(state: RAGState) -> Literal["chat", "question_rewriting"]:
    if state["intent"] == "chat":
        return "chat"
    return "question_rewriting"

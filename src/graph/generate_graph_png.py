from pathlib import Path
from langchain_core.runnables.graph_mermaid import draw_mermaid_png
from src.graph.graph import build_rag_graph

output = Path(__file__).parent.parent.parent / "graph_rag.png"

graph = build_rag_graph()
mermaid = graph.get_graph().draw_mermaid()

tool_block = """
	subgraph tools[\"🔧 chat 可用工具\"]
		get_weather(\"🌤 get_weather<br/>wttr.in\")
		search_web(\"🔍 search_web<br/>Tavily\")
	end
	chat --> get_weather;
	chat --> search_web;
"""

insert_before = "classDef default"
mermaid = mermaid.replace(insert_before, tool_block + "\n\t" + insert_before)

draw_mermaid_png(
    mermaid_syntax=mermaid,
    output_file_path=str(output),
    background_color="white",
)
print(f"Saved to {output}")

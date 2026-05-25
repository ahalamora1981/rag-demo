from pathlib import Path
from src.graph.graph import build_rag_graph

output = Path(__file__).parent.parent.parent / "graph_rag.png"
graph = build_rag_graph()
graph.get_graph().draw_mermaid_png(output_file_path=str(output))
print(f"Saved to {output}")

from langgraph.graph import END, START, StateGraph

from src.nodes_edges import _load_document, _split_document, _generate_summary, _group_partial_summaries, _collapse_summaries, _generate_final_summary, _map_input, _map_documents, _map_chunks, _should_collapse
from src.states import InputState, OverallState, OutputState


# Define the graph
builder = StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)

# Add nodes
builder.add_node("load_document", _load_document)
builder.add_node("split_document", _split_document)
builder.add_node("generate_summary", _generate_summary)
builder.add_node("group_partial_summaries", _group_partial_summaries)
builder.add_node("collapse_summaries", _collapse_summaries)
builder.add_node("generate_final_summary", _generate_final_summary)

# Add edges with conditional routing
builder.add_conditional_edges(START, _map_input, ["load_document"])
builder.add_conditional_edges("load_document", _map_documents, ["split_document"])
builder.add_conditional_edges("split_document", _map_chunks, ["generate_summary"])
builder.add_edge("generate_summary", "group_partial_summaries")
builder.add_conditional_edges("group_partial_summaries", _should_collapse, ["collapse_summaries", "generate_final_summary"])
builder.add_conditional_edges("collapse_summaries", _should_collapse, ["collapse_summaries", "generate_final_summary"])
builder.add_edge("generate_final_summary", END)

# Compile the graph
graph = builder.compile(
    interrupt_before=[],  # Add nodes here if you want to update state before execution
    interrupt_after=[],   # Add nodes here if you want to update state after execution
)
graph.name = "DocumentSummarizationGraph"

# Create main agent instance
app = graph
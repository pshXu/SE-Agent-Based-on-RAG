from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Literal

from .state import GraphState
from .nodes.router import route
from .nodes.se_process import run as se_run
from .nodes.synthesizer import run as syn_run
from .nodes.summarizer import run as sum_run

def router_condition(state: GraphState) -> Literal["se_process", "synthesizer"]:
    """
    Conditional logic to determine the next node based on the router's output.
    """
    next_step = state.get("next_step", "synthesizer")
    
    if next_step == "se_process":
        return "se_process"
    else:
        # Fallback to synthesizer for general queries or if unknown step
        return "synthesizer"

# 1. Initialize the Graph
workflow = StateGraph(GraphState)

# 2. Add Nodes
workflow.add_node("router", route)
workflow.add_node("se_process", se_run)
workflow.add_node("synthesizer", syn_run)
workflow.add_node("summarizer", sum_run)

# 3. Define Edges
# Entry point
workflow.set_entry_point("router")

# Conditional edges from Router
workflow.add_conditional_edges(
    "router",
    router_condition,
    {
        "se_process": "se_process",
        "synthesizer": "synthesizer"
    }
)

# Normal edges to Synthesizer
workflow.add_edge("se_process", "synthesizer")

# Synthesizer -> Summarizer -> End
workflow.add_edge("synthesizer", "summarizer")
workflow.add_edge("summarizer", END)

# 4. Initialize Checkpointer for In-Memory Memory
# This keeps the state in memory for the duration of the session
memory = MemorySaver()

# 5. Compile the Graph with checkpointer
app = workflow.compile(checkpointer=memory)

def execute(initial_state: GraphState, config: dict):
    """
    Executes the compiled graph with the initial state and thread config.
    """
    return app.invoke(initial_state, config=config)

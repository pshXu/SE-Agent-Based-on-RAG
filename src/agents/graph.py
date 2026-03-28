from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Literal

from .state import GraphState
from .nodes.router import route
from .nodes.planner import run as planner_run
from .nodes.validator import run as validator_run
from .nodes.executor import run as executor_run
from .nodes.synthesizer import run as syn_run
from .nodes.summarizer import run as sum_run

def router_condition(state: GraphState) -> Literal["planner", "synthesizer"]:
    """
    Conditional logic to determine the next node based on the router's output.
    """
    next_step = state.get("next_step", "synthesizer")
    
    if next_step == "se_process":
        return "planner"
    else:
        return "synthesizer"

def validator_condition(state: GraphState) -> Literal["planner", "executor"]:
    """
    Determines whether to go back to planner for reflection or proceed to execution.
    Used by the Validator node.
    """
    return state.get("next_step", "executor")

def planning_complexity_condition(state: GraphState) -> Literal["validator", "executor"]:
    """
    Decides whether to validate the plan based on its complexity.
    If the plan has only one query (simple), skip validation to save time.
    """
    plan = state.get("plan", [])
    if len(plan) > 1:
        return "validator"
    return "executor"

# 1. Initialize the Graph
workflow = StateGraph(GraphState)

# 2. Add Nodes
workflow.add_node("router", route)
workflow.add_node("planner", planner_run)
workflow.add_node("validator", validator_run)
workflow.add_node("executor", executor_run)
workflow.add_node("synthesizer", syn_run)
workflow.add_node("summarizer", sum_run)

# 3. Define Edges
workflow.set_entry_point("router")

# Conditional edges from Router
workflow.add_conditional_edges(
    "router",
    router_condition,
    {
        "planner": "planner",
        "synthesizer": "synthesizer"
    }
)

# Planner -> Validator (if complex) OR Executor (if simple)
workflow.add_conditional_edges(
    "planner",
    planning_complexity_condition,
    {
        "validator": "validator",
        "executor": "executor"
    }
)

# Planning -> Validation Loop (only used if in validator node)
workflow.add_edge("validator", "planner") # This edge is only taken if validator returns 'planner' via conditional edge below

# Replace the simple edge with conditional one for the loop
workflow.add_conditional_edges(
    "validator",
    validator_condition,
    {
        "planner": "planner",
        "executor": "executor"
    }
)

# Execution -> Synthesis
workflow.add_edge("executor", "synthesizer")

# Synthesizer -> Summarizer -> End
workflow.add_edge("synthesizer", "summarizer")
workflow.add_edge("summarizer", END)

# 4. Initialize Checkpointer
memory = MemorySaver()

# 5. Compile
app = workflow.compile(checkpointer=memory)

def execute(initial_state: GraphState, config: dict):
    return app.invoke(initial_state, config=config)

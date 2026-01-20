from typing import List, TypedDict, Annotated, Optional
import operator
from langchain_core.messages import BaseMessage

class GraphState(TypedDict):
    """
    Represents the state of our multi-agent graph.
    """
    # The original user query
    query: str
    
    # A list of messages in the conversation, using operator.add to append new messages
    messages: Annotated[List[BaseMessage], operator.add]
    
    # Stores the output from the SE Process Agent (context/answers)
    process_output: List[str]
    
    # The final answer produced by the Synthesizer
    final_answer: str
    
    # Control variable to decide the next node in the graph
    next_step: Optional[str]
    
    # Summary of the conversation so far (Long-term context)
    summary: str

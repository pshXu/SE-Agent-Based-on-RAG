import logging
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

from ..state import GraphState
from src.utils.llm_factory import get_llm
from config.prompts import ROUTER_SYSTEM_PROMPT

def _format_chat_history(messages) -> str:
    """Helper to format the chat history for the prompt."""
    if not messages:
        return "（无近期对话）"
    
    formatted_history = []
    # Take the last 5 messages to keep context relevant and concise
    for msg in messages[-5:]:
        if isinstance(msg, HumanMessage):
            formatted_history.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted_history.append(f"Agent: {msg.content}")
            
    return "\n".join(formatted_history)

def route(state: GraphState) -> GraphState:
    """
    Analyzes the user's query and routes it to the appropriate agent.
    Updates the 'next_step' key in the state.
    """
    print("--- Router: Analyzing Query Intent ---")
    query = state["query"]
    messages = state.get("messages", [])
    summary = state.get("summary", "（无历史摘要）")
    
    chat_history_str = _format_chat_history(messages)
    
    llm = get_llm()
    
    # Updated PromptTemplate to include summary and chat_history
    prompt = PromptTemplate(
        template="{system_prompt}\n\n{summary}\n\n{chat_history}\n\n用户当前查询：{query}",
        input_variables=["system_prompt", "summary", "chat_history", "query"]
    )
    
    chain = prompt | llm | StrOutputParser()
    
    next_step = "synthesizer" # Default fallback
    
    try:
        # Invoke the LLM to classify the intent
        intent = chain.invoke({
            "system_prompt": ROUTER_SYSTEM_PROMPT,
            "summary": f"【长期记忆】：{summary}",
            "chat_history": f"【近期对话】：{chat_history_str}",
            "query": query
        })
        
        # Clean up the output
        intent = intent.strip().lower()
        print(f"Router intent classification: {intent}")
        
        if "process" in intent:
            next_step = "se_process"
        else:
            # Fallback to synthesizer for general queries or unclear intent
            next_step = "synthesizer"
            
    except Exception as e:
        logging.error(f"Error in router: {e}")
        next_step = "synthesizer"
        
    print(f"Routing to: {next_step}")
    return {"next_step": next_step}

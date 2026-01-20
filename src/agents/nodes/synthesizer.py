import logging
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage

from ..state import GraphState
from src.utils.llm_factory import get_llm
from config.prompts import SYNTHESIZER_CRITIQUE_PROMPT

def run(state: GraphState) -> GraphState:
    """
    Synthesizes the outputs from various agents into a final, coherent answer.
    Performs self-critique and correction.
    """
    logging.info("--- Synthesizer: Generating Final Answer ---")
    
    query = state["query"]
    process_output = state.get("process_output", [])
    
    # Flatten lists to strings
    process_str = "\n\n".join(process_output) if process_output else "（无）"
    
    llm = get_llm()
    
    prompt = PromptTemplate(
        template=SYNTHESIZER_CRITIQUE_PROMPT,
        input_variables=["query", "process_output"]
    )
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        answer = chain.invoke({
            "query": query,
            "process_output": process_str
        })
        logging.info("Final answer synthesized.")
    except Exception as e:
        logging.error(f"Error in synthesizer: {e}")
        answer = "抱歉，在生成最终回答时遇到了问题。"
    
    # Update both final_answer and the messages list for persistent history
    return {
        "final_answer": answer,
        "messages": [AIMessage(content=str(answer))]
    }

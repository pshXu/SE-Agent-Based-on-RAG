import logging
import re
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage

from ..state import GraphState
from src.utils.llm_factory import get_llm
from config.prompts import SYNTHESIZER_CRITIQUE_PROMPT

def _format_chat_history(messages) -> str:
    """Helper to format recent messages for the synthesizer's context."""
    if not messages: return "（无近期对话）"
    formatted = []
    # Only take last 5 to keep context focused
    for msg in messages[-5:]:
        if isinstance(msg, HumanMessage): formatted.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage): formatted.append(f"Agent: {msg.content}")
    return "\n".join(formatted)

def run(state: GraphState) -> GraphState:
    """
    Synthesizes the outputs from various agents into a final, coherent answer.
    Distinguishes between 'Reference Docs' and 'Expert Draft' for precise critique.
    """
    logging.info("--- Synthesizer: Generating Final Answer ---")
    
    query = state["query"]
    process_output = state.get("process_output", [])
    summary = state.get("summary", "（无历史摘要）")
    messages = state.get("messages", [])
    
    # 1. 显式解析证据与初稿
    # 我们知道 se_process 存入的顺序是：[..., 原始文档, 专家初稿]
    reference_docs = "（无原始文档）"
    expert_draft = "（无专家初稿）"
    
    for item in process_output:
        if "--- 原始参考文档 ---" in item:
            reference_docs = item.replace("--- 原始参考文档 ---", "").strip()
        elif "--- 专家初稿答案 ---" in item:
            expert_draft = item.replace("--- 专家初稿答案 ---", "").strip()
    
    # 2. Format chat history
    chat_history_str = _format_chat_history(messages)
    
    llm = get_llm()
    
    # 3. 构建综合专家输出，明确区分事实来源与专家意见
    # 这样 Prompt 中的 "根据下方提供的【专家输出】" 就能包含所有维度
    combined_expert_input = f"""【原始证据库（含本地与网络）】：
{reference_docs}

【专家建议初稿】：
{expert_draft}"""

    prompt = PromptTemplate(
        template=SYNTHESIZER_CRITIQUE_PROMPT,
        input_variables=["query", "process_output", "summary", "chat_history"]
    )
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        # 4. Invoke with restructured input
        answer = chain.invoke({
            "query": query,
            "process_output": combined_expert_input,
            "summary": summary,
            "chat_history": chat_history_str
        })
        logging.info("Final answer synthesized.")
    except Exception as e:
        logging.error(f"Error in synthesizer: {e}")
        answer = "抱歉，在生成最终回答时遇到了问题。"
    
    current_intent = state.get("current_intent", "general")
    
    # Update both final_answer and the messages list for persistent history
    return {
        "final_answer": answer,
        "messages": [AIMessage(content=str(answer), additional_kwargs={"intent": current_intent})]
    }

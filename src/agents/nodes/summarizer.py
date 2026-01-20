import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import RemoveMessage, HumanMessage, AIMessage

from ..state import GraphState
from src.utils.llm_factory import get_llm

def run(state: GraphState) -> GraphState:
    """
    Summarizes relevant SE technical content and removes old messages.
    Implements selective memory: filters out small talk and unrelated chat.
    """
    logging.info("--- Summarizer: Processing Selective Memory ---")
    
    summary = state.get("summary", "")
    messages = state.get("messages", [])
    
    # Threshold for compression
    if len(messages) <= 6:
        return {}
    
    logging.info("Context length exceeded. Starting semantic compression...")
    
    # Summarize history except the last turn (2 messages)
    messages_to_summarize = messages[:-2]
    
    history_str = ""
    for msg in messages_to_summarize:
        if isinstance(msg, HumanMessage):
            history_str += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            history_str += f"Agent: {msg.content}\n"
            
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_template(
        """你是一个知识精炼专家。你的任务是将对话历史整合进现有的摘要中。
        
        【重要规则】：
        1. **选择性记忆**：只提取和精炼与软件工程（SE）流程、规范、文档、技术定义等相关的实质性信息。
        2. **过滤噪音**：必须完全忽略寒暄、闲聊（如天气、心情）、简单的致谢或无意义的对话片段。
        3. **保持连贯**：如果新增内容包含 SE 知识点，请将其与【现有摘要】有机融合。
        4. **静默机制**：如果【新增对话】中没有任何软件工程相关的有价值信息，请原样返回【现有摘要】，不要添加任何新内容。
        
        【现有摘要】：
        {summary}
        
        【待处理的新增对话】：
        {new_lines}
        
        请输出更新后的摘要："""
    )
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        new_summary = chain.invoke({"summary": summary, "new_lines": history_str})
        logging.info("Selective summary updated.")
    except Exception as e:
        logging.error(f"Error in selective summarization: {e}")
        return {}
    
    # Still remove the messages to free up context window, regardless of whether the summary grew.
    delete_messages = [RemoveMessage(id=m.id) for m in messages_to_summarize]
    
    return {
        "summary": new_summary,
        "messages": delete_messages
    }

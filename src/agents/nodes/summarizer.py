import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import RemoveMessage, HumanMessage, AIMessage

from ..state import GraphState
from src.utils.llm_factory import get_llm

def run(state: GraphState) -> GraphState:
    """
    Summarizes relevant SE technical content and removes old messages.
    Filters out 'general' (chit-chat) messages using intent labels from metadata.
    """
    logging.info("--- Summarizer: Processing Intent-Based Memory ---")
    
    summary = state.get("summary", "")
    messages = state.get("messages", [])
    
    # 阈值设置：当累积超过 30 条消息（约 15 轮对话）时触发压缩
    if len(messages) <= 30:
        return {}
    
    logging.info(f"Context window ({len(messages)} messages) is large. Summarizing technical history...")
    
    # 1. 确定待处理的消息范围（保留最近 10 轮，即 20 条消息）
    # 压缩 20 条之前的所有老消息
    messages_to_process = messages[:-20]
    
    # 2. 意图过滤：仅保留 SE 相关内容进行总结
    history_to_summarize = ""
    for msg in messages_to_process:
        # 只有在 router 中被标记为 'process' 的消息才参与总结
        # 注意：HumanMessage 本身可能没有 metadata，通常通过其后的 AIMessage 来判断这一轮的意图
        intent = msg.additional_kwargs.get("intent", "process") 
        
        if intent == "process":
            if isinstance(msg, HumanMessage):
                history_to_summarize += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                history_to_summarize += f"Agent: {msg.content}\n"
        else:
            logging.info(f"Skipping 'general' message from summary: {msg.content[:30]}...")
            
    # 3. 如果没有实质性 SE 知识，直接删除消息而不调用 LLM
    if not history_to_summarize.strip():
        logging.info("No SE technical content found in the processed window. Skipping LLM call.")
        return {
            "summary": summary,
            "messages": [RemoveMessage(id=m.id) for m in messages_to_process]
        }
            
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_template(
        """你是一个知识精炼专家。你的任务是将对话历史整合进现有的摘要中。
        
        【重要规则】：
        1. **选择性记忆**：只提取和精炼与软件工程（SE）流程、规范、文档、技术定义等相关的实质性信息。
        2. **保持连贯**：如果新增内容包含 SE 知识点，请将其与【现有摘要】有机融合。
        3. **格式要求**：保持专业、事实性的陈述。
        
        【现有摘要】：
        {summary}
        
        【待处理的新增对话（已过滤闲聊）】：
        {new_lines}
        
        请输出更新后的摘要："""
    )
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        new_summary = chain.invoke({"summary": summary, "new_lines": history_to_summarize})
        logging.info("Intent-based summary updated.")
    except Exception as e:
        logging.error(f"Error in selective summarization: {e}")
        return {}
    
    # 4. 生成删除指令，清理已处理的消息（无论是否总结）
    delete_messages = [RemoveMessage(id=m.id) for m in messages_to_process]
    
    return {
        "summary": new_summary,
        "messages": delete_messages
    }

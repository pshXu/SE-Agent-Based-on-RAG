import logging
from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

from ..state import GraphState
from src.utils.llm_factory import get_llm
from config.prompts import SE_PROCESS_PROMPT

class SubQuery(BaseModel):
    query: str = Field(description="具体的子查询问题")
    source: Literal["local", "web", "both"] = Field(description="该子查询的最佳检索来源")

class QueryPlan(BaseModel):
    sub_queries: List[SubQuery] = Field(description="拆解后的子查询列表")

def run(state: GraphState) -> GraphState:
    """
    Planner Node: Decomposes the query into sub-queries.
    """
    logging.info("--- Planner: Generating Query Plan ---")
    query = state["query"]
    chat_history = state.get("messages", [])
    summary = state.get("summary", "（无历史摘要）")
    feedback = state.get("planning_feedback", "")
    retry_count = state.get("retry_count", 0)

    llm = get_llm().with_structured_output(QueryPlan)
    
    # 格式化对话历史
    history_str = ""
    for msg in chat_history[-5:]:
        if isinstance(msg, HumanMessage): history_str += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage): history_str += f"Agent: {msg.content}\n"

    # 如果有反馈，要在 Prompt 中体现
    feedback_str = f"\n\n【上一次计划未通过验证，原因如下】：\n{feedback}\n请根据反馈改进你的拆解计划。" if feedback else ""

    prompt = ChatPromptTemplate.from_messages([
        ("system", SE_PROCESS_PROMPT),
        ("human", "请根据我的问题进行【显式拆解】，并为每个子查询指定 source (local/web/both)。{feedback}\n\n当前问题：{query}")
    ])
    
    try:
        result = llm.invoke({
            "query": query, 
            "summary": summary, 
            "chat_history": history_str or "（无近期对话）",
            "feedback": feedback_str
        })
        
        # 将 Pydantic 对象转换为字典列表以便存储在状态中
        plan_dicts = [{"query": sq.query, "source": sq.source} for sq in result.sub_queries]
        return {"plan": plan_dicts}
    except Exception as e:
        logging.error(f"Error in planner: {e}")
        return {"plan": [{"query": query, "source": "both"}]}

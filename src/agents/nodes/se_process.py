import logging
from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from ..state import GraphState
from src.rag.retriever import HybridRetriever
from src.rag.vector_db import get_vector_store
from src.tools.search_tool import search_documents
from src.utils.llm_factory import get_llm
from config.settings import BGE_MODEL_NAME
from config.prompts import SE_PROCESS_PROMPT

# Global variable to cache the retriever instance
_cached_retriever = None

def _get_retriever():
    global _cached_retriever
    if _cached_retriever is None:
        logging.info("Initializing HybridRetriever for SE Process Agent...")
        embedding_function = HuggingFaceEmbeddings(
            model_name=BGE_MODEL_NAME,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        vector_store = get_vector_store(embedding_function=embedding_function)
        _cached_retriever = HybridRetriever(vector_store=vector_store)
        logging.info("HybridRetriever initialized.")
    return _cached_retriever

# --- Data Models ---
class SubQuery(BaseModel):
    query: str = Field(description="具体的子查询问题")
    source: Literal["local", "web", "both"] = Field(description="该子查询的最佳检索来源")

class QueryPlan(BaseModel):
    """Output structure for query decomposition."""
    sub_queries: List[SubQuery] = Field(description="拆解后的子查询列表")

# --- Helper Functions ---
def _decompose_query(query: str, chat_history: List[BaseMessage], summary: str) -> List[SubQuery]:
    """
    Decomposes a complex query into simpler sub-queries with assigned sources.
    """
    logging.info(f"Decomposing query: {query}")
    llm = get_llm().with_structured_output(QueryPlan)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个问题拆解与规划专家。你的任务是分析用户的软件工程咨询请求。
        
        【长期记忆（摘要）】：
        {summary}
        
        1. **拆解**：如果问题复杂（涉及对比、多阶段、跨领域），拆解为 2-3 个独立子查询。如果简单，保持原样。
        2. **定源**：为每个子查询指定检索来源：
           - 'local': 关于标准流程、定义、学院规范、文档模板。
           - 'web': 关于最新趋势、行业新闻、具体工具的最新版本。
           - 'both': 如果不确定，或者两者都需要参考。
        
        请结合【历史对话】来消解代词。"""),
        ("placeholder", "{messages}"),
        ("human", "{query}")
    ])
    
    try:
        chain = prompt | llm
        result = chain.invoke({"query": query, "messages": chat_history, "summary": summary})
        
        fallback = SubQuery(query=query, source="local")
        if not any(sq.query == query for sq in result.sub_queries):
             result.sub_queries.append(fallback)
             
        logging.info(f"Plan generated: {result.sub_queries}")
        return result.sub_queries
    except Exception as e:
        logging.error(f"Error in decomposition: {e}")
        return [SubQuery(query=query, source="both")]

def run(state: GraphState) -> GraphState:
    """
    SE Process Agent Node: Implements Explicit Decomposition + Parallel Retrieval + Global Rerank.
    """
    logging.info("--- SE Process Agent: Starting Advanced Retrieval Pipeline ---")
    query = state["query"]
    chat_history = [m for m in state.get("messages", []) if isinstance(m, BaseMessage)]
    summary = state.get("summary", "（无历史摘要）")
    
    # 1. Decompose
    plan = _decompose_query(query, chat_history, summary)
    
    # 2. Parallel Retrieval (Conceptually parallel, sequential in loop for simplicity)
    all_documents = []
    retriever = _get_retriever()
    
    for item in plan:
        sub_q = item.query
        src = item.source
        logging.info(f"Fetching '{sub_q}' from {src}")
        
        try:
            # Local Retrieval
            if src in ["local", "both"]:
                # Note: retrieve() already does a mini-rerank, which is fine.
                # We collect these high-quality candidates.
                local_docs = retriever.retrieve(sub_q, k_final=5)
                all_documents.extend(local_docs)
            
            # Web Retrieval
            if src in ["web", "both"]:
                web_docs = search_documents(sub_q, max_results=3)
                all_documents.extend(web_docs)
                
        except Exception as e:
            logging.error(f"Error retrieving for {sub_q}: {e}")

    # 3. Deduplicate
    # Use dictionary comprehension to dedup by page_content
    unique_docs_map = {doc.page_content: doc for doc in all_documents}
    unique_docs = list(unique_docs_map.values())
    logging.info(f"Collected {len(all_documents)} docs, {len(unique_docs)} unique.")

    # 4. Global Re-ranking
    # Rerank all candidates against the ORIGINAL query to find the most relevant ones for the final answer
    if unique_docs:
        top_docs = retriever.rerank_documents(query, unique_docs, k=8) # Context window allows ~8 chunks
    else:
        top_docs = []
        
    logging.info(f"Top {len(top_docs)} docs selected for generation.")

    # 5. Generation
    # Format context
    context_str = ""
    for i, doc in enumerate(top_docs):
        src = doc.metadata.get("source", "Unknown")
        context_str += f"[文档{i+1}] (来源: {src})\n{doc.page_content}\n\n"
    
    if not context_str:
        context_str = "未找到相关文档，请根据你的知识尝试回答。"

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SE_PROCESS_PROMPT),
        ("human", "请根据以下【精选参考文档】回答我的问题。如果在文档中找到了答案，请引用来源。\n\n【精选参考文档】：\n{context}\n\n问题：{query}")
    ])
    
    try:
        response = llm.invoke(prompt.format_messages(
            context=context_str, 
            query=query, 
            chat_history=_format_chat_history(chat_history),
            summary=f"【长期记忆】：{summary}"
        ))
        answer = response.content
    except Exception as e:
        logging.error(f"Generation error: {e}")
        answer = "抱歉，生成回答时出错。"

    # 6. Return Update
    current_output = state.get("process_output", [])
    new_output = current_output + [str(answer)]
    
    return {
        "process_output": new_output,
        "messages": [AIMessage(content=str(answer))]
    }

# Need to import HuggingFaceEmbeddings inside the function to avoid circular imports or init issues
from langchain_huggingface import HuggingFaceEmbeddings

# Copy helper from router to avoid circular import dependency on Router node
def _format_chat_history(messages) -> str:
    if not messages: return "（无近期对话）"
    formatted = []
    for msg in messages[-5:]:
        if isinstance(msg, HumanMessage): formatted.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage): formatted.append(f"Agent: {msg.content}")
    return "\n".join(formatted)

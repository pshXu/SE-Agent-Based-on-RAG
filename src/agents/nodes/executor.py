import logging
import concurrent.futures
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from ..state import GraphState
from src.tools.retriever_tool import retrieve_knowledge
from src.tools.search_tool import search_documents
from src.utils.llm_factory import get_llm
from config.prompts import SE_PROCESS_PROMPT

def run(state: GraphState) -> GraphState:
    """
    Executor Node: Performs parallel retrieval of sub-queries and initial generation.
    Optimized for high-concurrency environments and data redundancy.
    """
    logging.info("--- Executor: Parallel Retrieval Pipeline ---")
    query = state["query"]
    plan = state.get("plan", [])
    retry_count = state.get("retry_count", 0)
    summary = state.get("summary", "（无历史摘要）")
    chat_history = [m for m in state.get("messages", []) if isinstance(m, BaseMessage)]
    
    # 策略：如果计划验证失败超过 3 次，降级为使用原始查询
    if retry_count >= 3:
        logging.warning("Fallback to original query due to validation failures.")
        plan = [{"query": query, "source": "both"}]

    # 用于存储所有子查询的结果，后续统一去重
    all_local_docs = []
    all_web_docs = []
    
    # 定义单个子查询的检索工作函数
    def fetch_results(item):
        sub_q = item["query"]
        src = item["source"]
        l_res, w_res = [], []
        try:
            logging.info(f"Launching parallel task for: '{sub_q}' (Source: {src})")
            if src in ["local", "both"]:
                l_res = retrieve_knowledge(sub_q)
            if src in ["web", "both"]:
                w_res = search_documents(sub_q, max_results=3)
        except Exception as e:
            logging.error(f"Error in parallel task for '{sub_q}': {e}")
        return l_res, w_res

    # --- 核心优化：并发执行 ---
    # 限制最大并发数为 3。因为每个检索任务都涉及 HyDE (LLM生成)，
    # 设为 3 可以在显著提升速度的同时，有效规避 API 并发限制 (Rate Limit)。
    max_workers = min(len(plan), 3) if plan else 1
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有子查询任务
        future_to_item = {executor.submit(fetch_results, item): item for item in plan}
        
        for future in concurrent.futures.as_completed(future_to_item):
            l_docs, w_docs = future.result()
            all_local_docs.extend(l_docs)
            all_web_docs.extend(w_docs)

    # --- 核心优化：基于内容的精准去重 ---
    # 使用字典推导式快速过滤重复内容。Key 为 page_content，Value 为 Document 对象。
    # 这种方式保证了即便不同子查询命中了同一知识片段，最终也只保留一份。
    unique_local_map = {doc.page_content: doc for doc in all_local_docs}
    unique_web_map = {doc.page_content: doc for doc in all_web_docs}
    
    unique_local = list(unique_local_map.values())
    unique_web = list(unique_web_map.values())
    
    # 策略：选取本地前 5 个精英片段 + 网络前 3 个补充片段
    top_docs = unique_local[:5] + unique_web[:3]
    
    logging.info(f"Deduplication complete: {len(all_local_docs)} local docs -> {len(unique_local)} unique.")

    # 构建上下文字符串，包含源文件和标题层级信息
    context_str = ""
    for i, doc in enumerate(top_docs):
        src_name = doc.metadata.get("file_name", "Unknown")
        headers = [doc.metadata.get(f"Header_{j}") for j in range(1, 5) if f"Header_{j}" in doc.metadata]
        header_path = " > ".join([h for h in headers if h]) or "根目录"
        context_str += f"[文档{i+1}] (来源: {src_name} | 层级: {header_path})\n{doc.page_content}\n\n"
    
    if not context_str: 
        context_str = "未找到相关文档，请根据你的知识尝试回答。"

    # 调用大模型生成“专家初稿”
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SE_PROCESS_PROMPT),
        ("human", "请根据以下【精选参考文档】回答我的问题。如果在文档中找到了答案，请引用来源。\n\n【精选参考文档】：\n{context}\n\n问题：{query}")
    ])
    
    try:
        # 格式化历史对话以提供上下文
        history_str = ""
        for msg in chat_history[-5:]:
            if isinstance(msg, HumanMessage): history_str += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage): history_str += f"Agent: {msg.content}\n"

        response = llm.invoke(prompt.format_messages(
            context=context_str, 
            query=query, 
            summary=summary,
            chat_history=history_str or "（无近期对话）"
        ))
        answer = response.content
    except Exception as e:
        logging.error(f"Generation error in executor: {e}")
        answer = "抱歉，在生成专家初稿时遇到了问题。"

    # 将参考原文和专家初稿一并打包，传递给后续的 Synthesizer 节点进行校对
    combined_output = [
        f"--- 原始参考文档 ---\n{context_str}",
        f"--- 专家初稿答案 ---\n{answer}"
    ]
    
    return {
        "process_output": combined_output,
        "messages": [AIMessage(content=str(answer), additional_kwargs={"intent": state.get("current_intent", "process")})]
    }

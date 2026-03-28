from langchain_core.tools import Tool
from src.rag.retriever_llama import LlamaIndexRetriever
from src.utils.llm_factory import get_llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import logging

# Global cache
_cached_retriever = None

def _init_retriever():
    global _cached_retriever
    if _cached_retriever is None:
        logging.info("Initializing LlamaIndexRetriever for Tool...")
        # Path to LlamaIndex persisted data
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        persist_dir = os.path.join(project_root, "data", "llama_vector_store")
        
        _cached_retriever = LlamaIndexRetriever(persist_dir=persist_dir)
    return _cached_retriever

def _generate_hyde_doc(query: str) -> str:
    """
    Generates a ultra-concise hypothetical document to speed up RAG retrieval.
    Less tokens = Faster response.
    """
    llm = get_llm()
    prompt = PromptTemplate(
        template="""你是一名软件工程专家。请针对以下问题，写一段【极简】且【专业】的事实性回答（100字以内）。
该内容将用于语义检索，请直接输出核心知识点，严禁任何客套话、引言或总结性陈述。

问题：{query}
回答：""",
        input_variables=["query"]
    )
    chain = prompt | llm | StrOutputParser()
    try:
        # 这里的调用是并发发起的，缩短字数能极大提升并行效率
        return chain.invoke({"query": query})
    except Exception as e:
        logging.error(f"HyDE generation failed: {e}")
        return query

def retrieve_docs(query: str) -> str:
    """
    Performs retrieval and returns a formatted string. (For simple tool use)
    """
    docs = retrieve_knowledge(query)
    if isinstance(docs, str): return docs # Error message
    
    if not docs:
        return "没有在本地知识库中找到相关文档。"
    
    # 3. Format results for the Agent
    result = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        result.append(f"[本地文档{i+1}] 来源: {source}\n内容: {doc.page_content}")
    
    return "\n\n".join(result)

def retrieve_knowledge(query: str) -> list:
    """
    Performs retrieval using HyDE + LlamaIndex and returns raw Document list.
    Used by advanced nodes like se_process for further processing.
    """
    try:
        # 1. HyDE Generation (Industrial trick)
        logging.info(f"Generating HyDE document for: {query}")
        hyde_doc = _generate_hyde_doc(query)
        
        # 2. Retrieval using LlamaIndex
        retriever = _init_retriever()
        docs = retriever.retrieve(hyde_doc, k_final=5)
        return docs
    except Exception as e:
        logging.error(f"Retrieval tool error: {e}")
        return []

def get_retriever_tool() -> Tool:
    """
    Returns the RAG retrieval tool powered by LlamaIndex.
    """
    return Tool(
        name="local_knowledge_retrieval",
        func=retrieve_docs,
        description="用于查询本地知识库中关于软件工程流程、规范、文档标准的详细信息。输入应该是具体的查询问题。"
    )

from ddgs import DDGS
from langchain_core.tools import Tool
from langchain_core.documents import Document
import logging
from typing import List

def duckduckgo_search_func(query: str) -> str:
    """
    Executes a search using the duckduckgo-search library directly.
    Returns a summarized string of the top results.
    """
    try:
        # Use DDGS context manager for better resource management
        with DDGS() as ddgs:
            # Get top 5 results
            results = list(ddgs.text(query, max_results=5))
            
            if not results:
                return "未找到相关的搜索结果。"
            
            # Format results into a single string
            formatted_results = []
            for i, res in enumerate(results):
                title = res.get("title", "No Title")
                snippet = res.get("body", "No Description")
                link = res.get("href", "No Link")
                formatted_results.append(f"[{i+1}] {title}\n摘要: {snippet}\n链接: {link}")
            
            return "\n\n".join(formatted_results)
            
    except Exception as e:
        logging.error(f"DuckDuckGo search error: {e}")
        return f"执行联网搜索时出错: {e}"

def search_documents(query: str, max_results: int = 5) -> List[Document]:
    """
    Executes search and returns a list of Document objects for re-ranking.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            docs = []
            for res in results:
                docs.append(Document(
                    page_content=res.get("body", ""),
                    metadata={
                        "source": res.get("href", ""),
                        "title": res.get("title", "")
                    }
                ))
            return docs
    except Exception as e:
        logging.error(f"DuckDuckGo search_documents error: {e}")
        return []

def get_search_tool() -> Tool:
    """
    Returns a custom-built DuckDuckGo search tool that bypasses the broken LangChain wrapper.
    """
    return Tool(
        name="web_search",
        func=duckduckgo_search_func,
        description="用于搜索互联网上的最新信息、实时数据或本地知识库缺失的内容。输入应该是具体的查询问题。"
    )

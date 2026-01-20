from langchain_core.tools import Tool
from src.tools.search_tool import get_search_tool
from src.tools.retriever_tool import get_retriever_tool
from typing import List

def get_se_tools() -> List[Tool]:
    """
    Returns tools for the SE Process Agent (RAG Retrieval & Web Search).
    """
    return [get_retriever_tool(), get_search_tool()]

import os
import sys
import asyncio
from typing import List
from mcp.server import Server
from mcp.types import Tool, TextContent, EmbeddedResource
import mcp.server.stdio

# 将项目根目录添加到 python 路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.tools.retriever_tool import retrieve_knowledge

# 初始化 MCP 服务器
server = Server("nju-se-expert")

@server.list_tools()
async def list_tools() -> List[Tool]:
    """
    向 MCP 客户端列出可用的工具。
    """
    return [
        Tool(
            name="retrieve_se_knowledge",
            description="查询南大软件工程专家知识库，获取关于流程规范、文档标准和专业定义的权威信息。输入应为具体的软件工程相关问题。",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "需要查询的问题，如 '什么是需求评审？'"
                    }
                },
                "required": ["query"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> List[TextContent]:
    """
    处理工具调用请求。
    """
    if name == "retrieve_se_knowledge":
        query = arguments.get("query")
        if not query:
            return [TextContent(type="text", text="Error: Query is required.")]
        
        print(f"[*] MCP Server: Processing query: {query}", file=sys.stderr)
        
        # 调用核心 RAG 检索逻辑 (由于 retrieve_knowledge 是同步的，我们用 run_in_executor 避免阻塞)
        loop = asyncio.get_event_loop()
        docs = await loop.run_in_executor(None, retrieve_knowledge, query)
        
        if not docs:
            return [TextContent(type="text", text="未在本地知识库中找到相关信息。")]
        
        # 格式化检索结果供客户端模型使用
        formatted_results = []
        for i, doc in enumerate(docs):
            src = doc.metadata.get("file_name", "Unknown")
            headers = [doc.metadata.get(f"Header_{j}") for j in range(1, 5) if f"Header_{j}" in doc.metadata]
            header_path = " > ".join([h for h in headers if h]) or "根目录"
            
            formatted_results.append(
                f"--- 证据 {i+1} ---\n"
                f"来源: {src} | 层级: {header_path}\n"
                f"内容: {doc.page_content}\n"
            )
            
        return [TextContent(type="text", text="\n".join(formatted_results))]
    
    raise ValueError(f"Unknown tool: {name}")

async def main():
    """
    启动 STDIO 模式的 MCP 服务器。
    """
    print("[*] NJU SE MCP Server starting...", file=sys.stderr)
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    # 加载环境变量
    from dotenv import load_dotenv
    load_dotenv()
    
    asyncio.run(main())

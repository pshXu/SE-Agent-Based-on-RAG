import os
import sys
import logging
from typing import List, Dict, Any

# Fix for Hugging Face tokenizers deadlock warning/error
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 优化 MPS 显存分配，允许更灵活的内存使用
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    Document as LlamaDocument,
)
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.settings import BGE_MODEL_NAME, COLLECTION_NAME

# Setup Logging
logging.basicConfig(level=logging.INFO)

# --- 核心切片器：三级切分逻辑 ---
def get_node_parser():
    """
    实现“标题 -> 段落 -> 固定长度”的三级切分策略。
    1. 优先按 Markdown 标题 (#) 切分 Section。
    2. 如果 Section 内部内容超过 chunk_size，则进行固定长度切分。
    3. 较短的段落会自动合并在同一个标题下的 Node 中。
    """
    return MarkdownNodeParser(
        chunk_size=1024,      # 第三级：单个 Node 的最大 Token 长度
        chunk_overlap=128     # 长度切分时的重叠度，保证语义连续
    )

def main():
    print("--- 启动结构化 RAG 数据入库流程 (Markdown 优化版) ---")
    
    # 1. 配置模型与切片器
    # 将 embed_batch_size 从 16 降低到 4 以解决 MPS 内存溢出问题
    embed_model = HuggingFaceEmbedding(
        model_name=BGE_MODEL_NAME, 
        embed_batch_size=4,
        device="mps" # 显式指定，如果显存依然不足可改为 "cpu"
    )
    Settings.embed_model = embed_model
    Settings.llm = None 
    Settings.node_parser = get_node_parser()

    # 2. 配置向量库 (ChromaDB)
    persist_dir = os.path.join(project_root, "data", "llama_vector_store")
    db = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 3. 初始化索引
    try:
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)
        print("成功加载已有索引。")
    except Exception:
        print("未发现现有索引，创建新索引。")
        index = VectorStoreIndex.from_documents([], storage_context=storage_context)

    # 4. 处理 Markdown 文件 (核心路径)
    md_path = os.path.join(project_root, "data", "processed", "markdown")
    if not os.path.exists(md_path):
        os.makedirs(md_path)
        print(f"创建 Markdown 存放目录: {md_path}")

    file_count = 0
    # 扫描处理后的 Markdown
    for root, _, files in os.walk(md_path):
        for file in files:
            if file.lower().endswith(".md"):
                file_path = os.path.join(root, file)
                print(f"\n正在处理 Markdown: {file}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 构建 Document 对象
                doc = LlamaDocument(
                    text=content, 
                    metadata={
                        "file_name": file, 
                        "format": "markdown",
                        "source_type": "processed"
                    }
                )
                
                # 使用 MarkdownNodeParser 自动进行三级切分
                nodes = Settings.node_parser.get_nodes_from_documents([doc])
                
                # 将节点插入索引
                index.insert_nodes(nodes)
                file_count += 1
                print(f"  - 已入库 {len(nodes)} 个结构化切片。")

    # 5. 持久化索引
    index.storage_context.persist(persist_dir=persist_dir)
    print(f"\n--- 入库完成。处理了 {file_count} 个 Markdown 文件。 ---")

if __name__ == "__main__":
    main()

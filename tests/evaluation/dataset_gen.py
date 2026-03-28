import os
import sys
import random
import asyncio
import logging
import json
from dotenv import load_dotenv

# 加载 .env 环境变量
load_dotenv()

# 屏蔽冗余日志
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from llama_index.core import SimpleDirectoryReader, PromptTemplate, Settings
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from src.utils.llm_factory import get_llm
from src.utils.logger import get_logger

logger = get_logger("dataset-gen")

async def generate_golden_dataset():
    input_dir = "data/processed/markdown"
    if not os.path.exists(input_dir):
        logger.error(f"目录不存在: {input_dir}")
        return

    reader = SimpleDirectoryReader(input_dir=input_dir, recursive=True, required_exts=[".md"])
    documents = reader.load_data()
    
    # 核心优化：将大文档切分为小片段 (Chunking)，避免将整个文档发送给 LLM 导致 Token 消耗巨大
    from llama_index.core.node_parser import SentenceSplitter
    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=50)
    all_nodes = splitter.get_nodes_from_documents(documents)
    
    random.shuffle(all_nodes)
    logger.info(f"📚 文档已切分，共 {len(all_nodes)} 个片段 (每个约 1024 tokens)")

    from llama_index.llms.openai import OpenAI
    actual_model = os.getenv("OPENAI_MODEL_NAME", "deepseek-chat")
    llama_llm = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base=os.getenv("OPENAI_API_BASE"),
        model="gpt-3.5-turbo",
        additional_kwargs={"model": actual_model},
        is_chat_model=True,
        timeout=120.0 # 增加到 120 秒，避免频繁断连
    )
    Settings.llm = llama_llm
    native_llm = get_llm()

    eval_data = []
    seen_queries = set()
    data_lock = asyncio.Lock()
    stats = {"simple": 0, "complex": 0}
    output_path = "tests/evaluation/golden_dataset.json"

    async def save_checkpoint():
        """实时保存进度"""
        async with data_lock:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(eval_data, f, ensure_ascii=False, indent=2)

    async def process_node(node, template, tag, semaphore, target_count):
        async with semaphore:
            if stats[tag] >= target_count: return
            
            gen = RagDatasetGenerator.from_documents([node], llm=llama_llm, text_question_template=template, num_questions_per_chunk=1)
            try:
                # 移除外部 wait_for，让任务自然完成
                ds = await gen.agenerate_dataset_from_nodes()
                for ex in ds.examples:
                    # --- 强化清洗与校验逻辑 ---
                    query = ex.query.strip()
                    
                    # 1. 移除所有可能的冗余标签和符号
                    noise = ["Question:", "软件工程问题：", "软件工程问题:", "**软件工程问题：**", "**软件工程问题:**", "问题：", "问题:", "**", "###", "Q:"]
                    for n in noise:
                        query = query.replace(n, "").strip()
                    
                    # 2. 严格校验：如果长度太短（可能是空题目）或者仅包含无意义字符，则跳过
                    if len(query) < 15: 
                        continue
                    
                    # 3. 校验是否只是重复了提示词
                    if query in ["请根据内容生成问题", "侧重于概念解释"]:
                        continue

                    async with data_lock:
                        if query not in seen_queries and stats[tag] < target_count:
                            seen_queries.add(query)
                            eval_data.append({
                                "query": query,
                                "reference_context": ex.reference_contexts[0] if ex.reference_contexts else "",
                                "reference_answer": ex.reference_answer,
                                "type": tag
                            })
                            stats[tag] += 1
                            print(f"✅ [{tag}] {stats[tag]}/{target_count} | {query[:40]}...")
            except Exception:
                pass

    async def run_parallel(nodes, template, target_count, tag):
        semaphore = asyncio.Semaphore(8) # 略微调低并发，保证稳定性
        tasks = [process_node(n, template, tag, semaphore, target_count) for n in nodes[:target_count*3]]
        # 使用 as_completed 保证谁好了谁就打印，不被慢任务拖累
        for coro in asyncio.as_completed(tasks):
            await coro

    # 格式指令
    format_hint = "\n【强制要求】：直接输出问题正文，严禁输出“软件工程问题：”或“问题：”等任何标签、标题或前缀。"

    # 1. 简单题
    logger.info("🚀 生成简单题...")
    simple_t = PromptTemplate("内容：{context_str}\n请根据上述内容生成一个侧重于【概念解释】的简洁软件工程问题（例如：请解释什么是xxx）及其参考答案。使用中文。" + format_hint)
    await run_parallel(all_nodes[:200], simple_t, 20, "simple")
    await save_checkpoint()

    # 2. 复杂题
    logger.info("🚀 生成复杂题...")
    complex_t = PromptTemplate(
        "内容：{context_str}\n"
        "请根据上述内容生成一个具有挑战性的软件工程专业问题及其参考答案。要求：\n"
        "1. 侧重于多个概念的辨析与深层理解区别；\n"
        "2. 或者是对不同软件工程方案/模型的优缺点进行对比分析；\n"
        "3. 或者是针对具体案例场景，考查软件工程知识的综合运用。\n"
        "使用中文。" + format_hint
    )
    await run_parallel(all_nodes[200:400], complex_t, 20, "complex")
    await save_checkpoint()

    # 3. 联网题
    logger.info("🌐 生成联网题...")
    try:
        resp = await asyncio.to_thread(native_llm.invoke, "生成 10 个关于 2024-2026 软件工程趋势的问题。JSON: [{\"query\": \"...\", \"reference_answer\": \"...\", \"type\": \"web_impact\"}]")
        web_samples = json.loads(resp.content.replace("```json", "").replace("```", "").strip())
        for s in web_samples:
            if s['query'] not in seen_queries:
                eval_data.append(s)
                seen_queries.add(s['query'])
    except Exception as e:
        logger.error(f"联网题失败: {e}")

    await save_checkpoint()
    logger.info(f"✨ 完成! 最终题目: {len(eval_data)}")

if __name__ == "__main__":
    asyncio.run(generate_golden_dataset())

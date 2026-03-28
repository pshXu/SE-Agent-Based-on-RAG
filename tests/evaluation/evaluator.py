import os
import sys
import json
import asyncio
import pandas as pd
import logging
import re
from typing import List, Dict

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.rag.retriever_llama import LlamaIndexRetriever
from src.agents.nodes.synthesizer import run as run_synthesizer
from src.utils.llm_factory import get_llm
from src.tools.search_tool import search_documents

from ragas import evaluate
from ragas.metrics import faithfulness
from datasets import Dataset
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 屏蔽冗余日志
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

logger = logging.getLogger("RAG-Evaluator-Judge")
logging.basicConfig(level=logging.INFO)

class RAGEvaluator:
    def __init__(self, persist_dir: str):
        self.retriever = LlamaIndexRetriever(persist_dir=persist_dir)
        self.llm = get_llm()
        
    def load_dataset(self, path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    async def llm_judge_retrieval(self, query: str, retrieved_texts: List[str], ground_truth_context: str) -> int:
        """
        使用 LLM 作为裁判，判定检索到的文档列表中哪一个才是真正的 '命中'。
        返回命中文档的 1-based 索引，若未命中则返回 0。
        """
        if not retrieved_texts or not ground_truth_context:
            return 0
            
        # 构建判定 Prompt
        docs_str = "\n".join([f"[{i+1}] {text[:500]}..." for i, text in enumerate(retrieved_texts)])
        
        prompt = f"""你是一名资深的软件工程评估专家。你的任务是判定检索到的文档是否包含回答特定问题所需的核心信息。

用户问题: {query}
标准参考内容 (Ground Truth): {ground_truth_context}

以下是检索到的 {len(retrieved_texts)} 个文档片段:
{docs_str}

任务要求:
1. 对比“标准参考内容”与上述检索片段。
2. 即使语言不同（如中文对英文），只要讨论的是同一个技术概念、设计模式或核心事实，即视为“命中”。
3. 找出检索片段中【第一个】包含“标准参考内容”核心信息的片段索引。

请只返回一个数字（1-{len(retrieved_texts)}），代表命中的片段索引。如果没有任何片段命中，请返回 0。
除了数字外，不要返回任何其他文字。"""

        try:
            # 调用 LLM 进行判定
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            match = re.search(r'\d+', response.content)
            rank = int(match.group()) if match else 0
            return rank if rank <= len(retrieved_texts) else 0
        except Exception as e:
            logger.error(f"LLM Judge 判定失败: {e}")
            return 0

    async def run_eval_for_mode(self, dataset: List[Dict], mode: str) -> pd.DataFrame:
        """运行特定模式的完整评估"""
        results = []
        for i, sample in enumerate(dataset):
            query = sample["query"]
            gt_context = sample.get("reference_context", "")
            
            # 1. 检索
            local_docs = self.retriever.retrieve(query, k_final=5)
            web_docs = []
            if mode == "hybrid":
                try:
                    web_docs = search_documents(query, max_results=3)
                except: pass
            
            all_docs = local_docs + web_docs
            contexts = [doc.page_content for doc in all_docs]
            
            # 2. 计算检索指标 (使用 LLM-as-a-Judge)
            # 我们针对前 5 个文档计算命中排名
            hit_rank = await self.llm_judge_retrieval(query, contexts[:5], gt_context)
            hit = 1 if hit_rank > 0 else 0
            mrr = 1 / hit_rank if hit_rank > 0 else 0
            
            # 3. 生成答案
            state = {"query": query, "process_output": contexts, "messages": []}
            try:
                output_state = run_synthesizer(state)
                answer = str(output_state.get("final_answer", ""))
            except Exception as e:
                answer = f"Error: {e}"

            results.append({
                "question": query,
                "contexts": contexts,
                "answer": answer,
                "ground_truth": sample.get("reference_answer", ""),
                "hit": hit,
                "mrr": mrr
            })
            print(f"  > [{mode}] {i+1}/{len(dataset)} | Judge 结果: {'✅' if hit else '❌'} (Rank: {hit_rank})")

        # 4. Ragas 忠实度评估
        df = pd.DataFrame(results)
        logger.info(f"正在计算 {mode} 模式的 Faithfulness...")
        try:
            ragas_ds = Dataset.from_pandas(df[["question", "contexts", "answer", "ground_truth"]])
            ragas_results = evaluate(ragas_ds, metrics=[faithfulness], llm=self.llm)
            df["faithfulness"] = ragas_results.to_pandas()["faithfulness"].tolist()
        except Exception as e:
            logger.error(f"Ragas 评估失败: {e}")
            df["faithfulness"] = 0.0
            
        return df

    def print_report(self, df: pd.DataFrame, title: str):
        print("\n" + "="*50)
        print(f"       {title} (LLM-Judge)")
        print("="*50)
        print(f"平均召回率 (Recall@5):  {df['hit'].mean():.4f}")
        print(f"平均 MMR:             {df['mrr'].mean():.4f}")
        print(f"平均忠实度 (Faithfulness): {df['faithfulness'].mean():.4f}")
        print("="*50)

async def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    persist_dir = os.path.join(project_root, "data", "llama_vector_store")
    input_data = os.path.join(project_root, "tests/evaluation/golden_dataset.json")
    
    evaluator = RAGEvaluator(persist_dir=persist_dir)
    dataset = evaluator.load_dataset(input_data)[:20] # 抽取 20 条

    print("🚀 开始 Local 模式评估 (LLM Judge 模式)...")
    df_local = await evaluator.run_eval_for_mode(dataset, mode="local")
    
    print("\n🚀 开始 Hybrid 模式评估 (LLM Judge 模式)...")
    df_hybrid = await evaluator.run_eval_for_mode(dataset, mode="hybrid")

    # 输出报告
    evaluator.print_report(df_local, "Local-Only RAG 性能报告")
    evaluator.print_report(df_hybrid, "Hybrid (Local+Web) RAG 性能报告")

    # 保存详细数据
    df_local.to_csv("tests/evaluation/results_local_llm.csv", index=False, encoding="utf-8-sig")
    df_hybrid.to_csv("tests/evaluation/results_hybrid_llm.csv", index=False, encoding="utf-8-sig")
    print(f"\n详细结果已保存至 tests/evaluation/ 目录下。")

if __name__ == "__main__":
    asyncio.run(main())

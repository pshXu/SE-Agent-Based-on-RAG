import logging
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from llama_index.core import Settings

from ..state import GraphState
from src.utils.llm_factory import get_llm

class ValidationResult(BaseModel):
    is_valid: bool = Field(description="计划是否通过验证")
    feedback: str = Field(description="改进建议")

def _calculate_scores(orig_query: str, plan: list) -> dict:
    """
    核心数学评估函数：计算覆盖度和带惩罚的独立性。
    """
    sub_queries = [sq["query"] for sq in plan]
    n = len(sub_queries)
    
    # 获取 Embedding 模型 (复用 LlamaIndex 全局设置)
    embed_model = Settings.embed_model
    
    # 1. 批量获取向量 (批处理提高效率)
    all_texts = [orig_query] + sub_queries
    embeddings = np.array(embed_model.get_text_embedding_batch(all_texts))
    
    orig_vec = embeddings[0]
    sub_vecs = embeddings[1:]
    
    # --- 维度 A: 语义累加覆盖度 (Semantic Accumulation Coverage) ---
    # 使用向量和而不是平均值，避免语义稀释
    combined_sub_vec = np.sum(sub_vecs, axis=0)
    
    # 计算余弦相似度
    coverage_score = np.dot(orig_vec, combined_sub_vec) / (
        np.linalg.norm(orig_vec) * np.linalg.norm(combined_sub_vec) + 1e-9
    )
    
    # --- 维度 B: 独立性与惩罚 (Independence & Penalty) ---
    penalty_weight = 0.1
    n_threshold = 3
    
    if n > 1:
        # 计算两两相似度矩阵
        norms = np.linalg.norm(sub_vecs, axis=1)
        sim_matrix = np.dot(sub_vecs, sub_vecs.T) / (np.outer(norms, norms) + 1e-9)
        # 取上三角部分（不含对角线）
        tri_indices = np.triu_indices(n, k=1)
        avg_redundancy = np.mean(sim_matrix[tri_indices])
        base_independence = 1.0 - avg_redundancy
    else:
        base_independence = 1.0

    # 计算数量惩罚
    penalty = max(0, n - n_threshold) * penalty_weight
    independence_score = base_independence - penalty
    
    return {
        "coverage": float(coverage_score),
        "independence": float(independence_score),
        "penalty": float(penalty),
        "count": n
    }

def run(state: GraphState) -> GraphState:
    """
    Validator Node: Reflects on the quality of the query plan using Embedding logic.
    """
    logging.info("--- Validator: Mathematical Reflection on Plan Quality ---")
    query = state["query"]
    plan = state.get("plan", [])
    retry_count = state.get("retry_count", 0)

    # 1. 基础校验：如果没生成计划
    if not plan:
        return {
            "next_step": "planner", 
            "planning_feedback": "未能生成任何有效的子查询，请尝试重新拆解。",
            "retry_count": retry_count + 1
        }

    # 2. 执行数学评估
    try:
        scores = _calculate_scores(query, plan)
        logging.info(f"Validator Scores: {scores}")
    except Exception as e:
        logging.error(f"Error during mathematical validation: {e}")
        # 如果向量计算失败，兜底由 LLM 决定
        return {"next_step": "executor"}

    # 3. 判定逻辑与反馈生成
    is_valid = True
    feedback_msgs = []
    
    # 覆盖度门槛：0.85 (硬指标)
    if scores["coverage"] < 0.85:
        is_valid = False
        feedback_msgs.append(f"【覆盖度不足({scores['coverage']:.2f})】: 子查询集合未能完整涵盖原始问题的语义，请检查是否遗漏了关键约束或背景。")
    
    # 独立性门槛：0.4 (软约束 + 惩罚)
    if scores["independence"] < 0.4:
        is_valid = False
        if scores["penalty"] > 0:
            feedback_msgs.append(f"【拆解过细({scores['independence']:.2f})】: 子查询数量({scores['count']})过多产生冗余，请合并相似意图的查询。")
        else:
            feedback_msgs.append(f"【独立性低({scores['independence']:.2f})】: 子查询之间存在严重内容重叠，请确保每个子查询侧重点不同。")

    # 4. 循环控制：如果重试达到 3 次，强制通过 (Executor 会使用兜底策略)
    if not is_valid and retry_count < 3:
        feedback = " ".join(feedback_msgs)
        print(f"Validator: Rejected. {feedback}")
        return {
            "next_step": "planner", 
            "planning_feedback": feedback, 
            "retry_count": retry_count + 1
        }
    
    # 通过验证或达到重试上限
    if retry_count >= 3:
        logging.warning("Max retries reached. Proceeding to execution with current plan.")
    else:
        print("Validator: Plan passed successfully.")
        
    return {"next_step": "executor", "planning_feedback": ""}

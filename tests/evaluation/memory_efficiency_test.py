import os
import sys
import asyncio
import pandas as pd
import logging
import time
from langchain_core.messages import HumanMessage, AIMessage

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.agents.nodes.router import route
from src.agents.nodes.se_process import run as run_se_process
from src.agents.nodes.synthesizer import run as run_synthesizer
from src.agents.nodes.summarizer import run as run_summarizer
from src.utils.llm_factory import get_llm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Memory-Stress-Test")

# --- 模拟深度的 SE 对话流 (20轮) ---
CONVERSATION_FLOW = [
    "我们要为南大设计一个图书馆管理系统，第一个硬性约束是：系统必须在每晚 23:00 准时进入‘冷备份’状态，期间拒绝所有查询。请记住这个约束。",
    "好的，现在讨论需求分析阶段。我们需要哪些核心用例？",
    "针对‘借书’用例，请描述其基本流。",
    "如何设计该系统的数据库架构以支持高并发？",
    "我们需要引入 Redis 吗？如果引入，主要解决什么问题？",
    "在南京大学的 SRS 规范中，非功能性需求应该包含哪些维度？",
    "请详细解释一下该系统的‘领域模型’，特别是‘书籍’和‘借阅记录’的关系。",
    "如果我们要支持跨校区借书，架构上需要做哪些调整？",
    "讨论一下安全设计，如何防止 SQL 注入？",
    "我们需要为这个系统编写详细的设计文档（SDD），南大软院对 SDD 的结构有什么要求？",
    "对于借书过程中的‘并发冲突’（两个人同时借最后一本书），你建议用乐观锁还是悲观锁？",
    "系统的测试计划应该如何制定？",
    "如果我们要引入微服务架构，应该如何拆分模块？",
    "微服务之间如何进行通信？建议使用 Feign 还是消息队列？",
    "如果系统在 2026 年需要升级支持 AI 推荐书籍，现在应该预留什么接口？",
    "讨论一下 DevOps 流程，我们需要配置 Jenkins 吗？",
    "如何实现系统的灰度发布？",
    "对于这个图书馆系统，前端建议使用什么框架以符合南大的教学实践？",
    "如果数据库发生宕机，我们的恢复策略是什么？",
    "【终极一致性测试】：回到最初的设定，该系统在每晚 23:00 有什么特殊的行为约束？"
]

class MemoryTester:
    def __init__(self):
        self.llm = get_llm()

    def estimate_tokens(self, state: dict) -> int:
        """粗略估算当前上下文的 Token 消耗 (字符数 / 2)"""
        text = str(state.get("summary", "")) + str(state.get("messages", "")) + str(state.get("process_output", ""))
        return len(text) // 2

    async def run_simulation(self, use_summarizer: bool):
        logger.info(f"\n--- 启动 {'[摘要压缩模式]' if use_summarizer else '[全量历史模式]'} ---")
        state = {
            "query": "",
            "messages": [],
            "process_output": [],
            "final_answer": "",
            "next_step": "",
            "summary": "",
            "current_intent": "process"
        }
        
        results = []
        
        for i, query in enumerate(CONVERSATION_FLOW):
            start_time = time.time()
            state["query"] = query
            state["messages"].append(HumanMessage(content=query))
            
            state["next_step"] = "se_process"
            state["current_intent"] = "process"
            
            output = run_se_process(state)
            state.update(output)
            
            final = run_synthesizer(state)
            state.update(final)
            
            if use_summarizer:
                mem_update = run_summarizer(state)
                if mem_update:
                    if "messages" in mem_update:
                        from langchain_core.messages import RemoveMessage
                        removals = [m.id for m in mem_update["messages"] if isinstance(m, RemoveMessage)]
                        state["messages"] = [m for m in state["messages"] if m.id not in removals]
                    if "summary" in mem_update:
                        state["summary"] = mem_update["summary"]

            tokens = self.estimate_tokens(state)
            duration = time.time() - start_time
            
            results.append({
                "turn": i + 1,
                "tokens": tokens,
                "answer": state["final_answer"]
            })
            
            logger.info(f"第 {i+1:02d} 轮 | 估算消耗: {tokens:5d} tokens | 耗时: {duration:.1f}s")
            
        # --- LLM Judge 环节 (针对最后一轮) ---
        last_answer = state["final_answer"]
        original_constraint = "系统必须在每晚 23:00 准时进入‘冷备份’状态，期间拒绝所有查询。"
        
        judge_prompt = f"""你是一名 AI 评估专家。请判定 AI 在第 20 轮对话中的表现。
        
原始设定：{original_constraint}
AI 的回答：{last_answer}

请根据以下标准给出一个 1-10 的评分：
1. 事实准确性：是否提到了 23:00 和冷备份？
2. 完整性：是否保留了拒绝查询的细节？
3. 语言专业性。

请只按以下 JSON 格式返回结果：
{{"score": 8, "reason": "AI 准确记住了时间点和备份操作，但漏掉了拒绝查询的细节说明。"}}
"""
        try:
            from langchain_core.output_parsers import JsonOutputParser
            judge_resp = self.llm.invoke(judge_prompt)
            # 解析 JSON (处理可能的 Markdown 标签)
            clean_content = judge_resp.content.replace("```json", "").replace("```", "").strip()
            score_data = json.loads(clean_content)
        except Exception as e:
            score_data = {"score": 0, "reason": f"Judge failed: {e}"}

        return pd.DataFrame(results), score_data

async def main():
    import json
    tester = MemoryTester()
    
    # 运行全量模式
    df_full, score_full = await tester.run_simulation(use_summarizer=False)
    
    # 运行摘要模式
    df_summary, score_summary = await tester.run_simulation(use_summarizer=True)
    
    # --- 最终报告 ---
    print("\n" + "="*70)
    print("         RAG 记忆效率与长程一致性深度对比报告")
    print("="*70)
    print(f"{'指标':<20} | {'全量模式 (Full)':<20} | {'摘要模式 (Summary)':<20}")
    print("-" * 70)
    print(f"{'最终单轮 Token':<20} | {df_full['tokens'].iloc[-1]:<20} | {df_summary['tokens'].iloc[-1]:<20}")
    print(f"{'累计总 Token':<20} | {df_full['tokens'].sum():<20} | {df_summary['tokens'].sum():<20}")
    print(f"{'记忆保持评分(1-10)':<20} | {score_full['score']:<20} | {score_summary['score']:<20}")
    print(f"{'评价理由':<20} | {score_full['reason'][:18]}... | {score_summary['reason'][:18]}...")
    
    saving = (1 - df_summary['tokens'].sum() / df_full['tokens'].sum()) * 100
    print("-" * 70)
    print(f"💰 总体成本节省: {saving:.2f}%")
    print("="*70)

    # 保存数据
    df_full.to_csv("tests/evaluation/stress_full.csv", index=False)
    df_summary.to_csv("tests/evaluation/stress_summary.csv", index=False)

if __name__ == "__main__":
    asyncio.run(main())

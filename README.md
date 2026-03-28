# 基于 RAG 的软件工程智能问答助手 (Agentic RAG System)

本项目是一个基于 **LangGraph**、**LlamaIndex** 和构建的垂直领域智能咨询系统。它专为解决复杂的软件工程（SE）流程咨询而设计，具备多智能体协作、计划反思循环、并行检索以及质量评估体系。

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.11-blue)

---

## ✨ 核心特性

### 1. 代理式架构 (Agentic Workflow)
*   **Plan-and-Execute + Reflection**: 采用三级代理逻辑：
    *   **Planner**: 自动将复杂问题拆分为高质量子查询。
    *   **Validator**: 基于 Embedding 的数学评估模型，验证子查询的 **语义累加覆盖度** 与 **独立性**。
    *   **Executor**: 并发执行检索任务，结合 HyDE 增强与内容级去重，大幅提升响应速度。
*   **智能记忆系统**: 采用 **Buffer + Summary** 双层架构，通过 Summarizer Agent 自动提取 SE 技术要点，清理冗余，实现长程对话稳定性。

### 2. 高级检索策略 (Advanced RAG)
*   **结构化切分**: 使用 `MarkdownNodeParser` 实现“标题 -> 段落 -> 块”的三级切分，保留文档的层级语义。
*   **混合检索 (Hybrid Search)**: 结合 **向量检索 (BGE-M3)** 与 **关键字检索 (BM25)**，通过 **RRF (Reciprocal Rank Fusion)** 进行得分融合。
*   **重排序 (Rerank)**: 使用 **Cross-Encoder** 进行精排，并引入 **MMR (Maximal Marginal Relevance)** 算法平衡相关性与多样性。
*   **HyDE 增强**: 引入假设性文档嵌入，利用“答案匹配答案”解决语义不对称问题。

### 3. 多模态解析
*   **OCR 引擎**: 集成 **Unstructured + Tesseract**，支持中英混杂的扫描件 PDF 自动识别，具备复杂的版面分析能力。

---

## 📊 评估与质量保障 (Evaluation)

项目内置了完整的评估框架（位于 `tests/evaluation/`），确保系统回答的忠实度与准确性。

*   **Ragas 自动化评估**: 
    *   **Faithfulness (忠实度)**: 检查答案是否完全基于参考文档。
    *   **Answer Relevance (答案相关性)**: 检查答案是否准确解决用户问题。
    *   **Context Recall (上下文召回率)**: 检查检索到的内容是否覆盖了标准答案。
*   **黄金数据集 (Golden Dataset)**: 包含 50+ 组由专家标注的 SE 领域问答对。
*   **压力测试**: 模拟长程对话与高并发检索，评估系统的 Token 消耗与内存效率。
*   **性能对比**: 提供本地模型（Local LLM）与混合模型（Hybrid LLM）的测试报告（`.csv`）。

---

## 🚀 快速开始

### 1. 环境准备 (macOS)
```bash
brew install tesseract poppler
conda create -n rag python=3.11 && conda activate rag
pip install -r requirements.txt
```

### 2. 数据入库
将 PDF/MD 放入 `data/raw/books/`，运行：
```bash
python src/rag/ingestion_llama.py
```

### 3. 启动交互
*   **CLI 交互**: `python main.py`
*   **MCP 模式**: 在 IDE 中配置 `mcp_server.py` 以接入标准协议。
*   **性能评估**: 运行 `python tests/evaluation/evaluator.py` 查看当前系统的 Ragas 得分。

---

## 📂 项目结构

```text
nju-se-agent/
├── src/
│   ├── agents/
│   │   ├── nodes/              # 各阶段 Agent 逻辑 (Planner, Validator, Executor...)
│   │   └── graph.py            # LangGraph 状态机编排
│   ├── rag/
│   │   ├── ingestion_llama.py  # Markdown 优化入库流程
│   │   ├── retriever_llama.py  # 混合检索与重排序核心
│   │   └── vector_db.py        # 向量库底层封装
│   ├── tools/                  # 检索与搜索工具集
│   └── utils/                  # LLM 工厂、OCR 转换器
├── tests/
│   ├── evaluation/             # 评估框架 (Dataset Gen, Ragas Eval, Stress Test)
│   └── test_rag.py             # 核心组件测试
├── mcp_server.py               # MCP 协议适配层
└── design.md                   # 系统详细设计方案

```

---

## ⚠️ 性能提示
*   **GPU 加速**: 默认启用 MPS (Metal Performance Shaders) 加速 Embedding。
*   **并发限制**: 并发检索数建议设为 3-5 以规避 API 频率限制。

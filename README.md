# 南大软件学院 SE 流程智能体 (Agentic RAG System)

本项目是一个基于 **LangGraph** 和 **LlamaIndex** 构建的垂直领域智能咨询系统。它专为解决复杂的软件工程（SE）流程咨询而设计，具备多智能体协作、自主检索、长程记忆以及对非结构化文档（扫描件 PDF）的高精度解析能力。

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.11-blue)

## ✨ 核心特性

*   **Agentic RAG (代理式检索)**: 引入 **ReAct** 循环与 **显式问题拆解 (Explicit Decomposition)**，智能体能自动将复杂问题拆分为多个子查询，并自主决定查库还是搜网。
*   **LlamaIndex 驱动**: 底层检索引擎全面升级为 LlamaIndex，支持 **Sentence Window Retrieval (句子窗口检索)**，实现毫秒级精准定位与完整上下文回填。
*   **多模态解析 (OCR)**: 集成 **Unstructured + Tesseract**，支持中英混杂的扫描件 PDF 识别。
*   **HyDE 增强**: 引入 **Hypothetical Document Embeddings**，利用“答案匹配答案”技术解决语义不对称问题。
*   **智能记忆系统**: 采用 **Buffer + Summary** 双层架构，通过 Summarizer Agent 实现选择性记忆压缩，有效控制 Token 消耗并过滤闲聊噪音。

---

## 🛠️ 环境准备

### 1. 克隆项目
```bash
git clone <repository_url>
cd nju-se-agent
```

### 2. 系统依赖 (macOS)
本项目依赖 **Tesseract** 进行 OCR 识别，依赖 **Poppler** 处理 PDF 图像。
```bash
brew install tesseract
brew install tesseract-lang  # 安装中文语言包 (chi_sim)
brew install poppler
```

### 3. Python 依赖
建议使用 Conda 创建独立环境：
```bash
conda create -n rag python=3.11
conda activate rag
pip install -r requirements.txt
```

### 4. 环境变量配置
复制 `.env.example` 为 `.env`，并填入您的 API Key：
```bash
cp .env.example .env
```
**关键配置项**：
*   `OPENAI_API_KEY`: 您的 LLM 服务商 Key。
*   `OPENAI_API_BASE`: 如果使用中转服务（如 DeepSeek），请在此配置 Base URL。
*   `BGE_MODEL_NAME`: 默认使用 `BAAI/bge-m3`。

---

## 📚 数据准备与入库

### 1. 放置文档
请将您的知识库文档（支持 PDF, HTML, Markdown）放入以下目录：
> **`data/raw/books/`**

*   系统支持递归扫描子目录。
*   支持 **扫描件 PDF**（会自动触发 OCR）。
*   支持 **中英混杂** 内容。

### 2. 构建索引 (Ingestion)
运行以下脚本，将原始文档转化为 LlamaIndex 向量索引：
```bash
python src/rag/ingestion_llama.py
```
*   该过程可能较慢（特别是包含 OCR 时），请耐心等待。
*   索引文件将持久化保存在 `data/llama_vector_store/`。

---

## 🚀 启动智能体

索引构建完成后，即可启动 CLI 交互界面：
```bash
python main.py
```

### 交互示例
*   **查规范**: "敏捷开发和瀑布模型有什么区别？"
*   **查细节**: "需求规格说明书（SRS）应该包含哪些章节？"
*   **多轮追问**: 
    *   User: "什么是 CMMI？"
    *   Agent: (回答...)
    *   User: "**它**有几个等级？" (系统能自动识别“它”指代 CMMI)

---

## 📂 项目结构

```text
nju-se-agent/
├── data/
│   ├── raw/books/          # [输入] 原始文档存放区
│   └── llama_vector_store/ # [输出] 向量索引持久化目录
├── src/
│   ├── agents/             # LangGraph 智能体逻辑 (Router, SE Process, Synthesizer)
│   ├── rag/                # LlamaIndex 引擎 (Ingestion, Retriever)
│   └── tools/              # 工具封装 (Local Retriever, Web Search)
├── config/                 # 配置文件
└── main.py                 # 启动入口
```

## ⚠️ 注意事项
*   **Token 消耗**: 由于开启了 HyDE 和多跳检索，复杂问题的 Token 消耗较大（单次可能几千 Token）。
*   **首次运行**: 第一次运行 `ingestion_llama.py` 时会自动下载 BGE-M3 和 Cross-Encoder 模型（约 2GB），请确保网络通畅。
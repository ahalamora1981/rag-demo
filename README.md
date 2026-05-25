# RAG Demo - 法律知识问答系统

基于 LangGraph + Chroma 的法律文档 RAG（检索增强生成）问答系统，支持 4 部中国法律的多轮对话问答。

## 架构

```
START
  │
context_expansion   ← 多轮对话：结合历史上下文扩展问题
  │
intent_recognition  ← 意图识别：过滤非法律问题
  ├─ legal
  │   question_rewriting  ← 优化检索语句
  │   retrieve            ← Chroma 向量检索 (bge-m3)
  │   generate_answer     ← DeepSeek-V4 生成回答 + 法条引用
  │   generate_references ← 格式参考来源
  │   generate_next_questions ← 推荐 3 个追问
  └─ non-legal
      reject_non_legal    ← 礼貌拒绝 + 引导法律话题
  END
```

## 快速开始

### 环境要求

- Python >= 3.12
- Docker (用于 Chroma)

### 1. 启动 Chroma

```bash
cd chroma-server
docker compose up -d
```

Chroma 运行在 `localhost:8100`。

### 2. 配置 API Key

编辑 `.env`：

```env
LLM_BASE_URL=https://api.deepseek.com
LLM_API_KEY=your_deepseek_key
LLM_MODEL=deepseek-v4-flash
EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_API_KEY=your_siliconflow_key
EMBEDDING_MODEL=BAAI/bge-m3
```

### 3. 安装依赖

```bash
uv sync
```

### 4. 构建索引

```bash
uv run python -m src.main index
```

### 5. 开始对话

```bash
uv run python -m src.main ask
```

交互命令：输入问题回车发送，`clear` 清空历史，`quit` 退出。

## 项目结构

```
rag-demo/
├── chroma-server/          # Chroma Docker 配置
│   ├── docker-compose.yaml
│   └── chroma-data/        # 持久化数据 (gitignore)
├── laws/                   # 法律文档
│   ├── 劳动法.md
│   ├── 劳动合同法.md
│   ├── 民法典.md
│   └── 消费者权益保护法.md
├── src/
│   ├── config.py           # 配置加载
│   ├── embeddings.py       # 嵌入模型 (SiliconFlow bge-m3)
│   ├── timer.py            # 计时装饰器
│   ├── indexing/
│   │   ├── chunker.py      # 文档加载 + 固定切片 (500字/80字重叠)
│   │   └── indexer.py      # Chroma 索引构建
│   ├── graph/
│   │   ├── state.py        # LangGraph 状态定义
│   │   ├── nodes.py        # 各节点实现
│   │   └── graph.py        # 图编排
│   └── main.py             # CLI 入口
└── graph_rag.png           # 流程图
```

## 关键技术点

| 环节 | 方案 |
|------|------|
| 意图识别 | LLM 判断是否为法律问题，非法律问题引导话题 |
| 多轮对话 | 每次先扩展问题（结合历史上下文），再走后续流程 |
| 问题重写 | LLM 将问题改写为关键词密集的检索语句 |
| 切片策略 | 固定 500 字切片、80 字重叠，按句号/换行等自然断点 |
| 嵌入模型 | BAAI/bge-m3 (1024维) |
| 检索 | Chroma L2 距离，召回 5 条 |
| 引用展示 | 法律名 + 法条范围 + 前 100 字原文 |
| 追问推荐 | LLM 基于问答内容生成 3 个相关法律问题 |

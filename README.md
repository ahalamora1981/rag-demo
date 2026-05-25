# RAG Demo - 法律知识问答系统

基于 **LangGraph + Chroma** 的法律文档 RAG（检索增强生成）问答系统，覆盖 4 部中国法律，支持多轮对话，具备意图识别、上下文扩写、问题重写、向量检索、答案生成、来源引用、追问推荐等完整能力。

## 系统流程

```
┌───────────────────────────────────────────────────┐
│                   START                           │
│                     │                             │
│          ┌──────────▼──────────┐                  │
│          │ context_expansion  │  ← 1. 结合历史对话 │
│          │ 上下文扩写          │    重写为独立问题   │
│          └──────────┬──────────┘                  │
│                     │                             │
│          ┌──────────▼──────────┐                  │
│          │ intent_recognition │  ← 2. 判断是否     │
│          │ 意图识别            │    法律相关问题     │
│          └──────┬─────┬───────┘                  │
│          legal  │     │  non-legal               │
│    ┌────────────▼─┐  ┌▼────────────────┐        │
│    │question       │  │reject           │        │
│    │_rewriting     │  │_non_legal       │        │
│    │  ← 3. 检索优化│  │  ← 礼貌拒绝      │        │
│    └──────┬───────┘  └──────────────────┘        │
│           │                                       │
│    ┌──────▼───────┐                              │
│    │ retrieve     │  ← 4. Chroma L2 向量检索      │
│    │              │     (bge-m3, top-5)           │
│    └──────┬───────┘                              │
│           │                                       │
│    ┌──────▼───────┐                              │
│    │generate      │  ← 5. DeepSeek-V4 生成        │
│    │_answer       │     法律条文引用 + 通俗解释     │
│    └──────┬───────┘                              │
│           │                                       │
│    ┌──────▼───────┐                              │
│    │generate      │  ← 6. 提取引用来源             │
│    │_references   │     法条 + 原文预览 + URL      │
│    └──────┬───────┘                              │
│           │                                       │
│    ┌──────▼───────────┐                          │
│    │generate_next     │  ← 7. LLM 生成 3 个       │
│    │_questions        │     相关法律追问           │
│    └──────────────────┘                          │
│                     │                             │
│                   END                             │
└───────────────────────────────────────────────────┘
```

### 各环节详解

#### 1. context_expansion（上下文扩写）

多轮对话场景下，用户追问往往高度依赖上文（如"那工资呢？"），只看当前问题无法理解完整语义。因此**在所有其他步骤之前先执行扩写**，将片段化追问还原为独立完整的问题。

- 首轮单条消息时跳过（零开销）
- 多轮时取最近 2 轮对话，让 LLM 将当前问题与历史合并为自包含的独立问题
- 扩写结果同时用于意图识别和最终回答，保证全链路语义一致

> 例如：上文在讨论试用期，用户问"那工资呢？" → 扩写为"试用期的工资标准是什么？"

#### 2. intent_recognition（意图识别）

使用扩写后的问题（不是原始问题）做意图分类，避免省略型追问被误判为非法律问题。LLM 输出 JSON 格式：

```json
{"is_legal": true, "reason": "属于劳动法范畴"}
```

若为非法律问题（如"今天天气怎么样？"），转入拒绝节点；若为法律问题，继续后续流程。解析失败时**默认按法律问题处理**，保证不丢答。

#### 3. question_rewriting（问题重写）

将自然语言问题改写为**更适合向量检索**的形式：提取法律概念和关键词、去除冗余表达、保持法律术语准确性。改写后的问题作为 `retrieve` 节点的输入。

#### 4. retrieve（向量检索）

- 使用 `BAAI/bge-m3` 将检索语句编码为 1024 维向量
- 在 Chroma 中执行 L2 距离最近邻搜索，返回 top-5 结果
- 每条结果附带：原始文本、来源法律、法条范围、L2 距离分数

#### 5. generate_answer（生成回答）

将检索到的 5 条法律条文作为上下文注入 prompt，要求 LLM：
- 准确引用法律条文（标注"根据《XXX法》第X条"）
- 用通俗中文解释法律术语
- 信息不足时明确告知用户

#### 6. generate_references（引用展示）

从检索结果中提取引用元数据，输出格式：
```
📖 《中华人民共和国劳动合同法》 第十九条~第二十条
   第十九条劳动合同期限三个月以上不满一年...
```
每条引用包含法律名、法条范围、前 100 字原文预览、相关度分数和来源 URL。

#### 7. generate_next_questions（追问推荐）

LLM 基于当前问答内容和涉及法律领域，生成 3 个用户可能继续追问的相关法律问题。短小、独立、有实际价值。

### 为什么 context_expansion 放在 intent 之前

传统的先意图再扩展的流程对多轮对话有害：

```
❌ 旧流程: intent → context_expansion
   "那工资呢？" → 意图判断 → 非法律问题 → 被拒绝
   （明明是合法的追问，因为缺上下文被误杀）

✅ 新流程: context_expansion → intent
   "那工资呢？" → 扩写为"试用期的工资标准是什么？" → 意图判断 → 法律问题 ✓
```

## 快速开始

### 环境要求

- Python >= 3.12
- Docker（用于 Chroma）

### 1. 启动 Chroma

```bash
cd chroma-server
docker compose up -d
```

Chroma 运行在 `localhost:8100`，数据持久化到 `chroma-data/` 目录。

### 2. 配置 API

编辑 `.env`（已 gitignore，需自行创建）：

```env
LLM_BASE_URL=https://api.deepseek.com
LLM_API_KEY=your_deepseek_key
LLM_MODEL=deepseek-v4-flash
LLM_THINKING_ENABLED=false    # 设为 true 开启思考模式
LLM_REASONING_EFFORT=high     # high 或 max
EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_API_KEY=your_siliconflow_key
EMBEDDING_MODEL=BAAI/bge-m3
```

> **为什么 LLM 和 Embedding 用不同的 API？** DeepSeek 官方不提供 Embedding API，因此聊天用 DeepSeek 官方、嵌入用 SiliconFlow（支持 bge-m3）。

### 3. 安装依赖

```bash
uv sync
```

### 4. 构建索引

```bash
uv run python -m src.main index
```

将 `laws/` 目录下的 4 部法律 Markdown 文档加载、切片（500 字/块，80 字重叠，共 329 块），通过 SiliconFlow bge-m3 编码后存入 Chroma。

### 5. 开始对话

```bash
uv run python -m src.main ask
```

命令：输入问题回车，`clear` 清空历史，`quit` 退出。

## 问答示例

### 单轮问答

```
You: 试用期最长是多久？

AI: 根据《中华人民共和国劳动合同法》第十九条的规定，试用期的最长时限取决于
    劳动合同的期限，具体如下：

    - 劳动合同期限三个月以上不满一年的，试用期不得超过一个月；
    - 劳动合同期限一年以上不满三年的，试用期不得超过二个月；
    - 三年以上固定期限和无固定期限的劳动合同，试用期不得超过六个月。

    此外，同一用人单位与同一劳动者只能约定一次试用期...

--- 参考来源 ---
  📖 《中华人民共和国劳动合同法》 第十九条~第二十条
  📖 《中华人民共和国劳动法》 第二十条
  ...

--- 您可能还想问 ---
  1. 试用期工资可以低于转正后的工资吗？有最低标准吗？
  2. 试用期内被公司辞退，能要求经济补偿吗？
  3. 如果公司约定的试用期超过法定最长期限，员工可以主张什么权利？
```

### 非法律问题拒绝

```
You: 今天天气怎么样？

AI: 您好，您的问题不属于法律咨询范畴。我仅能为您提供法律相关问题的解答。

    您可以就以下领域进行咨询：
    - 劳动法（劳动合同、工资、工时、社保等）
    - 劳动合同法（签订、解除、经济补偿等）
    - 民法典（婚姻家庭、继承、合同、侵权等）
    - 消费者权益保护法（消费维权、退货、欺诈赔偿等）
```

### 多轮对话

```
You: 试用期最长是多久？

AI: 试用期最长不得超过六个月...

You: 那工资呢？     ← 省略型追问，但不会误杀

AI: [自动扩写为"试用期的工资标准是什么？"]
    根据《劳动合同法》第二十条规定，试用期工资不得低于本单位相同岗位
    最低档工资或合同约定工资的 80%，且不得低于当地最低工资标准...
```

## 性能

| 环节 | 平均耗时 | 说明 |
|------|---------|------|
| context_expansion | ~0s（首轮） / ~1.5s（多轮） | 首轮跳过 |
| intent_recognition | ~2s | DeepSeek-V4-Flash |
| question_rewriting | ~1.5s | DeepSeek-V4-Flash |
| retrieve | ~0.4s | bge-m3 嵌入 + Chroma 搜索 |
| generate_answer | ~3s | DeepSeek-V4-Flash |
| generate_next_questions | ~2s | DeepSeek-V4-Flash |
| **总耗时** | **~10s** | 不含首轮冷启动 |

> 各节点已内置 `@timer` 装饰器，运行时自动输出耗时日志，方便定位瓶颈。

## 项目结构

```
rag-demo/
├── laws/
│   ├── 劳动法.md                    # 498 行，107 条
│   ├── 劳动合同法.md                # 560 行，98 条
│   ├── 民法典.md                    # 999+ 行，1260 条
│   └── 消费者权益保护法.md          # 356 行，63 条
├── chroma-server/
│   ├── docker-compose.yaml          # Chroma Docker 配置（端口 8100）
│   └── chroma-data/                 # 向量数据持久化（gitignore）
├── src/
│   ├── config.py                    # 全部可调参数
│   ├── embeddings.py                # bge-m3 嵌入（SiliconFlow，批量 32）
│   ├── timer.py                     # @timer 计时装饰器
│   ├── indexing/
│   │   ├── chunker.py               # 文档加载 + RecursiveCharacterTextSplitter
│   │   └── indexer.py               # Chroma 索引构建/获取
│   ├── graph/
│   │   ├── state.py                 # RAGState TypedDict
│   │   ├── nodes.py                 # 7 个节点 + router
│   │   └── graph.py                 # StateGraph 编排
│   └── main.py                      # CLI（index / ask）
├── .env                             # API 密钥配置（gitignore）
├── .gitignore
├── pyproject.toml                   # uv 项目配置
├── uv.lock
└── README.md
```

## 配置参考

`src/config.py` 中的可调参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `LLM_MODEL` | `deepseek-v4-flash` | 对话模型 |
| `LLM_THINKING_ENABLED` | `false` | 是否开启思考模式 |
| `LLM_REASONING_EFFORT` | `high` | 思考强度（`high`/`max`） |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | 嵌入模型 |
| `CHUNK_SIZE` | 500 | 切片字符数 |
| `CHUNK_OVERLAP` | 80 | 切片重叠字符数 |
| `RETRIEVAL_K` | 5 | 检索返回数 |
| `REFERENCE_PREVIEW_CHARS` | 100 | 引用原文预览字符数 |
| `COLLECTION_NAME` | `laws_rag` | Chroma 集合名 |
| `CHROMA_HOST` / `CHROMA_PORT` | `localhost` / `8100` | Chroma 服务地址 |

## 切片策略

使用 `RecursiveCharacterTextSplitter`，按优先级递归切分：

```
"\n\n" → "\n" → "。" → "；" → "，" → " " → ""
```

- 优先在段落边界切分，再退而求其次到句子、逗号，最后按字符硬切
- 500 字/块 + 80 字重叠，兼顾检索精度和上下文完整性
- 切分后自动用正则 `第[一二三四五六七八九十百千零\d]+条` 提取每个块中覆盖的法条范围，存入元数据

4 部法律共产生 329 个 chunks，每条 chunk 携带元数据：`law_name`、`source_file`、`source_url`、`article_range`。

## 思考模式（Thinking Mode）

DeepSeek-V4 支持在输出答案前进行内部推理（思维链），显著提升复杂法律问题的准确性。当前**仅对 `generate_answer` 节点启用**，其余节点（意图、扩写、重写、追问）使用普通模式以降低延迟和成本。

### 启用方式

`.env` 中设置：

```env
LLM_THINKING_ENABLED=true
LLM_REASONING_EFFORT=high    # high 或 max
```

### 延迟对比（问题："劳动者被违法辞退，可以获得哪些赔偿？"）

| 指标 | Thinking 禁用 | Thinking 启用 |
|------|:----:|:----:|
| generate_answer | 10.8s | 14.0s |
| 全流程耗时 | 16.9s | 19.4s |
| 回答字数 | 1596 | 1809 |
| 结构 | 列表式 | 分节式 + 法条原文引用 |

> 思考模式额外开销约 15%，但回答更结构化、分析更深入，适合复杂法律推理场景。

### 参数控制

| 参数 | 位置 | 含义 |
|------|------|------|
| `reasoning_effort: "high"/"max"` | ChatOpenAI 构造参数 | 强度（`low`/`medium` 映射为 `high`） |

说明：DeepSeek 默认 thinking 开关已启用，只需设置 `reasoning_effort` 即可。无需额外配置。

### 节点分布

| 节点 | 思考模式 | 理由 |
|------|---------|------|
| context_expansion | 关闭 | 简单问题合并，无需推理 |
| intent_recognition | 关闭 | 二分类判断，无需推理 |
| question_rewriting | 关闭 | 关键词提取，无需推理 |
| **generate_answer** | **开启** | 法条解读 + 法律推理，适合思维链 |
| generate_next_questions | 关闭 | 简单问题生成，无需推理 |

## 模型选择

| 用途 | 模型 | API | 理由 |
|------|------|-----|------|
| 对话 | `deepseek-v4-flash` | DeepSeek 官方 | 速度快（~2s/次），中文法律理解好，成本低 |
| 嵌入 | `BAAI/bge-m3` | SiliconFlow | 1024 维，中英多语言，法律文本检索效果好 |

## 技术栈

- **LangGraph** — 有状态的多步骤工作流编排（条件分支、状态传递）
- **LangChain** — LLM 调用、向量存储、文本切片的底层抽象
- **Chroma** — 开源向量数据库，Docker 部署，L2 距离检索
- **DeepSeek-V4-Flash** — 主力对话模型
- **BAAI/bge-m3** — 中文法律文本嵌入模型
- **uv** — Python 包管理和虚拟环境

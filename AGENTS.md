# IntelliOps Agent 项目指南

## 项目概述

企业级智能对话和运维助手，支持 RAG 知识库问答和 AIOps 智能故障诊断。
基于 **FastAPI + LangChain + LangGraph** 构建，使用 **阿里云 DashScope (通义千问)** 作为 LLM，
**Milvus** 作为向量数据库，通过 **MCP (Model Context Protocol)** 集成日志查询和监控工具。

## 技术栈

| 层级 | 技术 |
|------|------|
| Web 框架 | FastAPI + Uvicorn |
| LLM 框架 | LangChain + LangGraph |
| LLM 服务 | 阿里云 DashScope (qwen-max / text-embedding-v4) |
| 向量数据库 | Milvus (Docker Compose 部署) |
| 工具协议 | MCP (FastMCP / streamable-http) |
| 前端 | 纯静态 HTML + JS |
| 依赖管理 | uv (推荐) / pip |
| 代码质量 | Ruff + Black + isort + mypy + bandit + pre-commit |
| 日志 | Loguru |
| 测试 | pytest + pytest-cov |
| Python | >=3.11, <3.14 |

## 架构

```
API 层 (api/) → 服务层 (services/) → Agent 层 (agent/) + 工具层 (tools/)
                                        → 核心层 (core/: LLM工厂、Milvus客户端)
                                        → MCP 客户端 (agent/mcp_client.py)
```

关键数据流：
- **Chat**: HTTP/SSE → rag_agent_service (LangGraph) → LLM + 向量检索 → 流式输出
- **AIOps**: HTTP SSE → aiops_service → Planner → Executor (MCP 工具) → Replanner → 诊断报告
- **Document**: HTTP Upload → document_splitter → embedding → Milvus 向量存储

## 目录结构

```
app/
├── main.py                   # FastAPI 入口 + 生命周期 + CORS + 路由注册
├── api/                      # API 路由层
│   ├── chat.py               #   对话接口 (普通/流式)
│   ├── aiops.py              #   AIOps 诊断（流式 SSE）
│   ├── file.py               #   文件上传
│   └── health.py             #   健康检查
├── services/                 # 业务服务层
│   ├── rag_agent_service.py  #   LangGraph RAG Agent
│   ├── aiops_service.py      #   Plan-Execute-Replan 诊断
│   ├── vector_embedding_service.py
│   ├── vector_index_service.py
│   ├── vector_search_service.py
│   └── document_splitter_service.py
├── agent/
│   ├── mcp_client.py         #   MCP 工具调用客户端
│   └── aiops/                #   AIOps Agent 内部状态机
│       ├── state.py          #   状态定义 (Pydantic)
│       ├── planner.py        #   计划制定
│       ├── executor.py       #   步骤执行
│       └── replanner.py      #   重规划
├── models/                   # Pydantic 数据模型
├── tools/                    # Agent 工具 (知识库查询、时间等)
├── core/                     # 核心组件
│   ├── llm_factory.py        #   LLM 工厂
│   └── milvus_client.py      #   Milvus 客户端
└── utils/
    └── logger.py             #   Loguru 配置
mcp_servers/                  # MCP 服务端
static/                       # 前端静态文件
aiops-docs/                   # 运维知识库文档
```

## 编码规范

- **行宽**: 100 字符
- **格式化**: Black + Ruff (`ruff format` / `black`)
- **Import 排序**: isort (profile=black, line_length=100)
- **类型检查**: mypy (宽松模式，忽略第三方库)
- **安全检查**: bandit
- **pre-commit**: 提交前自动运行格式化和检查

### 命名约定

- 文件名、变量、函数: snake_case
- 类名: PascalCase
- 常量: UPPER_SNAKE_CASE
- API 路由: `/api/{resource}`

### 分层约束

- API 层 (`api/`) — 只负责 HTTP 路由解析和参数校验，不包含业务逻辑
- 服务层 (`services/`) — 编排业务逻辑，调用 Agent 和 Core
- Agent 层 (`agent/`) — LangGraph 状态机定义和节点逻辑
- 模型层 (`models/`) — Pydantic 请求/响应模型，不含业务逻辑
- Core 层 (`core/`) — 第三方客户端封装 (LLM, Milvus)
- 禁止 API 层直接调用 Core 层或模型层以外的模块

### 错误处理

- 使用 HTTPException 在 API 层返回错误
- 服务层通过返回值或自定义异常传递错误
- 关键路径必须加 try-except 并记录日志

## 环境配置

通过 `.env` 文件配置（见 README）：

```
DASHSCOPE_API_KEY=your-key
DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen-max
MILVUS_HOST=localhost
MILVUS_PORT=29530
MILVUS_TIMEOUT=10000
RAG_TOP_K=3
CHUNK_MAX_SIZE=800
CHUNK_OVERLAP=100
```

## 常用命令

```bash
# 启动（Linux/macOS）
make start                    # 启动所有服务（MCP + FastAPI）
make dev                      # 开发模式（热重载）
make stop                     # 停止所有服务
make init                     # 一键初始化

# Windows
.\start-windows.bat           # 启动所有服务
.\stop-windows.bat            # 停止所有服务

# 依赖管理
uv pip install -e .           # 安装生产依赖
uv pip install -e ".[dev]"    # 安装开发依赖
uv pip install -e . --upgrade # 同步依赖

# Docker
make up                       # 启动 Milvus
make down                     # 停止 Milvus

# 代码质量
make format                   # 格式化代码
make lint                     # 代码检查
make test                     # 运行测试
make check-all                # 全部检查
make fix                      # 自动修复

# 文档
make upload                   # 上传文档到向量库
make test-upload              # 测试上传单个文件
```

## 测试

- 测试框架: pytest (asyncio_mode=auto)
- 测试目录: `tests/`
- 文件匹配: `test_*.py` / `*_test.py`
- 覆盖率: pytest-cov (app 包)
- 运行: `make test` 或 `pytest tests/ -v`
- 快速: `make test-quick`

## 关键模式说明

### AIOps Plan-Execute-Replan

1. **Planner** — 根据会话上下文生成 4-6 步诊断计划
2. **Executor** — 依次执行计划步骤，调用 MCP 工具（日志查询、监控）
3. **Replanner** — 评估执行结果，决定继续执行、调整计划或生成最终报告
4. 使用 LangGraph 状态机管理流程，通过 SSE 流式输出中间结果

### RAG

1. 文档上传 → 文本分割 (chunk_max_size=800, overlap=100)
2. Embedding (text-embedding-v4) → Milvus 向量存储
3. 用户问题 → 向量检索 (top_k=3) → LLM 上下文增强 → 回答

### MCP 集成

- 使用 streamable-http 传输协议
- 预置两个 MCP 工具: CLS 日志查询 + Monitor 监控数据
- 通过 `agent/mcp_client.py` 统一管理 MCP 连接和工具调用

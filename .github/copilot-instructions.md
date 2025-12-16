# Copilot / AI agent 使用说明（仓库专用）

目的：快速让 AI 编码代理在本仓库中立即可用并高效工作，说明架构要点、开发/运行步骤、以及项目内的约定与示例。

要点概览
- 核心包：`llm_utils/` —— 提供统一 LLM 客户端接口、会话管理、数据类型与日志。参见 `llm_utils/base_client.py`、`llm_utils/conversation.py`。
- 客户端实现：
  - 本地模型：`llm_utils/local_client.py`（基于 HuggingFace，自动下载/加载，默认 `load_on_init=True`，支持 `mirror_url`/`HF_ENDPOINT`）。
  - 在线模型：`llm_utils/online_client.py`（封装 OpenAI/Anthropic/Google/Paratera 等 API；依赖 `openai` 或相应 SDK）。
- 工厂与便捷方法：使用 `llm_utils/factory.py` 中的 `LLMClientFactory` 或 `create_llm_client(...)` 创建客户端。

重要数据流与约定
- API surface：所有客户端继承 `BaseLLMClient`，实现 `chat(...)`, `chat_with_history(...)`, `get_token_probabilities(...)`。
- 响应对象：统一使用 `llm_utils.data_types.LLMResponse`（字段 `content`, `raw_response`, `metadata`）。
  - token 相关：`metadata` 中可能包含 `token_probabilities`（按生成顺序的 token 概率列表）和 `token_distributions`（每个位置的 top-k 分布）。
  - Chain-of-Thought 支持：可用 `LLMResponse.crop_thinking()` 去除 `</think>` 之前的“思考”部分，示例见 `llm_utils/data_types.py`。
- 会话：使用 `client.start_conversation(system_prompt=..., max_history=...)` 或 `Conversation` 管理多轮；`max_history` 会保留首条 system 消息并修剪旧条目。

常见任务 / 开发工作流
- 安装依赖（示例）：
```powershell
python -m pip install torch transformers huggingface_hub tokenizers sentencepiece openai loguru numpy
```
（注意：本项目在 Windows 下使用 `snapshot_download(..., local_dir_use_symlinks=False)`；对于中国网络建议使用 `mirror_url` 或设置 `HF_ENDPOINT` 环境变量。）
- 运行示例/测试：
  - 本地模型示例：运行 `local_model_example.py`（会自动下载并加载模型，可能占用大量显存/磁盘）。
  - 在线模型示例：运行 `online_model_example.py`（需要有效 API key）。
  - 白盒测试/示例：`whitebox_TEST.py` 展示了使用 `crop_thinking()` 与 `get_token_distributions()` 做因果判断的实用样例。
- 避免意外大模型下载：创建 `LocalLLMClient(..., load_on_init=False)`，并在需要时手动调用 `_ensure_model_ready()` / `_load_model()`。

项目特有约定（不要泛化）
- `return_token_probs`/概率追踪：本地客户端默认支持并倾向于返回 token 级概率（见 `LocalLLMClient._generate()`）。代码中多数示例依赖 `response.metadata['token_probabilities']` 与 `token_distributions`。
- token 聚合策略：`get_token_probabilities(..., aggregate='first'|'max'|'mean'|'all')` 的实现及默认行为在 `local_client.py` 中有明确实现，AI 修改时请保留相同字段名与结构。
- 日志：项目使用 `llm_utils.logging_config.setup_logging()`，logger 名称前缀为 `llm_utils`。示例文件在顶部调用 `setup_logging(level='INFO')`。

集成点与外部依赖
- HuggingFace Hub：`huggingface_hub.snapshot_download()`（本地模型自动下载），可通过 `mirror_url` 控制源。
- Transformers / PyTorch：本地推理依赖 `transformers` 的 `AutoTokenizer` 与 `AutoModelForCausalLM`。
- 在线 API：`openai`（或相应 SDK）。`OnlineLLMClient` 的 `API_CONFIGS` 列出了项目常用后端（`paratera`、`openai`、`anthropic`、`google`）。

修改/扩展提示（对 AI 代理）
- 修改客户端接口时，优先保持 `LLMResponse.metadata` 的兼容性（别改字段名）；例如不要替换 `token_probabilities` 或 `token_distributions` 的结构。
- 若增加新 API 后端：将配置放入 `API_CONFIGS` 并在 `_initialize_client()` 中安全地按需导入 SDK；不要假设所有 API 都支持 token-level logprobs。
- 对于自动化测试/CI，避免真实大模型下载：mock `LocalLLMClient._ensure_model_ready()` 或在测试时使用 `load_on_init=False` 并用轻量模型/本地伪造 tokenizer。

参考文件（快速跳转）
- `llm_utils/base_client.py`
- `llm_utils/local_client.py`
- `llm_utils/online_client.py`
- `llm_utils/conversation.py`
- `llm_utils/data_types.py`
- `llm_utils/logging_config.py`
- `local_model_example.py`, `online_model_example.py`, `whitebox_TEST.py`

如果有需要我可以：
- 把以上要点翻译为英文/更长的开发者文档；
- 将常用运行命令添加到 `README.md`；
- 或根据你的 CI 环境写一个不触发大模型下载的测试 harness。

请告知哪些部分不清楚或需要补充，我会迭代更新此文件。

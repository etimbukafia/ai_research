# Harness — File summaries

Short descriptions of the key modules in this package.

- [experiments/harness/agent.py](experiments/harness/agent.py): Base agent abstractions — `AgentConfig`, `BaseAgent` lifecycle, hook/tool/context/session wiring and an `Agent` wrapper for third‑party frameworks.
- [experiments/harness/evaluation.py](experiments/harness/evaluation.py): Lightweight evaluation framework — `EvalCase`/`EvalResult`, evaluators, `TraceHook`, and `EvaluationRunner` which runs datasets and emits JSON reports.
- [experiments/harness/hooks.py](experiments/harness/hooks.py): Flexible hook system — `HookContext`, `HookDecision`, `HookPhase`, `HookRegistry`, built-in hooks (logging, rate-limit, timeout, filter), and decorator helpers.
- [experiments/harness/loop_controller.py](experiments/harness/loop_controller.py): Generic async loop controller — iteration, retry/backoff, safety limits (timeout, budget), and lifecycle hooks.
- [experiments/harness/prompt.py](experiments/harness/prompt.py): `PromptAssembler` using Jinja2 to render system prompts and auto-inject tool descriptors from the `ToolRegistry`.
- [experiments/harness/retriever.py](experiments/harness/retriever.py): Retrieval primitives — `Document`/`SearchResult`, lexical (BM25) and vector retrievers, and a `HybridRetriever` that fuses results.
- [experiments/harness/session.py](experiments/harness/session.py): Session logging API — `SessionEvent`, `SessionManager` protocol, and `JSONLSessionManager` for append/replay of per-session JSONL files.
- [experiments/harness/sub_agents.py](experiments/harness/sub_agents.py): Sub-agent spec and registry — `AgentSpec`, `BaseSubAgent` interface, and `SubAgentRegistry` for registering/discovering sub-agents.
- [experiments/harness/tools.py](experiments/harness/tools.py): Tool registry and `Tool` dataclass — register/execute tools, fire PRE/POST tool hooks, and produce LLM-ready tool descriptors.
- [experiments/harness/utils.py](experiments/harness/utils.py): Small helpers — file read/write/edit, `bash` runner, and `grep_file` utility.
- [experiments/harness/context/manager.py](experiments/harness/context/manager.py): `ContextManager` that holds message history, memory blocks, token accounting, overflow checks, and applies compression strategies.
- [experiments/harness/context/memory.py](experiments/harness/context/memory.py): `MemoryBlock` and `MemoryType` — structured memory with token limits, compression hooks, metadata and lifecycle helpers.
- [experiments/harness/context/policy.py](experiments/harness/context/policy.py): `OverflowPolicy` enum — policies for handling token/context overflow (compress, warn, truncate, evict, etc.).
- [experiments/harness/context/strategies.py](experiments/harness/context/strategies.py): Compression strategies — hierarchical summarization, JIT retrieval, observation masking, token pruning, task-boundary, importance scoring, deduplication, sliding window, `HybridStrategy`, and `production_preset`.
- [experiments/harness/context/tokenizer.py](experiments/harness/context/tokenizer.py): Tokenizer abstractions and implementations — `SimpleTokenizer`, `TiktokenTokenizer`, `HuggingFaceTokenizer`, and `get_tokenizer()` helper.

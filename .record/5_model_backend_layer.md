# SkillOpt 内部文档 5 — 模型后端层（Model Backend Layer）

> **目的 / Purpose**：讲清楚 `skillopt/model/` 怎么把 `chat_target` / `chat_optimizer` 路由到具体后端、optimizer 与 target 如何独立配置、token 怎么计费、重试与 reasoning_effort 怎么生效。改模型相关代码、加新后端、排查“模型没调对/超时/计费不对”时看这里。
>
> 配套：调用点全图见 [`1_architecture_map.md` §5/§6](1_architecture_map.md)；红线见 [`4_invariants_and_gotchas.md` §H](4_invariants_and_gotchas.md)。
> 锚点 `file:line` 基于撰写时 HEAD。

---

## 0. 一句话 / One-liner

> `skillopt/model/__init__.py` 是**唯一活跃的模型 API**：它维护 **optimizer 和 target 两套独立 backend+deployment**，`chat_optimizer` 按 `get_optimizer_backend()` 路由、`chat_target` 按 `get_target_backend()` 路由，分发到 azure/claude/qwen/minimax 具体模块；exec 类后端（codex/claude_code）不走 `chat_target`，由各 env 的 rollout 代码自己驱动。

---

## 1. ⚠️ 两个同名 router，别认错

| 模块 | 状态 | 谁用它 | 特征 |
|---|---|---|---|
| `skillopt/model/__init__.py` | ✅ **活跃/主 API** | trainer、reflect、rollout、aggregate、clip 全部 `from skillopt.model import ...` | optimizer/target **分离**路由；支持 openai/claude/qwen/minimax/codex_exec/claude_code_exec |
| `skillopt/model/router.py` | 🟡 **旧版/遗留** | 几乎无（历史脚本兼容） | **单一全局** backend（仅 azure/codex/claude），用 `REFLACT_MODEL_BACKEND` 环境变量 |

> 🔴 **改模型路由请改 `__init__.py`，不是 `router.py`。** 两者函数同名（`chat_optimizer`/`chat_target`/`set_backend`），极易改错文件。验证：`trainer.py:633` 之后的 `from skillopt.model import (chat_optimizer, ...)` 导入的是 `__init__.py`。

---

## 2. 双角色模型 / Optimizer vs Target（核心设计）

两个角色**各自独立**有 backend 和 deployment，可完全不同（如 optimizer=gpt-5.5，target=claude_code_exec）。

| 维度 | optimizer（改 skill） | target（做任务） | 定义 |
|---|---|---|---|
| backend 变量 | `OPTIMIZER_BACKEND` | `TARGET_BACKEND` | `backend_config.py:15-16` |
| backend setter | `set_optimizer_backend` | `set_target_backend` | `backend_config.py:49/64` |
| deployment 变量 | `OPTIMIZER_DEPLOYMENT` | `TARGET_DEPLOYMENT` | `azure_openai.py:104-105` |
| deployment setter | `set_optimizer_deployment` | `set_target_deployment` | `__init__.py:191/186`（fan-out 到所有模块） |
| 调用入口 | `chat_optimizer` | `chat_target` | `__init__.py:80/119` |

**trainer 启动时的配置序列**（`trainer.py:632-754`）：
```
configure_azure_openai(... 每角色 endpoint/key/auth ...)   trainer.py:634
推断 optimizer_backend / target_backend（按 model.backend 给默认）  trainer.py:670-689
set_optimizer_backend / set_target_backend                trainer.py:690-691
set_optimizer_deployment(cfg["optimizer_model"])          trainer.py:692
set_target_deployment(cfg["target_model"])                trainer.py:693
configure_codex_exec / configure_claude_code_exec / configure_qwen_chat / configure_minimax_chat  trainer.py:694-738
set_reasoning_effort(cfg["reasoning_effort"])             trainer.py:748
```

---

## 3. 后端取值表 / Backend values

| backend 值 | 角色 | 实现模块 | 类型 |
|---|---|---|---|
| `openai_chat` | both | `azure_openai.py` | chat（Azure/OpenAI 兼容） |
| `claude_chat` | both | `claude_backend.py` | chat |
| `qwen_chat` | both | `qwen_backend.py` | chat（本地 vLLM） |
| `minimax_chat` | target only | `minimax_backend.py` | chat |
| `codex_exec` | **target only** | `codex_backend.py` + `codex_harness.py` | **exec**（agentic CLI） |
| `claude_code_exec` | **target only** | `claude_backend.py` | **exec** |

**合法性校验**（`backend_config.py:52/67`）：
- optimizer 只能是 4 个 chat 后端之一（不能 exec）。
- target 可以是 chat 也可以是 exec。

**别名归一化**（`common.py:31-47`，`normalize_backend_name`）：注意几个坑——
- `"openai"` → `"codex"`（不是 openai_chat！）
- `"claude"`/`"anthropic"` → `"claude_chat"`
- `"azure"`/`"azure-openai"` → `"azure_openai"`
- 空值 → `"azure_openai"`

每后端默认模型见 `common.py:19-29`（`_BACKEND_DEFAULT_MODELS`）。

---

## 4. 路由逻辑 / Routing（chat_optimizer / chat_target）

```
chat_optimizer(system, user, ...)                          __init__.py:80
   ├─ get_optimizer_backend()=="claude_chat"  → _claude.chat_optimizer   :89
   ├─ =="qwen_chat"                            → _qwen.chat_optimizer     :98
   └─ else (openai_chat)                       → _openai.chat_optimizer   :108

chat_target(system, user, ...)                             __init__.py:119
   ├─ =="claude_chat"   → _claude.chat_target       :128
   ├─ =="qwen_chat"     → _qwen.chat_target         :137
   ├─ =="minimax_chat"  → _minimax.chat_target      :147
   ├─ not is_target_chat_backend() → raise NotImplementedError  :156
   └─ else (openai_chat) → _openai.chat_target      :161
```

> 🔴 **exec 后端（codex_exec / claude_code_exec）调 `chat_target` 会抛 `NotImplementedError`**（`__init__.py:156-160`）。它们**不走统一 chat 接口**，由 env 专属 rollout 代码用 `get_codex_exec_config()` / `get_claude_code_exec_config()`（`backend_config.py:133/177`）拿配置后自己驱动 CLI。ALFWorld 用的是 chat 路径，所以不涉及；SpreadsheetBench/Codex 类 env 才走 exec。

还有 `*_messages` 变体（多轮/工具调用）：`chat_optimizer_messages` `__init__.py:172`、`chat_target_messages` `__init__.py:220`，路由同构，多了 `tools`/`tool_choice`/`return_message`。

---

## 5. 单次调用内部 / Inside one chat call（以 azure 为例）

`azure_openai.py` 的 `chat_optimizer`（`:732`）/ `chat_target`（`:772`）都委托 `_chat_impl`（`:356`）：

```
_chat_impl(client, deployment, system, user, max_completion_tokens, retries, stage, reasoning_effort, timeout):
   for attempt in range(retries):                    azure_openai.py:371
       ├─ needs_responses_api(model)? 用 Responses API（max_output_tokens）  :376
       │   else 用 Chat Completions（max_completion_tokens）                 :408
       ├─ actual_effort = reasoning_effort or REASONING_EFFORT              :380/410
       ├─ tracker.record(stage, prompt_tokens, completion_tokens)          :424
       └─ 出错 → time.sleep(指数退避) 重试                                   :434
```

### 关键点
- **retries**：`chat_*` 默认 5，但**各阶段实际传 3**（reflect/aggregate/clip 都 `retries=3`）。重试是指数退避（`time.sleep` `:434`）。
- **reasoning_effort 优先级**：单次调用传入 > 全局 `REASONING_EFFORT`（`set_reasoning_effort` 设置，`__init__.py` fan-out 到所有模块 `router.py:181`）。留空字符串 → 回退环境变量（见 `0_start.md` Azure 排错记录：`reasoning_effort` 配错会 400）。
- **Responses API vs Chat Completions**：`needs_responses_api(model)`（`common.py:62`）对 `gpt-5-codex`/`gpt-5.4-pro` 等用 Responses API，参数名和 usage 解析都不同（`max_output_tokens` + `usage_from_responses_usage` `common.py:175`）。

---

## 6. Token 计费 / Token tracking

```
TokenTracker（线程安全）                                    common.py:70
   record(stage, prompt_tokens, completion_tokens)         common.py:75
   summary() → {stage: {calls, prompt_tokens, completion_tokens, total_tokens}, _total: {...}}  common.py:88

全局单例 tracker                                            common.py:117
get_token_summary() / reset_token_tracker()                __init__.py:333/385 → router fan-out
```

- 每次模型调用按 `stage` 累加（`stage` ∈ rollout/analyst/merge/ranking/target/optimizer…）。`stage` **只影响计费分类，不影响路由**。
- trainer 每个 step 做**增量快照**（`tokens_before`/`tokens_after` 差值，`trainer.py:1567-1581`），写进 `step_record.json` 的 `tokens.<stage>`。
- 想知道一次训练各阶段花了多少 token → 看 `step_record.json` 累加或 `summary()` 的 `_total`。

---

## 7. 配置入口 / configure_* 函数（trainer 启动时调）

| 函数 | 配什么 | 定义 |
|---|---|---|
| `configure_azure_openai` | 每角色 endpoint/api_version/api_key/auth_mode/ad_scope/managed_identity | `__init__.py:196`、底层 `azure_openai.py:392` |
| `configure_codex_exec` | codex CLI：path/sandbox/profile/full_auto/reasoning_effort/use_sdk/network/web_search/approval_policy | `backend_config.py:91` |
| `configure_claude_code_exec` | claude CLI：path/profile/use_sdk/effort/max_thinking_tokens | `backend_config.py:148` |
| `configure_qwen_chat` | base_url/api_key/temperature/timeout/max_tokens/enable_thinking（含每角色覆盖） | `qwen_backend.py` |
| `configure_minimax_chat` | base_url/api_key/temperature/max_tokens/enable_thinking | `minimax_backend.py` |

- 所有 exec 配置同时写进**环境变量**（`backend_config.py` 每个 setter 都 `os.environ[...]=`），方便子进程 CLI 读取。
- exec 空响应重试：`EXEC_EMPTY_RESPONSE_RETRIES`（默认 1，`backend_config.py:42`）。
- codex 轨迹回传给 optimizer：`REFLACT_CODEX_TRACE_TO_OPTIMIZER`（`trainer.py:742`，控制 reflect 是否带 codex trace，见 `reflect.py:192`）。

---

## 8. 加一个新后端 / Adding a backend（步骤）

参考官方 `docs/guide/new-backend.md`，要点：

1. **写后端模块** `skillopt/model/yourbackend.py`，至少实现：`chat_optimizer`、`chat_target`、`chat_*_messages`、`set_target_deployment`、`set_optimizer_deployment`、`set_reasoning_effort`、`get_token_summary`、`reset_token_tracker`（签名对齐现有后端；token 用 `common.tracker.record`）。
2. **在 `common.py` 登记**：`_BACKEND_ALIASES`（`:31`）、`_BACKEND_DEFAULT_MODELS`（`:19`）。
3. **在 `backend_config.py` 放行**：`set_target_backend`/`set_optimizer_backend` 的合法集合（`:52/67`），以及 `is_target_chat_backend` 等判定（`:79-88`）。
4. **在 `__init__.py` 加路由分支**：`chat_optimizer`（`:80`）/`chat_target`（`:119`）里 `import` 你的模块并加 `if get_*_backend()=="your_chat"` 分支。
5. **trainer 配置**（可选）：若需专属参数，加 `configure_yourbackend` 并在 `trainer.py:632-738` 调用 + config 项。

> ⚠️ 别忘了 fan-out 函数：`set_reasoning_effort`/`set_*_deployment`/`get_token_summary`/`reset_token_tracker` 在 `__init__.py`/`router.py` 是遍历所有后端模块的（`router.py:181-193`），你的模块要实现这些同名函数否则 fan-out 时报错。

---

## 9. 排错速查 / Symptom → cause

| 症状 | 根因 | 看 |
|---|---|---|
| `NotImplementedError: chat_target only supported with...` | target 是 exec 后端却走了 chat 路径 | env rollout 应自己驱动 exec（§4） |
| 模型调用打到了错误的模型 | optimizer/target deployment 没设对 / 改错了 router.py | `trainer.py:692-693`、§1 |
| 400 reasoning_effort 错误 | effort 值非法 / 模型不支持 | `set_reasoning_effort`、`0_start.md` |
| token 统计为 0 / 不分阶段 | 后端没调 `tracker.record` 或 stage 没传 | `common.py:75`、§6 |
| 改了 `router.py` 没效果 | 改错文件，主 API 是 `__init__.py` | §1 |
| 新后端 fan-out 报 AttributeError | 缺 `set_*_deployment` 等同名函数 | §8 |

---

## 附：改后端层自检 / Checklist
- [ ] 改的是 `__init__.py`（活跃）而非 `router.py`（遗留）？
- [ ] optimizer/target 两套 backend+deployment 都对？
- [ ] exec 后端没误走 chat_target？
- [ ] 新后端在 common.py 别名/默认模型 + backend_config 合法集合 + __init__ 路由分支都登记？
- [ ] fan-out 同名函数（deployment/effort/token）都实现？
- [ ] token 用 `common.tracker.record(stage,...)` 计费？

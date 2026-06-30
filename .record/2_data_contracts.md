# SkillOpt 内部文档 2 — 数据契约（Data Contracts）

> **目的 / Purpose**：把在模块之间“隐式”流动的数据结构全部显式化。SkillOpt 大量用裸 `dict` 在阶段间传递（rollout result、patch、merged_patch…），字段名拼错 / 漏字段 / 语义搞反是**最高频的 bug 来源**。修改 `skillopt/` 任何阶段前，先来这里对一遍契约。
>
> 阅读顺序：本文是“名词表 / schema 手册”；流程怎么串见 [`3_pipeline_stages.md`](3_pipeline_stages.md)。
>
> 约定：所有 `file:line` 锚点均可点击跳转；行号基于撰写时的 `HEAD`，大改后请重新核对。

---

## 0. 全局速查表 / Cheat Sheet

| 契约 Contract | 形态 | 定义处 file:line | 生产者 Producer | 消费者 Consumer |
|---|---|---|---|---|
| **RolloutResult** | `dict`（dataclass 可选） | `skillopt/types.py:104` | `run_alfworld_batch` `rollout.py:298` | `compute_score`、`reflect`、gate |
| **conversation.json** | `list[dict]` | （无类，逐步记录） | `run_alfworld_batch` `rollout.py:256` | `fmt_trajectory` `reflect.py:65` |
| **RawPatch**（analyst 输出） | `dict` | `skillopt/types.py:205` | `run_error/success_analyst_minibatch` `reflect.py:256/366` | `_normalise_patches` `trainer.py:149`、`merge_patches` |
| **Patch / Edit** | `dict` | `skillopt/types.py:72/26` | aggregate / select | `apply_patch_with_report` `skill.py:165` |
| **merged_patch** | `dict`（含 `edits`/`revise_suggestions`/`skill_candidates`） | — | `merge_patches` `aggregate.py:143` | `rank_and_select` `clip.py:25` |
| **BatchSpec** | `dataclass(slots)` | `skillopt/datasets/base.py:41` | dataloader | `adapter.build_env_from_batch` `adapter.py:297` |
| **ALFWorldBatchRun** | `frozen dataclass` | `skillopt/envs/alfworld/adapter.py:24` | `build_env_from_batch` | `adapter.rollout` `adapter.py:320` |
| **GateResult** | `frozen dataclass` | `skillopt/evaluation/gate.py:34` | `evaluate_gate` `gate.py:76` | `trainer.train` `trainer.py:1483` |
| **(hard, soft)** | `tuple[float,float]` | `compute_score` `utils/scoring.py:7` | rollout 后打分 | gate、history |
| **SlowUpdateResult** | `dict` | `skillopt/types.py:249` | `run_slow_update` | trainer epoch 末 `trainer.py:1770` |
| **step_record / history / runtime_state** | `dict` / `list[dict]` | （无类，落盘 JSON） | `trainer.train` | resume、WebUI、人工复盘 |

> **核心心智模型**：dataclass（`types.py`、`gate.py`、`datasets/base.py`）只是**文档化的契约 + round-trip 工具**；运行时阶段间传的几乎都是**裸 dict**，靠字段名约定对齐。所以“字段名”就是契约本身。

---

## 1. RolloutResult — rollout 阶段的产物 / 一切的起点

**定义**：`skillopt/types.py:104`（`RolloutResult` dataclass，但运行时多以 dict 流动）。
**生产**：`run_alfworld_batch` 末尾构造，`skillopt/envs/alfworld/rollout.py:298-309`。
**消费**：`compute_score`（`utils/scoring.py:7`）、`adapter.reflect`（`adapter.py:428`）、gate 验证。

### 字段契约 / Field contract

| 字段 | 类型 | 必填? | 含义 | 谁用 |
|---|---|---|---|---|
| `id` | str | ✅ | 任务唯一 ID，决定 `predictions/<id>/conversation.json` 路径 | reflect 读轨迹靠它 |
| `hard` | int(0/1) | ✅ | 硬指标：任务是否**完整成功**。打分、gate 的主依据 | `compute_score`、gate |
| `soft` | float[0,1] | ✅ | 软指标：部分得分 / F1。ALFWorld 里 = hard | gate（metric=soft/mixed 时） |
| `n_turns` | int | ⬜ | 走了多少步 | 日志 / 失败分析 header |
| `fail_reason` | str | ⬜ | 失败原因（"Timeout..." / "Episode ended..."），见 `rollout.py:291-296` | reflect header |
| `task_type` | str | ⬜ | 任务类型（`pick_and_place` 等） | 分桶统计 `_compute_task_type_buckets` `trainer.py:447` |
| `task_description` | str | ⬜ | 任务自然语言描述 | reflect header |
| `reference_text` | str | ⬜ | 隐藏参考答案 / 计划（ALFWorld 用 `build_reference_text` `adapter.py:196`） | reflect header `reflect.py:160` |
| `target_system_prompt` / `target_user_prompt` | str | ⬜ | agent 当时看到的 prompt | reflect header `reflect.py:168/180` |
| `extras` | dict | ⬜ | env 专属字段的兜底容器 | `from_dict` 自动收集 `types.py:142` |

### ⚠️ 不变量 / Invariants（违反 = bug）

1. **`id` 必须与 `predictions/<id>/conversation.json` 的目录名一致**。reflect 阶段按 `item["id"]` 去找轨迹文件（`reflect.py:140-143`），对不上 → 该轨迹被静默跳过、无梯度。
2. **`hard` 的判定口径**：`compute_score` 把 `hard` 当 float 求平均（`utils/scoring.py:21`），所以 `hard` 可以是连续值（平滑奖励），但 ALFWorld 只用 0/1。失败/成功分流用 `hard < 1e-9` 判失败（`reflect.py:537`）。
3. **新增 env 字段**：要么进 `RolloutResult` 的显式字段，要么靠 `extras` 兜底。如果你直接往 dict 塞新 key，`from_dict` 会归到 `extras`（`types.py:142`），`to_dict` 再展开（`types.py:175`）——但**只有 reflect 显式读取的字段才会进 prompt**，光塞不读 = 没用。

---

## 2. conversation.json — 逐步轨迹 / target 模型的“答卷”

**生产**：`run_alfworld_batch` 每步追加，落盘于 `rollout.py:256-266` + `313-317`。
**消费**：`fmt_trajectory`（`reflect.py:65`）渲染成 analyst 可读文本。

每个元素（一步）的契约：

| key | 含义 | 渲染为 |
|---|---|---|
| `step` | 第几步 | `[step N ...]` 前缀 |
| `action` | `<action>` 标签内动作（`_extract_action` `rollout.py:40`） | `[step N action]` |
| `reasoning` | `<think>` 标签内推理 | `[step N think]` |
| `model_response` | target 原始完整响应 | （存档，不直接渲染） |
| `env_feedback` | 环境返回观测 | `[step N obs]` |
| `reward` / `done` | 即时奖励 / 是否结束 | （存档） |

> `fmt_trajectory` 还能识别**别的格式**（Codex 的 `tool_call`/`cmd`/`obs`、通用 `role`/`content`），见 `reflect.py:83-104`。给新 env 写 rollout 时，只要 conversation 元素含 `action`+`env_feedback` 或 `content`，就能被复用。
> **关键**：`_clip_text`（`reflect.py:54`）**故意不截断**——全量喂给 optimizer。轨迹很长时这是 token 成本与上下文窗口的主要压力点。

---

## 3. Edit / Patch — “文本梯度”的原子单位

**定义**：`Edit` `skillopt/types.py:26`，`Patch` `skillopt/types.py:72`，`EditOp` 字面量 `types.py:23`。
**消费（应用处）**：`apply_patch_with_report` `skillopt/optimizer/skill.py:165` → 逐条 `_apply_edit_with_report` `skill.py:85`。

### Edit 字段 / Fields

| 字段 | 类型 | 含义 |
|---|---|---|
| `op` | `"append"\|"insert_after"\|"replace"\|"delete"` | 操作类型 |
| `content` | str | append/insert/replace 要写入的 markdown |
| `target` | str | **精确文本锚点**：insert_after/replace/delete 用来定位 |
| `support_count` | int? | 该编辑由多少条轨迹支持（合并时累加，`trainer.py:169`） |
| `source_type` | `"failure"\|"success"?` | 来自失败还是成功分析 |
| `merge_level` | int? | aggregate 第几层合并产生（`aggregate.py:57`） |

### 四种 op 的精确语义 / Exact op semantics（`skill.py:98-145`）

| op | target 要求 | 行为 | target 找不到时 |
|---|---|---|---|
| `append` | 不需要 | 追加到文档尾部（受保护区之前） | — |
| `insert_after` | 精确文本 | 在 target 所在行之后插入 | **fallback 成 append**（`skill.py:109-117`） |
| `replace` | 精确文本 | `str.replace(target, content, 1)` 只替换第一处 | **跳过**，status=`skipped_replace_target_not_found` |
| `delete` | 精确文本 | `str.replace(target, "", 1)` | **跳过**，status=`skipped_delete_target_not_found` |

### ⚠️ 不变量（这是 patch 类 bug 的重灾区）

1. **`target` 必须是 skill 文档里逐字节存在的子串**，否则 replace/delete 被静默跳过（不报错、不改文档）。LLM 经常给出“近似但不精确”的 target → 编辑丢失。排查时看 `steps/step_XXXX/edit_apply_report.json` 的 `status`。
2. **保护区不可改**：`<!-- SLOW_UPDATE_START/END -->` 和 `<!-- APPENDIX_START/END -->`（`skill.py:14-30`）。target 落在保护区内 → status=`skipped_protected_region`（`skill.py:94`）。append/insert 的 fallback 会插到**最早一个保护区之前**（`skill.py:99-104`），保证保护块永远在文档尾部。
3. **edit 的 content 里若含保护区 marker 会被剥掉**（`_strip_slow_update_markers` `skill.py:66`），防止重复 marker 破坏区块解析。
4. **`apply_patch_with_report` 永不抛异常**：单条 edit 出错记 `status="error"` 继续（`skill.py:176-184`）。所以“编辑没生效”不会让训练崩，只会**静默无效**——必须看 report 才知道。

---

## 4. RawPatch — analyst 的原始输出（带出处）

**定义**：`skillopt/types.py:205`。**生产**：`reflect.py:344-360`（失败）/ `reflect.py:438-444`（成功）。

LLM 实际返回、`extract_json` 解析后的 dict 形态（契约由 prompt 规定，见 `skillopt/envs/alfworld/prompts/analyst_error.md` 末尾）：

```jsonc
{
  "batch_size": 8,                       // 本组分析了几条轨迹
  "failure_summary": [                   // 仅失败侧；进 step_buffer 供后续参考
    {"failure_type": "wrong_sequence", "count": 5, "description": "..."}
  ],
  "patch": {                             // ← 真正的梯度
    "reasoning": "...",
    "edits": [ {"op":"append","content":"..."}, ... ]
  },
  "source_type": "failure",              // 由代码注入，不是 LLM 给的（reflect.py:345）
  "appendix_notes": [...]                // 仅 skill-aware 开启时
}
```

### ⚠️ 不变量

1. **`source_type` 由代码强制写入**，不信任 LLM 输出：失败 analyst 一律 `"failure"`（`reflect.py:345`），成功一律 `"success"`（`reflect.py:439`）。
2. **预算双重裁剪**：analyst 内 `truncate_payload(result["patch"], edit_budget, mode)`（`reflect.py:347`）保证单组 ≤ L 条；后续 SELECT 阶段再裁一次（见文档 3）。
3. **空 patch 也可能合法**：skill-aware 模式下某组可能只产 appendix notes、`edits: []`（`reflect.py:355-360`）——这是为了让 notes 能传回 trainer，`_normalise_patches` 会把空 edits 丢出 body 流水线（`trainer.py:166-167`）。
4. **解析失败返回 `None`**：`extract_json` 失败 / 无 `"patch"` → 返回 None（`reflect.py:341-363`），调用方 `run_minibatch_reflect` 只把非 None 的存盘（`reflect.py:624`）。`raw_patches` 列表里可能混 None，下游 `_normalise_patches` 用 `isinstance(p, dict)` 过滤（`trainer.py:161`）。

---

## 5. payload 多态：edits / revise_suggestions / skill_candidates

这是一个**容易踩的设计**：patch 内部的“载荷 key”随 `skill_update_mode` 变化。统一由 `skillopt/optimizer/update_modes.py` 处理。

| update_mode（归一化后） | payload key | 单位词 | 定义 |
|---|---|---|---|
| `patch`（默认） | `edits` | edit | `update_modes.py:6` |
| `rewrite_from_suggestions` | `revise_suggestions` | suggestion | `update_modes.py:7` |
| `full_rewrite_minibatch` | `skill_candidates` | skill candidate | `update_modes.py:8` |

**永远用 helper 取载荷，不要硬编码 `patch["edits"]`**：
- `payload_key(mode)` `update_modes.py:36`
- `get_payload_items(container, mode)` `update_modes.py:52` ← 取载荷列表
- `set_payload_items` `update_modes.py:59`
- `truncate_payload` `update_modes.py:64`
- `normalize_update_mode` `update_modes.py:11`（接受一堆别名）

> ⚠️ 如果你新写代码直接 `patch.get("edits")`，在 rewrite 模式下会拿到空列表 → 编辑静默丢失。**改任何涉及 patch 的代码，先过一遍 `get_payload_items`。**

---

## 6. BatchSpec — dataloader → adapter 的批次请求

**定义**：`skillopt/datasets/base.py:41`（`@dataclass(slots=True)`）。

| 字段 | 含义 |
|---|---|
| `phase` | `"train"` / `"eval"` |
| `split` | split 名（`"train"` / eval split 如 `"valid_seen"`） |
| `seed` | 决定批次的确定性种子 |
| `batch_size` | 该批 item/episode 数 |
| `payload` | env 专属载荷；数据集型 env 是 item list，模拟器型(ALFWorld) 可为 None |
| `metadata` | 结构化元信息（ALFWorld 在这里塞 `gamefiles`/`result_ids`/`eval_dataset`/`is_train`，见 `adapter.py:298-305`） |

**消费**：`adapter.build_env_from_batch(batch)`（`adapter.py:297`）把它翻译成具体 env 句柄（ALFWorld → `ALFWorldBatchRun`）。

### 确定性种子约定（resume 依赖）
- `make_base_seeds(steps_per_epoch, accumulation, seed)` → `[seed+1, seed+2, ...]`（`datasets/base.py:97`）。
- `shuffle_epoch_seeds(base_seeds, epoch, seed)` 用 `Random(seed + epoch*1000)` 洗牌（`datasets/base.py:104`）。
- ⚠️ **改种子公式 = 改变批次组成 = 破坏断点续训的可复现性**。

---

## 7. ALFWorldBatchRun — 惰性批次描述（防 OOM）

**定义**：`skillopt/envs/alfworld/adapter.py:24`（`@dataclass(frozen=True)`）。

| 字段 | 含义 |
|---|---|
| `env_num` | 这一批要跑多少个 episode（`__len__` 返回它，`adapter.py:44`） |
| `eval_dataset` | `eval_in_distribution` / `eval_out_of_distribution` / train |
| `seed` / `is_train` / `workers` | 跑批参数 |
| `specific_gamefiles` | 指定要跑的 ALFWorld gamefile 列表 |
| `result_ids` | 与 gamefiles 对齐的结果 ID（决定 result["id"]） |
| `items` | item 元数据（`__iter__` 遍历它，`adapter.py:42`） |

**关键设计**：它**不持有任何打开的模拟器**，只是清单。真正开模拟器在 `_run_batch`（`adapter.py:371`）里**按 `workers` 分块**进行：每块 `build_alfworld_env`（`rollout.py:71`，此处才 import vendor）→ `run_alfworld_batch` → `finally: _close_env`（`adapter.py:424`）。这样几百个任务不会同时开几百个模拟器。详见 [`3_pipeline_stages.md` §ROLLOUT](3_pipeline_stages.md)。

---

## 8. GateResult + (hard, soft) — 验证门控的输入/输出

**定义**：`GateResult` `skillopt/evaluation/gate.py:34`（frozen），`evaluate_gate` `gate.py:76`，`select_gate_score` `gate.py:46`。

### (hard, soft) 投影成单一比较分（`gate.py:46-73`）
```
metric="hard"  → hard
metric="soft"  → soft
metric="mixed" → (1-w)*hard + w*soft     # w = gate_mixed_weight ∈ [0,1]
```

### GateResult 字段
| 字段 | 含义 |
|---|---|
| `action` | `"accept_new_best"` / `"accept"` / `"reject"`（`gate.py:30`） |
| `current_skill` / `current_score` | 接受→候选；拒绝→回退原值 |
| `best_skill` / `best_score` / `best_step` | 最优快照（仅 new_best 时更新） |

### ⚠️ 不变量
1. **严格大于才接受**：`cand_score > current_score`（`gate.py:123`），相等也拒绝。这是“只在严格改进时更新”的核心。
2. **gate 是纯函数**：不调模型、不落盘、不打印。所有副作用（rollout 打分、缓存、状态写回）由 trainer 负责（`gate.py:7-8`）。trainer 在 `trainer.py:1483-1493` 把 GateResult 的字段写回训练状态。
3. **`use_gate=False` 时不走 evaluate_gate**：trainer 自己构造一个 `action="force_accept"` 的 GateResult（`trainer.py:1470-1477`），验证仍跑（分数照记），只是无条件接受。

---

## 9. 落盘产物契约 / On-disk artifacts（resume + 复盘的真相源）

输出根目录 `out_root = outputs/<run_name>/`，结构（trainer 各处 `_save_*`）：

```
out_root/
├── config.json                 # 扁平化+脱敏后的完整 cfg（trainer.py:816, _redact_cfg:342）
├── history.json                # list[step_record]，每步一条（_save_history:357）
├── runtime_state.json          # resume 指针（_save_runtime_state:404）
├── best_skill.md               # 当前验证最优 skill（trainer.py:1593）
├── lr_history.jsonl            # autonomous LR 决策流水（trainer.py:1266）
├── skills/
│   ├── skill_v0000.md          # 初始 skill 快照（_save_skill:363）
│   └── skill_vXXXX.md          # 每步结束的 current_skill 快照
├── selection_eval_baseline/    # 初始 skill 的基线验证 rollout（trainer.py:1001）
├── steps/step_XXXX/
│   ├── rollout/
│   │   ├── results.jsonl       # 该批 RolloutResult（adapter.py:359；resume 靠它）
│   │   └── predictions/<id>/conversation.json
│   ├── patches/minibatch_{fail,succ}_NNN.json  # analyst 输出（reflect.py:625）
│   ├── merged_patch.json       # AGGREGATE 输出（trainer.py:1229）
│   ├── ranked_edits.json       # SELECT 输出（trainer.py:1280）
│   ├── lr_decision.json        # autonomous LR（trainer.py:1264）
│   ├── candidate_skill.md      # UPDATE 后候选（trainer.py:1361）
│   ├── edit_apply_report.json  # 每条 edit 的 status（trainer.py:1364）
│   ├── selection_eval/         # EVALUATE 的验证 rollout
│   ├── appendix_notes.json     # skill-aware（trainer.py:111）
│   ├── trajectory_digest.json  # step_buffer 条目（trainer.py:1562）
│   └── step_record.json        # 该步全量记录（trainer.py:1598）
├── slow_update/epoch_XX/        # epoch 末慢更新（comparison_pairs.json, candidate_skill.md, slow_result.json）
└── meta_skill/epoch_XX/meta_skill_result.json
```

### runtime_state.json 字段（resume 的核心，`trainer.py:931-946`）
`last_completed_step`、`current_skill_path`、`current_score`、`current_origin`、`best_skill_path`、`best_score`、`best_step`、`best_origin`。

### step_record.json 关键字段（`trainer.py` 全程累加进 `step_rec`）
`step/epoch/step_in_epoch`、`rollout_hard/soft/n`、`n_failure_patches/n_success_patches`、`n_edits_merged/ranked`、`edit_budget`、`lr_control_mode`、`candidate_hash`、`selection_hard/soft`、`candidate_gate_score`、`action`、`current_score/best_score/best_step`、`current_origin/best_origin`、`skill_len`、`timing.{rollout,reflect,aggregate,select,evaluate}_s`、`tokens.<stage>.{calls,prompt_tokens,completion_tokens}`、`wall_time_s`。

### ⚠️ resume 不变量
1. **resume 靠两层**：`runtime_state.json`（优先）或 `history.json`（回退），见 `trainer.py:856-917`。两者缺失才从头。
2. **阶段级断点**：每个 `results.jsonl`（`adapter.py:331-340`）和每个 `minibatch_*.json`（`reflect.py:560-575`）存在即跳过重算。**改产物路径/文件名 = 破坏断点续训**，已跑的步会被重跑或读不到。
3. **`candidate_hash` = `skill_hash(candidate_skill)`**（`utils/scoring.py:26`，sha256 前16位）。验证分数按 hash 缓存（`sel_cache`，`trainer.py:949`、`1420`），同一候选不重复 rollout。

---

## 10. 配置键映射 ⚠️ 极易踩坑 / Config key remapping

YAML 里的**嵌套键**在进 trainer 前被 `flatten_config`（`skillopt/config.py:185`）**扁平化并重命名**。trainer 读的是扁平名，和 YAML 名**不一样**：

| YAML 键（你在 config 里写的） | trainer 里读的扁平键 | 定义 |
|---|---|---|
| `optimizer.learning_rate` | `cfg["edit_budget"]` | `config.py:111` |
| `optimizer.min_learning_rate` | `cfg["min_edit_budget"]` | `config.py:112` |
| `evaluation.sel_env_num` | `cfg["sel_env_num"]` | `config.py:128` |
| `env.skill_init` | `cfg["skill_init"]` | `config.py:132` |
| `env.out_root` | `cfg["out_root"]` | `config.py:133` |
| `model.backend` | `cfg["model_backend"]` | `config.py:32` |

> ⚠️ **后果**：在 `trainer.py` 里 grep `learning_rate` 搜不到任何东西——它叫 `edit_budget`（scheduler 用 `max_lr=cfg["edit_budget"]`，`trainer.py:823`）。改配置项时要同时知道两个名字：CLI/YAML 用嵌套名，代码里用扁平名。新增配置项记得在 `config.py` 的映射表里登记，否则 trainer 读不到。

---

## 附：契约级改动检查清单 / Contract change checklist

改任何阶段前，按此核对（防回归）：

- [ ] 新增 rollout 字段 → 是进 `RolloutResult` 显式字段还是 `extras`？reflect 是否真的读它？
- [ ] 改 patch 结构 → 是否用了 `get_payload_items`/`payload_key` 而非硬编码 `["edits"]`？三种 update_mode 都测了吗？
- [ ] 改 `result["id"]` 生成逻辑 → 是否仍与 `predictions/<id>/` 目录名一致？
- [ ] 改产物路径/文件名 → resume（`results.jsonl`/`minibatch_*.json`/`runtime_state.json`）还成立吗？
- [ ] 改 gate 比较逻辑 → 是否保持“严格大于”和纯函数性质？
- [ ] 新增配置项 → 在 `config.py` 扁平化映射里登记了吗？YAML 名与代码名都对吗？
- [ ] 改保护区逻辑 → slow_update / appendix 两个区块的 marker 都同步了吗？

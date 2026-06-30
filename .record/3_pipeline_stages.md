# SkillOpt 内部文档 3 — 六阶段流水线逐段深挖（Pipeline Stages）

> **目的 / Purpose**：把 `ReflACTTrainer.train()`（`skillopt/engine/trainer.py:597`，约 800 行的主循环）拆成可查的小节。每段给出：**输入 / 输出 / 调不调大模型 / 关键函数 file:line / 落盘产物 / 跳过(skip)条件 / 改动红线**。
>
> 配套：数据结构定义见 [`2_data_contracts.md`](2_data_contracts.md)；本文聚焦“流程怎么串、何时调模型”。
>
> 术语：**target** = 被训练/被测模型（做任务）；**optimizer** = 改 skill 的模型（反思/合并/排序）。两者由 `model.target` / `model.optimizer` 指定，可不同。

---

## 0. 总览 / The big picture

```
ReflACTTrainer.train()                                    trainer.py:597
│
├─ 准备：setup → 配置模型后端 → 载入初始 skill → 算 train_size/steps → resume 检查 → 基线评估
│
└─ for epoch in 1..num_epochs:                            trainer.py:1026
     │  载入上一 epoch 的 meta_skill；新建 step_buffer
     └─ for step in steps_per_epoch:                      trainer.py:1062
          │
          ├─ for a in accumulation:                       trainer.py:1096
          │    ① ROLLOUT   adapter.rollout()      → target 模型   trainer.py:1119
          │    ② REFLECT   adapter.reflect()      → optimizer 模型 trainer.py:1131
          │
          ├─ ③ AGGREGATE  merge_patches()         → optimizer 模型 trainer.py:1220
          ├─ ④ SELECT     rank_and_select()       → optimizer 模型 trainer.py:1238
          ├─ ⑤ UPDATE     apply_patch_with_report()→ 通常不调模型  trainer.py:1308
          ├─ ⑥ EVALUATE   adapter.rollout()+gate  → target 模型   trainer.py:1418
          └─ 写 step_buffer / 存盘 / 更新 runtime_state
     │
     ├─ SLOW UPDATE (epoch≥2)  run_slow_update()   → optimizer 模型 trainer.py:1617
     └─ META SKILL  run_meta_skill()               → optimizer 模型
│
└─ 最终 TEST 评估（valid_unseen / test 集，用 best_skill）
```

**模型调用点共 6 类**：target 在 ①ROLLOUT、⑥EVALUATE、SLOW UPDATE 的 rollout；optimizer 在 ②REFLECT、③AGGREGATE、④SELECT、（可选 ⑤rewrite）、SLOW UPDATE 反思、META SKILL。
**部署期零模型调用**：训练产物只是一份 `best_skill.md`。

---

## 准备阶段 / Setup（train() 开头）

| 子步骤 | file:line | 说明 |
|---|---|---|
| adapter.setup + 取 dataloader | `trainer.py:605-606` | 一次性初始化 |
| 配置 Azure / Codex / Claude / Qwen / MiniMax 后端 | `trainer.py:634-746` | 决定后续 `chat_target`/`chat_optimizer` 打到哪 |
| 推断 optimizer/target backend | `trainer.py:670-693` | 没显式配则按 `model.backend` 推默认 |
| 设 reasoning_effort | `trainer.py:747-748` | 全局 |
| 载入初始 skill | `trainer.py:768-776` | 读 `cfg["skill_init"]`（ALFWorld: `envs/alfworld/skills/initial.md`），不存在则空串 |
| 算训练规模 | `trainer.py:801-804` | `steps_per_epoch=ceil(train_size/(batch_size*accumulation))`，`total_steps=num_epochs*steps_per_epoch` |
| 建 LR scheduler | `trainer.py:821-826` | `max_lr=cfg["edit_budget"]`（注意：YAML 叫 `optimizer.learning_rate`，见文档2 §10） |
| resume 检查 | `trainer.py:856-917` | `runtime_state.json` 优先，否则 `history.json`，否则从头 |
| 基线评估 | `trainer.py:991-1017` | `current_score<0` 时，用初始 skill 在 selection 集(`valid_seen`)跑一遍打基准分 → **这里调一次 target** |

> ⚠️ 基线分 `current_score` 是后续 gate 的初始“门槛”。如果初始 skill 很强，后面很难超过它 → 大量 reject 属正常。

---

## ① ROLLOUT — 采样 / “前向传播”

**作用**：用**当前 skill** 让 target 模型实际去做一批任务，得到成败轨迹。类比 NN 的前向传播（产生预测）。

**调用链**：
```
trainer.py:1122  adapter.rollout(train_env, current_skill, rollout_dir, use_eval_feedback=True)
  └─ adapter.py:320  ALFWorldAdapter.rollout
       ├─ adapter.py:331  resume：results.jsonl 存在则直接读回，跳过整段
       ├─ adapter.py:342  isinstance(ALFWorldBatchRun) → _run_batch（分块开模拟器）
       │    └─ adapter.py:371  _run_batch：按 workers 切块
       │         ├─ rollout.py:71   build_alfworld_env  ← 此处才 import vendor 并开模拟器
       │         ├─ rollout.py:129  run_alfworld_batch  ← 真正跑
       │         │    └─ rollout.py:211  chat_target(...)  ← 唯一的 target 模型调用
       │         └─ adapter.py:424  finally _close_env    ← 跑完立刻关，防 OOM
       └─ adapter.py:359  把 results 写 results.jsonl
```

**输入**：`current_skill`（str）、`train_env`（`ALFWorldBatchRun`）、`rollout_dir`。
**输出**：`list[RolloutResult-dict]`（见文档2 §1）+ `predictions/<id>/conversation.json`（文档2 §2）+ `results.jsonl`。
**模型**：target，每个 episode 每步一次，`max_api_workers` 并发（`rollout.py:228`）。

**skill 如何起作用**：`_build_skill_prompt`（`rollout.py:50`）把 skill 包成 `## Skill Knowledge ...`，每步拼到 `obs["text"][i]` 前（`rollout.py:199-201`），再喂给 `chat_target`。`obs["text"]` 本身是 vendor 用 `rollout_with_history.md` 等模板渲染的（见文档1）。

**容错**：空响应 / 缺 `<action>` / 异常 → 兜底成 `<action>look</action>`（`rollout.py:221-226`），保证 episode 不崩。

### ⚠️ 红线
- **resume 依赖 `results.jsonl`**：`rollout_dir` 下已有它就整段跳过（`adapter.py:331`）。调试时若想强制重跑，删掉对应 step 的 rollout 目录。
- **分块大小 = `workers`**（`adapter.py:384`）。`workers` 太大 → 同时开太多模拟器 → OOM；太小 → 慢。
- `accumulation>1` 时一个 step 跑多批（`trainer.py:1096`），各批 rollout 结果用 `n_envs` 加权平均（`trainer.py:1173-1175`）。

---

## ② REFLECT — 反思 / “求梯度”

**作用**：optimizer 模型读失败/成功轨迹，产出对 skill 的修改建议（RawPatch）。类比反向传播算梯度——“文本梯度”。

**调用链**：
```
trainer.py:1138  adapter.reflect(rollout_results, current_skill, batch_dir, ...)
  └─ adapter.py:428  ALFWorldAdapter.reflect → 取 analyst prompt：
       │   get_error_minibatch_prompt() / get_success_minibatch_prompt()  base.py:285/298
       │   （load_prompt 优先 envs/alfworld/prompts/analyst_*.md，回退 skillopt/prompts/）
       └─ reflect.py:472  run_minibatch_reflect
            ├─ reflect.py:537-538  失败/成功分流（hard<1e-9 判失败）
            ├─ reflect.py:540-545  确定性洗牌 + 按 minibatch_size(M) 切组
            ├─ reflect.py:560-575  resume：minibatch_*.json 存在则读回
            └─ reflect.py:613-633  ThreadPool 并行，每组一次：
                 ├─ reflect.py:256  run_error_analyst_minibatch
                 │    ├─ reflect.py:304  fmt_minibatch_trajectories（读 conversation.json 拼文本）
                 │    └─ reflect.py:334  chat_optimizer(stage="analyst")  ← optimizer 调用
                 └─ reflect.py:366  run_success_analyst_minibatch（同构）
```

**输入**：`rollout_results`、`current_skill`、`step_buffer_context`（本 epoch 历史失败模式+被拒编辑）、`meta_skill_context`。
**输出**：`raw_patches: list[dict|None]`（文档2 §4）+ `patches/minibatch_*.json`。
**模型**：optimizer，调用次数 = 失败组数 + 成功组数（**不是每条轨迹一次**，这是 minibatch 反思省钱处）。

**prompt 组装**（`reflect.py:308-331`）：`## Current Skill` + `## Edit Budget (L=...)` + `## Previous Steps`（step_buffer）+ meta_skill + `## Failed Trajectories`（渲染的轨迹）。

### ⚠️ 红线
- **`failure_only`**（`reflect.py:538`）：True 时不分析成功轨迹，省一半成本但丢失“固化成功经验”信号。
- **`random_seed` 决定分组**（`reflect.py:540`）：resume 时必须一致，否则 minibatch 组成变化、缓存失效。
- analyst 返回 None（解析失败）很常见，下游已用 `isinstance(p, dict)` 兜底（`trainer.py:161`）；但**大量 None = optimizer prompt 或模型有问题**，要看 patches 目录是否为空。
- 详细的“轨迹→patch”机制见已有讲解（analyst prompt schema 在 `analyst_error.md` 末尾）。

---

## 归一化 / Normalize（②③ 之间）

**作用**：把 raw_patches 拆成 failure / success 两组，丢掉空载荷。

```
trainer.py:1145  _normalise_patches(raw_patches, update_mode)   定义 trainer.py:149
   ├─ 过滤非 dict（含 None）              trainer.py:161
   ├─ get_payload_items 取载荷，空则丢弃   trainer.py:166-167
   ├─ 给每条 item 注入 source_type/support_count  trainer.py:170-173
   └─ 按 source_type 分到 failure/success  trainer.py:174-177
```

**skip 分支①**：若 failure 和 success 都空（`trainer.py:1197`）→ `action="skip_no_patches"`，skill 不变，存盘后 `continue`。
⚠️ skill-aware 开启时，**skip 前必须 flush appendix notes**（`trainer.py:1201-1204` → `_flush_skill_aware_appendix` `trainer.py:81`），否则 lapse-only 步的 notes 被静默丢弃。

---

## ③ AGGREGATE — 聚合 / “梯度累积”

**作用**：把多组独立 patch 分层合并成一份。失败优先于成功。类比梯度累积/平均。

**调用链**：
```
trainer.py:1222  merge_patches(current_skill, all_failure_patches, all_success_patches,
                               batch_size=merge_bs, workers, update_mode, meta_skill_context)
  └─ aggregate.py:143  merge_patches
       ├─ aggregate.py:168-180  按 update_mode 选 merge_*.md prompt
       ├─ aggregate.py:182  _hierarchical_merge(failure)  ← 失败组分层合并（并行）
       ├─ aggregate.py:188  _hierarchical_merge(success)  ← 成功组
       └─ aggregate.py:230  最终合并两组（失败 HIGH priority）→ chat_optimizer(stage="merge")
            └─ _hierarchical_merge 内每层 _merge_batch  aggregate.py:28 → chat_optimizer
```

**输入**：failure/success patch 列表、`current_skill`。
**输出**：`merged_patch` dict（含 `edits`/`revise_suggestions`/`skill_candidates`）+ `merged_patch.json`（`trainer.py:1229`）。
**模型**：optimizer，多次（分层，每层每批一次），`workers` 并发（`aggregate.py:116`）。

**分层逻辑**（`_hierarchical_merge` `aggregate.py:70`）：N 个 patch 每 `batch_size` 个一组合并成 1 个，反复直到剩 1 个。每组单独 LLM 调用、同层并行。
**兜底**：LLM 失败时**拼接所有 edits**（`aggregate.py:62-67` / `250-253`），不丢梯度。

### ⚠️ 红线
- 失败/成功**任一为空**走捷径直接返回另一组（`aggregate.py:199-202`），不做最终合并。
- 合并 prompt 随 update_mode 变（`merge_failure.md` / `_rewrite` / `_full_rewrite`），改 update_mode 要确认对应 prompt 存在。

---

## ④ SELECT — 选择 / “梯度裁剪 / 学习率”

**作用**：用 `edit_budget`（= learning_rate）限制本步最多应用几处编辑，optimizer 排序后取 top-L。类比梯度裁剪 / 控制步长。

**调用链**：
```
trainer.py:1238  ④ SELECT
  ├─ full_rewrite_minibatch 模式：跳过 LR/select，直接用 merged  trainer.py:1241-1250
  ├─ lr_control_mode=="autonomous"：                              trainer.py:1252
  │    decide_autonomous_learning_rate(...) → optimizer 决定 L    （写 lr_decision.json, lr_history.jsonl）
  ├─ 否则：edit_budget = scheduler.step()                         trainer.py:1273
  └─ rank_and_select(current_skill, merged_patch, max_edits=edit_budget, ...)  trainer.py:1274
       └─ clip.py:25  rank_and_select
            ├─ clip.py:54  载荷 ≤ budget → 原样返回（不调模型）
            └─ clip.py:75  否则 chat_optimizer(stage="ranking") 选 top-L 索引
                 └─ clip.py:104  解析失败兜底：简单截断 edits[:max_edits]
```

**输入**：`merged_patch`、`edit_budget`。
**输出**：`ranked_patch` + `ranked_edits.json`（`trainer.py:1280`）。
**模型**：optimizer，**仅当载荷数 > budget 时**才调（`clip.py:54`）；否则零调用。

**三种 LR 控制**（`lr_control_mode`，归一化 `trainer.py:206`）：
- `fixed`：用 scheduler（constant/linear/cosine，`build_scheduler` `trainer.py:821`）。
- `autonomous`：optimizer 自己每步决定 L（`trainer.py:1252`）。
- `none`：full_rewrite_minibatch 模式强制（`trainer.py:787-788`），不裁剪。

### ⚠️ 红线
- **预算双重裁剪**：reflect 阶段已 `truncate_payload` 到 L（文档2 §4），这里再裁一次。两处都受 `edit_budget` 影响。
- `edit_budget` 来自 `cfg["edit_budget"]`（YAML `optimizer.learning_rate`，文档2 §10）。

---

## ⑤ UPDATE — 应用 / “optimizer.step()”

**作用**：把选中的编辑真正打进 skill 文档，生成 `candidate_skill`。

**三种模式**（`trainer.py:1308-1360`）：
| 模式 | 行为 | 调模型? | file:line |
|---|---|---|---|
| `patch`（默认） | `apply_patch_with_report` 确定性应用 op/target/content | ❌ | `trainer.py:1360` → `skill.py:165` |
| `rewrite_from_suggestions` | optimizer 按建议重写整篇 | ✅ | `trainer.py:1311-1319` |
| `full_rewrite_minibatch` | 取 merged 里的整篇候选 | ❌（候选已在前面生成） | `trainer.py:1328-1358` |

**输出**：`candidate_skill.md`（`trainer.py:1361`）+ `edit_apply_report.json`（`trainer.py:1364`，仅 patch 模式有）+ `candidate_hash`（`trainer.py:1367`）。

**edit 应用细节**（patch 模式）：四种 op 语义、target 精确匹配、保护区跳过——全在 `skill.py:85-145`，详见文档2 §3。

**skip 分支②**：rewrite 模式没生成有效新 skill（`trainer.py:1386-1412`）→ `action="skip_no_rewrite"`，skill 不变。同样**先 flush appendix**（`trainer.py:1395-1398`）。

### ⚠️ 红线
- patch 模式下 edit 静默失效（target 不匹配/在保护区）**不会报错**，只在 `edit_apply_report.json` 的 `status` 里体现。candidate 看起来“没怎么变”时先查这个文件。
- candidate 此刻只是“候选”，**还没被接受**——接受与否由 ⑥ 决定。

---

## ⑥ EVALUATE — 验证门控 / “validation + early-accept”

**作用**：把 candidate 在 selection 集(`valid_seen`)上 rollout 打分，只有**严格优于当前分**才接受为新 current；否则丢弃、回退。类比带验证集的模型选择。

**调用链**：
```
trainer.py:1418  ⑥ EVALUATE
  ├─ trainer.py:1420  candidate_hash 命中 sel_cache → 直接用缓存分（不重跑）
  ├─ trainer.py:1427  否则 _build_eval_env(split="valid_seen", env_num=cfg["sel_env_num"])
  │    trainer.py:1434  adapter.rollout(sel_env, candidate_skill, ...)  ← target 调用
  │    trainer.py:1435  compute_score → (cand_hard, cand_soft)，存入 sel_cache
  ├─ trainer.py:1441  evaluate_gate(...)  ← 纯函数，不调模型（gate.py:76）
  ├─ trainer.py:1457  use_gate=False 时构造 force_accept GateResult
  └─ trainer.py:1483-1493  把 GateResult 写回 current/best 状态
```

**输入**：`candidate_skill`、`current_score`、`best_score`。
**输出**：更新后的 `current_skill/score`、`best_skill/score/step`；`action ∈ {accept_new_best, accept, reject, force_accept}`。
**模型**：target（验证 rollout），命中缓存则零调用；gate 本身不调模型。

**接受判据**（`gate.py:123`）：`cand_score > current_score` 才 accept；`> best_score` 才 accept_new_best。**相等也拒绝**。

**skill-aware flush**：接受/拒绝后都 flush appendix（`trainer.py:1495-1498`）。

### ⚠️ 红线
- **selection 集 = `valid_seen`**（`trainer.py:1428`），train/test 之外的独立 split。它太小 → hard 不敏感，可换 `gate_metric=soft/mixed`（`gate.py:46`）。
- **`sel_cache` 按 candidate_hash 缓存**（`trainer.py:1420`），同一份 skill 不重复验证。改 skill 内容（哪怕空白字符）→ hash 变 → 重新 rollout。
- reject 时 `ranked_edits` 进 step_buffer 的 `rejected_edits`（`trainer.py:1549-1557`），提醒后续步别重犯。

---

## step 收尾 / Per-step finalize

```
trainer.py:1532  写 step_buffer（失败模式 + 被拒编辑）  _extract_failure_patterns:468
trainer.py:1562  存 trajectory_digest.json
trainer.py:1567  token 快照（按 stage 增量统计）
trainer.py:1583  写回 step_rec，_save_skill(global_step)，更新 best_skill.md
trainer.py:1595  history.append + _save_history + _persist_runtime_state
trainer.py:1598  存 step_record.json
```

> **step_buffer 是 epoch 内记忆**：累积本 epoch 每步的失败模式和被拒编辑（`trainer.py:1045`），下一步的 REFLECT/SELECT 都会看到（`step_buffer_context`），避免反复提同样无效的编辑。

---

## Epoch 末：SLOW UPDATE — “动量 / EMA”

**作用**：epoch 结束时对比相邻两 epoch 的表现，让 optimizer 产出一段“慢更新指导”写入保护区，稳住训练。类比动量。

**触发**：`use_slow_update=True`（`trainer.py:1618`）。
- **epoch 1**：只注入空占位符（`trainer.py:1661-1675`，`inject_empty_slow_update_field`）。
- **epoch ≥ 2**：纵向对比（`trainer.py:1676-`）：
  1. 取上一 epoch 末 skill（`trainer.py:1691`）。
  2. 从 train 集采 `slow_update_samples`(默认20) 个（`trainer.py:1694-1711`）。
  3. **用 prev_skill 和 curr_skill 各 rollout 一遍**（`trainer.py:1718-1719`）← target 调用 ×2。
  4. 构造 longitudinal pairs（improved/regressed/persistent_fail/stable_success，`_build_longitudinal_pairs` `trainer.py:255`）。
  5. `run_slow_update(...)` optimizer 分析（`trainer.py:1770`）。
  6. 接受方式两种：`slow_update_gate_with_selection`（在 selection 集 gate）或 force-accept（`trainer.py:1800-`）。

**输出**：`slow_update/epoch_XX/{comparison_pairs.json, candidate_skill.md, slow_result.json}`。

### ⚠️ 红线
- 慢更新内容写入 `<!-- SLOW_UPDATE_START/END -->` 保护区（文档2 §3），step 级编辑改不动它。
- force-accept 模式下慢更新只改 `current_skill`，**不污染 `best_skill`**（best 必须是验证最优步的忠实快照，`trainer.py:1654-1660`）。

---

## Epoch 末：META SKILL — “优化器记忆”

**作用**：把跨 step 的经验沉淀成 meta_skill，作为下一 epoch optimizer 的额外上下文（不是直接改 target 的 skill）。

**触发**：`use_meta_skill=True`。下一 epoch 开头 `_load_meta_skill_content(out_root, epoch-1)`（`trainer.py:1046-1050`、定义 `trainer.py:376`）载入，传给 REFLECT/AGGREGATE/SELECT 作 `meta_skill_context`。

**输出**：`meta_skill/epoch_XX/meta_skill_result.json`。

---

## 最终 TEST 评估 / Final eval

训练循环结束后，用 `best_skill` 在 held-out 集（`valid_unseen` / test）上评估（`eval_test=True` 时）。这是报告里的最终分数，与训练中的 selection 分数是**不同 split**，防止过拟合 selection 集。

> 单独评估某个 skill 不训练：用 `scripts/eval_only.py`（见 README / `0_start.md`）。

---

## 附：阶段改动检查清单 / Stage change checklist

- [ ] 改 ROLLOUT → resume(`results.jsonl`)、分块大小(`workers`)、容错兜底还在吗？
- [ ] 改 REFLECT → `random_seed` 分组一致性、failure/success 分流、None 兜底？
- [ ] 改 AGGREGATE/SELECT → 三种 update_mode 的 payload key 都处理了？兜底路径还在？
- [ ] 改 UPDATE → 看 `edit_apply_report.json` 确认编辑真生效，不是静默跳过？
- [ ] 改 EVALUATE/gate → 严格大于、纯函数、sel_cache 一致性？
- [ ] 任何 skip 分支 → skill-aware 的 appendix flush 跟上了吗（3 处：no_patches / no_rewrite / 正常）？
- [ ] 改 epoch 末逻辑 → 保护区 marker、best_skill 不被污染？

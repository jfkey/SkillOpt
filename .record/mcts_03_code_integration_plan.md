# MCTS-SkillOpt 设计文档 3 — 代码集成方案（Code Integration Plan）

> **目的 / Purpose**：把 [`mcts_01`](mcts_01_mcts_algorithm_spec.md)（算法）与 [`mcts_02`](mcts_02_value_and_cost.md)（估值引擎）**逐文件落到 SkillOpt 代码上**——抽哪个函数、加哪个类、改哪段配置、哪些假设线性历史的机制会在树里出错、不许碰哪些红线、产物怎么保持下游兼容。
>
> 配套：概念见 [`mcts_00`](mcts_00_idea_and_contribution.md)；算法/伪代码见 [`mcts_01`](mcts_01_mcts_algorithm_spec.md)；价值/成本估计见 [`mcts_02`](mcts_02_value_and_cost.md)；论文体 Algorithm 1 见 [`mcts_05`](mcts_05_consolidated_algorithm_and_tradeoffs.md)。SkillOpt 内部结构见 `.record/1_architecture_map.md`～`6_env_adapter_contract.md`（下称 doc1–doc6）。
>
> 术语：**target** = 做任务的冻结模型 M，**optimizer** = 改 skill 的模型 O。`file:line` 锚点基于撰写时 HEAD，大改后请重新核对。
>
> ⚠️ 本文已**对齐二维价值 + Option B**（backup 回传二维向量、selection 时才标量化）+ Pareto 前沿输出。早期单目标草稿见 [`_archive/`](_archive/)，勿引用。

---

## 0. 一句话 / One-liner

> SkillOpt 已经是「一步 PromptAgent」：当前 6 阶段循环 = 一次 state transition + 贪心 gate。集成 MCTS **不重写核心**，只做三件事：(1) 把 step 循环的 ①~⑤ 抽成可复用的 `transition()`；(2) 在其上套 `SkillWorldModel`（transition + **二维** `estimate_value`）与 `SearchAlgo`（`linear` | `mcts`）；(3) 小心处理那些**假设线性历史**的记忆/缓存机制。env adapter / model router / prompts **完全不动**。

---

## 1. 三层架构 / Layering（对齐 PromptAgent 的 agent / world_model / search_algo）

```
scripts/train.py (读 yaml, search.algo 开关)
  └─ Trainer / Agent
       ├─ EnvAdapter            # 不变：rollout / reflect / evaluator / dataloader（doc6）
       ├─ ModelRouter           # 不变：optimizer/target backend（doc5）
       ├─ SkillWorldModel       # 新增：封装 transition + 二维估值
       │     ├─ transition(node, batch) → 候选子 skill（= 现六阶段的 ①~⑤，不含 gate）
       │     └─ estimate_value(child, parent, budget) → (Δsucc, Δcost)  # CRN+SH, mcts_02 §3
       └─ SearchAlgo
            ├─ LinearSearch     # = 现状（greedy + gate），保留为 baseline / 向后兼容
            └─ MCTS             # 新增：UCT / expand / 估值 / 二维 backup / 前沿提取
```

🔵 **默认 `search.algo: linear`**，保证现有实验、CI、webui、`eval_only.py` 完全不受影响；只有显式 `--search.algo mcts` 才走树搜索。

---

## 2. 抽取 `transition()` —— 不要 fork `trainer.py`

`engine/trainer.py` 的 `ReflACTTrainer.train()`（`trainer.py:597`，主循环约 800 行，强耦合）。**不要复制整文件**；把 step 循环里 ①~⑤ 剥成一个纯函数 / `SkillWorldModel.transition`：

```python
def transition(parent_skill, batch, path_context) -> (candidate_skill, transition_log):
    # ① rollout(parent_skill, batch)                      # 复用 EnvAdapter.rollout
    # ② reflect(..., path_context)                        # reflect.py:472；path_context = root→parent 证据
    # ③ aggregate                                         # aggregate.py:143
    # ④ select(rank/clip to edit_budget)                  # clip.py:25
    # ⑤ apply_patch_with_report → candidate_skill         # skill.py:165
    # 注意：不含 gate；gate 是 LinearSearch 专属
```

🔴 **保持 `skillopt/` 与 `skillopt_sleep/` 解耦**（CLAUDE.md 红线）：MCTS 只加在 `skillopt/`（建议新模块 `skillopt/search/`），不得在两包间引入 import。

---

## 3. `SkillWorldModel.estimate_value()` —— 二维、CRN+SH（不是标量 reward）

⚠️ 与早期草稿最大差异：**estimate_value 返回二维配对差 `(Δsucc, Δcost)`，不是标量 reward**。实现细节全在 [`mcts_02` §3–5](mcts_02_value_and_cost.md)，这里只给代码挂点：

```python
def estimate_value(child, parent, n0, eta, cap) -> (Δsucc, Δcost):
    h = skill_hash(child.skill)
    if h in sel_cache:                       # transposition table（trainer.py:1420 已有）
        return aggregate_per_task(sel_cache[h])
    # CRN：在 parent.eval_tasks 的共享前缀子集上、低温/定种子评估 child
    # SH：先 n0，落后早砍，×eta 追加；排序依据用配对 Δ（mcts_02 §5）
    # 记录 per-task (succ,cost) → sel_cache[h]（增量缓存）
    ...
```

🔴 **`sel_cache` 必须从「存聚合分」扩成「存每任务分数」**（`{task_id: (succ, cost)}`），否则配对评估/SH 的增量复用无从谈起（mcts_02 §3、自检清单）。缓存 key 带上 seed / 评估子集 id，避免跨配置串味。

🔵 复用 `compute_score`（`gate.py:46`，hard/soft/mixed）当 success 维；cost 维新增，写进 `RolloutResult.extras`（mcts_02 §1）。

---

## 4. 节点与树 / SkillNode（二维价值 + Option B）

```python
class SkillNode:
    skill_hash                 # 节点 id（内容哈希）= state；不要 deepcopy 整篇文档
    skill_path                 # outputs/<run>/mcts/nodes/<id>/skill.md（落盘引用）
    parent, children           # 树结构
    edge_edits                 # 由 parent 生成本节点的那组 edit（RawPatch/selected）
    value: (success, cost)     # 二维绝对价值，telescoping 锚定：v = parent.value + Δ（mcts_01 §5）
    W: (Σsucc, Σcost)          # 🔴 Option B：二维向量累加（backup 不焼 λ）
    N                          # 访问次数
    # Q 不落盘：导航时按需 Q = scal_λ(W / N)（见 §UCT；λ 只在 selection 出现）
    eval_tasks                 # {task_id: (succ,cost)}，供 CRN/SH 增量复用
    train_evidence             # 本节点 train-rollout 轨迹缓存（首次展开时跑，之后复用）
    expanded, terminal
```

🔴 **Option B（与 [`mcts_06` 支柱三](mcts_06_skillorchestra_pareto_borrowings.md) 一致）**：backup 累加**二维向量** W，**不**在 backup 把 λ 烤进标量；UCT 在 selection 读出时才 `scal_λ(W/N)`。这样**一棵树的前沿提取对 λ 免费**（非支配集与 λ 无关），λ-sweep 只在提取期发生，无需为每个 λ 重建树。

🔵 **大状态存储**：节点存 `skill_hash` + 落盘路径，不 `deepcopy`；规划 `outputs/<run>/mcts/nodes/<id>/`，复用现有 steps 目录约定。

---

## 5. 配置设计 / Config（对齐 PromptAgent `search_setting`）

在 `configs/_base_/default.yaml` 新增 `search:` 段（任何键都可 `--search.xxx` 覆盖，沿用 SkillOpt 的 CLI 覆盖约定）：

```yaml
search:
  algo: linear            # linear | mcts （默认 linear，向后兼容）
  budget: 24              # 搜索总预算 B：selection-rollout 调用次数上界（mcts_01 §6）
  branch: 2               # b = expand_width，每节点候选子数
  depth_limit: 4          # D，树深上限（mcts_01 §6：建议 4–6）
  c_explore: 2.5          # UCT 探索常数 c（PromptAgent 用 2.5）
  patience: 5             # 无 Pareto 改进早停
  lambda: 0.3             # 标量化权重；仅 selection/SH 用（提取期扫 λ，见 mcts_02 §6）
  n0: 4                   # SH 初始评估任务数
  sh_eta: 2               # SH 衰减/追加因子 η（2–3）
  granularity_moves: true # EXPAND 是否含 abstract/refine 双向粒度 move（mcts_06 支柱一）
```

🟡 `optimizer.learning_rate`（代码里叫 `cfg["edit_budget"]`，`config.py:111`）= 每次 transition 的编辑预算，MCTS 里建议**固定**或按节点深度定义，别用全局 `scheduler.step()` 计数（见 §6）。

---

## 6. 与现有线性机制的冲突表（迁移红线）/ Linear-history conflicts

这些机制**都假设线性历史**，在树里会串味或计数错乱，必须显式处理：

| 机制 | 文件 | 树里的问题 | v1 建议 |
|---|---|---|---|
| `step_buffer`（失败模式 + 被拒 edits 向后传） | `trainer.py:_format_step_buffer` | 全局线性 → **兄弟节点信息互相泄漏** | 改 **path-scoped**（只含 root→node 路径），对应 PromptAgent 的 `trajectory_prompts` |
| `meta_skill`（优化器记忆，按 epoch） | `optimizer/meta_skill.py` | 无树语义 | v1 **禁用**（mcts_01 §3） |
| `slow_update`（动量，跨连续 step） | `optimizer/slow_update.py` | "连续 step" 在树里无意义；写保护区 | v1 **禁用**；其成对比较能力被复用为 CRN（mcts_02 §4） |
| `scheduler.step()`（全局 LR 计数） | `optimizer/scheduler.py` | 全局计数与树深不对应 | 固定 `edit_budget`，或按节点深度定义 |
| `sel_cache`（按 hash 缓存分数） | `trainer.py:1420` | 假设 reward 确定；随机 rollout 会让缓存失真 | 固定 selection seed + 缓存 key 带 seed/子集 id（mcts_02 §4、mcts_04 风险） |
| `skip_no_patches` / `skip_no_rewrite` | `trainer.py` | 可能产出与父节点**完全相同**的子 skill | 子节点按 `skill_hash` **去重**；等于父则标 `terminal`，不浪费估值 |
| 保护区 `SLOW_UPDATE` / `APPENDIX` | `optimizer/skill_aware.py` | edit 改不动；v1 关 slow/meta 后这两区基本闲置 | 不碰；`apply_patch` 仍须尊重（doc4 §D/§E） |

🔴 **edit 应用红线（doc4 §D/§E/§J）**：`target` 必须逐字节匹配否则**静默跳过**；用 `get_payload_items` 不要硬编码 `["edits"]`；导出的每个前沿点必须是**真实通过评估节点**的忠实快照（best_skill 不被污染——v1 关 slow-update 所以无污染风险）。

---

## 7. 产物与下游兼容 / Artifacts & downstream

最终交付从「单个 best_skill.md」升级为**前沿菜单 + 按 prior 选点**（[`mcts_06` 支柱二](mcts_06_skillorchestra_pareto_borrowings.md)）：

- `outputs/<run>/mcts/frontier.json`：全部非支配工作点 + 各自 **test** 分与 cost（菜单）。
- `outputs/<run>/best_skill.md`：按声明 prior 选出的产物，**保持原文件名/位置**以兼容 `eval_only.py`、webui：
  - `perf` prior：test success 最高的前沿点（不看 cost）。
  - `efficiency` prior：success 掉 ≤ ε（默认如 1pt）内、cost 最低的点。
- `outputs/<run>/mcts/tree.json`：所有 nodes/paths（对应 PromptAgent 的 `data.json`），便于分析。
- 🔴 **不破坏既有约定**：仍写 `history.json` / run-dir / `best_skill.md`，否则 `scripts/eval_only.py` 与 webui 会断。

---

## 8. 落地顺序 / Landing order

1. **先跑通 linear baseline**，记录开销基线（reward 次数 / token / wall time / 单次 selection rollout 成本）——这是 mcts_04 §1 算力对齐的锚。
2. **抽取 `transition()`**：从 step 循环剥离 ①~⑤；linear 模式回归测试**分数不变**（正确性闸）。
3. **`estimate_value()`**：二维 + CRN + SH（mcts_02），先把 `sel_cache` 扩成每任务分数。
4. **`SkillNode` + `MCTS`**：先小配置 `budget≈24, branch=2, depth=3`（mcts_04 §3）。
5. **前沿提取 + frontier.json + perf/efficiency best_skill.md + tree.json**。
6. **最便宜 benchmark 端到端验证**（searchqa，mcts_04 §8）：先证明 MCTS ≥ linear baseline，再迁贵环境、再放大 budget。

---

## 9. 不许碰的红线 / Do-not-touch

- 🔴 不 fork `trainer.py`；用抽函数 + 组合，别复制 800 行主循环。
- 🔴 不在 `skillopt/` 与 `skillopt_sleep/` 之间引入任何 import（CLAUDE.md）。
- 🔴 不改 EnvAdapter / ModelRouter / prompts 的对外契约（doc5/doc6）——MCTS 只在其上编排。
- 🔴 不改模型权重、不引入 RL（mcts_00 §7：MCTS 是纯文本搜索，M 与 O 全程冻结）。

---

## 10. 文档地图 / Doc map

| 文档 | 层 | 回答什么 |
|---|---|---|
| [`mcts_00`](mcts_00_idea_and_contribution.md) | 概念 | 做什么、为什么、定位、范围 |
| [`mcts_01`](mcts_01_mcts_algorithm_spec.md) | 算法 | 四阶段、树结构、伪代码 |
| [`mcts_02`](mcts_02_value_and_cost.md) | 承重墙 | value/cost 估计、降方差/控成本 |
| `mcts_03`（本文） | 工程 | 逐文件改 SkillOpt、线性机制冲突、红线、产物兼容 |
| [`mcts_04`](mcts_04_experiment_plan.md) | 评测 | 基线、指标、消融、算力对齐、过拟合曲线 |
| [`mcts_05`](mcts_05_consolidated_algorithm_and_tradeoffs.md) | 汇总 | 论文体 Algorithm 1 + 优缺点 + 待改进 |
| [`mcts_06`](mcts_06_skillorchestra_pareto_borrowings.md) | 横向借鉴 | SkillOrchestra 的 Pareto 净增量与指导 |

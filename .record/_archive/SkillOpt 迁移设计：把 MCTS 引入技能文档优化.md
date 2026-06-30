# SkillOpt 迁移设计：把 PromptAgent 的 MCTS 引入技能文档优化

> 目标：把 PromptAgent（ICLR 2024）的 **MCTS 策略规划**思路迁移到 SkillOpt（`/data3/liujunfeng/code/SkillOpt`），
> 用"前瞻 + 回溯"的树搜索替换当前的线性贪心优化，提升 `best_skill.md` 的质量。
>
> 配套文档：`SkillOpt MCTS 迁移：开销与风险评估.md`（务必先读，开销是头号约束）。
> 参考：`PromptAgent 论文笔记：基于 MCTS 策略规划的专家级 Prompt 优化.md`（§十一 代码级复现指南）。

---

## 一、TL;DR

SkillOpt 当前的训练循环本质上 = **PromptAgent 的一次 state transition + 贪心 gate**。
迁移 MCTS **不需要重写核心逻辑**，只需要做三件事：

1. 把 trainer 里"单步转移"（`rollout → reflect → aggregate → select → update`，**去掉 gate**）抽成一个可复用的 `transition()`；
2. 在其上套一层 `WorldModel`（reward = selection 集分数）+ `MCTS`（UCT 选择 / 扩展 / 模拟 / 回溯）；
3. 用 `search_algo: linear | mcts` 配置开关切换，env adapter / model router / prompts **完全不动**。

**关键认知**：当前 `evaluation/gate.py:evaluate_gate`（"严格优于 current 才接受"的单调爬山）正是 PromptAgent 论文里被
MCTS 击败的 Greedy/Beam baseline。MCTS 要替换的就是它——把"接受/拒绝"换成"所有节点都进树 + UCT + 回溯"。

---

## 二、概念映射（PromptAgent ↔ SkillOpt）

| PromptAgent 概念 | SkillOpt 现状 | 对应代码 |
| --- | --- | --- |
| **state** $s_t$ = prompt 文本 | `current_skill`（一篇 skill.md 文档） | `engine/trainer.py` |
| **action** $a_t$ = error feedback (gradient) | `reflect→aggregate→select→update` 产出的 candidate（一束 add/delete/replace edits） | `gradient/reflect.py`, `gradient/aggregate.py`, `optimizer/clip.py`, `optimizer/skill.py` |
| **transition** $T$（optimizer LLM 生成新 state） | 一个 training step（六阶段前五段） | `trainer.py` 内层 step 循环（≈ L1062+） |
| **reward** $r$ = eval 集 accuracy | selection 集（`valid_seen`）的 `cand_hard / cand_soft` | `compute_score`, `evaluation/gate.py:select_gate_score` |
| **trajectory_prompts**（root→node 路径，喂给 optimize 模板） | `step_buffer`（rejected edits / failure patterns）+ `meta_skill` | `trainer.py:_format_step_buffer`, `optimizer/meta_skill.py` |
| **（无树；贪心 gate）** | `evaluate_gate`：candidate 严格 > current 才接受，否则保持 current | `evaluation/gate.py` |
| MCTS 的 `expand_width`（采样多个 batch 得多个子节点） | 用不同 `batch_seed` 跑多次 transition | `trainer.py` 的 `batch_seed`/`shuffled_seeds` 机制 |
| `num_new_prompts`（每步生成几个新 prompt） | reflect/rewrite 一次产出几个 candidate skill | `skill_update_mode` + `optimizer/rewrite.py` |

> 结论：**state / action / reward / 多样化采样**这四个 MCTS 必需原语，SkillOpt 已经全部具备。
> 缺的只是"把它们组织成一棵树 + UCT + 回溯 + 输出选择"。

---

## 三、当前 SkillOpt 循环回顾（被替换的部分）

`engine/trainer.py` 的 `ReflACTTrainer.train()`，每个 step 六阶段（环境无关）：

```
① ROLLOUT    冻结 target model + current_skill 在 train batch 上跑 → 打分 (hard/soft)
② REFLECT    optimizer 把失败轨迹 → patches（add/delete/replace 建议）
③ AGGREGATE  merge_patches 分层合并
④ SELECT     rank_and_select 按 edit budget(=learning_rate) 裁剪
⑤ UPDATE     apply_patch → candidate_skill
⑥ EVALUATE   candidate 在 selection 集(valid_seen)上 rollout → evaluate_gate 接受/拒绝
```

状态变量：`current_skill / current_score / best_skill / best_score / best_step`，
缓存 `sel_cache`（按 `skill_hash` 缓存 selection 分数），记忆 `step_buffer`（跨 step 携带失败模式 + 被拒 edits）。

**这是一条线性贪心链**：accept 则 current 前进，reject 则原地不动。没有分支、没有回溯。
MCTS 改的就是 ⑥ 之后的"如何组织多个 candidate"。

---

## 四、目标架构

### 4.1 分层（对齐 PromptAgent 的 agent / world_model / search_algo 三层）

```
scripts/train.py (读 yaml, search_algo 开关)
  └─ Trainer / Agent
       ├─ EnvAdapter            # 不变：rollout / reflect / evaluator / dataloader
       ├─ ModelRouter           # 不变：optimizer/target backend
       ├─ SkillWorldModel       # 新增：封装 transition + reward
       │     ├─ transition(node, batch) → child candidate skill   # = 现有六阶段的前五段
       │     └─ reward(skill) → selection 分数（复用 compute_score / sel_cache）
       └─ SearchAlgo            # 新增：
            ├─ LinearSearch     # = 现状（greedy + gate），保留为 baseline
            └─ MCTS             # 新增：UCT / expand / simulate / backprop / 输出选择
```

### 4.2 关键改动点

**(A) 抽取 `transition()`——不要 fork trainer.py（2376 行，强耦合）**

把 step 循环里 ①~⑤ 抽成一个纯函数 / `SkillWorldModel.transition`：

```
transition(parent_skill, batch, path_context) -> candidate_skill, transition_log
    # ① rollout(parent_skill, batch)
    # ② reflect(... , path_context)        # path_context = root→parent 的 trajectory/buffer
    # ③ aggregate
    # ④ select(edit_budget)
    # ⑤ update → candidate_skill
    # 注意：不含 gate；gate 是 LinearSearch 专属
```

**(B) reward 与 gate 解耦**

```
reward(skill) -> float
    # 复用 evaluation 的 compute_score + select_gate_score
    # 先查 sel_cache[skill_hash]，命中直接返回
    # 未命中：在 selection 子集上 rollout（开销见风险评估文档）
```

MCTS 里**所有子节点都进树并算 reward，绝不丢弃更差的节点**。
`evaluate_gate` 仅 LinearSearch 使用。

**(C) 节点与树**

```
class SkillNode:
    id, depth, parent, children
    skill_hash                  # state 用 hash + 落盘引用，不要 deepcopy 整篇文档
    skill_path                  # outputs/<run>/mcts/nodes/<id>/skill.md
    action_log                  # 产生它的那次 transition 的日志/edits
    reward                      # selection 分数
    cum_rewards: list[float]    # 回溯累积；Q = mean(cum_rewards)
    is_terminal
```

### 4.3 MCTS 主循环（照搬 PromptAgent，见论文笔记 §11.3）

```
search(init_skill):
    root = SkillNode(init_skill); root.reward = reward(init_skill)
    min_threshold = mcts_threshold = root.reward
    for it in range(iteration_num):
        path = select(root)                 # 从根用 UCT 下探到叶/可扩展节点
        if not depth_limit_reached(path[-1]):
            expand(path[-1])                # 采样 expand_width 个 batch → 子节点 + reward
            simulate(path)                  # 反复 expand 并走 reward 最高子节点，直到终止/早停
        back_propagate(path)                # 累积 reward 到路径上每个节点
    return select_output(root)              # 见 4.4
```

- **UCT**：`Q + w_exp * sqrt(ln(N_parent+1) / max(1, len(cum_rewards)))`
- **扩展**：每个子节点 = 用一个**不同 `batch_seed`** 的 minibatch 跑一次 `transition`（保证多样性，对应
  PromptAgent 每次扩展采不同 batch）。
- **终止/早停**（相对化阈值）：
  - `depth >= depth_limit`
  - min 阈值早停：`reward < (min_threshold + parent.reward)/2 且 depth > min_depth`
  - max 早停：`reward > mcts_threshold 且 depth > min_depth`，并抬高全局 `mcts_threshold`
- **simulation 不是随机 rollout**：每次走 `argmax(child.reward)` 的子节点。

### 4.4 输出选择（产物兼容）

照搬 PromptAgent 的"最佳平均 reward 路径上的最佳 reward 节点"：

```
best_reward_path = argmax_path( mean(node.reward for node in path) )
selected_node    = argmax_reward( node in best_reward_path )
write selected_node.skill -> best_skill.md
```

- 额外落 `outputs/<run>/mcts/tree.json`（所有 paths/nodes，对应 PromptAgent 的 `data.json`），便于分析。
- **保留 `history.json` / run-dir / `best_skill.md` 约定**，否则 `scripts/eval_only.py` 和 webui 会断。

---

## 五、配置设计

在 `configs/_base_/default.yaml` 新增一段（对齐 PromptAgent 的 `search_setting`）：

```yaml
search:
  algo: linear            # linear | mcts   （linear = 现状，默认，保证向后兼容）
  iteration_num: 8        # MCTS 迭代次数（开销敏感，先小）
  expand_width: 2         # 每个节点扩展的子节点数（= 采样几个不同 batch）
  depth_limit: 4          # 最大深度
  min_depth: 1            # 早停最小深度
  w_exp: 2.5              # UCT 探索权重 c
  num_candidates: 1       # 每次 transition 产出几个候选（≈ num_new_prompts）
  reward_subset_size: 0   # >0 时搜索期用 selection 子集算 reward，0=全量
```

> 默认 `algo: linear` 保证现有实验、CI、webui 完全不受影响；只有显式 `--search.algo mcts` 才走树搜索。

---

## 六、与现有机制的冲突点（迁移红线）

这些机制**都假设线性历史**，在树里会出错，必须显式处理：

| 机制 | 文件 | 树里的问题 | 第一版建议 |
| --- | --- | --- | --- |
| `step_buffer`（失败模式 + 被拒 edits 向后传） | `trainer.py:_format_step_buffer` | 全局线性 → 兄弟节点信息互相泄漏 | 改为 **path-scoped**（只含 root→node 路径），对应 `trajectory_prompts` |
| `meta_skill`（优化器记忆，按 epoch） | `optimizer/meta_skill.py` | 无树语义 | MCTS 第一版**禁用** |
| `slow_update`（动量，跨连续 step） | `optimizer/slow_update.py` | "连续 step"在树里无意义 | MCTS 第一版**禁用** |
| `scheduler.step()`（全局 LR 计数） | `optimizer/scheduler.py` | 全局计数器与树深度不对应 | 用固定 `edit_budget`，或按节点深度定义 |
| `sel_cache`（按 hash 缓存分数） | `trainer.py` | **假设 reward 确定**；rollout 随机会让缓存失真 | 固定 selection seed 或多 seed 平均（见风险文档） |
| `skip_no_patches` / `skip_no_rewrite` | `trainer.py` | 可能产出与父节点**完全相同**的子节点 | 子节点去重（按 hash）/ 标记 terminal，避免 no-op 重复 |

---

## 七、落地顺序（详见实现清单 / 风险文档）

1. **先跑通 linear baseline**，记录开销基线（token / 评估次数 / wall time）。
2. 抽取 `transition()`：从 trainer step 循环剥离 ①~⑤，linear 模式回归测试不变。
3. 实现 `SkillWorldModel.reward()`（复用 `compute_score` + `sel_cache`）。
4. 实现 `SkillNode` + `MCTS`（先小配置：`iteration_num=4, expand_width=2, depth_limit=3`）。
5. 实现输出选择 + `tree.json` + `best_skill.md`。
6. 用最便宜的 benchmark（如 searchqa）端到端验证：MCTS 是否 ≥ linear baseline。

---

## 八、一句话总结

> SkillOpt 已经是"一步 PromptAgent"。迁移 MCTS = **抽出那一步当成 `transition`，
> 把贪心 gate 换成树搜索，再小心处理那些假设线性历史的记忆/缓存机制**。
> 头号约束不是算法，而是 reward = 完整 agent rollout 的**开销**（见风险评估文档）。

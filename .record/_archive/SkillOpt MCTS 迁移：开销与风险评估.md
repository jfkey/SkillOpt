# SkillOpt MCTS 迁移：开销与风险评估

> 配套 `SkillOpt 迁移设计：把 MCTS 引入技能文档优化.md`。
> 一句话：**算法不难，难的是开销**——SkillOpt 的 reward 是一次完整 agent rollout，MCTS 会把它乘上一个大系数。
> 这份文档量化开销、给出缓解策略，并列出非开销类风险与对策。

---

## 一、为什么开销是头号约束

PromptAgent 的 reward = 对一段文本算 accuracy（一次 LLM 前向，便宜、确定、温度 0）。
SkillOpt 的 reward = 让**冻结的 agent** 在 selection 集（`valid_seen`）上跑**完整 rollout**：
多轮对话、工具调用、可能十几步交互。这是数量级的差异。

线性循环里这个开销每个 step 只付一次（⑥ EVALUATE）。MCTS 会把"算 reward"的次数乘上
`expand_width × depth × iteration_num` 量级。

---

## 二、评估次数量级推算

记：
- `I` = `iteration_num`（MCTS 迭代数）
- `W` = `expand_width`（每次扩展的子节点数）
- `D` = `depth_limit`
- 每个**新子节点**都要算一次 reward（= 一次 selection 集 rollout）

### 2.1 reward 调用次数（最坏上界）

每次迭代里：select 下探（不新增 reward）→ expand 产生 `W` 个新节点 → simulate 反复 expand 直到深度上限，
最坏每层再 `W` 个。单次迭代新节点数上界 ≈ `W × D`，所以：

```
reward 调用次数 ≈ I × W × D        （最坏，无缓存）
```

对比线性 baseline：每个 step 1 次 reward，`num_steps` 个 step → `num_steps` 次。

### 2.2 具体数字感受（每次 reward = 1 次 selection-set rollout）

| 配置 | I | W | D | reward 调用上界 | 相对 linear(≈10 step) |
| --- | --- | --- | --- | --- | --- |
| 论文 Standard | 12 | 3 | 8 | ~288 | ~29× |
| **建议起步** | 4 | 2 | 3 | ~24 | ~2.4× |
| 中等 | 8 | 2 | 4 | ~64 | ~6.4× |

> 还要叠加：每个 transition 本身含 ① rollout(train batch) + ② reflect(optimizer LLM)，
> 这部分也乘 `I × W × D`。**所以总 token / 总钱 / 总时间都按同一系数膨胀**，不只是 reward。

### 2.3 单次 selection rollout 的成本要素

实测前先量一次 linear baseline 的：
- selection 集大小 `sel_env_num`（每次 reward 跑多少 item）
- 每个 item 平均交互步数 × 每步 token
- optimizer LLM 每次 reflect 的 token

把这三个数代入 §2.1，就能在跑之前估出 MCTS 一次完整 search 的钱和时间。

---

## 三、开销缓解策略（按性价比排序）

1. **复用 `sel_cache`（几乎免费，必做）**
   现有 `trainer.py` 已按 `skill_hash` 缓存 selection 分数。MCTS 里同一 skill 会被反复触达
   （simulate / 不同路径汇聚到同一 hash），缓存命中能砍掉一大块重复 reward。
   → 把 `reward(skill)` 第一行写成"查 `sel_cache[skill_hash]` 命中即返回"。

2. **搜索用子集，最终用全量（对应 PromptAgent eval 集 vs test 集）**
   配置 `search.reward_subset_size`：搜索期间 reward 只在 selection 的**小子集**上算（便宜、用于排序），
   搜索结束后只对**选出的 top-k 节点**做全量 selection / test 评估。
   PromptAgent 正是这样分工的（reward=eval 集小，test 只测最终选出的节点）。

3. **从小配置起步，确认有效再放大**
   先 `I=4, W=2, D=3`。先验证"MCTS ≥ linear baseline"，再逐步加大 `I`。
   论文消融：`iteration_num` 太大反而过拟合训练集（16 不如 12），所以不是越大越好。

4. **挑最便宜的 benchmark 调通**
   先用交互步数最少、rollout 最便宜的环境（如 searchqa）跑通整条 MCTS 链路，
   再迁到 alfworld / spreadsheetbench 这类长交互、贵的环境。

5. **限制分支宽度优先于增加深度**
   开销对 `W` 和 `D` 都是乘性。经验上"窄而深"比"宽而浅"更省，且更贴近 PromptAgent 的轨迹式优化。

6. **并行化扩展**
   一个节点的 `W` 个子节点彼此独立，reward rollout 可并行（受 API 限速 / 显存约束）。
   现有 backend router 若支持并发，扩展阶段可批量发。

---

## 四、非开销类风险与对策

### 4.1 reward 噪声 / 非确定性（高危）

- **问题**：agent rollout 随机（温度、工具非确定）。Q 值带噪 → UCT 选择失稳；更糟的是
  `sel_cache` 按 hash 缓存**本身假设了确定性**，同一 skill 两次跑分数不同会让缓存"骗"了搜索。
- **对策**：
  - 给 selection rollout **固定 seed**（代码已传 `seed`，确认 evaluator 真的吃这个 seed）；
  - 或对每个节点 reward 做**多 seed 平均**（更稳但更贵，与开销权衡）；
  - 缓存 key 里带上 seed / 评估子集 id，避免跨配置串味。
- **诊断**：跑前先测"同一 skill 重复评估的方差"，方差大就必须多 seed 或扩大子集。

### 4.2 线性记忆机制串味（高危，正确性问题）

`step_buffer` / `meta_skill` / `slow_update` / `scheduler` 全部假设线性历史（详见设计文档 §六）。
在树里直接复用会让**兄弟节点互相泄漏信息**、动量/LR 计数错乱。
- **对策**：第一版禁用 `meta_skill` / `slow_update`，`step_buffer` 改为 **path-scoped**
  （只含 root→当前节点路径，对应 PromptAgent 的 `trajectory_prompts`），`edit_budget` 用固定值。

### 4.3 退化 / 重复子节点

`skip_no_patches` / `skip_no_rewrite` 分支可能产出与父节点**完全相同**的 skill。
- **对策**：子节点按 `skill_hash` 去重；若等于父节点则标记 terminal，不浪费 reward。

### 4.4 大状态的存储 / 内存

每个节点是一整篇 skill.md + 一个 rollout 目录，树有几十到上百节点。
- **对策**：节点存 **hash + 落盘路径**，不要 `deepcopy` 整篇文档；
  规划 `outputs/<run>/mcts/nodes/<id>/` 布局，复用现有 steps 目录约定。

### 4.5 产物 / 下游兼容

最终交付仍是 `best_skill.md`，且 `eval_only.py`、webui 依赖 `history.json` / run-dir 结构。
- **对策**：MCTS 输出选择后写 `best_skill.md`；额外落 `mcts/tree.json`；不破坏既有文件约定。

### 4.6 阈值定标

PromptAgent 的早停阈值相对根/父 reward 定标。SkillOpt 分数尺度（hard/soft/mixed）不同，
且可能贴近天花板。
- **对策**：阈值一律相对化（用根节点 reward 初始化 `min_threshold`/`mcts_threshold`），
  不要写死绝对数值；先关掉早停跑一轮观察分数分布，再调。

---

## 五、上线前检查清单

- [ ] 测过 linear baseline 的开销基线（reward 次数 / token / wall time / 单次 selection rollout 成本）。
- [ ] 测过"同一 skill 重复评估的方差"，确定 seed 策略（固定 seed or 多 seed 平均）。
- [ ] `reward()` 第一行查 `sel_cache`；缓存 key 含 seed / 子集 id。
- [ ] 搜索用 `reward_subset_size` 子集，最终对 top-k 全量评估。
- [ ] 第一版禁用 meta_skill / slow_update；step_buffer 改 path-scoped；edit_budget 固定。
- [ ] 子节点按 hash 去重；no-op 子节点标 terminal。
- [ ] 节点存 hash+路径，不 deepcopy 文档。
- [ ] 输出写 `best_skill.md` + `mcts/tree.json`，不破坏 history.json / run-dir。
- [ ] 先在 searchqa（最便宜）跑通，再迁到贵环境。
- [ ] 小配置 `I=4,W=2,D=3` 起步，先证明 MCTS ≥ linear，再放大 I。

---

## 六、一句话总结

> 把 reward 的开销算清楚再动手。**缓存 + 子集评估 + 小配置起步**是三件套；
> seed 确定性和线性记忆串味是两个最容易翻车的正确性坑。

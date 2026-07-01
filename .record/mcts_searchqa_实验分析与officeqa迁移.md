# MCTS-SkillOpt × SearchQA 实验分析 & 向多轮任务（OfficeQA）迁移方向

- **日期**：2026-06-30
- **分支**：`feat/mcts-pareto-search`（commit `ed6e3eb`）
- **MCTS 运行目录**：`outputs/mcts_compare_20260630_174516/`
- **配置**：`configs/searchqa/mcts_compare.yaml`（`lambda_cost=0`、`sel_eval_num=40`、`test_eval_num=150`、`batch_size=20`、`budget=8`、`branch=2`、`depth_limit=4`、`n0=8`、`sh_eta=2`、`value_mode=paired`）
- **对标 baseline**：`SkillOpt-baseline/outputs/skillopt_searchqa_gpt-5.5_20260630_111929/`（linear 贪心训练器，commit `43a8419`≈main）
- **模型**：optimizer = target = **gpt-5.5**，`reasoning_effort=medium`，`max_turns=1`

---

## 0. 一句话结论

> **MCTS 在精度上与 linear baseline 打平（测试 hard-EM 0.92 vs 0.9267，差 1/150 = 噪声），token 少 15%，但墙钟慢 60%。真正的发现不是胜负，而是：搜索在 depth 1 就撞到天花板——第一刀吃满了全部增益（+7 题），后续 3 层全是"加成本不加精度"的 plateau。当前 SearchQA 设置根本没有发挥 MCTS 树搜索的优势。这指向一个明确结论：SearchQA（单轮、gpt-5.5）太简单，应迁移到 `max_turn > 1` 的多步任务（如 OfficeQA）。**

---

## 1. 与 baseline 的对比

| 指标 | linear baseline | MCTS（本次） | 解读 |
|---|---|---|---|
| 初始空技能 测试 hard | 0.8800 (132/150) | 0.8733 (131/150) | 差 1 题 = 模型随机性，**等价** |
| **部署技能 测试 hard** | **0.9267** (139/150) | **0.9200** (138/150) | 差 1 题 = **统计等价** |
| 部署技能 验证 hard | 0.875 (35/40)¹ | 0.925 (37/40) | 验证集已饱和（见 §3） |
| 测试增益（部署 − 初始） | +0.0467（+7 题） | +0.0467（+7 题） | **完全相同，都涨 7 题** |
| tokens 总 | 1,612,983 | **1,375,240** | MCTS 省 15% |
| └ target rollout | 1,421,463 | 1,021,362 | MCTS −28%（CRN+SH 剪枝有效） |
| └ optimizer | 191,520 | 353,878 | MCTS +85%（8 次 transition vs 4 步） |
| 墙钟 | **537s** | 856s | MCTS 慢 60%（SH rung 串行，并行度低） |
| 选择集 task-rollouts | ~360 | **264** | CRN+SH 省评测 |
| 部署技能体积 | 3.1KB | 2.0KB | MCTS 更紧凑 |

> ¹ baseline 报告中 v0003 验证 hard = 0.875；初始 = 0.775。本表对齐"部署口径"。

**要点**：两套方法在 SearchQA 上**精度无显著差异**。MCTS 的"省 token"来自 CRN-paired + successive halving 的评测复用；"慢墙钟"来自 SH 各 rung 串行、并行度不如 baseline 的大批量评测阶段。

---

## 2. 搜索动态（核心诊断）

```
node0 空技能  sel 0.750  cost 1255  test 0.8733   ←(root)
 │  iter1 展开
 ├─ node1  sel 0.925  cost 1635  test 0.9200   ★ perf 部署点 — 全部增益在这一刀
 └─ node2  sel 0.925  cost 1697
       iter2 展开 node2 → node3 sel 0.900 / node4 sel 0.675   ← 回退！
       iter3 展开 node1 → node5 0.925 / node6 0.925           ← 平，但更贵 → 被支配
       iter4 展开 node5 → node7 0.925 / node8 0.925           ← 平，更贵 → 被支配
```

| iter | 展开节点 | depth | 子节点 sel_succ | 前沿是否增长 | no_improve |
|---|---|---|---|---|---|
| 1 | node0 | 0 | 0.925, 0.925 | ✅ | 0 |
| 2 | node2 | 1 | 0.900, **0.675** | ❌ | 1 |
| 3 | node1 | 1 | 0.925, 0.925 | ❌ | 2 |
| 4 | node5 | 2 | 0.925, 0.925 | ❌ | 3 |

搜索在 `n_evals=8 == budget` 时停止（`patience=4` 也快触发）。

**节点体积/成本随深度单调上涨**：104 → 2057 → 3186 → 4372 B；每题 token 1255 → 1635 → 1863 → 2108。`lambda_cost=0` 时导航不罚成本，技能自由膨胀，深层节点全部被 Pareto 支配（前沿正确地只留 root + node1）。

---

## 3. 三个硬事实（为什么 plateau）

1. **验证集在 depth 1 饱和**：8 个候选里 **5 个精确等于 0.925（=37/40）**。`sel=40` 时 1 题 = 0.025，候选之间分不出高下 → UCT 实际在噪声上导航，退化成"随便挑个 0.925 的兄弟"。搜索没有梯度可爬。

2. **成本单调上涨、精度零收益**：depth≥2 全是"加规则、加长度、加成本，但 success 不变甚至下降"。这反过来**验证了成本感知的动机**——即使本次跑 `lambda=0`，数据已显示一个正的 `lambda_cost` 会把导航从臃肿后代引开。

3. **固定 train batch → 兄弟节点"啃同一笔旧账，甚至把对的改反"**：所有 expansion 都对**同一个 `seed=42` 的 20 条 train**反思（见 [mcts.py:104](../skillopt/search/mcts.py#L104) 用固定 `self.train_items`）。具体证据：
   - **node1（制胜，0.925）**：`clue 指示姓氏时只答姓氏`。
   - **node4（崩溃，0.675）**：`答全名，不要只答姓氏`——**直接反转了 node1 的制胜规则**。

optimizer 反复在同样 20 个失败上打转，于是要么收敛到同一批规则，要么把正确的洞见改反。

---

## 4. 根因：SearchQA 太简单，不该用它评判 MCTS

- **单轮任务**（`max_turns=1`，每 rollout 1 次模型调用）：成本 = 一次调用的 prompt+completion，几乎完全由"技能被前置进 context"主导 → **成本轴 ≈ 技能长度的单调函数**，没有有趣的二维权衡。
- **gpt-5.5 本身太强**：空技能已 0.873，一次好反思（+7 题）就吃满增益，后续全 plateau。**SearchQA 缺少"每步 edit 解锁下一步"的多步梯度**，而那正是 MCTS 树搜索发光的地方。
- 结论：在 SearchQA 上 MCTS 最多打平 linear，无法体现增量价值。**要证明 MCTS 的价值，必须换到多步、有过程成本权衡的任务。**

---

## 5. 下一步方向：迁移到多轮任务 OfficeQA（`max_turn > 1`）

OfficeQA 是带工具的多轮检索问答（`configs/officeqa/default.yaml`：`max_tool_turns=24`、`max_queries_per_turn=4`、本地/在线 search 工具）。它恰好补齐 SearchQA 缺的两件事：

1. **多步精度梯度**：一条好技能可以改变检索策略（查什么、查几轮、如何调和冲突证据）→ 每步 edit 可解锁下一步，给树搜索真实的爬坡空间。
2. **真正二维的成本轴**：更好的技能可能**减少**工具轮数（更少 search → 更便宜）或**增加**轮数（更彻底 → 更贵）→ (精度, 成本) 形成非平凡 Pareto 权衡，而非 SearchQA 那种"成本=技能长度"的单调关系。这才是 cost-aware MCTS 相对 linear 的核心卖点。

### 5.1 ⚠️ 前置阻塞项：OfficeQA 必须先接好成本通道

当前 [officeqa/rollout.py:695](../skillopt/envs/officeqa/rollout.py#L695) 的 result dict 有 `hard`/`soft`/**`n_turns`**，但**没有 `cost_total_tokens`、也没有 `prompt_tokens`/`completion_tokens`**。而 MCTS 的成本由 [search/cost.py](../skillopt/search/cost.py) 的 `task_cost()` 读取，回退链是：

```
cost_total_tokens  →  prompt_tokens + completion_tokens  →  len(final response)/4
```

所以**直接跑 MCTS 会"能跑但成本轴失真"**：它只会数最终答案的字符数 / 4，完全忽略 1–24 轮工具调用 + 检索证据的 token——而那才是多轮任务的真实部署成本。

**修复（迁移第一步，必做）**：在 officeqa 多轮循环里累计每轮 prompt+completion，写入 `result["cost_total_tokens"]`，对齐 searchqa 的 [rollout.py:237/323](../skillopt/envs/searchqa/rollout.py#L237)。
- 备选/补充：officeqa 已有 `n_turns`，**可直接把"工具轮数"作为成本轴**——对多轮任务比 token 更可解释（"这条技能平均省了几轮检索"）。值得作为第二个成本口径一并产出。

### 5.2 OfficeQA MCTS 起步配置（草案）

仿照 `mcts_compare.yaml`，关键差异：
- `search.lambda_cost: 0.3`（**打开成本感知**——这次目的就是展示 Pareto，不再是纯精度对标）。
- `search.sel_eval_num: ≥80`（避免 SearchQA 那样的小样本饱和；OfficeQA 更难，天花板更低，留得出梯度）。
- `search.depth_limit: ≥4`、`branch: 2~3`、`budget: 12+`（多步梯度值得更深探索）。
- `env.max_tool_turns`、`max_completion_tokens` 沿用 officeqa 默认。
- 成本通道修好后，前沿的 `*_cost` 才有意义。

### 5.3 迁移检查清单

- [ ] **接成本通道**：officeqa rollout 累计多轮 token → `cost_total_tokens`（+ 暴露 `n_turns` 作为备选成本轴）。§5.1
- [ ] 确认 officeqa 的 split（`data/officeqa_split`）与 search 凭证（`OFFICEQA_CUSTOM_SEARCH_AUTH`）就绪，`source .env`。
- [ ] 新建 `configs/officeqa/mcts_compare.yaml`（含 `search:` 块，§5.2）。
- [ ] 先跑 linear baseline（`configs/officeqa/default.yaml`）拿对标点，再跑 MCTS。
- [ ] 验证多轮任务是否提供真实多步梯度（depth≥2 是否仍能涨分）——这是判断 MCTS 价值的关键观察。

---

## 6. SearchQA 本身的次要调优（若仍要在 SearchQA 上继续）

按优先级（但整体优先级低于"迁移到 OfficeQA"）：

1. **扩大验证集** `sel_eval_num: 100~150`——消除 0.925 饱和，让 UCT 有真实分差可导航。
2. **打开 `lambda_cost` 0.1~0.3**——省 budget + 产出真正的 Pareto 前沿（linear 永远只有单点）。
3. **增加 train-evidence 多样性**——在 `expand()` 里按 `node_id` 派生 seed 重取 `train_items`（现为固定），或加大 `batch_size`，让深层 edit 攻击新失败模式而非反转旧规则（§3.3）。
4. **效率**：提高并行度/`max_api_workers` 缓解 SH 串行；前沿点多时 test eval 按 top-k 限。

---

## 附录 A：部署技能（node1，2.0KB，sel 0.925 / test 0.92）

学到的全是"答案跨度精度"规则（与 baseline v0003 同类）：最短直答、姓氏槽位、地点/共同属性答具体名而非描述短语、匹配全部 clue 词、Jeopardy cloze、不带解释。完整见 `outputs/mcts_compare_20260630_174516/best_skill.md`。

## 附录 B：关键产物

| 文件 | 内容 |
|---|---|
| `summary.json` | 总览（tokens 分阶段、wall_time、n_evals、n_task_rollouts、cost_ref） |
| `mcts/frontier.json` | success-cost 前沿（2 点：root + node1），含 sel/test 双口径 |
| `mcts/tree.json` | 全部 9 节点 + 4 条 iter_log |
| `mcts/nodes/000X/skill.md` | 每个候选技能快照（看 node4 的反转规则） |
| `best_skill.md` / `mcts/best_skill_efficiency.md` | perf / efficiency 两个 Pareto 选点 |

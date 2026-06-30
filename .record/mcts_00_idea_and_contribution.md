# MCTS-SkillOpt 设计文档 0 — 概念与贡献（Idea & Contribution）

> **目的 / Purpose**：定义本项目的研究命题、问题形式化、相对 SkillOpt 与 PromptAgent 的定位、范围与成功标准。这是**概念层索引文档**——算法细节去 [`mcts_01`](mcts_01_mcts_algorithm_spec.md)，value/cost 估计去 [`mcts_02`](mcts_02_value_and_cost.md)，代码改造去 [`mcts_03`](mcts_03_code_integration_plan.md)，实验去 [`mcts_04`](mcts_04_experiment_plan.md)，论文体汇总去 [`mcts_05`](mcts_05_consolidated_algorithm_and_tradeoffs.md)，SkillOrchestra 借鉴去 [`mcts_06`](mcts_06_skillorchestra_pareto_borrowings.md)。
>
> 术语沿用 SkillOpt：**target** = 做任务的模型，**optimizer** = 改 skill 的模型。`file:line` 锚点基于 SkillOpt 内部文档撰写时的 HEAD。

---

## 1. 一句话 / Thesis

> 把 SkillOpt 的 skill 优化从**贪心爬山**升级为**成本感知的多目标 MCTS 规划**：节点是一份 skill 文档，动作是一次有预算的 edit，节点价值是 **(任务成功率, −skill 诱导成本)** 的二维向量；在「skill 版本树」上做带 lookahead 的 **UCT 树搜索**（v1：二维价值回传、selection 时标量化 UCB 并在提取期扫 λ；Pareto-UCT 留 v2），最终产出一条 **success–cost 的 Pareto 前沿**，而不是单个 `best_skill.md`。**target 与 optimizer 模型全程冻结**，MCTS 是纯文本空间的离散搜索，**部署期零模型调用**这一卖点不变。

---

## 2. 动机与缺口 / Motivation & Gap

| 维度 | SkillOpt（现状） | PromptAgent | 本项目 |
|---|---|---|---|
| 搜索结构 | 线性链 S0→S1→…（贪心爬山） | prompt 版本树（MCTS） | skill 版本树（MCTS） |
| 接受准则 | 严格 `>` gate，reject 即退回 current（`gate.py:123`） | UCT，可回溯 | 标量化 UCT（v1）→ Pareto-UCT（v2），可回溯 |
| 优化目标 | 仅 success | 仅 success | **success + cost（多目标）** |
| 对象 | 长结构化 skill 文档 | 短 prompt | 长结构化 skill 文档 |
| 评估成本 | 高（多轮 agentic rollout） | 低（单轮分类） | 高（多轮 agentic rollout） |
| 输出 | 单个 best_skill.md | 单个 prompt | **Pareto 前沿（多个工作点）** |

**两个具体缺口：**
1. **SkillOpt 是贪心的**：skill 沿单链前进，一旦 candidate 不严格优于 current 就退回，**无回溯、无 lookahead**，极易困在局部最优。这是 SkillOpt 方法层面最脆的点。
2. **对部署成本盲视**：SkillOpt 只统计**训练**成本，从不计 skill 的**部署**开销——而 skill 被 prepend 到每个任务的 context（常驻 token），还会诱导 agent 多做步骤/工具调用。单目标 success 会乐于学出**最啰嗦、最贵**的 skill。

**没人做的事**：在真实 agentic harness 上，对 skill edit 做**成本感知的多目标规划**。PromptAgent 证明了 MCTS-over-edits 可行，但它 success-only、对象是评估便宜的短 prompt。

---

## 3. 问题形式化 / MDP

| 元素 | 定义 | 复用 SkillOpt 的 |
|---|---|---|
| state `s` | 一份 skill 文档 | `skill_hash(s)` 直接作节点 id（`utils`） |
| action `a` | 一次有预算的 edit | reflect→aggregate→select→update 流水线产出（`reflect.py`/`aggregate.py`/`clip.py`/`skill.py`） |
| transition | 应用 edit（给定 edit 后确定性） | `apply_patch_with_report`（`skill.py:165`） |
| reward `r(s)` | 向量 `(success, −cost)`，在 selection 集上评估 | rollout 在 `valid_seen` 上的 `hard/soft` + 新增 cost 通道 |
| 目标 | 在 skill 版本树上找 Pareto 最优的 skill 集合 | — |

`r(s)` 的 success 维沿用 SkillOpt 的 `compute_score`（hard/soft/mixed，`gate.py:46`）；cost 维是**新增**的（token / turn / tool-call，定义见 `02_value_and_cost.md`）。

---

## 4. 为什么用 MCTS（它修掉 SkillOpt 的什么）

- **回溯**：可以回到树中任一非叶节点，换一条 edit 路径，而非贪心只能从 current 往前。
- **lookahead**：Q 值沿路径前传，评估一条 edit 路径的长期收益，而非只看单步 Δ。
- **explore / exploit**：UCT 平衡，主动跳出严格 `>` gate 卡住的局部最优。
- **粒度非单调，需要先降后升**：success 对 skill 详细度**非单调**——过细的 skill 会被误用 / 稀释注意力而**直接降准确率**（不只是更贵，见 [`mcts_06` §2.1](mcts_06_skillorchestra_pareto_borrowings.md)）。贪心严格 `>` gate 结构上做不到「先压缩降一点分、再换更稳的高分」，UCT 回溯可以——这是比「省 token」强得多的用 MCTS 理由。
- 一句话：MCTS 正面回应 SkillOpt 的核心弱点（贪心局部最优），而代价（更多节点评估）正是 [`mcts_02`](mcts_02_value_and_cost.md) 要用配对评估 + successive halving 去压的东西。

---

## 5. 新颖性边界（诚实定位）

- **不主张** "MCTS over text 是新的"——PromptAgent 已经做过，硬卖这点会被判定为 incremental。
- **真正的 delta**：(a) **成本感知的多目标节点价值 + Pareto 前沿输出**；(b) 对象是**持久、可审计、agentic** 的 skill 而非单 prompt；(c) 在真实工具执行 harness（Codex / Claude Code / ALFWorld）上。
- **基线必须包含 success-only MCTS**（= PromptAgent 思路直接搬到 skill 上），用来证明增益来自 **cost 维度与多目标搜索**，而不是 MCTS 本身。

---

## 6. 核心技术挑战（不藏着）

节点评估 = 一遍 selection 集 rollout，三重困难：**贵 + 高方差 + Dsel 小**（SkillOpt doc4 §F 明确：selection 集小、hard 不敏感时要换 soft/mixed）。naive MCTS 在这种 reward 上跑不动。因此 value 估计走（细节见 `02`）：

- **配对评估 / CRN（降方差）**：父子节点在**同一批采样任务**上评，比较 Δ 而非绝对值。
- **successive halving（控成本）**：有希望的节点追加评估预算，差的早砍。
- **（未来）廉价 value 模型**：回归预测 (success, cost)，兼作 verifier-free gate。
- **待研究的张力**：搜索越多 → 越容易对小 Dsel 过拟合（多重比较）。把「**搜索预算 vs Dsel 过拟合**」当成一个要正面量化的问题，而不是回避——这本身可以是论文的一个 finding。

---

## 7. 范围 / Scope（v1）

**做**：单域、单 skill 的 Pareto 前沿；在 SkillOpt 现有 env 上改造；先打 2-3 个 benchmark（建议 1 个程序性 Spreadsheet/OfficeQA + 1 个 ALFWorld）。

**不做（v1 非目标）**：skill library / 跨域共享；改任何模型权重；RL 微调任何网络（MCTS 是纯搜索，LLM 全程冻结）。

**复用**：EnvAdapter 接口、`sel_cache`(`skill_hash`) 节点缓存、TokenTracker 计费、resume 三件套。

---

## 8. 成功标准 / Success criteria

- 在 **success–cost 平面**上 **Pareto-dominate** SkillOpt：等成本更高 success，或等 success 更低成本。
- 至少在 1-2 个程序性 benchmark 上显著（这些 benchmark 上 skill 诱导的步数/格式开销最大，cost 维最有发挥空间）。
- 额外交付：**搜索预算–过拟合特性曲线**；配对评估 / successive halving 的消融，证明它们让 MCTS 在昂贵 reward 下变得可行。

---

## 9. 文档地图 / Doc map

| 文档 | 层 | 回答什么 |
|---|---|---|
| `mcts_00`（本文） | 概念 | 做什么、为什么、定位、范围 |
| [`mcts_01`](mcts_01_mcts_algorithm_spec.md) | 算法 | 四阶段精确定义、树结构、伪代码 |
| [`mcts_02`](mcts_02_value_and_cost.md) | 承重墙 | value 怎么估、cost 怎么度量、方差/成本怎么压 |
| [`mcts_03`](mcts_03_code_integration_plan.md) | 工程 | 逐文件改 SkillOpt、不碰哪些红线 |
| [`mcts_04`](mcts_04_experiment_plan.md) | 评测 | 基线、指标、消融、算力对齐 |
| [`mcts_05`](mcts_05_consolidated_algorithm_and_tradeoffs.md) | 汇总 | 论文体 Algorithm 1 + 优缺点 + 待改进 |
| [`mcts_06`](mcts_06_skillorchestra_pareto_borrowings.md) | 横向借鉴 | SkillOrchestra 的 Pareto 净增量与指导 |

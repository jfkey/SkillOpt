# MCTS-SkillOpt 设计文档 6 — SkillOrchestra 的 Pareto 借鉴与指导（三支柱重组）

> **目的 / Purpose**：把 SkillOrchestra（`2602 CITE_11`）的 **Pareto-optimal handbook selection（§4.3）** 这套*产物选择方法论*，对齐到我们已成型的 MCTS 设计上。本版**按三支柱（二维价值 / Pareto 前沿 / 扫 λ）重新组织**，每条主轴下统一走「SkillOrchestra 证据 → 印证 records 什么 → 🟢 净增量/锐化 → 落地」，并新增一张**对照 PDF 的事实核对表**。
>
> 关联：概念见 [`00`](mcts_00_idea_and_contribution.md)、承重墙见 [`02`](mcts_02_value_and_cost.md)、合并算法见 [`05`](mcts_05_consolidated_algorithm_and_tradeoffs.md)。
>
> ⚠️ **先声明边界**：SkillOrchestra **不搜索、不 RL**（它是「对比挖掘 + 贝叶斯能力估计 + 贪心查表」）。我们借的是它的**目标函数 / 选择准则 / 粒度观**，**不是**它的求解结构——求解结构由我们的 MCTS 提供。混淆这点就会错搬（见 §4）。

---

## 0. 一句话 + 锚点公式 / One-liner

> records 已经独立想到「二维价值 + Pareto 前沿 + 扫 λ」这层。SkillOrchestra 的真正净增量是把这三支柱各**锐化一刀**：① 二维价值里 **cost 不是惩罚项，粒度是耦合的第三轴，且 success 对 detail 非单调**（太详细会**降准确率**，不只是更贵）；② Pareto 前沿是**菜单不是终产物**，挑部署点要**声明的 prior**；③ 扫 λ 上，它的「per-λ 重选」对我们是灾难，必须翻译成**「一次搜索、提取时扫 λ」**，且**最终选点要从搜索期的相对 Δ 切回绝对、整轨迹、留出**。

**锚点公式**——SkillOrchestra §4.3 的选择目标一行装下三支柱：

```
H*_base = argmax_{H ⊆ H*}  E_{q∼D_val} [ R(τ_H(q)) − λ · Σ_t C(ψ_t, A_t) ]
                                          └── 二维价值 (R, C) ──┘
          └─ 扫 λ 描出 H*_base(λ) 一族 = Pareto 前沿 ─┘
          └─ argmax 在留出集 D_val、按整条轨迹 τ_H ─┘
```

---

## 1. 不重复造轮子：records 已有 vs SkillOrchestra 净增量

下表标清楚哪些 records 已覆盖（别再写），哪些是 SkillOrchestra 的净增量（本文重点）。后文按三支柱把净增量逐条展开。

| 维度 | records 现状 | SkillOrchestra 是否补充 | 归属 |
|---|---|---|---|
| 二维节点价值 (success, cost) | ✅ [`02`§1](mcts_02_value_and_cost.md)、[`00`§3](mcts_00_idea_and_contribution.md) | 已有，**不补** | 支柱一 |
| Pareto 前沿作为产物 | ✅ [`05` EXTRACTFRONTIER](mcts_05_consolidated_algorithm_and_tradeoffs.md) | 已有，**不补** | 支柱二 |
| λ 标量化 + 扫 λ + cost 归一化 | ✅ [`02`§6](mcts_02_value_and_cost.md) | 已有，**不补** | 支柱三 |
| confound（贵但更成功）用前沿化解 | ✅ [`02`§2](mcts_02_value_and_cost.md) | 已有，**不补** | 支柱一 |
| Dsel 过拟合 / 嵌套验证 / test-only | ✅ [`02`§7](mcts_02_value_and_cost.md)、[`05`§5](mcts_05_consolidated_algorithm_and_tradeoffs.md) | 已有，**强化** | 支柱三 |
| **粒度是决策轴 + success 对 detail 非单调** | ❌（cost 只当惩罚项，默认 success 随 detail 单调↑） | 🟢 **净增量** | 支柱一 |
| **从前沿挑部署点的 prior** | ⚠️（只说扫 λ 给前沿，没说怎么收成一个产物） | 🟢 **净增量** | 支柱二 |
| **λ 与树解耦（一次搜索扫 λ）+ 选点切回绝对** | ✅（[`05` 待改进 #4](mcts_05_consolidated_algorithm_and_tradeoffs.md) 现已写进主算法 [`mcts_01` §5](mcts_01_mcts_algorithm_spec.md)，Option B） | 🟢 **净增量/锐化** | 支柱三 |
| **「押注 Pareto」的外部背书 + 叙事定位** | ⚠️（[`05`§3.3](mcts_05_consolidated_algorithm_and_tradeoffs.md) 自己担心这是最弱主张） | 🟢 **净增量** | meta §3 |
| **双列 (Acc, Cost) 消融模板** | ✅（已落进 [`mcts_04` §4](mcts_04_experiment_plan.md)） | 🟢 **可直接抄** | meta §3 |

---

## 2. 三支柱 / The three pillars

### 2.1 支柱一：二维价值 (success, cost) — 净增量最大

- **SkillOrchestra 证据**：每个路由决策都在 `E[competence] − λ·cost` 上权衡（§4.1 `A*_t = argmax_A [E_σ φ_{A,σ} − λ_c·Ĉ_A]`）；Fig 1 / Fig 3 的坐标轴就是 (Accuracy, Cost)。
- **印证 records**：节点价值 = (success, cost)（[`02`§1](mcts_02_value_and_cost.md)、[`00`§3](mcts_00_idea_and_contribution.md)）。**已有，不补。**
- **🟢 净增量（本文最重要）：cost 不只是惩罚项，粒度是耦合的第三轴，且 success 对 detail 非单调。**
  - Table 2：`No FG Skills`（去细粒度）80.4% / \$15.1，`Full System` 85.0% / \$9.3 —— 去掉细粒度技能**既更差又更贵**；粒度配得好，两轴**同时改善**。
  - Step 1（p.7–8）：强 orchestrator 能可靠区分 `symbolic_logic` vs `numerical_approximation`；**弱 orchestrator 会认错激活技能、引入路由偏差→准确率下降**，用粗粒度（`data_processing`）反而更稳。
  - 这推翻 records 的隐含假设——[`02`§6](mcts_02_value_and_cost.md) 只罚超基线 cost、以免逼 skill 一味缩短，隐含「success 随 detail 单调↑、只是越长越贵」。SkillOrchestra：**对冻结 target，过度详细的 skill 会直接拉低 success**（被误用、注意力稀释、触发错误分支）。于是 success-最大的 skill 未必最长，可能在**中等粒度**；cost 轴从「省钱」升级为「accuracy 杠杆」，坐实 [`00`§2 缺口 2](mcts_00_idea_and_contribution.md)「单目标会学出最啰嗦最贵的 skill」——现在能加一句：**而且很可能更差**。
- **落地指导**：
  - 🔴 **EXPAND 必须含双向粒度 move**，不只是 reflect→aggregate→clip→apply 的增量累加（对应 SkillOrchestra Phase 2 的 merge/split）：
    - `abstract / compress`：把若干细规则合并、抽象成高层指导（往短走）。
    - `refine / split`：把一条粗规则拆成带触发条件的子规则（往细走）。
  - 🟡 写进「为什么用 MCTS」（[`00`§4](mcts_00_idea_and_contribution.md)）：贪心严格 `>` gate 结构上做不到「先压缩降一点分、再换更稳的高分」，UCT 回溯可以——这是 MCTS 比贪心值钱的硬理由之一（比「省 token」强得多）。
  - 🟡 粒度甜点是 **per-target** 的（Fig 6 右 + §4.1）：每个 run 冻结一个 target，选出的粒度是为这个 target 调的，**别假设能迁到更强/更弱的 target**。→ 顺手变成一个实验（§3 消融）。

### 2.2 支柱二：Pareto 前沿 — 前沿是菜单，不是终产物

- **SkillOrchestra 证据**：Fig 1 —— SkillOrchestra/+ 落在 (Acc, Cost) 前沿（比所有 baseline 更准更省）；Fig 3(中) —— 候选手册 `H^(1..K)` 在 (Reward, Cost) 平面，非支配集构成前沿。
- **印证 records**：输出一条前沿而非单个 best_skill（[`05` EXTRACTFRONTIER](mcts_05_consolidated_algorithm_and_tradeoffs.md)）。**已有，不补。**
- **🟢 净增量：从前沿挑部署点要有声明的 prior。** Fig 3(中) 明确画了 `Perf-prior selection` 与 `Efficiency-prior` 两个箭头，各取**一个**点 `H^(O)_base` 作为实际部署手册。records 把「产出一条前沿」当终点，但**没回答下游到底 ship 哪个 `best_skill.md`**。SkillOrchestra 给了干净两段式：**前沿 = 菜单**（我们扫 λ 产出）；**prior = 怎么点菜**（部署期把菜单收敛成一个产物）。
- **落地指导**：产物分两层——
  - (a) `frontier.json`：全部非支配工作点 + 各自 test 分与 cost（菜单）。
  - (b) 按 prior 各导一个 `best_skill.md`：`perf` = test success 最高（不看 cost）；`efficiency` = success 掉 ≤ ε（默认如 1pt）内 cost 最低。
  - 这同时让 [`05` 待改进 #4](mcts_05_consolidated_algorithm_and_tradeoffs.md) 落地更顺：**一棵树服务所有 λ → 前沿免费 → prior 只是前沿上的一次挑选**，无需为每个 prior 重跑（见支柱三）。

### 2.3 支柱三：扫 λ — λ 与树解耦 + 选点切回绝对

- **SkillOrchestra 证据**：λ 是 §4.3 目标里显式的 tradeoff 系数，扫 λ 得到不同 `H*_base` → 描出前沿；§4.3 强调选手册「**directly evaluates entire trajectories rather than local routing accuracy**」，在留出 `D_val` 上按整条轨迹的 `argmax_H E[R − λΣC]` 挑。
- **印证 records**：λ 标量化 + 扫 λ + cost 归一化（[`02`§6](mcts_02_value_and_cost.md)）；最终选点用独立留出 + 整条 rollout 的 (success, cost)（[`05`§5](mcts_05_consolidated_algorithm_and_tradeoffs.md)、[`02`§7](mcts_02_value_and_cost.md)）。**已有，强化。**
- **🟢 净增量/锐化（两点，都关系到「别把 SkillOrchestra 的便宜操作直接搬」）**：
  1. **λ 必须与树解耦。** SkillOrchestra 每个 λ 重选手册是因为它**贪心查表、几乎免费**；我们每个 λ 重建 MCTS 树则是灾难（开销文档的头号威胁）。对策（= [`05` 待改进 #4](mcts_05_consolidated_algorithm_and_tradeoffs.md)）：**backup 回传二维向量 W、不把 λ 烤进 Q，selection 时才标量化** → 一棵树服务所有 λ，前沿提取免费、可搜索中自适应 λ（向 Pareto-UCT 过渡）。把它们的「per-λ reselect」翻译成我们的「一次搜索、提取时扫 λ」。
  2. **最终选点的度量要从「搜索期相对 Δ」切回「绝对、整轨迹、留出」。** 我们用配对 Δ(CRN) 做树内导航/SH 排序（[`02`§3–5](mcts_02_value_and_cost.md)）——那是**局部相对量**；SkillOrchestra §4.3 提醒最终选点不能用这个局部量。一句写死：**Δ 只用于树内导航；前沿提取与最终选点一律用 telescoping 锚定后的绝对二维价值 + 独立 D_test。**

---

## 3. 跨支柱 meta：背书 / 切割 / 消融模板

- **🟢 外部背书（呼应支柱二/三）**：[`05`§3.3](mcts_05_consolidated_algorithm_and_tradeoffs.md) 诚实结论是「贡献押在多目标前沿，而非 MCTS>greedy on success（最弱主张）」。SkillOrchestra 正是这个赌注的外部证据——它**整篇靠 Pareto** 打赢 RL baseline（Router-R1/ToolOrchestra），且明确「higher per-token price ≠ higher total cost」「skill-aware 分配比死磕大模型更省更准」。说明「显式 performance-cost 前沿 + 让产物复杂度匹配消费者容量」是站得住、能发表的主张。
- **🟡 related work 必须显式切割（风险）**：SkillOrchestra 已占住「skill + Pareto + performance-cost」这组词。务必点名区分：**它 = 路由/编排的 Pareto（选 agent/model）、无搜索**；**我们 = 单 skill 文档优化的 Pareto（选 edit 路径）、核心是 MCTS lookahead/回溯**。否则易被审稿人判 incremental。叙事可定位成 **「PromptAgent（success-only MCTS-over-text）× SkillOrchestra（cost-aware Pareto selection）的交叉点，对象是 agentic、可审计、持久的 skill」**——单独哪个都不新，组合 + agentic harness 才是 delta（[`00`§5](mcts_00_idea_and_contribution.md)）。
- **🟢 消融模板（直接抄 Table 2 双列结构）**：[`mcts_04` §4](mcts_04_experiment_plan.md) 已套这个骨架，每个组件同时报 Acc% 和 Cost\$。

  | 设置 | 回溯(UCT) | 降粒度 move | CRN+SH | Pareto 选点 | **test success** | **cost** |
  |---|---|---|---|---|---|---|
  | 退化(b=1,c=0) = 贪心 SkillOpt | ✗ | ✗ | — | ✗ | — | — |
  | success-only MCTS = PromptAgent 搬运 | ✓ | ✗ | ✓ | ✗(只取 max success) | — | — |
  | + 降粒度 move（支柱一） | ✓ | ✓ | ✓ | ✗ | — | — |
  | Full（cost-aware Pareto） | ✓ | ✓ | ✓ | ✓ | — | — |

  - **关键消融**：去掉「降粒度 move」那一行 ≈ SkillOrchestra 的 `No FG Skills`，证明双向粒度搜索**同时改善 success 和 cost**（复刻 85.0/\$9.3 vs 80.4/\$15.1）。
  - **per-target 粒度甜点实验**（支柱一末尾）：固定 benchmark，换 1 强 1 弱 target，各画 Pareto 前沿——若甜点粒度随容量右移，是一条干净 finding，与 SkillOrchestra Fig 6 互证。

---

## 4. ⚠️ 不要错搬 / Honest boundaries

- 它的 Pareto 候选是**少量预枚举的粒度档**（98/10/3 skills）；我们的候选由 **MCTS 树搜索生成**。借「目标 + 选择准则 + 粒度观」，**别借「候选从哪来」**。
- 它的 cost 是**部署期路由的 token/latency**；我们的 cost 是 **skill 文档自身体积 + 诱导步数**（[`02`§1](mcts_02_value_and_cost.md)）。两者都属「部署 cost」可类比，但与本系列一再强调的「**MCTS 搜索/评估 cost**」（开销文档头号威胁）是两回事，写作别混进同一句。
- 它**无 lookahead、无回溯、无 credit assignment**（贪心查表），所以**给不了「搜索怎么做」的任何指导**——那部分仍由 [`01`](mcts_01_mcts_algorithm_spec.md)/[`05`](mcts_05_consolidated_algorithm_and_tradeoffs.md) 的 UCT/SH 负责。

---

## 5. 事实核对（对照 PDF）

| records / 本文说法 | PDF 出处 | 核对 |
|---|---|---|
| Full System 85.0/\$9.3 vs No FG Skills 80.4/\$15.1（更准更便宜） | Table 2 (p.13) | ✅ |
| perf-prior / efficiency-prior 从前沿挑一个部署点 | Fig 3(中) (p.4) | ✅ |
| 选手册按整条轨迹、留出集 `argmax_H E[R−λΣC]` | §4.3 (p.9–10) | ✅ |
| 弱模型下过细粒度→认错激活技能→掉分；粗粒度更稳 | Step 1 (p.7–8) | ✅ |
| 手册跨 backbone 迁移、强 backbone 增益更大 | Fig 6(右)/Obs 4 (p.12) | ✅ |
| Pareto 打赢 RL baseline、per-token 价 ≠ 总成本 | Fig 1/5、Obs 2 (p.1,11) | ✅ |
| Phase 2 split/merge（高方差→split，画像难分→merge） | §4.2 (p.9) | ✅（对应降/升粒度 move 的语义来源） |

---

## 6. 落地 checklist / Actionable

- [ ] EXPAND 增加 **abstract/compress** 与 **refine/split** 两类降/升粒度 move（支柱一）。
- [ ] [`00`§4](mcts_00_idea_and_contribution.md)「为什么用 MCTS」加一条：**success 对 detail 非单调，需要能先降后升的回溯**（支柱一）。
- [ ] 产物分两层：`frontier.json`（菜单）+ 按 `perf`/`efficiency` prior 各导一个 `best_skill.md`（支柱二）。
- [ ] backup 回传**二维向量**、selection 时才标量化（一次搜索扫 λ）；写死「**Δ 只用于树内导航；前沿提取/最终选点用绝对二维价值 + 独立 D_test**」（支柱三）。
- [ ] related work 显式区分 SkillOrchestra（路由 Pareto、无搜索）vs 本工作（单 skill、MCTS 搜索）（meta §3）。
- [ ] 按 [`mcts_04` §4](mcts_04_experiment_plan.md) 跑 Table 2 双列 (Acc,Cost) 消融 + 「去掉降粒度 move」消融 + per-target 粒度甜点实验（meta §3）。

---

## 7. 文档地图 / Doc map

| 文档 | 层 | 回答什么 |
|---|---|---|
| [`mcts_00`](mcts_00_idea_and_contribution.md) | 概念 | 做什么、为什么、定位、范围 |
| [`mcts_01`](mcts_01_mcts_algorithm_spec.md) | 算法 | 四阶段、树结构、伪代码 |
| [`mcts_02`](mcts_02_value_and_cost.md) | 承重墙 | value/cost 估计、降方差/控成本 |
| [`mcts_03`](mcts_03_code_integration_plan.md) | 工程 | 逐文件改 SkillOpt、线性机制冲突、红线 |
| [`mcts_04`](mcts_04_experiment_plan.md) | 评测 | 基线、指标、消融、算力对齐、过拟合曲线 |
| [`mcts_05`](mcts_05_consolidated_algorithm_and_tradeoffs.md) | 汇总 | 论文体 Algorithm 1 + 优缺点 + 待改进 |
| `mcts_06`（本文） | 横向借鉴 | SkillOrchestra 的 Pareto 给本设计的净增量与指导（三支柱重组） |

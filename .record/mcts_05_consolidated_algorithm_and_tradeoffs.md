# MCTS-SkillOpt 设计文档 5 — 合并版 Algorithm 1 与优缺点评估

> **目的 / Purpose**:把分散在 [`mcts_01`](mcts_01_mcts_algorithm_spec.md)(四阶段)与 [`mcts_02`](mcts_02_value_and_cost.md)(估值引擎)里的内容,**收敛成一份论文体的 Algorithm 1**(可直接对照 SkillOpt 原 Algorithm 1),并附上对该方法的**诚实优缺点评估、是否会更好的判断、以及待改进清单**。工程落地见 [`mcts_03`](mcts_03_code_integration_plan.md),实验/算力对齐见 [`mcts_04`](mcts_04_experiment_plan.md)。
>
> 概念/定位见 [`mcts_00`](mcts_00_idea_and_contribution.md)。术语沿用 SkillOpt:**target** = 做任务的冻结模型 M,**optimizer** = 改 skill 的模型 O。
>
> v1 决策:**二维价值回传 + selection 时标量化 UCB**(Option B:backup 不焼 λ、提取期扫 λ;见 §2 的 BACKUP/SELECT,与 [`mcts_06` 支柱三](mcts_06_skillorchestra_pareto_borrowings.md) 一致),Pareto-UCT 留 v2;关闭 slow-update / meta-skill(见 §3 待改进第 2 条 —— 这是一个**有争议**的决策)。

---

## 0. 一句话 / One-liner

> SkillOpt 现在的 6 阶段循环 = **分支因子 1、永远选 current、严格 `>` 接受、不回溯**的退化 MCTS。本算法把它泛化成 **skill 版本树上的成本感知多目标 MCTS**:`reflect→aggregate→select→update` 变成 **Expansion**,`gate` 的评估变成 **Simulation(配对估值)**,`accept/reject` 变成 **Backpropagation**,新增 **UCT Selection** 让搜索能回到任意节点换路径;最终产出一条 **success–cost Pareto 前沿**而非单个 `best_skill.md`。M 与 O 全程冻结,部署期零模型调用不变。

---

## 1. 与原 SkillOpt Algorithm 1 的阶段映射

| 原 Alg1 阶段 | MCTS 对应 | 复用 / 新增 |
|---|---|---|
| `s_cur` 单指针前进 | **Selection**:root 按 UCT 下降到可扩展节点 | 全新(`search/mcts.py`) |
| L8–13 rollout→reflect→aggregate→clip→apply | **Expansion**:在选中节点上长出 `b` 个候选子 skill | 原样复用 ②③④⑤ |
| L17 `EVALUATE(·,D_sel)` | **Simulation**:CRN 配对 + successive halving 估二维价值 | 复用 rollout/`sel_cache`,新增配对+SH |
| L20–27 严格 `>` gate + rejected buffer | **Backprop**:telescoping 锚定的**二维价值向量回传**(Option B,不焼 λ) | gate 的"比较"语义改成树回传 |
| L29–35 epoch 末 slow/meta update | (v1 关闭)其成对比较能力被复用为 CRN | 见 §3 待改进第 2 条 |
| L37 单个 `best_skill` | **Frontier extraction**:非支配集 →(success,cost)前沿 | 新增 |

---

## 2. Algorithm 1 — MCTS-SkillOpt (cost-aware multi-objective)

```
Algorithm 1  MCTS-SkillOpt skill optimization
─────────────────────────────────────────────────────────────────────────────
Require: frozen target M, optimizer O, harness h, splits D_train, D_sel, D_test,
         initial skill s0, scalarization weight λ, exploration const c,
         branch factor b, depth limit D, init eval size n0, SH factor η,
         search budget B (selection-rollout calls), patience π,
         reflection minibatch size B_m, edit-budget schedule L_t
Ensure:  Pareto frontier F of validation-gated skills, and held-out test scores
─────────────────────────────────────────────────────────────────────────────
 1: root ← NODE(s0); EVAL ← ∅                       # EVAL = 全局已评估节点账本
 2: root.value ← ESTIMATEVALUE_FULL(root)           # 较大预算的绝对锚 (success,cost)
 3: cache[HASH(s0)] ← root.eval_tasks               # transposition table = sel_cache
 4: EVAL ← EVAL ∪ {root};  no_improve ← 0;  v* ← root.value
 5: while budget B not exhausted and no_improve < π do
 6:     ▷ 1. SELECTION ───────────────────────────── (纯 bookkeeping, 0 模型调用)
 7:     node ← SELECT(root, c, λ, D)
 8:     if node.terminal then continue
 9:     ▷ 2. EXPANSION ────────────────────────────── (= 一个 SkillOpt edit step)
10:     if node.train_evidence = ∅ then
11:         node.train_evidence ← ROLLOUT(M, h, node.skill, D_train)   # 首次缓存
12:     children ← EXPAND(node, b)
13:     ▷ 3. SIMULATION (价值估计, CRN + SH) ────────
14:     children ← ESTIMATEVALUE_SH(children, node, n0, η)             # 共享预算分配
15:     for each child ∈ children do
16:         child.value ← node.value + child.Δ        # telescoping 锚定到 root
17:         cache[HASH(child.skill)] ← child.eval_tasks
18:         EVAL ← EVAL ∪ {child};  node.children.append(child)
19:     node.expanded ← (len(node.children) ≥ b) or no feasible edit
20:     ▷ 4. BACKPROPAGATION (Option B: 二维向量回传, 不焼 λ) ─────
21:     for each child ∈ children do
22:         BACKUP(child)                             # ∀祖先: N+=1; W += child.value (二维); Q 不落盘
23:     ▷ 改进追踪 (用于 patience 早停, 锚定在二维价值上)
24:     if ∃ child with child.value ≻ v* (Pareto-dominates) then
25:         v* ← child.value;  no_improve ← 0
26:     else  no_improve ← no_improve + 1
27: end while
28: ▷ 提取与汇报 ────────────────────────────────────
29: F ← EXTRACTFRONTIER(EVAL)                         # (success,cost) 非支配集
30: for each s ∈ F do  score_test[s] ← EVALUATE(M, h, s, D_test)   # 仅此处碰 test
31: return F, score_test
```

### Procedure SELECT — UCT 下降(可回溯)

```
SELECT(root, c, λ, D):
 1: node ← root
 2: while node.expanded and node.children ≠ ∅ and depth(node) < D do
 3:     node ← argmax_{ch ∈ node.children} UCT(ch, c)
 4: return node
 5:
 6: UCT(ch, c, λ):                                    # ch.N=0 → +∞ (强制至少访一次)
 7:     return scal_λ(ch.W / ch.N) + c · sqrt( ln(N(parent(ch))) / N(ch) )
        # 🔴 λ 只在此处出现: ch.Q = scal_λ(二维均值 ch.W/ch.N) 在 selection 时才标量化
        # scal_λ(v) = v.success − λ·max(0, ĉ−1),  ĉ = cost/cost_noskill  (mcts_02 §6)
```

### Procedure EXPAND — 一个 edit step,复用 SkillOpt ②③④⑤

```
EXPAND(node, b):
 1: children ← [ ]
 2: for k = 1 … b do                                  # b 个互异候选: 变 seed 或 O 采样温度>0
 3:     P_fail ← O.reflect(failure-minibatches of node.train_evidence, B_m)   # §3.3
 4:     P_succ ← O.reflect(success-minibatches of node.train_evidence, B_m)
 5:     E ← O.merge(P_fail) ⊕ O.merge(P_succ)  with failure-priority           # aggregate
 6:     E ← rank/clip(E, L_t)                          # 有界编辑预算 (textual lr)
 7:     s̃ ← apply_patch(node.skill, E)                 # 尊重 target 逐字匹配/保护区红线
 8:     children.append( NODE(s̃, parent=node, edge_edits=E) )
 9: return children
```

### Procedure ESTIMATEVALUE_SH — CRN 配对 + successive halving(承重墙)

```
ESTIMATEVALUE_SH(children, parent, n0, η):
 1: S ← parent.eval_tasks                             # 固定顺序/定种子的 D_sel 任务列表 (CRN 锚)
 2: alive ← children;  m ← n0
 3: while |alive| > 1 and m ≤ |S| do
 4:     T ← first m tasks of S
 5:     for each ch ∈ alive do
 6:         if HASH(ch.skill) ∈ cache then  reuse per-task scores  ; continue
 7:         run ch on T  (低温/定种子 → CRN 前提)      # ROLLOUT on D_sel
 8:         ch.Δ_succ ← mean_{t∈T}[ succ_ch(t) − succ_parent(t) ]   # 配对差, 低方差
 9:         ch.Δ_cost ← mean_{t∈T}[ cost_ch(t) − cost_parent(t) ]
10:         ch.eval_tasks ← per-task (succ, cost) on T              # 增量缓存
11:     keep top ⌈|alive|/η⌉ of alive by (Δ_succ − λ·Δ_cost)       # SH 排序用配对 Δ
12:     alive ← kept;  m ← m · η
13: for each ch ∈ children do  ch.Δ ← (ch.Δ_succ, ch.Δ_cost)
14: return children
        # cost 归一化: ĉ = cost / cost_noskill;  标量化只罚超基线部分 max(0, ĉ−1)
```

### Procedure BACKUP — telescoping 二维价值向量回传 (Option B, 不焼 λ)

```
BACKUP(child):                                          # 不收 λ —— 这是 Option B 的关键
 1: x ← child
 2: while x ≠ ∅ do
 3:     x.N ← x.N + 1
 4:     x.W.success ← x.W.success + child.value.success    # 二维向量累加(导航用)
 5:     x.W.cost    ← x.W.cost    + child.value.cost
 6:     x ← parent(x)
        # Q 不落盘: SELECT 时按需 x.Q = scal_λ(x.W / x.N); λ 只在 SELECT 出现
        # ⇒ 一棵树服务所有 λ, 前沿提取与 λ 无关(见 §4 待改进 #4 —— 已落进主算法)
        # 导航(Q=标量化均值)与提取(child.value, 二维全局账本 EVAL)分离 —— 化解 mean/max 之争
```

### Procedure EXTRACTFRONTIER — 输出 Pareto 前沿(取代单个 best_skill)

```
EXTRACTFRONTIER(EVAL):
 1: return { n.skill : n ∈ EVAL,  ∄ n' ∈ EVAL s.t. n'.value ≻ n.value }
        # 非支配集; 单次 λ 几乎免费给一条前沿; 扫 λ 网格 → 更完整前沿
        # 每个前沿点必须是真实通过评估的节点快照 (best_skill 不被污染)
```

---

## 3. 优缺点评估 / Tradeoffs

### 3.1 优点

- 🟢 **正面修掉 SkillOpt 最脆的点(贪心局部最优)**。严格 `>` gate 一旦 reject 就退回 `s_cur`、无回溯;MCTS 能回到树中任意节点换路径。
- 🟢 **explore/exploit 有原则**:UCT 把昂贵评估预算按潜力分配,而非死磕当前最优。
- 🟢 **容忍"先降后升"的编辑路径**:Q 沿路径前传,可换取更高长期收益 —— 严格 `>` gate 结构上做不到。
- 🟢 **天然多目标**:节点带 (success, cost),输出 Pareto 前沿,补上 SkillOpt 对部署成本的盲视。**这是最稳、最像新贡献的部分。**
- 🔵 **白送复用**:`sel_cache` = transposition table;reflect/aggregate/clip/apply 原样当 expansion;slow_update 成对比较 → CRN。

### 3.2 缺点 / 风险

- 🔴 **评估成本爆炸(头号威胁)**。节点价值 = 一遍 `D_sel` 的 agentic rollout(多轮、带工具),比 PromptAgent 的单轮分类贵几个数量级。MCTS 评估候选数远多于贪心;CRN+SH 只压乘数、不改量级。**这是存亡问题。**
- 🔴 **小 `D_sel` 多重比较过拟合被放大**。选 max-of-many → 乐观偏置,selection-best 易不迁移到 test,可能打破 SkillOpt Figure 3 的 selection↔test 对齐。
- 🟡 **telescoping 误差沿深度累积**,深节点价值不可靠 → 实际可用深度被压到 D≈4–6。而深度正是用 MCTS 的主要理由,自相矛盾。
- 🔴 **丢了 slow-update/meta-skill**。论文消融里"同时去掉 meta+slow"是最大的一次掉分(SpreadsheetBench 77.5→55.0)。MCTS 拿回回溯却扔掉真正驱动大涨的纵向巩固。
- 🟡 **CRN 假设脆**:配对降方差要求可复现解码(低温/定种子),ALFWorld 等随机环境/带工具 harness 不总能确定化;CRN 失效则 SH 误判,成本控制引擎塌。
- 🟡 **超参负担重**:c、λ 网格、b、n0、η、D、B、π + reflect 旋钮,很脆。
- 🟡 **edit economy 削弱回溯收益**:最终增益只来自 1–4 次被接受的编辑(Table 6),有效深度很浅 → lookahead/回溯收益小,成本乘数大。

### 3.3 会更好吗 / Verdict(诚实版)

分轴看:

- **"修方法弱点"轴(跳出局部最优)**:会更好。一定存在贪心卡住、MCTS 找到更优 skill 的 benchmark。
- **"headline 指标"轴(等成本下 test success)**:**不确定,甚至可能更差**。理由:① SkillOpt 消融说增益"对 batch/minibatch/lr 不敏感,对 gate、rejected buffer、slow/meta **非常敏感**" —— MCTS 留了 gate、却弱化/丢了另两个最关键杠杆;② edit economy → 有效深度浅;③ D_sel 过拟合放大。
- **真正的赢点是 cost-aware Pareto 前沿**,不是"MCTS 在 success 上打败贪心"。前者新颖稳健,后者是最弱主张。

> 📌 **结论**:贡献叙事押在**多目标前沿**上,而非"MCTS>greedy on success"。

---

## 4. 待改进清单 / Improvements(按优先级)

1. 🔴 **make-or-break 基线:等算力下对比"贪心 + 随机重启"**(同 token 预算,非同迭代数)。若 MCTS 打不过 random-restart greedy,树的机器不值这个钱。鉴于 edit economy,**宽而浅 + 重启**很可能优于一棵深树。
2. 🔴 **别丢 slow-update/meta-skill**:改成沿"当前最优路径"每 K 次迭代周期性巩固(写保护区、仍过 gate),把 SkillOpt 最大的杠杆捡回来。
3. 🔴 **廉价 value 模型 / 过程信号**(`02` §8):回归器预测 edit 的 Δ(success,cost) 当 prior/预筛,只在有希望的节点跑真 rollout —— 让 agentic 评估成本可控的唯一现实路径。
4. ✅ **已采纳进主算法(见 §2 BACKUP/SELECT)**:UCT 不在 backup 把 λ 烤进 Q,改回传**二维向量 W**、selection 时才标量化 → 一棵树服务所有 λ、前沿提取免费。**剩余 v2**:真正的 **Pareto-UCT**(用支配关系而非标量化做 selection)+ **搜索中自适应 λ**,把"扫 λ"从提取期进一步前移到导航期。
5. 🟡 **嵌套验证(selection-2)做最终选点** + 量化"搜索预算 vs 过拟合"曲线(SkillOpt Figure 3 同款),把放大的过拟合从风险变 finding。
6. 🟡 **progressive widening + 周期性 re-anchoring**:只在高访问节点加子节点(替代固定 b);对高访问深节点偶尔做一次全量评估,重置 telescoping 漂移。
7. 🟢 **确定性 transition → 可选 max-backup**:应用 edit 是确定性的,均值回传偏保守,max-backup 在确定性树上常更利于 exploit(列为旋钮)。
8. 🟢 **保留 rejected-edit 负记忆**:别完全依赖 Q 编码失败;rejected buffer 是论文里被实测验证有效的杠杆,可作每节点负记忆喂回 reflect。

---

## 5. v1 退化检验 / Degenerate checks(必须做的对照)

- `λ=0, b>1, c>0, D` 足够 → **success-only MCTS** = PromptAgent 思路直接搬到 skill 上;用来证明增益来自 **cost 维 + 多目标**而非 MCTS 本身。
- `b=1, c=0, D=∞` → 退回原 SkillOpt 贪心链。
- 最终分**只在独立 D_test 报**(held-out),与 SkillOpt 一致。

---

## 6. 文档地图 / Doc map

| 文档 | 层 | 回答什么 |
|---|---|---|
| [`mcts_00`](mcts_00_idea_and_contribution.md) | 概念 | 做什么、为什么、定位、范围 |
| [`mcts_01`](mcts_01_mcts_algorithm_spec.md) | 算法 | 四阶段精确定义、树结构、伪代码 |
| [`mcts_02`](mcts_02_value_and_cost.md) | 承重墙 | value 怎么估、cost 怎么度量、方差/成本怎么压 |
| [`mcts_03`](mcts_03_code_integration_plan.md) | 工程 | 逐文件改 SkillOpt、线性机制冲突、红线 |
| [`mcts_04`](mcts_04_experiment_plan.md) | 评测 | 基线、指标、消融、算力对齐、过拟合曲线 |
| `mcts_05`(本文) | 汇总 | 论文体 Algorithm 1 + 优缺点 + 是否更好 + 待改进 |
| [`mcts_06`](mcts_06_skillorchestra_pareto_borrowings.md) | 横向借鉴 | SkillOrchestra 的 Pareto 净增量与指导 |

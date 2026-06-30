# MCTS-SkillOpt 设计文档 1 — 算法规格（MCTS Algorithm Spec）

> **目的 / Purpose**：把「skill 版本树上的 MCTS」逐阶段讲精确——树节点存什么、四阶段怎么定义、价值怎么锚定与回传、何时终止、最终怎么从树里取前沿。每一步都映射到 SkillOpt 现有阶段代码（复用 > 重写）。
>
> 配套：概念/定位见 [`mcts_00`](mcts_00_idea_and_contribution.md)；**value/cost 估计的细节（配对评估、successive halving、cost 度量）见 [`mcts_02`](mcts_02_value_and_cost.md)**——本文的 §4 Simulation 只给接口，引擎在 `mcts_02`；代码改造见 [`mcts_03`](mcts_03_code_integration_plan.md)。
>
> v1 决策：**二维价值回传 + selection 时标量化 UCB**（Option B——backup 累加二维向量、不焼 λ，提取期扫 λ 得整条前沿；见 §5/§8，与 [`mcts_06` 支柱三](mcts_06_skillorchestra_pareto_borrowings.md) 一致）；Pareto-UCT 留 v2。

---

## 0. 一句话 / One-liner

> SkillOpt 现在的 6 阶段循环 = **分支因子 1、永远选 current 节点、贪心接受、不回溯**的退化 MCTS。本项目把它泛化成树：**reflect→aggregate→select→update 变成「expansion」（从一个节点长出候选子 skill），gate 的评估变成「simulation/价值估计」，accept/reject 变成「backpropagation」**，再新增 UCT selection 让搜索能回到任意节点换路径。绝大多数阶段代码可直接复用。

---

## 1. 树与节点 / Tree & Node

每个**节点 = 一份 skill 文档版本**。节点存：

| 字段 | 含义 | 复用/来源 |
|---|---|---|
| `skill_hash` | 节点 id（内容哈希） | SkillOpt `skill_hash`（`utils`） |
| `skill_content` | skill 文本（或由 hash→store 映射） | — |
| `parent` / `children` | 树结构 | 新增 |
| `edge_edits` | 由 parent 生成本节点的那组 edit（RawPatch/selected） | reflect→select 产出 |
| `n_visits` `N`、`W` | 访问次数 + **二维价值累加向量** `(Σsucc, Σcost)`（Option B 回传，§5） | UCT 用 |
| `Q` | 导航用标量，**不落盘**：按需 `scal_λ(W/N)`（§2/§5） | 新增 |
| `value_success` `value_cost` | 二维**绝对**价值分量（telescoping 锚定，提取/前沿用） | rollout 打分 + cost 通道（`mcts_02`） |
| `eval_tasks` | 已评估的 selection 任务 id + 每任务分数 | 供配对评估/SH 增量复用 |
| `train_evidence` | 本节点的 train-rollout 轨迹缓存（expansion 用） | 见 §3 🔴 |
| `expanded` `terminal` | 是否已展开/终止 | 深度/预算/无可行 edit |

🔵 **节点缓存白送**：SkillOpt 的 `sel_cache` 已按 `skill_hash` 缓存验证分（`trainer.py:1420`），不同 edit 产出同一份 skill 时不重复 rollout——这正是 MCTS 的 transposition table。`02` 会把它扩成「存每任务分数」以支持配对与 SH。

---

## 2. 选择 / Selection（纯 bookkeeping，零模型调用）

从 root 出发，按 UCT 下降，直到到达一个**未完全展开**（还有未尝试动作）或**叶**节点：

```
UCT(child) = scal_λ( W(child) / N(child) ) + c · sqrt( ln N(parent) / N(child) )
   其中   scal_λ(v) = v.success − λ · max(0, ĉ − 1),   ĉ = cost / cost_noskill
```

- `scal_λ(W/N)` = **在 selection 时**对二维均值价值标量化（公式 = `mcts_02` §6 的归一化罚分，cost 归一化 ĉ 见 [`mcts_02` §6](mcts_02_value_and_cost.md)）。
- 🔴 **λ 只在这里出现**：backup 累加二维向量、不焼 λ（§5），所以一棵树的前沿提取对 λ 免费、可搜索中自适应 λ。
- `c` = 探索常数（PromptAgent 用 2.5，待调）。
- 未访问子节点 `N=0` 视作 +∞ 优先（强制至少访一次）。
- 🔵 这一步只读统计、不评估、不调模型——和 SkillOpt 的 gate 同理（`gate.py` 纯函数精神）。

---

## 3. 扩展 / Expansion（= 一个 SkillOpt edit step）

在被选中的节点 `s` 上，生成 `b` 个候选子节点（分支因子 `b`），每个子节点 = 在 `s` 上跑**一遍** SkillOpt 的 ②③④⑤：

```
reflect(train_evidence, s)         → RawPatch        reflect.py:472
aggregate(merge)                   → merged edits    aggregate.py:143
select(rank/clip to edit_budget)   → selected edits  clip.py:25
update(apply_patch_with_report)    → candidate skill skill.py:165
```

**怎么从同一个 `s` 长出 `b` 个不同的子节点（关键）：**
- 🔴 reflect 需要 `s` 的 **train-rollout 轨迹**作证据，而非 selection 轨迹。**每个节点缓存 `train_evidence`**（首次展开时跑一批 train rollout，之后复用），否则每次展开都重跑 train rollout，成本爆。
- 多样化手段（拿到 `b` 个不同 edit）：① 变 `random_seed` → reflect 的 minibatch 分组变（`reflect.py:540`）→ 提出不同 edit；② optimizer 采样温度 >0（PromptAgent：生成温度 1.0、评分温度 0）。
- 🟢 **动作空间含双向粒度 move**（[`mcts_06` §2.1](mcts_06_skillorchestra_pareto_borrowings.md)）：除 reflect→aggregate→clip→apply 的**增量累加**外，EXPAND 还应能 `abstract/compress`（合并细规则、抽象成高层指导，往短走）与 `refine/split`（把粗规则拆成带触发条件的子规则，往细走），让树在**粒度轴**上**双向**移动——因为 success 对 detail **非单调**（[`mcts_00` §4](mcts_00_idea_and_contribution.md)），最优 skill 未必最长。由 `search.granularity_moves` 开关（[`mcts_03` §5](mcts_03_code_integration_plan.md)）。
- 🔴 尊重 SkillOpt 红线（doc4 §D/§E）：edit 的 `target` 必须逐字节匹配否则**静默跳过**；用 `get_payload_items` 不要硬编码 `["edits"]`；保护区 `SLOW_UPDATE`/`APPENDIX` 改不动。

**v1 简化决策**：🔵 **关闭 slow-update / meta-skill**。它们是 epoch 语义，在树里没有干净对应，且引入保护区复杂度。v1 树纯靠 step 级 edit 生长；slow/meta 留待 v2（可在「当前最优路径」上周期性跑）。

每个候选子 skill 进入 §4 估值；估出值后才正式挂为子节点。

---

## 4. 模拟 / Simulation（= 价值估计，引擎在 `02`）

经典 MCTS 的 simulation = 随机 rollout 到终局。这里**没有博弈终局**，所以沿用 PromptAgent 的做法：**节点价值 = 该 skill 在 selection 集上的评估**。这是唯一昂贵的一步，绝不全量评估，接口如下（实现见 `02`）：

```
EstimateValue(child, parent, budget) → (success_est, cost_est, tasks_used)   # 02 §3
   ├─ 命中 sel_cache[skill_hash] → 直接复用每任务分数
   ├─ 配对评估 CRN：child 与 parent 在【同一批】selection 任务、同种随机性下评 → 低方差 Δ
   ├─ successive halving：先小预算 n0，有希望者追加，差者早砍
   └─ 返回锚定价值（见 §5 的 telescoping）
```

v1 **不做额外的 edit-rollout 前瞻**：以节点自身价值作为该路径的 return（等价于 PromptAgent 用 score-set 评估当返回）。浅层贪心前瞻留 v2。

---

## 5. 回传 / Backpropagation + 锚定价值 / Anchored value

**锚定（telescoping）价值估计**——解决「配对评估给低方差 Δ，但 UCT 需要可比的绝对值」的矛盾：

- root 做**一次较大预算的全量评估**，得到绝对锚 `v(root) = (success, cost)`。
- 任何其它节点：`v(node) = v(parent) + Δ(node, parent)`，Δ 由配对评估在共享任务子集上测得（`02` §2）。
- 于是所有节点价值都挂在 root 这一个锚上、尺度一致、且每条边只承担一个低方差 Δ。
- 🟡 代价：深节点的价值 = 一串 Δ 的累加，**估计误差沿深度累积**——这是限制树深的另一个理由（§6）。

**回传（Option B：二维向量回传，不焼 λ）**：从被估值的子节点沿路径回到 root，每个祖先：
- `N += 1`；把子节点的**二维价值**累加进二维向量 `W`：`W.success += child.value.success`、`W.cost += child.value.cost`。
- 🔴 **不在 backup 标量化**：`Q` 不落盘，导航时按需 `Q = scal_λ(W/N)`（λ 只在 selection 出现，§2）。这样**一棵树服务所有 λ**——前沿提取与 λ 无关、可在搜索中自适应 λ（向 Pareto-UCT 过渡）。这正是 [`mcts_05` 待改进 #4](mcts_05_consolidated_algorithm_and_tradeoffs.md) 提出、现已落进主算法的那一刀。

🔵 **导航与提取分离**：`Q`（二维均值的标量化）只用于 UCT 选哪里搜；**最终 skill 不从 Q 取**，而从「所有已评估节点的二维**绝对**价值全局账本」里取非支配集（§7）。这干净地化解了 mean/max 之争——均值利于探索，提取看全局最优。

---

## 6. 终止 / Termination

满足任一即停：
- **节点评估预算耗尽**（真正的硬约束，按 selection-rollout 次数或 token 计）；
- 达到最大迭代数；
- **树深上限 `D`**（限 horizon；§5 的误差累积 + Dsel 过拟合 `02` §7 都要求 D 不宜大，建议先 D≈4–6）；
- `patience` 轮无改进早停（参考同类 prompt-MCTS 的 patience=5）。

---

## 7. 前沿提取 / Frontier extraction

- v1 单次运行固定一个 λ → 得一个工作点。**扫描 λ 网格**（多次运行）→ 整条 success–cost 前沿。
- 🔵 **省钱技巧**：即使**单次** λ 运行，也可对「所有已评估节点的 `(success, cost)`」取**非支配集**，直接读出一条前沿（覆盖度可能不如 λ-sweep/Pareto-UCT，但几乎免费）。v1 默认两者都报。
- 🔴 **best_skill 不被污染**（doc4 §J）：导出的每个前沿点必须是某个**通过验证的真实节点**的忠实快照；v1 关了 slow-update 所以无污染风险。
- 最终分数在**独立 test 集**上报（held-out），与 SkillOpt 一致。

---

## 8. 伪代码 / Pseudocode

```
MCTS_SkillOpt(initial_skill s0, λ, budget B, branch b, depth D, c):
    root ← Node(s0)
    root.value ← EstimateValue_full(root)          # 较大预算的锚 (二维), 02 §3
    while budget B not exhausted and not early_stop:
        # 1. SELECTION  (Q 在此处才标量化, λ 只在这里出现)
        node ← root
        while node.expanded and node.children and depth(node) < D:
            node ← argmax_{ch in node.children} UCT(ch, c, λ)
            # UCT(ch,c,λ) = scal_λ(ch.W / ch.N) + c·sqrt(ln N(parent)/N(ch));  ch.N=0 → +∞
        # 2. EXPANSION (= 一个 SkillOpt edit step, 复用 ②③④⑤ + 可选粒度 move §3)
        if depth(node) < D and not node.terminal:
            if node.train_evidence is None:
                node.train_evidence ← rollout(node.skill, split=train)   # 缓存
            for k in range(b):
                edits ← reflect→aggregate→select(node.train_evidence, seed=k or temp>0)
                child_skill ← apply_patch_with_report(node.skill, edits)  # skill.py:165
                child ← Node(child_skill, parent=node, edge_edits=edits)
                # 3. SIMULATION / 价值估计 (02), 二维配对差
                Δ ← EstimateValue_paired(child, node, budget=n0)         # CRN + SH, (Δsucc,Δcost)
                child.value ← node.value + Δ                            # telescoping 二维, §5
                node.children.append(child)
                # 4. BACKPROP  (Option B: 二维向量回传, 不焼 λ, §5)
                backup(child → root): for x in path: x.N += 1; x.W += child.value   # W 是二维向量
            node.expanded ← True
    # 提取: 对二维绝对价值取非支配集 (与 λ 无关; 扫 λ 只发生在标量化挑点时)
    return pareto_nondominated({n.value for n in all_evaluated_nodes})    # §7
```

---

## 9. 与 SkillOpt 阶段的对应表 / Mapping

| MCTS 阶段 | SkillOpt 复用 | 新增代码 |
|---|---|---|
| Selection | — | UCT（新模块 `search/mcts.py`） |
| Expansion | reflect/aggregate/clip/skill（②③④⑤原样） | 节点循环 + 多样化采样 + train_evidence 缓存 |
| Simulation | rollout on `valid_seen` + `compute_score` + `sel_cache` | 配对评估 + SH 分配器（`02`） |
| Backprop | gate 的「比较」语义 | 树回传 + 锚定价值 |
| 终止/提取 | test 评估、`skill_hash` | 预算管理 + 非支配集提取 |

---

## 10. v1 待定旋钮 / Open knobs（交给 `04` 调）

`c`（探索）、`λ` 网格、`b`（分支）、`n0`（初始评估任务数）、SH 衰减因子 `η`、深度 `D`、总预算 `B`、`patience`、评估温度（建议低温/定种子以支持 CRN，见 `02` §2）。

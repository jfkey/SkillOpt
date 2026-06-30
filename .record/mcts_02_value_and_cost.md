# MCTS-SkillOpt 设计文档 2 — 价值与成本估计（Value & Cost Estimation）

> **目的 / Purpose**：这是整个项目的**承重墙**。`01` 的 simulation 阶段只给了接口，引擎全在这里：**cost 到底怎么度量（及那个 confound 怎么去）、配对评估（CRN）怎么降方差、successive halving 怎么控成本、价值怎么锚定、怎么缓存复用、λ 怎么归一化、Dsel 过拟合怎么办**。
>
> 配套：被 [`mcts_01`](mcts_01_mcts_algorithm_spec.md) §4/§5 调用；概念见 [`mcts_00`](mcts_00_idea_and_contribution.md)；落地见 [`mcts_03`](mcts_03_code_integration_plan.md)；实验/算力对齐见 [`mcts_04`](mcts_04_experiment_plan.md)。

---

## 0. 一句话 / One-liner

> 节点价值是二维向量 `(success, cost)`。**cost = skill 在部署期诱导的总开销**（常驻 skill token + 诱导的多余步数/工具/生成 token），这是 SkillOpt 完全没算的盲点。估值的全部工程目标是：在**贵+噪+Dsel 小**的条件下，用**配对评估降方差 + successive halving 控成本 + 缓存复用**，把每个节点的价值估得**又便宜又稳**。

---

## 1. Cost 是什么 / Defining skill-induced cost

skill 在部署期诱导两类开销：

| 来源 | 含义 | 怎么测 |
|---|---|---|
| **常驻 skill token** | skill 被 prepend 到**每个任务** context，固定输入 token 成本 | `len(skill)` token（即时可得） |
| **诱导 rollout 成本** | skill 改变 agent 行为 → 步数/工具调用/生成 token 变化 | 从轨迹统计：avg turns、avg tool calls、avg total tokens |

🔵 SkillOpt 现状：只统计**训练**成本（Table 6），从不计部署期的 skill-token + 诱导步数。我们把它补成一等公民。

**主指标（headline cost）建议：每任务总 token** = 输入(含 prepend 的 skill) + 全部 completion + 工具往返 token。理由：harness 间可比、客观、同时吃下「skill 自身体积」和「诱导步数」两块。
- **每 benchmark 备选**：ALFWorld 用 **steps/episode** 更自然；Codex/Claude Code 类用 **tool-call 数**。
- 🟡 **latency 不做主指标**：harness 相关、机器相关、方差大；只作旁证。

数据流：SkillOpt 已按 stage 计 token（`step_record.json` 的 `tokens.<stage>`），ALFWorld 已有步数。我们把**每任务 cost** 写进 `RolloutResult.extras`（见 `03`），与 `hard/soft` 并列。

---

## 2. 那个 confound：买来的成本 vs 浪费的成本

更谨慎的 skill **既更贵也更成功**。naive 罚 cost 会连有用步骤一起砍。处理：

- 🔵 **Pareto/前沿框架本身就化解 confound**：贵但更成功的 skill **不被支配**，前沿自动保留它。**你永远不必判断「这点成本是否浪费」——直接报整条 tradeoff 曲线**。这正是坚持多目标/前沿（而非单个罚分）的根本理由。
- 罚分只活在标量化的 λ 里：`success − λ·ĉ`，**λ 就是「愿意拿多少 success 换多少 cost」**；扫 λ 描出的，恰恰就是 tradeoff 曲线本身。
- 🟡 **效率型备选指标**（旁证用）：`tokens / success`（成本/成功率）或「成功轨迹内的平均 cost」，隔离纯效率；但 success→0 时不稳，只作补充。

---

## 3. 估值主流程 / EstimateValue（CRN + SH）

`01` §4 调用的 `EstimateValue_paired(child, parent, budget)`：

```
EstimateValue_paired(child, parent, n0, η, cap):
    if skill_hash(child) in sel_cache:                      # 命中即复用每任务分数
        return aggregate(sel_cache[child])
    S ← parent.eval_tasks  (共享任务子集, 固定顺序/种子)      # CRN 锚
    tasks ← first n0 of S
    while True:
        run child on `tasks`  (低温/定种子, §2 CRN 前提)      # rollout valid_seen
        Δ_succ ← mean_{t∈tasks}[ succ_child(t) − succ_parent(t) ]   # 配对差, 低方差
        Δ_cost ← mean_{t∈tasks}[ cost_child(t) − cost_parent(t) ]
        record per-task scores → sel_cache[child]            # 增量缓存
        if 该 child 在兄弟中排名落后(SH) or budget 用尽 or tasks==S:
            break
        tasks ← grow(tasks, ×η)                              # successive halving: 追加预算
    return (Δ_succ, Δ_cost)        # 由 01 §5 锚定成绝对值 child.value = parent.value + Δ
```

---

## 4. 配对评估 / CRN（降方差，最便宜的大赢）

- **同任务、同随机性**比较 `child` 与 `parent`：在**同一批 selection 任务**上、尽量**同种解码随机性**（低温/定种子）评估两者，则 **paired Δ 的方差远低于各自绝对值的方差**（消掉了任务间方差这个主导项）。
- 维护一个**固定顺序、固定种子**的 selection 任务列表；每节点评估它的一个**前缀子集**；比较时只用双方共享的子集。
- 🟡 CRN 要求评估**可复现**：建议 gate/估值用**低温或定种子**的 target 解码（PromptAgent 评分用温度 0）。agentic harness 不总能温度 0，但定种子/降温是降方差的关键杠杆。
- 🔵 **直接复用 SkillOpt 的 slow-update 套路**：它已经在做「同样采样任务下 prev/curr 两版 skill 的纵向对比」（`slow_update.py:309`、`build_comparison_pairs:159`）——我们把这个「成对比较同一批任务」的能力推广到树里**所有**父子/兄弟比较。

---

## 5. Successive Halving / Hyperband（控成本）

- 设定：一个节点展开出 `b` 个候选子，全量评估每个都贵。SH：先给所有候选**小预算 n0**，按临时价值排序，保留 top 1/η，预算 ×η，重复，直到剩 1 个或到 cap。η 取 2–3。
- 两个层级都用：(a) 节点展开时在 `b` 个子里选谁值得全评；(b) 全局在「值得评的叶节点」间分配评估预算。
- 🔴 **SH + CRN 必须组合**：小 n0 + 二值噪声 reward 上，SH 单独排序会误判。所以**排序依据用「相对共享 parent 的配对 Δ」**（§4），而非各自绝对分——这样早期排序才可靠。
- 🔵 与 UCT 正交：**SH 决定「每个节点评多少」，UCT 决定「下一个展开谁」**，互补。（也可用 UCB 直接当评估分配器，作为 v2 备选。）

---

## 6. λ 与 cost 归一化 / Normalization

success∈[0,1]，cost 是几百~几千 token，**不归一化 λ 没有意义**。
- ĉ = `cost / cost_noskill`（相对无 skill 基线：1.0=基线，<1 更省）。可读、稳，**推荐**。
- 标量化用 `success − λ·max(0, ĉ−1)`：**只罚超过基线的部分**，不惩罚比基线更省的 skill（避免逼 skill 一味缩短）。
- 🟡 树生长时 cost 分布会漂；ĉ 的参考(`cost_noskill`)**固定为无 skill 基线**、不随树更新，保证尺度稳定。
- λ 网格建议先 `{0, 0.1, 0.3, 1.0}`：λ=0 即纯 success（退化成 PromptAgent 式基线，正好做对照）。

---

## 7. Dsel 过拟合（搜索越多越危险）

🔴 这是 MCTS 相对贪心 SkillOpt 被**放大**的风险：MCTS 对**同一个小 Dsel** 评估的候选远多于贪心 → 多重比较 → 易选中只拟合 Dsel 噪声的 skill。缓解 + 把它变成 finding：
- 最终分**只在独立 test 报**（SkillOpt 已如此）。
- **限制总候选评估数相对 |Dsel|**（预算 §`01`-6 与 |Dsel| 挂钩）。
- **嵌套验证**：再留一个小 selection-2 做最终节点选择，避免在同一集上既搜又选。
- **量化它**：画 selection-best 与 test 的 gap 随「累计评估候选数」的曲线（SkillOpt Figure 3 的同款）——**这条曲线本身就是论文的一个 finding**：「在给定 Dsel 下，搜索预算到哪一点开始过拟合」。

---

## 8. 未来：廉价 value 模型（v2/v3，闭环到无 verifier）

- 训个小回归器，从 `(skill embedding, edit 特征)` 预测 `(success, cost)`——更好是预测 **edit 的 Δ**（更低方差、更可学）。当 MCTS 的 value prior / SH 预筛器，**只在确认有希望的节点才跑真评估**。
- 🔵 **这就是前几轮说的 verifier-free gate，同一个东西**——它同时(i)让搜索便宜、(ii)给开放域提供无真值的评估。
- 冷启动：先用少量真评估 warm-start，在线训练逐步摊销。
- **过程信号**（更稠的代理）：到达正确子目标、冗余步数、格式合规——比终局二值又稠又便宜，可做预筛特征。

---

## 9. 自检 / Checklist

- [ ] cost 三块都进了 `RolloutResult.extras`（skill token / 步数 / 总 token）？
- [ ] 估值走 CRN（共享任务子集、低温/定种子）而非各自绝对分？
- [ ] SH 排序用「相对 parent 的配对 Δ」而非绝对分？
- [ ] 价值锚定：root 全量评估，其余 = parent + Δ（`01` §5）？
- [ ] λ 用归一化 ĉ，且 λ=0 退化成纯 success 基线？
- [ ] `sel_cache` 扩成「存每任务分数」以支持配对/SH 增量复用？
- [ ] selection-best vs test gap 曲线在记录，用于 §7 过拟合分析？

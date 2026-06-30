# MCTS-SkillOpt 设计文档 4 — 实验方案（Experiment Plan）

> **目的 / Purpose**：定义怎么**证明**这套方法值得——基线、指标、消融、**算力对齐**、过拟合曲线、benchmark 选择与上线前检查。头号约束不是算法而是**评估开销**（节点价值 = 一遍 agentic rollout），所以实验设计先回答「**等算力下打不打得过贪心 + 随机重启**」和「**增益是否来自 cost 维而非 MCTS 本身**」。
>
> 配套：概念/成功标准见 [`mcts_00` §8](mcts_00_idea_and_contribution.md)；估值/过拟合见 [`mcts_02` §7](mcts_02_value_and_cost.md)；优缺点/待改进见 [`mcts_05` §3–5](mcts_05_consolidated_algorithm_and_tradeoffs.md)；消融模板见 [`mcts_06` §3](mcts_06_skillorchestra_pareto_borrowings.md)；落地/产物见 [`mcts_03`](mcts_03_code_integration_plan.md)。
>
> ⚠️ 已对齐二维价值 + Pareto 前沿输出。开销模型源自 [`_archive/`](_archive/) 旧稿但已翻译到二维框架，勿引用旧稿。

---

## 0. 一句话 / One-liner

> 押注叙事是 **cost-aware Pareto 前沿**，不是 "MCTS 在 success 上打败贪心"（[`mcts_05` §3.3](mcts_05_consolidated_algorithm_and_tradeoffs.md) 的诚实结论）。所以实验三件套：① **算力对齐的 make-or-break 基线**（等 token 预算、非等迭代数）；② **双列 (Acc, Cost) 消融**证明增益来自 cost 维 + 双向粒度搜索；③ **过拟合特性曲线**把「搜索越多越过拟合」从风险变 finding。最终分**只在独立 D_test 报**。

---

## 1. 算力对齐 / Compute alignment（头号约束）

PromptAgent 的 reward = 对文本算 accuracy（一次 LLM 前向，便宜、确定、温度 0）。**SkillOpt 的 reward = 冻结 agent 在 D_sel 上跑完整 rollout**（多轮、带工具、十几步），贵几个数量级。线性循环每 step 只付一次（⑥ EVALUATE）；MCTS 把它乘上一个大系数。

**评估次数量级（最坏上界，无缓存）**：每个新子节点都要一次 D_sel rollout，

```
reward 调用 ≈ I × W × D        # I=迭代数, W=branch(expand_width), D=depth_limit
```

| 配置 | I | W | D | reward 调用上界 | 相对 linear(≈10 step) |
|---|---|---|---|---|---|
| 论文 Standard 风格 | 12 | 3 | 8 | ~288 | ~29× |
| **建议起步** | 4 | 2 | 3 | ~24 | ~2.4× |
| 中等 | 8 | 2 | 4 | ~64 | ~6.4× |

> 🔴 注意 transition 自身也含 ① train rollout + ② optimizer reflect，**同样乘 I×W×D**——总 token / 总钱 / 总时间按同系数膨胀，不只是 reward。CRN+SH（[`mcts_02` §3–5](mcts_02_value_and_cost.md)）只压乘数、不改量级。
> 🔴 **预算与 |D_sel| 挂钩**：总候选评估数必须相对 |D_sel| 受限，否则过拟合（§5）。`search.budget`（[`mcts_03` §5](mcts_03_code_integration_plan.md)）是硬上界。

**跑前先量一次 linear baseline 的三个数**（代入上表估钱/时间）：D_sel 大小 `sel_env_num`、每 item 平均交互步数 × 每步 token、optimizer 每次 reflect 的 token。

---

## 2. 指标 / Metrics

| 轴 | 定义 | 来源 |
|---|---|---|
| **success** | `compute_score`（hard/soft/mixed，`gate.py:46`） | SkillOpt 已有 |
| **cost（headline）** | 每任务总 token = 输入(含 prepend skill) + completion + 工具往返 | [`mcts_02` §1](mcts_02_value_and_cost.md) |
| cost（备选） | ALFWorld 用 steps/episode；Codex/Claude Code 用 tool-call 数 | per-benchmark |

- 🔴 **所有汇报指标只在独立 `D_test`（held-out）上算**，与 SkillOpt 一致；搜索期的配对 Δ / train 分**不得**进最终表（[`mcts_06` §2.3](mcts_06_skillorchestra_pareto_borrowings.md)：Δ 仅树内导航）。
- 主结果图：**success–cost 平面**，画各方法的工作点 / 前沿，看是否 **Pareto-dominate** SkillOpt（等成本更高 success，或等 success 更低成本——[`mcts_00` §8](mcts_00_idea_and_contribution.md)）。

---

## 3. 基线 / Baselines（含退化检验）

🔴 **make-or-break（[`mcts_05` 待改进 #1](mcts_05_consolidated_algorithm_and_tradeoffs.md)）**：**等算力下对比「贪心 + 随机重启」**（同 token 预算，**不是**同迭代数）。鉴于 edit economy（增益只来自 1–4 次被接受编辑），**宽而浅 + 重启**很可能优于一棵深树；若 MCTS 打不过 random-restart greedy，树的机器不值这个钱。这是最关键的对照。

**退化检验（必须做，证明各部件的贡献）**：

| 设置 | 旋钮 | 等价于 | 用途 |
|---|---|---|---|
| 退回贪心链 | `b=1, c=0, D=∞` | 原 SkillOpt | 正确性闸（应复现 linear baseline 分数） |
| success-only MCTS | `λ=0, b>1, c>0, D` 足够 | PromptAgent 思路搬到 skill | 证明增益来自 **cost 维 + 多目标**，而非 MCTS 本身 |
| 等算力 random-restart greedy | 同 token 预算的多次贪心重启 | — | **make-or-break 对照** |

---

## 4. 消融 / Ablations（双列 Acc, Cost，抄 SkillOrchestra Table 2）

每个组件 ablate，**同时报 test success 和 cost**（[`mcts_06` §3](mcts_06_skillorchestra_pareto_borrowings.md)）：

| 设置 | 回溯(UCT) | 降粒度 move | CRN+SH | Pareto 选点 | **test success** | **cost** |
|---|---|---|---|---|---|---|
| 退化(b=1,c=0) = 贪心 SkillOpt | ✗ | ✗ | — | ✗ | — | — |
| success-only MCTS = PromptAgent 搬运 | ✓ | ✗ | ✓ | ✗(只取 max success) | — | — |
| + 降粒度 move | ✓ | ✓ | ✓ | ✗ | — | — |
| Full（cost-aware Pareto） | ✓ | ✓ | ✓ | ✓ | — | — |

- 🟢 **关键消融**：去掉「降粒度 move」那一行 ≈ SkillOrchestra 的 `No FG Skills`，证明**双向粒度搜索同时改善 success 和 cost**（复刻 85.0/\$9.3 vs 80.4/\$15.1 的故事）。降粒度 move 定义见 [`mcts_01` §3](mcts_01_mcts_algorithm_spec.md) / [`mcts_06` §2.1](mcts_06_skillorchestra_pareto_borrowings.md)。
- 🟢 **CRN+SH 消融**：关掉配对评估/SH，看估值方差与成本（证明它们让昂贵 reward 上的 MCTS 可行——[`mcts_00` §8](mcts_00_idea_and_contribution.md) 承诺的交付）。
- 🟢 **per-target 粒度甜点实验**：固定 benchmark，换 1 强 1 弱 target，各画 Pareto 前沿——若甜点粒度随容量右移，是一条干净 finding，与 SkillOrchestra Fig 6 互证（[`mcts_06` §3](mcts_06_skillorchestra_pareto_borrowings.md)）。

---

## 5. 过拟合特性曲线 / Overfitting study（把放大的风险变 finding）

🔴 MCTS 对**同一个小 D_sel** 评估的候选远多于贪心 → 多重比较 → 易选中只拟合 D_sel 噪声的 skill（[`mcts_02` §7](mcts_02_value_and_cost.md)）。

- **量化它**：画 **selection-best 与 test 的 gap** 随「累计评估候选数」的曲线（SkillOpt Figure 3 同款）——**这条曲线本身就是论文的一个 finding**：「给定 D_sel，搜索预算到哪一点开始过拟合」。
- **嵌套验证**：再留一个小 `selection-2` 做最终节点选择，避免在同一集上既搜又选。
- 🟡 论文消融提示：`iteration_num` 太大反而过拟合（PromptAgent 16 不如 12）——**不是越大越好**，曲线会自证拐点。

---

## 6. 开销缓解策略 / Cost mitigation（按性价比）

1. **复用 `sel_cache`（几乎免费，必做）**：`estimate_value` 第一行查 `sel_cache[skill_hash]`，命中即返回每任务分数（[`mcts_03` §3](mcts_03_code_integration_plan.md)）。
2. **搜索用子集、最终用全量**：搜索期 reward 在 D_sel 小子集上算（SH 的 n0 前缀）；搜索结束只对前沿 top-k 做全量 + D_test 评估。
3. **小配置起步**：先 `budget≈24, b=2, D=3`，证明 MCTS ≥ linear 再放大 I。
4. **挑最便宜 benchmark 调通**（§8）。
5. **限制 W 优先于 D**：开销对 W、D 都是乘性；"窄而深" 更省且更贴 PromptAgent 轨迹式优化。
6. **并行化扩展**：一个节点的 b 个子彼此独立，reward rollout 可并行（受 API 限速约束）。

---

## 7. 非开销实验风险 / Non-cost risks（先量再跑）

| 风险 | 现象 | 对策 |
|---|---|---|
| 🔴 reward 噪声/非确定 | agent rollout 随机 → Q 带噪、`sel_cache` 按 hash 缓存被"骗" | **跑前先测「同一 skill 重复评估的方差」**；固定 selection seed 或多 seed 平均；缓存 key 带 seed/子集 id（[`mcts_02` §4](mcts_02_value_and_cost.md)、[`mcts_03` §6](mcts_03_code_integration_plan.md)） |
| 🔴 线性记忆串味 | step_buffer/meta/slow/scheduler 假设线性历史 | v1 禁用 meta/slow；step_buffer 改 path-scoped；edit_budget 固定（[`mcts_03` §6](mcts_03_code_integration_plan.md)） |
| 🟡 阈值定标 | 早停阈值绝对值不适配 hard/soft/mixed 尺度 | 阈值一律相对化（用 root reward 初始化）；先关早停跑一轮看分布再调 |
| 🟡 CRN 假设脆 | ALFWorld 等随机环境无法温度 0 → SH 误判 | 降温/定种子尽力而为；CRN 失效则回退多 seed 平均（成本换稳） |

---

## 8. Benchmark 选择与顺序 / Benchmark roadmap

1. **SearchQA**（最便宜，单轮 `max_turns:1`）：先端到端跑通整条 MCTS 链路、量开销基线、做正确性闸。
2. **程序性 benchmark**（SpreadsheetBench / OfficeQA）：skill 诱导的步数/格式开销最大，**cost 维最有发挥空间**——主结果与降粒度消融押在这里（[`mcts_00` §8](mcts_00_idea_and_contribution.md)）。
3. **ALFWorld**（具身多步，最重）：验证长交互、随机环境下 CRN 退化与缓解。

> 至少在 1–2 个程序性 benchmark 上显著 Pareto-dominate SkillOpt，就足以支撑主张。

---

## 9. 上线前检查清单 / Pre-launch checklist

- [ ] 测过 linear baseline 开销基线（reward 次数 / token / wall time / 单次 selection rollout 成本）。
- [ ] 测过「同一 skill 重复评估的方差」，定下 seed 策略。
- [ ] `estimate_value` 第一行查 `sel_cache`；缓存 key 含 seed / 子集 id；`sel_cache` 已扩成每任务分数。
- [ ] 搜索用子集，最终对前沿 top-k 全量 + D_test 评估。
- [ ] v1 禁用 meta/slow；step_buffer path-scoped；edit_budget 固定；子节点按 hash 去重、no-op 标 terminal。
- [ ] 产物：`frontier.json` + `perf`/`efficiency` 各导一个 `best_skill.md` + `tree.json`；不破坏 `history.json` / run-dir。
- [ ] 退化检验（`b=1,c=0` 复现 linear；`λ=0` = success-only MCTS）+ 等算力 random-restart greedy 对照就位。
- [ ] selection-best vs test gap 曲线在记录（§5）。
- [ ] 先 SearchQA 跑通，小配置 `budget≈24,b=2,D=3` 起步，证明 MCTS ≥ linear 再放大。

---

## 10. 文档地图 / Doc map

| 文档 | 层 | 回答什么 |
|---|---|---|
| [`mcts_00`](mcts_00_idea_and_contribution.md) | 概念 | 做什么、为什么、定位、范围 |
| [`mcts_01`](mcts_01_mcts_algorithm_spec.md) | 算法 | 四阶段、树结构、伪代码 |
| [`mcts_02`](mcts_02_value_and_cost.md) | 承重墙 | value/cost 估计、降方差/控成本 |
| [`mcts_03`](mcts_03_code_integration_plan.md) | 工程 | 逐文件改 SkillOpt、红线、产物兼容 |
| `mcts_04`（本文） | 评测 | 基线、指标、消融、算力对齐、过拟合曲线 |
| [`mcts_05`](mcts_05_consolidated_algorithm_and_tradeoffs.md) | 汇总 | 论文体 Algorithm 1 + 优缺点 + 待改进 |
| [`mcts_06`](mcts_06_skillorchestra_pareto_borrowings.md) | 横向借鉴 | SkillOrchestra 的 Pareto 净增量与指导 |

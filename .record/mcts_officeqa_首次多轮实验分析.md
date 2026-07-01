# MCTS-SkillOpt × OfficeQA 首次多轮实验分析（成本感知 Pareto 的关键证据）

- **日期**：2026-06-30
- **分支**：`feat/mcts-pareto-search`
- **运行目录**：`outputs/officeqa_mcts_20260630_195530/`
- **配置**：`configs/officeqa/mcts_compare.yaml`（`lambda_cost=0.3`、`sel_eval_num=12`、`test_eval_num=16`、`batch_size=8`、`budget=4`、`branch=2`、`depth_limit=2`、`n0=4`、`value_mode=paired`）
- **模式**：offline + local-tools（多轮，`max_tool_turns=24`，本地文档 glob/grep/read 工具）
- **模型**：optimizer = target = **gpt-5.5**，`reasoning_effort=medium`
- **对照**：SearchQA（单轮）见 [[mcts_searchqa_实验分析与officeqa迁移]]

---

## 0. 一句话结论

> **OfficeQA 给出了 SearchQA 给不出的东西：一个真实的成本感知 Pareto 收益。** 学到的技能在**精度持平（0.667）**的前提下，把平均工具轮数从 **8.33 → 5.00（−40%）**、每题 token 成本从 **26596 → 17868（−33%）**。技能文本更长了（887→4367B），但因为它教会 agent"先用 oracle 证据、定向检索、少读多验"，反而**更省**。于是这条技能在 (success, cost) 上**直接支配空技能基线**——这正是 cost-aware MCTS 相对 linear 的核心卖点，且**只在多轮任务上才会出现**。

---

## 1. 让 OfficeQA 跑起来做了什么（迁移工作）

| 步骤 | 内容 |
|---|---|
| 成本通道 | `officeqa/rollout.py` 全部 5 条路径接入 `cost_total_tokens`（per-task `cost_sink` 累加多轮 prompt+completion；exec 走 char/4 兜底）。`n_turns` 也作为备选成本轴。 |
| 数据 materialize | HF gated。本机 huggingface_hub 走代理报 FileMetadataError，但 `curl` 走同一代理可用 → 用 token 直接下载 `officeqa_full.csv`（246 行），join `officeqa_id_split` → `data/officeqa_split/{train,val,test}`（50/24/172，0 缺失）。 |
| 文档语料 | HF repo 的 `treasury_bulletins_parsed/jsons/*.json`（`_locate_parsed_json` 把 `<stem>.txt`→`<stem>.json`）。下载 285 个引用到的 JSON，并**渲染出 285 个 `.txt`** 供多轮 local 工具检索（~382MB，全部 gitignore）。 |
| **修 bug** | local-tools 循环假设 OpenAI-SDK pydantic tool_calls（`tc.model_dump()`/`tc.function.name`/`tc.id`），但 **gpt-5.5 走 Responses API，tool_calls 是普通 dict** → 第 1 轮即崩 `'dict' object has no attribute 'model_dump'`。新增 `_tool_call_fields(tc)`（dict/pydantic 通吃，返回可回灌 `_messages_to_responses_input` 的 dict）。 |
| 验证 | 1 条 live 冒烟：UID0003 hard=1（答对）、n_turns=6、cost 13824（单轮 3922 的 3.5×）。 |

> ⚠️ `run_batch` 是 resume-aware（跳过已有预测的 item）；重跑冒烟要换新 out_root，否则拿到旧缓存结果。

---

## 2. 结果

### 2.1 总览（`summary.json`）

| 指标 | 值 |
|---|---|
| 节点数 / 评估候选 / value calls | 5 / 5 / 4 |
| sel task-rollouts | 52 |
| 墙钟 | 1545s（~26 min） |
| tokens 总 / target rollout / optimizer | 3.06M / 2.52M / 0.54M |
| 前沿点数 | **1**（node 3 支配全部） |

### 2.2 逐节点（选择集 sel，n=12；node4 被 SH 早剪只评了 4 条）

| node | depth | skill_len | **avg_turns** | **avg_cost** | sel_acc |
|---|---|---|---|---|---|
| 0（空技能 root） | 0 | 887 | **8.33** | 26596 | 0.667 |
| 1 | 1 | 2844 | 5.92 | 22610 | 0.667 |
| 2 | 1 | 2567 | 5.17 | 18238 | 0.583 |
| **3（★ perf 部署点）** | 2 | 4367 | **5.00** | **17868** | **0.667** |
| 4 | 2 | 4099 | 6.50 | 26214 | 0.750 *(n=4)* |

- 前沿 perf 点 node 3：sel 0.667 @ 17868 → 测试集（n=16）**test 0.625 @ 18054**。
- node 3 **支配 root**：精度相同（0.667），成本低 33% → 前沿塌成单点（不是 bug，是"免费午餐"型支配）。

---

## 3. 核心机制：技能用"减少工具轮数"来降成本

```
空技能  8.33 轮 / 26596 tok   ── 大量探索式 search
node1   5.92 轮 / 22610 tok
node3   5.00 轮 / 17868 tok   ── 精度不变，轮数 −40%，成本 −33%
```

node 3 学到的是一套 **"检索纪律"**（技能开头几条直接驱动轮数下降）：
- *"有 oracle 解析页/来源提示时，先把它当主证据；local 工具主要用来核对、消歧、定位精确 span。"*
- *"先收敛到最可能的单个文件，再读长段落。"*
- *"用点名实体/期间/度量/表概念的定向检索词。"*
- *"命中后只读小范围邻域并核对年份/口径/单位。"*

空技能的 agent 反复盲搜（8.33 轮），有纪律的 agent 直奔 oracle 证据 + 定向检索 + 少读多验（5.0 轮）。外加表格算术/格式规则把精度**稳在** 0.667（没有因为少读而掉分）。

---

## 4. 与 SearchQA 的对照（为什么必须换多轮任务）

| 维度 | SearchQA（单轮） | OfficeQA（多轮） |
|---|---|---|
| 空技能起点 | 0.87（太高，没空间） | **0.625–0.667（有空间）** |
| 成本轴含义 | = 技能长度（单调，技能=纯 prompt 开销） | = 工具轮数×上下文（技能能**减少**轮数） |
| 成本随技能 | 单调**上涨**（越学越贵） | **下降**（学会少绕路） |
| Pareto | 平凡（lambda 无所谓） | **真实 2D**：好技能支配空基线（同精度，省 33%） |
| MCTS 价值 | ≈ linear，无增量 | 体现 cost-aware 价值（linear 无成本轴/无此发现） |

**结论：SearchQA 的成本轴是假的（只反映技能长度），OfficeQA 的成本轴是真的（反映部署期推理开销）。** 成本感知 MCTS-Pareto 的意义只有在多轮任务上才显现。

---

## 5. 注意事项 / 局限

1. **精度没涨（停在 0.667）**。三个原因：(a) `sel=12` 太小，1 题=0.083，天花板量化噪声大（0.667=8/12）；(b) `budget=4`/`depth=2` 太浅，只 2 次 expansion；(c) **λ=0.3 把高精度但更贵的 node4（sel 0.75）在 SH 阶段剪掉了**——它 6.5 轮、成本高，scalarized 值（success−λ·cost）输给了便宜的 node3。即成本惩罚可能误伤高精度候选。
2. **node4 的 0.75 只在 4 条上**（SH 早剪未全评），是"可能更高精度"的信号，不是定论。
3. 前沿单点 = 没有"更贵更准"的点存在（精度没爬起来），一旦精度能涨，前沿会展开成多点。

---

## 6. 下一步

1. **加一个 λ=0 对照臂**（同 budget/sel）：看不罚成本时精度能否爬到 0.75（node4 暗示可能）——区分"精度天花板"是任务本身还是 λ 误剪。
2. **扩大 sel→24（用满 val）**：消除 0.667 量化噪声，给搜索真实可分的精度梯度。
3. **加深搜索**：budget 8–12、depth 3–4，配多轮成本预算（每条 rollout ~18K tok，注意总量）。
4. **λ-sweep**：0 / 0.15 / 0.3 / 0.6，画出精度-成本前沿，正面展示"同精度更省 / 更省略降精度"的权衡曲线——这是相对 linear 单点的决定性对比图。
5. 报告口径：除 `cost_total_tokens` 外，**把 `avg_turns` 作为主成本轴出图**（对多轮更可解释："这条技能把平均检索轮数从 8.3 降到 5.0"）。

---

## 7. 产物

| 文件 | 内容 |
|---|---|
| `summary.json` / `mcts/tree.json` / `mcts/frontier.json` | 总览 / 5 节点+2 iter / 单点前沿 |
| `mcts/nodes/000X/skill.md` | 各候选技能（node3 = 部署的"检索纪律"技能） |
| `mcts/sel_eval/<hash>/results.jsonl` | 每个技能在 12 条 sel 上的逐条 n_turns/cost/hard（轮数证据来源） |
| `best_skill.md` | = node3（perf 部署点） |

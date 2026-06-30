# MCTS-SkillOpt × SearchQA 实验报告（成本感知 Pareto 搜索 / smoke test）

- **运行目录**：`outputs/mcts_smoke/`
- **日期**：2026-06-30
- **分支**：`main`（`skillopt/search/` 新增模块，Milestone 1+2：端到端骨架 + CRN/SH）
- **模型**：optimizer = target = **gpt-5.5**；backend `azure_openai`（`openai_compatible` / Edge Proxy 网关，Responses API）；`reasoning_effort = medium`；M、O 全程冻结
- **一句话结论**：在真实 gpt-5.5 上跑通了**整条 MCTS 成本感知 Pareto 链路**。从空技能出发，2 轮迭代、评估 5 个技能节点，产出一条 **success–cost 二维前沿**：node0（空技能）测试 EM **0.90 @ 1285 tok/题** vs node1（学到的 2KB 技能）测试 EM **1.00 @ 1617 tok/题**。全程 **27.2 万 tokens、~4.5 分钟、108 次 target rollout**。CRN+successive halving 把选择集评测从 60 次 rollout 压到 **44 次（省 27%）**。部署期零额外模型调用不变。

> 这是**小规模 smoke test**，目的是验证链路正确性与刻画 MCTS 的开销特性，**不是**主结果。测试集仅 20 题，绝对分数有噪声；结论看**方向**与**开销结构**，不看小数点。

---

## 1. 实验脚本（可复现）

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate skillopt
cd /data3/liujunfeng/code/SkillOpt
set -a; source ./.env; set +a          # ⚠️ 必须 source，代码不会自动 load_dotenv，否则 0 调用全 0 分

python scripts/train.py \
    --config configs/searchqa/mcts_smoke.yaml \
    --optimizer_model gpt-5.5 --target_model gpt-5.5 \
    --out_root outputs/mcts_smoke
```

- `search.algo: mcts` 触发 `scripts/train.py` 分发到 `skillopt/search/runner.py`（线性 trainer 完全不受影响）。
- 与线性 baseline 的关键差异：**MCTS 不调 `_resolve_train_size`**，直接 `build_train_env(batch_size)` 取一个固定训练批 + `build_eval_env(sel_eval_num, "valid_seen")` 取固定选择子集，无需构造派生 split。

---

## 2. 关键超参（`configs/searchqa/mcts_smoke.yaml`）

| 类别 | 参数 | 值 |
|---|---|---|
| 搜索 | algo / value_mode | mcts / **paired**（CRN Δ + successive halving） |
| 搜索 | budget（候选节点评估上限） | **4** |
| 搜索 | branch b / depth_limit D | **2 / 2** |
| 搜索 | c_explore / lambda_cost / patience | 1.5 / **0.0**（纯 success 导航）/ 3 |
| 搜索 | sel_eval_num \|S\|（CRN 锚任务数） | **12** |
| 搜索 | n0 / sh_eta（SH 初始任务 / 衰减因子） | **4 / 2** |
| 搜索 | test_eval_num（前沿点留出测试） | **20** |
| 训练 | batch_size（每次展开的 stage① 训练 rollout） | 12 |
| 梯度 | minibatch_size / merge_batch_size | 6 / 6 |
| 优化器 | edit_budget / skill_update_mode | 4 / `patch` |
| 优化器 | slow_update / meta_skill | **关 / 关**（v1 设计：树里无 epoch 语义） |
| 门控 | gate_metric | `hard`（严格 EM；此处仅作 success 维打分） |
| 模型 | reasoning / max_completion_tokens / max_turns | medium / 16384 / 1 |

> 与 baseline（train40/sel40/test150）相比，本次刻意更小：**sel 12 / test 20 / 2 次展开**。

---

## 3. 实验结果

### 3.1 success–cost Pareto 前沿（部署口径）

| 节点 | 技能 | 验证集(sel,12) EM | **测试集(20) EM** | **测试成本(tok/题)** | 是否在前沿 |
|---|---|---|---|---|---|
| **node0** 根 S₀（空，104B） | baseline | 0.667 | **0.900** | **1285** | ✅（最省点） |
| **node1**（学到的，2055B） | iter1 | 0.833 | **1.000** | **1617** | ✅（最准点） |

- **两点互不支配**：node1 准确率更高(+0.10)，node0 成本更省(−26% token)。原版 SkillOpt 只看准确率会无脑选 node1，**看不到** node0 这个"省 25% token 只低 0.1 分"的工作点。
- 产物：`best_skill.md` = node1（perf prior）；`mcts/best_skill_efficiency.md`（efficiency prior，本次因只有 2 点且分差大，也收敛到 node1）。
- **cost 轴是本方法的新增贡献**：原版只统计训练成本，从不计 skill 被 prepend 到每题 context 的**部署 token**。这里它是一等公民，且在**留出测试集**上测得。

### 3.2 搜索树全貌（5 节点 / 2 迭代，`mcts/tree.json`）

| id | depth | parent | sel succ | sel cost | skill_len | **sel 评测覆盖** | 角色 | 前沿? |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | — | 0.667 | 1258 | 104 | **12** | 根锚点（全量评估） | ✅ |
| 1 | 1 | 0 | 0.833 | 1675 | 2055 | **12** | iter1 SH **胜者**（晋级全量） | ✅ |
| 2 | 1 | 0 | 0.667 | 1658 | 2152 | **4** | iter1 SH **败者**（n0=4 处剪枝） | ✗ 被 node0 支配 |
| 3 | 2 | 1 | 0.833 | 1892 | 3189 | **12** | iter2 SH 胜者 | ✗ 被 node1 支配 |
| 4 | 2 | 1 | 0.833 | 1885 | 3097 | **4** | iter2 SH 败者（剪枝） | ✗ 被 node1 支配 |

**逐迭代轨迹（`iter_log`）**：

| iter | 展开节点 | 新子节点 | 评估累计 | 前沿大小 | 前沿增长? | no_improve |
|---|---|---|---|---|---|---|
| 1 | node0(根) | node1, node2 | 2 | 2 | ✅ | 0 |
| 2 | node1 | node3, node4 | 4 | 2 | ❌ | 1 |

- **iter1 抓到全部增益**：空技能→学到的技能，sel 0.667→0.833。
- **iter2 边际收益归零**：node3/node4 的 success 与 node1 **持平(0.833)**、成本更高(1892/1885 > 1675) → 被 node1 支配，前沿没长。`budget=4` 用尽 → 停。
- **这正是设计预测的 "edit economy"**：增益几乎全来自**第一次被接受的编辑**，深层搜索很快进入递减回报。

---

## 4. 成本 / 计算量分析（本报告重点）

### 4.1 token 总账（`tokens_total` = 272,277）

| 来源 | 阶段 | calls | tokens | 占比 |
|---|---|---|---|---|
| **target** | ① train rollout（展开证据） | 24 | 34,915 | 12.8% |
| **target** | 价值估计 rollout（sel，含 SH） | 44 | 71,913 | 26.4% |
| **target** | 前沿留出测试 rollout（test） | 40 | 58,048 | 21.3% |
| **target 小计** | | **108** | **164,876** | **60.6%** |
| **optimizer** | reflect + aggregate + clip（推导=总−target） | — | **107,401** | **39.4%** |
| **合计** | | | **272,277** | 100% |

- **target 108 次 rollout = 24(train) + 44(sel) + 40(test)**，与 `max_turns=1`（每 rollout 1 次调用）一致。
- **optimizer 占 39%**：MCTS 每展开一个子节点要跑一次完整 transition（reflect→merge→rank），4 次 transition 共 12 次 analyst（minibatch）调用 + merge/rank。

> ⚠️ **与 baseline 的占比差异要解释清楚**：baseline 里 rollout 占 88%、optimizer 仅 11%，**不是**因为 MCTS 更省 optimizer，而是因为 baseline 的评测规模大得多（test 150×3=450 次 rollout 把 target 占比抬高）。本次 sel/test 都很小，所以 optimizer 相对更显眼。**评测集放大后 target 会重新主导**（趋向 baseline 的 ~88%）。**结论：MCTS 的开销大头同样在 rollout 评测，要省钱优先压 `sel_eval_num` / `test_eval_num` / `budget`。**

### 4.2 CRN + Successive Halving 的实测收益

- 实际选择集 rollout = **44 次**；若每个节点都全量评估 5×12 = **60 次** → **省 16 次（27%）**。
- 省在哪：node2、node4 这两个**较弱的兄弟**在 n0=4 处被剪枝，没浪费后续 8 次/个；胜者 node1、node3 才晋级到全量 12（保证胜者价值精确）。
- **CRN 配对**：每个子节点都与**同一批锚任务上的父节点**逐题做差，`child.value = parent.value + Δ`，消掉任务间方差——这是在 EM 这种 0/1 噪声信号上能可靠排序的关键。
- 规模放大后收益更大：b 越大、\|S\| 越大，SH 砍掉的"全量评测"越多。

### 4.3 MCTS 开销模型（用于估算放大后的成本）

单次运行的 target rollout 次数 ≈：

```
  根锚点 |S|
+ (展开次数) × [ batch_size(train①)  +  Σ_children SH评测(n0…|S|) ]
+ (前沿点数) × test_eval_num
```

- 展开次数 ≈ `budget / branch`（本次 4/2 = 2）。
- 本次代入：12 + 2×(12 + ~22) + 2×20 = 12 + 68 + 40 ≈ 120（实测 108，因 SH 剪枝省下部分）。
- **放大到 baseline 规模的估算**（sel 40 / test 150 / budget 12 / b 3 / 前沿~3）：
  根 40 + 4×(12 + 3×~30) + 3×150 ≈ 40 + 408 + 450 ≈ **~900 次 target rollout**，约为线性 baseline(810) 的 ~1.1×——**同量级**，因为本次 budget 小；budget/depth 拉大才会显著超过线性（设计 mcts_04 的 I×W×D 膨胀）。

---

## 5. Best Skill 学到了什么（`best_skill.md` = node1）

从空白一步建立 QA 框架，规则与 baseline 的"好技能"同类（精确跨度 / 最短规范答案 / Jeopardy 填空 / 共享属性）：

- **Prefer the exact clue-matching span**：多段落时选词面最贴合 clue 的那段，别因为某实体出现频繁就选它。
- **Return the minimal canonical answer**：`<answer>` 内只给满足问题的最短跨度，去掉 "a place in…" 之类修饰。
- **Clue-style questions may expect shortened forms**：Jeopardy/数据库片段里答案以简写/分隔符给出时，用简写（人名可能只给姓氏）。
- **Shared-association questions**：列两个实体问共同点时，只答共享属性本身（如国家/地点名），别答描述句。
- **Use indirect and scoped clue wording**：用 context 解析昵称/译名/demonym/wordplay，并尊重 "first name / this state / this land" 这类范围约束。

> **关键观察（支撑 Milestone 3 动机）**：node3/node4 把技能继续加长到 3.1KB/3.0KB，**success 并未超过 node1(2.0KB)，只是更贵**。这正面印证"**success 对技能详细度非单调，越长不一定越好**"——动机即**降粒度 move（abstract/compress）**：让搜索能找到"更紧凑但同样强"的技能。

---

## 6. 结论与 MCTS 方法的行为画像（指导后续）

1. **链路正确**：在真实 gpt-5.5 上端到端跑通 transition→CRN+SH 估值→二维前沿→留出测试，产物 `frontier.json`/`tree.json`/`best_skill.md` 齐全；重组的 `transition()` 产出与线性 trainer 同质量的编辑。
2. **新增 cost 轴有效**：把部署 token 当第二目标，输出**菜单**而非单点，化解"贵但更准"的 confound（前沿自动两点都留）。
3. **开销大头在 rollout 评测**（target 61%，放大后趋向 ~88%）；**省钱优先级**：`sel_eval_num` > `test_eval_num` > `budget`/`branch`/`depth`。CRN+SH 已实测省 27% 选择集评测，规模越大省越多。
4. **edit economy 真实存在**：增益集中在第一次被接受的编辑；iter2 深搜无新增益。→ 等算力下，**宽而浅 + 多 λ / 重启** 很可能优于一棵深树（与设计 mcts_04 的 make-or-break 担忧一致，需正面做"等算力 random-restart greedy"对照）。
5. **越长不一定越好**：深层更长的技能没提分只增本 → 需要 **λ>0** 让成本进入导航，以及 **Milestone 3 的降粒度 move**。

### 后续建议
- **(a) 放大到可信规模**：sel 40 / test 150 / budget 12 / b 3 / **λ=0.3**，画出 ≥3 点的完整前沿；并跑 `value_mode=full` 做 **CRN+SH 消融**（对比 token 与方差）。
- **(b) Milestone 3 降粒度 move**：abstract/compress + refine/split，验证"双向粒度搜索同时改善 success 与 cost"。
- **(c) 等算力对照**：固定 token 预算，跑 random-restart greedy vs MCTS，判定树机器是否值这个钱。

---

## 7. 讨论：方法定位与三点反思

> 把不利于"MCTS"这个 framing 的话也说清楚。

### 7.1 纯 success 贪心是否已经够用?——很大程度上是

本次数据正面支持"贪心够用"：**增益全部来自 iter1 的第一次编辑**，iter2 深搜的更深节点（node3/node4）与 node1 持平、只更贵。一条严格 `>` gate 的贪心链会同样找到 node1，而且更省。在 target 很强（0.88 余量小）、地形 benign 的任务上，"回溯跳出局部最优"这台机器没什么可跳的。设计文档 `mcts_05 §3.3` 的自我结论也是：**"MCTS > greedy on success" 是最弱的主张。**

贪心**结构上赢不了**的只有两点：
- **cost 轴**：纯 success 贪心会乐于学最贵的技能（多规则 ≥ 同分，gate 不拦）；node0 省 26% token 只低 0.1 分，这个工作点贪心产不出。
- **success 对 detail 非单调**：node3/node4 更长却没提分；贪心严格 `>` 做不到"先压缩降一点、再换更稳的高分"。

**关键反问 = 决定性实验**：拿到 cost 轴**不一定需要 MCTS**——贪心链把每个候选（接受+拒绝）的 (success, cost) 都记下、取非支配集，几乎免费就有一条前沿。真正要证的是：**MCTS 的前沿是否 dominate "贪心+成本日志"的前沿。** SearchQA 上差距可能很小；程序性任务（啰嗦技能真掉分）更可能显著。→ **第一优先级对照（见 §8.1）。**

### 7.2 best_skill 的上限取决于 optimizer model——基本对，加三条修正

技能里每条规则都是 optimizer 想出来的，它的诊断+归纳能力封顶了质量。但：
1. **上限是 optimizer × target 联合**：规则要 target 照着做才有用。强 optimizer 写的微妙规则，弱 target 用错反而掉分（SkillOrchestra：弱模型+细粒度→更差）。甜点 = optimizer ≥ target 且规则在 target 的操作粒度。
2. **search/gate 不抬天花板，只采样/筛选**：gate 防退化，MCTS 更充分地**探索 optimizer 的提议分布**，能更可能命中它"能产出但低概率"的好规则，但**产不出分布外的规则**。一句话：**搜索摊销 optimizer 的生成分布，不提高它的天花板。**
3. **可主动利用的杠杆**：部署只用 target → 训练期**可用更强（更贵）的 optimizer**，把强模型的诊断力**蒸馏进一份 target 能用、部署零成本的技能**——本方法很有说服力的一个 framing。

### 7.3 宽而浅 + 多样性 优于深树

与 §7.1/§7.2 同一根逻辑：增益集中在第 1 次编辑 → 深度买的"路径前瞻"没价值；optimizer 是上限、好规则是它的低概率样本 → 最该做的是**浅层多采样**，width × diversity = 更好覆盖 optimizer 的提议分布。

但**光加宽不够**：现在 `expand` 只用 `expand_seed+k` 改 minibatch 分组，b 个子节点高度相关。要让"宽"有用，得让提议**真多样**：① optimizer 采样温度 >0；② 多反思视角（failure/success/cost lens、不同切片）；③ **不同动作类型做兄弟**（增量加规则 vs 降粒度压缩 vs 改写）——这把 M3 的降粒度 move 直接变成多样性来源。

---

## 8. 后续方向 与 CRN/SH 再评估

### 8.1 改进方向（按优先级）

1. 🔴 **决定性 baseline 对照（最高优先）**：给线性贪心链加"候选 (success,cost) 日志 + 非支配集导出"，几乎免费得一条**贪心前沿**；同 token 预算比 MCTS 前沿是否 dominate 它。**若贪心前沿已拿到 ~90%，这本身是一个诚实且有价值的 finding。**
2. 🔴 **引擎从深树 → 宽 + 多样 + 浅**：`depth 1–2, branch 4–6`，expand 加真多样性（温度 + 多视角 + 降粒度 move）。比"深树"更贴 edit economy。
3. 🟢 **强 optimizer → 弱/廉价 target 的蒸馏 framing**（§7.2）：训练期换更强 optimizer，量化技能质量随 optimizer 容量的提升；部署成本不变。
4. 🟢 **cost-aware Pareto 是稳态内核**：无论引擎是贪心/beam/MCTS 都成立；定位押这里，而非"MCTS>greedy on success"。
5. 🟡 **廉价 value 模型 / 过程信号**（`mcts_02 §8`）：回归预测 edit 的 Δ(success,cost) 当 prior/预筛——让 agentic 评估在大规模下可控的唯一现实路径，也等于 verifier-free gate。
6. 🟡 若保留树：**λ-sweep / Pareto-UCT** 让导航偏向前沿未覆盖区段（2D Option-B backup 已让一棵树服务所有 λ）。

### 8.2 CRN 是不是作用不大了?——分两层看

- **CRN 的"原则"（在同一批固定任务上比较）永远有用，而且我们已经免费拿到**：只要所有节点都评在**同一个固定 selection 子集**上，任意两节点的比较天然就是 common random numbers，无需额外机制。
- **CRN 的"机器"（显式 paired-Δ + telescoping）只是 successive halving 的配套**：它存在的唯一理由是 SH 给不同节点**不同/部分**的任务覆盖，这时才需要在共享前缀上做配对差来可靠排序。**它是 SH 的代价，不是免费的胜利。**

那么 **SH 还值不值?**
- **SH 的价值随 branch 上升**：候选越多，早砍弱者省得越多。所以**宽而浅（高 branch）恰恰更需要 SH** 来控宽度成本——与"放弃深树"互补，不矛盾。本次 sel=12 只省 27%；sel 大、branch 大时省得更多。
- **但在 diverse 兄弟下要更保守**：多样候选在极小前缀（n0）上更难分辨（本次单测复现过"前 2 题不区分→误剪枝"），应**调大 n0 / 调缓 η**，否则误杀"慢热但终局强"的候选。
- **小而便宜的 eval 集可以干脆 full-mode**：`sel_eval_num` 小、单题便宜时，直接全量评估所有兄弟（自动满足 CRN、无 telescoping 漂移、更简单）往往比"SH+paired+telescoping"更划算。`value_mode: full` 已保留为这条路径 + 消融基线。

> **一句话**：**CRN 原则保留（免费）；SH 用于控"宽度"成本、但在多样性下放缓；paired-Δ+telescoping 仅作 SH 的配套——eval 便宜时 full-mode 更简单。**

---

## 9. 产物清单（`outputs/mcts_smoke/`）

| 文件 | 内容 |
|---|---|
| `mcts/frontier.json` | 前沿菜单：每个非支配点的 sel/test 分数与 cost |
| `mcts/tree.json` | 全部节点（值/访问数/父子/终止）+ 逐迭代日志 `iter_log` |
| `best_skill.md` | perf prior 产物（= node1） |
| `mcts/best_skill_efficiency.md` | efficiency prior 产物 |
| `summary.json` | 总览：节点数 / 候选评估数 / task_rollouts / cost_ref / tokens_total |
| `mcts/nodes/<id>/skill.md` | 每个节点的技能快照 |
| `mcts/nodes/<id>/train_rollout/`、`mcts/sel_eval/<hash>/`、`mcts/test_eval/...` | 各阶段逐题 rollout（`results.jsonl` 含 `cost_total_tokens`，本报告的成本数据即由此精确求和） |

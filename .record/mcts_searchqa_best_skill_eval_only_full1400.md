# MCTS best_skill × SearchQA：eval_only 全量 1400 测试集评测与分析

- **日期**：2026-06-30
- **分支**：`feat/mcts-pareto-search`（commit `ed6e3eb`）
- **被评技能**：`outputs/mcts_compare_20260630_174516/best_skill.md`（MCTS 前沿 **perf** 部署点 = node1，2055 字符）
- **来源 MCTS 运行**：`outputs/mcts_compare_20260630_174516/`（`configs/searchqa/mcts_compare.yaml`，`lambda_cost=0`）
- **本次评测产物**：`outputs/eval_mcts_best_searchqa_test1400_20260630_192250/`
- **模型 / 设置**：target = **gpt-5.5**，`reasoning_effort=medium`，`max_turns=1`，`max_completion_tokens=16384`，`workers=24`
- **数据**：`data/searchqa_split/test/`（**全部 1400 条**，file order；非 MCTS 时的 150 条子样本）

---

## 0. 一句话结论

> **best_skill 在完整 1400 条 held-out 测试集上 hard-EM = 0.8600 / F1 = 0.9257**。MCTS 当初报告的 **test 0.92 是只评了 file 前 150 条的乐观值**——那 150 条恰好是测试文件里偏简单的一段（前 150 = 0.9133，其余 1250 = 0.8536，差约 6 个点）。在**完全相同的 150 条 + 同一技能**上，本次独立评测复现到 **0.9133 vs 0.92（差 1 题，逐题一致率 149/150 = 99.3%）**，唯一翻盘项是一个**弯引号 `'`(U+2019) 归一化漏网的评测 bug**，并非模型推理差异。换言之：rollout 近乎确定性，0.92 与 0.86 的差距**不是噪声、而是子样本选取偏差**——真实 held-out hard-EM 应以 **0.86** 为准。另：196 个 EM=0 里 **120 个 sub_em=1、133 个 F1≥0.5**，hard-EM 严重低估了实际答题质量。

---

## 1. 跑了什么 / 代码路径

### 1.1 命令

```bash
set -a; source ./.env; set +a
python scripts/eval_only.py \
  --config configs/searchqa/mcts_compare.yaml \
  --skill outputs/mcts_compare_20260630_174516/best_skill.md \
  --optimizer_model gpt-5.5 --target_model gpt-5.5 \
  --split test \
  --out_root outputs/eval_mcts_best_searchqa_test1400_20260630_192250
```

- `--split test` 走 [eval_only.py:419-421](../scripts/eval_only.py#L419-L421) 的 else 分支，`test_env_num` 默认 0 → `build_eval_env(0,"test",42)` → 不截断 → **全部 1400 条**。
- split 别名：`test → test/` 见 [datasets/base.py:154-161](../skillopt/datasets/base.py#L154-L161)（`_SPLIT_ALIAS`）。`env_num=0` 取全量见 [datasets/base.py:496-512](../skillopt/datasets/base.py#L496-L512)。

### 1.2 真正被执行的链路（无训练、无优化器调用）

```
eval_only.main()
  → get_adapter(cfg) = SearchQAAdapter            scripts/eval_only.py:106
  → adapter.rollout(items, skill, out_root)        envs/searchqa/adapter.py:74
      → run_batch(...)                             envs/searchqa/rollout.py:374  (ThreadPoolExecutor, workers=24, resume-aware)
          → process_one(item,...)                  envs/searchqa/rollout.py:142
              → _build_system(skill) + _build_user(question, context)
              → chat_target(system,user, max_completion_tokens=16384, stage="rollout")   ← 唯一一次 gpt-5.5 调用 / 题
              → evaluate(response, gold)            envs/searchqa/evaluator.py:88
  → compute_score(results)                         skillopt/utils/scoring.py:7
```

- **单轮**：`max_turns=1` ⇒ 每题 1 次模型调用；本次 `n_turns` 全为 1、`agent_ok=1400/1400`。
- **答案抽取**：`<answer>...</answer>`，回退到最后一非空行（[evaluator.py:28-39](../skillopt/envs/searchqa/evaluator.py#L28-L39)）。
- **指标定义**（[evaluator.py](../skillopt/envs/searchqa/evaluator.py) + [scoring.py:7](../skillopt/utils/scoring.py#L7)）：
  - `hard` = **EM**（SQuAD 归一化后精确匹配，转 int）；`compute_score.hard` = 均值。
  - `soft` = **token-level F1**（对所有 gold 取 max）；`compute_score.soft` = 均值。
  - `sub_em` = 归一化后任一 gold 与预测互为子串则 1.0（仅记录，不进 headline）。
- **成本通道**（[rollout.py:307-323](../skillopt/envs/searchqa/rollout.py#L307-L323)）：`cost_total_tokens = prompt_tokens + completion_tokens`，技能被前置进 system prompt，故 `prompt_tokens` 已吸收常驻技能 token。

---

## 2. 实验结果

### 2.1 Headline（全量 1400）

| 指标 | 值 | 计数 |
|---|---|---|
| **hard-EM（全量 1400）** | **0.8600** | 1204 / 1400 |
| **F1（全量 1400）** | **0.9257** | — |
| agent_ok | 100% | 1400 / 1400 |
| n_turns | 全 = 1 | 单轮 |

`eval_summary.json`：`{"split":"test","n_items":1400,"hard":0.86,"soft":0.9257159...}`

### 2.2 子样本 vs 全量（核心发现）

| 切片（file order） | n | hard-EM | F1 | 备注 |
|---|---|---|---|---|
| 前 150（= MCTS test_eval 同一批） | 150 | **0.9133** (137/150) | 0.9540 | MCTS 当初报 **0.9200** (138/150) |
| 其余 151–1400 | 1250 | **0.8536** | — | 比前 150 低 ~6 个点 |
| **全量** | 1400 | **0.8600** (1204/1400) | 0.9257 | 真实 held-out 口径 |

**每 200 条分箱**（file order，可见前段明显更易）：

```
items    0- 199: hard=0.9000   items  600- 799: hard=0.8400
items  200- 399: hard=0.8950   items  800- 999: hard=0.8300
items  400- 599: hard=0.8400   items 1000-1199: hard=0.8850
                               items 1200-1399: hard=0.8300
```

### 2.3 复现性 / gpt-5.5 run-to-run 噪声（同 150 条 + 同技能）

| 对比 | hard | 计数 |
|---|---|---|
| MCTS test_eval（`e1893a40fd6ff02d`） | 0.9200 | 138/150 |
| 本次 eval_only（同 150 条） | 0.9133 | 137/150 |
| **逐题一致率** | **149/150 = 0.9933** | 仅 1 题翻盘 |
| flips ✓→✗ / ✗→✓ | 1 / 0 | net −1 |

**唯一翻盘项 = 评测归一化 bug，非推理差异**：

- Q：`George & Martha are the boozy twosome in this Albee play`
- MCTS 答 `Who's Afraid of Virginia Woolf?`（直引号 `'` U+0027）→ EM=1
- 本次答 `Who's Afraid of Virginia Woolf?`（**弯引号 `'` U+2019**）→ EM=0
- 根因：`normalize_answer` 只删 `string.punctuation`（ASCII），**弯引号 U+2019 不在其中** → 残留 → 与 gold `whos...` 不匹配。已实测确认（`'' in string.punctuation == False`）。

### 2.4 成本（部署口径）

| 量 | 值 |
|---|---|
| `cost_total_tokens` 均值 / 中位 | **1590.8** / 1590 |
| └ prompt / completion 均值 | 1504.0 / 86.8 |
| min / max | 650 / 3070 |
| 全量 target token 合计 | ~2,227,189 |

对照 MCTS 报告的 node1 `test_cost = 1596.6`（150 条）→ 与本次 1590.8（1400 条）**几乎一致**，印证"成本 ≈ 常驻技能长度的函数"（单轮任务，成本轴退化为技能体积；与 `mcts_searchqa_实验分析与officeqa迁移.md` §4 一致）。

### 2.5 EM 的严苛性 / hard-EM 低估质量

- 196 个 EM=0 中：**sub_em=1（实为子串命中，仅跨度/形态差）= 120（61%）**；**F1≥0.5（基本答对）= 133（68%）**。
- 典型"假失败"（pred | gold）：
  - `'by bread alone'` | `'on bread alone'`（介词）
  - `'William Clark'` | `'Clark'`（多给名）；`'Venus'` | `'Venus Williams'`（少给姓）
  - `'Aloha Airlines'` | `'Aloha'`；`'D-Day'` | `'the Normandy invasion'`（同指不同表述）
  - 弯引号 / 反斜杠转义类 gold（§2.3）
- 这解释了 **F1 0.9257 与 hard-EM 0.86 的 6.6 点剪刀差**：真实答题质量更接近 F1。

---

## 3. 分析与结论

1. **0.92 是子样本乐观值，0.86 才是真实 held-out。** 文件前 150 条恰是较易的一段（前 400 条 ≈0.90，中后段掉到 0.83–0.84）。MCTS（含其对标的 linear baseline）都只评了这前 150 条，绝对分被系统性抬高约 6 个点。**相对结论不受影响**（见 4），但**绝对分要以全量 0.86 为准**。

2. **rollout 近乎确定性，方法对比里的"±1 题"确属噪声但极小。** 同技能同题逐题一致率 99.3%，唯一差异是评测 bug。这反向佐证了原 §3 的判断——验证/测试集上"1 题 = 0.0067~0.025"的波动几乎全是评测/Unicode 噪声，UCT 在 depth≥1 没有可爬的真实梯度。

3. **存在可修的评测缺陷（评测低估）。** (a) `normalize_answer` 不删 Unicode 标点（弯引号、长破折号）→ 平白丢分；(b) EM 对"多/少一个修饰词、介词、同指异表"零容忍，68% 的失败 F1≥0.5。建议：归一化里把 `str.translate` 扩展到 Unicode 标点类（`unicodedata.category(c).startswith('P')`），可立即回收一批假失败（至少 §2.3 那 1 题，量级上 120 个 sub_em=1 里有相当部分）。

4. **关于增益 Δ 的诚实声明（重要 caveat）。** 本次按用户要求**只评了 best_skill，未在全量 1400 上跑 seed 基线**。因此：
   - 可比的是**同一批 150 条**：best=0.9133 > seed=0.8733（MCTS 报告）≈ **+0.04**，与原实验一致，技能确有正增益。
   - **不可**用全量 best=0.86 去对 seed-on-150=0.8733 得出"技能掉分"——那是跨不同题集的伪比较。全量真实 Δ **尚未测**；若要，需在同 1400 条上补跑 seed（`outputs/mcts_compare_20260630_174516/mcts/nodes/0000/skill.md`）。

5. **与既有结论一致并强化它。** 成本轴 ≈ 技能长度（§2.4）、精度在 depth1 吃满、单轮 gpt-5.5 太强——都印证 `mcts_searchqa_实验分析与officeqa迁移.md` 的主结论：**SearchQA 不足以体现 cost-aware MCTS 的价值，应迁移到多轮 OfficeQA**（`max_turn>1`、真实二维成本权衡），且迁移前须先接好 OfficeQA 的 `cost_total_tokens` 成本通道。

---

## 4. 下一步（按价值排序）

- [ ] **（可选，闭合 Δ）** 在同 1400 条上补跑 seed 基线 → 得到全量真实增益：
      `python scripts/eval_only.py --config configs/searchqa/mcts_compare.yaml --skill outputs/mcts_compare_20260630_174516/mcts/nodes/0000/skill.md --split test --out_root outputs/eval_seed_searchqa_test1400`
- [ ] **修评测归一化**：`evaluator.normalize_answer` 增删 Unicode 标点（回收弯引号类假失败），并在 record/paper 里同时报 EM + sub_em/F1，避免低估。
- [ ] **报告口径统一**：后续 MCTS/baseline 对比一律在**全量 test**或**随机抽样**上评，勿用 file 前 N（前段偏易，系统性偏置）。
- [ ] 主线仍是 **迁移 OfficeQA**（见 `mcts_searchqa_实验分析与officeqa迁移.md` §5：先接 `cost_total_tokens`，再建 `configs/officeqa/mcts_compare.yaml`，`lambda_cost=0.3`）。

---

## 附录：关键产物

| 文件 | 内容 |
|---|---|
| `outputs/eval_mcts_best_searchqa_test1400_20260630_192250/eval_summary.json` | headline：hard=0.86, soft=0.9257, n=1400 |
| `outputs/eval_mcts_best_searchqa_test1400_20260630_192250/results.jsonl` | 1400 条逐题（em/f1/sub_em/cost_total_tokens/predicted/gold） |
| `outputs/eval_mcts_best_searchqa_test1400_20260630_192250/predictions/<id>/` | 每题 system/user prompt + conversation |
| `outputs/eval_mcts_best_searchqa_test1400_20260630_192250.log` | 运行日志（逐题滚动 acc） |
| `outputs/mcts_compare_20260630_174516/mcts/test_eval/.../e1893a40fd6ff02d/results.jsonl` | MCTS 当初的 150 条 test_eval（用于 §2.3 复现对照） |

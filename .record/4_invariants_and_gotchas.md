# SkillOpt 内部文档 4 — 不变量与改代码红线（Invariants & Gotchas）

> **目的 / Purpose**：把散落在 [`2_data_contracts.md`](2_data_contracts.md) / [`3_pipeline_stages.md`](3_pipeline_stages.md) 里的 ⚠️ 收口成**一页速查 + 排错手册**。改 `skillopt/` 前扫一遍；出 bug 时按“症状→根因”表反查。
>
> 标注 🔴 = 违反必出 bug；🟡 = 容易踩、但有兜底；🔵 = 设计约定，理解后能省很多排错时间。
> 锚点 `file:line` 基于撰写时 HEAD。

---

## A. 最容易踩的 5 条（TL;DR）

1. 🔴 **配置键改名**：YAML `optimizer.learning_rate` 在代码里叫 `cfg["edit_budget"]`（`config.py:111`）。grep 代码搜 `learning_rate` 一无所获是正常的。改配置项要在 `config.py` 映射表登记。→ §B
2. 🔴 **patch 的 `target` 必须逐字节精确匹配** skill 文本，否则 replace/delete **静默跳过**、不报错（`skill.py:128/138`）。candidate “没怎么变”先看 `edit_apply_report.json`。→ §D
3. 🔴 **patch 载荷 key 随 update_mode 变**（`edits`/`revise_suggestions`/`skill_candidates`）。永远用 `get_payload_items`，别硬编码 `["edits"]`（`update_modes.py:52`）。→ §D
4. 🔴 **改产物路径/文件名 = 破坏 resume**：`results.jsonl`、`minibatch_*.json`、`runtime_state.json` 存在即跳过重算。→ §C
5. 🟡 **三处 skip 分支都必须 flush skill-aware appendix**，否则 lapse-only 步的 notes 被丢（`trainer.py:1201/1395/1495`）。→ §E

---

## B. 配置 / Config

| 级别 | 不变量 | 锚点 |
|---|---|---|
| 🔴 | YAML 嵌套键被 `flatten_config` 扁平化**并改名**，trainer 读扁平名 | `config.py:185` |
| 🔴 | `optimizer.learning_rate`→`edit_budget`；`optimizer.min_learning_rate`→`min_edit_budget`；`evaluation.sel_env_num`→`sel_env_num`；`env.skill_init`→`skill_init`；`env.out_root`→`out_root`；`model.backend`→`model_backend` | `config.py:32/111/112/128/132/133` |
| 🟡 | 新增配置项：在 `config.py` 映射登记，否则 trainer `cfg.get(...)` 拿默认值、你以为生效其实没生效 | `config.py:185` |
| 🔵 | `_base_:` 继承用深合并（`_deep_merge` `config.py:139`）；CLI `--key val` 覆盖最后生效（`apply_overrides` `config.py:236`） |
| 🔵 | `config.json`（落盘）已脱敏（`_redact_cfg` `trainer.py:342`，含 api_key 等） |

**排错**：改了 config 没效果 → 先确认扁平名；打印 `config.json` 看 trainer 实际收到什么。

---

## C. Resume / 断点续训

| 级别 | 不变量 | 锚点 |
|---|---|---|
| 🔴 | resume 两层：`runtime_state.json` 优先，回退 `history.json`，都无才从头 | `trainer.py:856-917` |
| 🔴 | 阶段级断点靠产物存在性：rollout 看 `results.jsonl`，reflect 看 `minibatch_*.json` | `adapter.py:331`、`reflect.py:560` |
| 🔴 | 改产物路径/文件名/目录结构 → 已跑步被重跑或读不回 | 文档2 §9 |
| 🟡 | `random_seed` 决定 reflect 分组（`reflect.py:540`）；改种子公式 → 分组变 → 缓存失效、不可复现 | `datasets/base.py:97-108` |
| 🔵 | 验证分按 `candidate_hash=skill_hash(skill)` 缓存（`sel_cache`），同份 skill 不重复 rollout | `trainer.py:1420`、`scoring.py:26` |
| 🔵 | 想强制重跑某步：删该 `steps/step_XXXX/` 下对应子目录 |

---

## D. Patch / Edit 应用（最高频 bug 区）

| 级别 | 不变量 | 锚点 |
|---|---|---|
| 🔴 | `target` 必须是 skill 里逐字节存在的子串，否则 replace/delete 静默跳过 | `skill.py:128/138` |
| 🔴 | 载荷 key 多态：用 `get_payload_items(container, mode)`，勿硬编码 `["edits"]` | `update_modes.py:52` |
| 🟡 | `insert_after` target 找不到 → **降级为 append**（不是跳过） | `skill.py:109-117` |
| 🟡 | `replace`/`delete` 只替换/删**第一处**（`str.replace(.., 1)`） | `skill.py:132/142` |
| 🟡 | `apply_patch_with_report` 永不抛异常，单条出错记 `status="error"` 继续 | `skill.py:176-184` |
| 🔵 | 每条 edit 结果在 `edit_apply_report.json` 的 `status`：`applied_*` / `skipped_*` / `error` | `trainer.py:1364` |
| 🔵 | 预算双重裁剪：reflect 内 `truncate_payload`（`reflect.py:347`）+ SELECT `rank_and_select`（`clip.py:25`），都受 `edit_budget` | — |

**排错**：candidate skill 没按预期变 →
1. 看 `ranked_edits.json` 确认 edit 真被选中；
2. 看 `edit_apply_report.json` 的 `status` 是 `applied_*` 还是 `skipped_replace_target_not_found`；
3. 若大量 `skipped_*target_not_found` → optimizer 给的 target 不精确（prompt 问题）。

---

## E. 保护区 + skill-aware appendix

| 级别 | 不变量 | 锚点 |
|---|---|---|
| 🔴 | 两个保护区 `<!-- SLOW_UPDATE_START/END -->`、`<!-- APPENDIX_START/END -->`，step 级 edit 改不动（target 落区内 → `skipped_protected_region`） | `skill.py:14-30/94` |
| 🔴 | **三处 skip 分支都要 flush appendix**：no_patches（`1201`）、no_rewrite（`1395`）、正常路径（`1495`） | `trainer.py` + `_flush_skill_aware_appendix:81` |
| 🟡 | append/insert fallback 插到“最早保护区之前”，保证保护块永在文档尾 | `skill.py:99-104` |
| 🟡 | edit content 里的 marker 会被剥掉，防重复 marker 破坏区块解析 | `skill.py:66` |
| 🔵 | skill-aware 开启时初始 skill 会注入空 appendix 占位（`inject_empty_appendix_field` `trainer.py:929`） |
| 🔵 | appendix 超阈值会被 LLM 整理（`skill_aware_consolidate_threshold>0`，`trainer.py:130`） |

---

## F. Gate / 验证门控

| 级别 | 不变量 | 锚点 |
|---|---|---|
| 🔴 | **严格大于才接受**：`cand > current` 才 accept，`> best` 才 new_best，相等也 reject | `gate.py:123` |
| 🔵 | `evaluate_gate` 是纯函数：不调模型、不落盘、不打印；副作用全在 trainer | `gate.py:7-8`、`trainer.py:1483` |
| 🔵 | selection 集 = `valid_seen`，与 train/test 独立 | `trainer.py:1428` |
| 🔵 | `gate_metric` ∈ hard/soft/mixed；selection 集小、hard 不敏感时用 soft/mixed | `gate.py:46` |
| 🟡 | `use_gate=False` 不走 evaluate_gate，trainer 自构 `force_accept`，验证仍跑（分照记） | `trainer.py:1457-1477` |

---

## G. Rollout（target 调用侧）

| 级别 | 不变量 | 锚点 |
|---|---|---|
| 🔴 | `result["id"]` 必须等于 `predictions/<id>/conversation.json` 的目录名，否则 reflect 找不到轨迹、静默丢失 | `rollout.py:289/314`、`reflect.py:140` |
| 🟡 | ALFWorld 分块大小 = `workers`：太大 OOM（同时开太多模拟器），太小慢 | `adapter.py:384` |
| 🟡 | target 空响应/缺 `<action>`/异常 → 兜底 `<action>look</action>`，episode 不崩但该步无效 | `rollout.py:221-226` |
| 🔵 | `hard` 失败判定用 `hard < 1e-9`（支持连续奖励） | `reflect.py:537` |
| 🔵 | `compute_score` 把 hard/soft 都当 float 求平均 | `scoring.py:21` |

---

## H. 模型后端 / Token

| 级别 | 不变量 | 锚点 |
|---|---|---|
| 🔵 | optimizer 与 target 可不同后端/模型，分别配置 | `trainer.py:690-693` |
| 🔵 | `stage=`（rollout/analyst/merge/ranking）只用于 token 分类，不影响路由 | `model/__init__.py:333` |
| 🟡 | 后端切换走 `REFLACT_MODEL_BACKEND` 环境变量 + `router.set_backend`；改默认后端注意全局副作用 | `router.py:11/29` |
| 🔵 | reasoning_effort 留空 → 回退环境变量（见 `0_start.md` 的 Azure 排错记录） | `trainer.py:747` |

---

## I. 阶段间数据流 / Normalize

| 级别 | 不变量 | 锚点 |
|---|---|---|
| 🟡 | `raw_patches` 可能含 None（analyst 解析失败），下游用 `isinstance(p, dict)` 过滤 | `trainer.py:161` |
| 🟡 | 空 edits 的 patch 被 `_normalise_patches` 丢出 body 流水线（skill-aware 下仍保留 notes） | `trainer.py:166`、`reflect.py:355` |
| 🔵 | `source_type` 由代码强制注入，不信 LLM 输出 | `reflect.py:345/439` |
| 🔵 | aggregate/select LLM 失败都有兜底（拼接/截断），不丢梯度但质量下降 | `aggregate.py:62`、`clip.py:104` |

---

## J. Epoch 末（slow update / meta）

| 级别 | 不变量 | 锚点 |
|---|---|---|
| 🔴 | force-accept 模式下慢更新只改 `current_skill`，**绝不污染 best_skill**（best 须是验证最优步的忠实快照） | `trainer.py:1654-1660` |
| 🔵 | slow update：epoch 1 注空占位，epoch≥2 才纵向对比 | `trainer.py:1661/1676` |
| 🔵 | meta_skill 是 optimizer 的额外上下文，**不直接改 target 的 skill** | `trainer.py:1046`、`meta_skill.py:33` |

---

## K. 通用排错速查 / Symptom → root cause

| 症状 | 最可能根因 | 先看 |
|---|---|---|
| 改了 config 没效果 | 用了 YAML 名而非扁平名 / 没在映射登记 | `config.json`、`config.py:185`（§B） |
| candidate skill 几乎没变 | edit 静默跳过（target 不匹配/保护区） | `edit_apply_report.json`（§D） |
| 编辑全丢 / 拿到空载荷 | update_mode 下硬编码了 `["edits"]` | `get_payload_items`（§D） |
| reflect 没产出 patch | 轨迹文件找不到（id 不匹配）/ analyst 全 None | `predictions/`、`patches/`（§G/§I） |
| resume 重跑已完成的步 | 产物路径被改 / runtime_state 缺失 | `runtime_state.json`（§C） |
| 一直 reject、skill 不进步 | 基线太强 / selection 集太小 hard 不敏感 | 换 `gate_metric=soft`（§F） |
| OOM | ALFWorld `workers` 太大，同时开太多模拟器 | 降 `workers`（§G） |
| best_skill 被慢更新污染 | 误把 slow content 写进 best | `trainer.py:1654`（§J） |
| skill-aware notes 丢失 | 某 skip 分支漏 flush appendix | `trainer.py:1201/1395/1495`（§E） |

---

## L. 提交前自检 / Pre-commit checklist

- [ ] 新增配置项 → `config.py` 映射登记，YAML 名与代码扁平名都对？
- [ ] 碰 patch → 用 `get_payload_items`，三种 update_mode 都验证？
- [ ] 碰产物路径 → resume 三件套（results.jsonl / minibatch_*.json / runtime_state.json）仍成立？
- [ ] 碰 rollout → `result["id"]` 与 conversation 目录名一致？兜底/分块还在？
- [ ] 碰 gate → 严格大于 + 纯函数？
- [ ] 任意 skip 分支 → appendix flush 跟上（3 处）？
- [ ] 碰 epoch 末 → best_skill 不被污染、保护区 marker 同步？
- [ ] 跑一遍最小命令（见 `0_start.md`）确认 end-to-end 不崩？

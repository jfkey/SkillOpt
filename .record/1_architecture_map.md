# SkillOpt 内部文档 1 — 全局架构地图（Architecture Map）

> **目的 / Purpose**：一张“从命令行到 best_skill.md”的总览。先看懂模块怎么分工、调用怎么串、两个大模型在哪几个点被调用，再深入细节。这是**索引文档**——具体数据结构去 [`2_data_contracts.md`](2_data_contracts.md)，具体每阶段去 [`3_pipeline_stages.md`](3_pipeline_stages.md)。
>
> 约定：`file:line` 锚点基于撰写时的 HEAD；术语 **target** = 做任务的模型，**optimizer** = 改 skill 的模型。

---

## 1. 一句话架构 / One-liner

> SkillOpt 把 **skill 文档当作可训练状态**，用一个固定的 6 阶段循环（rollout→reflect→aggregate→select→update→gate）反复改它：**target 模型**用当前 skill 做任务产生轨迹，**optimizer 模型**把轨迹变成有预算的文本编辑，只有通过验证集门控的编辑才被接受。模型权重永不改变；部署产物只有一份 `best_skill.md`。

---

## 2. 顶层入口流 / Entry flow（CLI → 训练）

```
$ python scripts/train.py --config configs/alfworld/default.yaml [--key val ...]
        │
scripts/train.py:520  main()
   ├─ :522  load_config(args)          # 读 YAML + _base_ 继承 + CLI 覆盖
   │        scripts/train.py:388 → skillopt/config.py:261 load_config
   │                              → skillopt/config.py:185 flatten_config  # 嵌套→扁平+改名(见文档2 §10)
   ├─ :551  get_adapter(cfg)           # 按 cfg["env"] 实例化环境适配器
   │        scripts/train.py:106  （懒加载注册表 _register_builtins，缺依赖则跳过该 env）
   ├─ :555  ReflACTTrainer(cfg, adapter)
   └─ :556  trainer.train()            # ← 进入主循环，skillopt/engine/trainer.py:597
```

两个安装好的控制台命令（`pyproject.toml [project.scripts]`）：
- `skillopt-train` → `scripts.train:main`
- `skillopt-eval` → `scripts.eval_only:main`（只评估单个 skill，不训练）

---

## 3. 模块职责表 / Module responsibilities

| 模块 | 职责（一句话） | 关键入口 file:line |
|---|---|---|
| `scripts/train.py` | CLI 入口、env 注册表、组装 cfg+adapter | `main` `:520`，`get_adapter` `:106` |
| `scripts/eval_only.py` | 只跑评估，不训练 | `main` |
| `skillopt/config.py` | YAML 加载 / `_base_` 继承 / 扁平化改名 / CLI 覆盖 | `load_config` `:261`，`flatten_config` `:185` |
| `skillopt/engine/trainer.py` | **总指挥**：6 阶段循环 + epoch 末慢更新/meta + resume | `ReflACTTrainer.train` `:597` |
| `skillopt/envs/base.py` | `EnvAdapter` 抽象接口 + prompt 加载 | `:285/298` prompt getters |
| `skillopt/envs/<name>/` | 各 benchmark 适配器（rollout/reflect/dataloader/skills/prompts） | ALFWorld: `adapter.py:48` |
| `skillopt/datasets/base.py` | 批次规划（`BatchSpec` + 确定性种子） | `BatchSpec` `:41`，`plan_train_epoch` `:111` |
| `skillopt/gradient/reflect.py` | ② 反思：轨迹→RawPatch（minibatch analyst） | `run_minibatch_reflect` `:472` |
| `skillopt/gradient/aggregate.py` | ③ 聚合：分层合并 patch | `merge_patches` `:143` |
| `skillopt/optimizer/clip.py` | ④ 选择：按 budget 排序裁剪 | `rank_and_select` `:25` |
| `skillopt/optimizer/skill.py` | ⑤ 更新：把 edit 应用到文档 | `apply_patch_with_report` `:165` |
| `skillopt/optimizer/scheduler.py` | 学习率（edit budget）调度 | `build_scheduler` `:104` |
| `skillopt/optimizer/lr_autonomous.py` | optimizer 自主决定 LR | `decide_autonomous_learning_rate` |
| `skillopt/optimizer/slow_update.py` | epoch 末动量/纵向对比 | `run_slow_update` `:309`，`build_comparison_pairs` `:159` |
| `skillopt/optimizer/meta_skill.py` | optimizer 记忆 | `run_meta_skill` `:33` |
| `skillopt/optimizer/skill_aware.py` | 技能缺陷 vs 执行失误分流 + appendix | `augment_*_prompt`、`extract_appendix_notes` |
| `skillopt/optimizer/update_modes.py` | patch/rewrite/full_rewrite 载荷多态 helper | `get_payload_items` `:52` |
| `skillopt/evaluation/gate.py` | ⑥ 验证门控（纯决策函数） | `evaluate_gate` `:76` |
| `skillopt/model/` | 模型后端路由 + 计费 | `chat_target` `__init__.py:119`，`chat_optimizer` `:80` |
| `skillopt/prompts/` | 通用 prompt（.md）+ 加载器 | `load_prompt` `__init__.py:32` |
| `skillopt/utils/` | 打分 / 哈希 / JSON 解析 | `compute_score`、`skill_hash`、`extract_json` |
| `skillopt/types.py` | 阶段间数据契约 dataclass | `:26/72/104/205` |

---

## 4. 主调用图 / The call graph（训练一个 step）

```
ReflACTTrainer.train()                                              trainer.py:597
│
└─ for step:                                                        trainer.py:1062
   │
   ├─① ROLLOUT ─ adapter.rollout(current_skill)                    trainer.py:1122
   │   └─ ALFWorldAdapter.rollout                                  adapter.py:320
   │       └─ _run_batch（分块开/关模拟器）                          adapter.py:371
   │           ├─ build_alfworld_env  → vendor                     rollout.py:71
   │           └─ run_alfworld_batch                               rollout.py:129
   │               └─ chat_target ──────────────────────▶【target】 rollout.py:211 → model/__init__.py:119
   │
   ├─② REFLECT ─ adapter.reflect(rollout_results)                  trainer.py:1138
   │   └─ run_minibatch_reflect                                    reflect.py:472
   │       ├─ run_error_analyst_minibatch                          reflect.py:256
   │       │   └─ chat_optimizer ───────────────────────▶【optim】 reflect.py:334 → model/__init__.py:80
   │       └─ run_success_analyst_minibatch                        reflect.py:366
   │           └─ chat_optimizer ───────────────────────▶【optim】 reflect.py:431
   │   └─ _normalise_patches（拆 failure/success，丢空）            trainer.py:1145 → 149
   │
   ├─③ AGGREGATE ─ merge_patches                                   trainer.py:1222 → aggregate.py:143
   │   └─ _hierarchical_merge → _merge_batch                       aggregate.py:70/28
   │       └─ chat_optimizer ───────────────────────────▶【optim】 aggregate.py:46
   │
   ├─④ SELECT ─ rank_and_select(max_edits=edit_budget)            trainer.py:1274 → clip.py:25
   │   ├─ scheduler.step()  或  decide_autonomous_learning_rate    trainer.py:1273/1252
   │   └─ chat_optimizer（仅当载荷>预算）──────────────▶【optim】 clip.py:75
   │
   ├─⑤ UPDATE ─ apply_patch_with_report                           trainer.py:1360 → skill.py:165
   │   （patch 模式不调模型；rewrite 模式 chat_optimizer）          trainer.py:1311
   │
   └─⑥ EVALUATE ─ adapter.rollout(candidate, "valid_seen")        trainer.py:1434
       │   └─ chat_target ───────────────────────────────▶【target】（命中 sel_cache 则跳过）
       └─ evaluate_gate（纯函数，不调模型）                          trainer.py:1441 → gate.py:76
```

epoch 末（非每 step）：
```
SLOW UPDATE ─ run_slow_update                                      trainer.py:1770 → slow_update.py:309
   └─ 2× adapter.rollout(prev/curr) ─▶【target】 + chat_optimizer ─▶【optim】
META SKILL  ─ run_meta_skill                                       → meta_skill.py:33  ─▶【optim】
```

---

## 5. 大模型调用点总表 / Where the LLMs are called

| # | 阶段 | 角色 | 函数 | file:line | 调用次数（每 step） |
|---|---|---|---|---|---|
| 1 | ① ROLLOUT | **target** | `chat_target` | `rollout.py:211` | episodes × steps |
| 2 | ② REFLECT | **optimizer** | `chat_optimizer(stage="analyst")` | `reflect.py:334/431` | 失败组数+成功组数 |
| 3 | ③ AGGREGATE | **optimizer** | `chat_optimizer(stage="merge")` | `aggregate.py:46` | 分层合并，多次 |
| 4 | ④ SELECT | **optimizer** | `chat_optimizer(stage="ranking")` | `clip.py:75` | 0 或 1（载荷>预算才调） |
| 5 | ⑤ UPDATE | optimizer | `chat_optimizer`（仅 rewrite 模式） | `trainer.py:1311` | 0 或 1 |
| 6 | ⑥ EVALUATE | **target** | `chat_target` | `rollout.py:211` | 0（缓存）或 sel_set 规模 |
| 7 | SLOW UPDATE | target + optimizer | `adapter.rollout` ×2 + `run_slow_update` | `trainer.py:1718/1770` | epoch≥2，每 epoch 一次 |
| 8 | META SKILL | optimizer | `run_meta_skill` | `meta_skill.py:33` | 每 epoch 一次 |

> **部署期 = 0**。以上全部发生在训练期。最终交付的 `best_skill.md` 直接喂给目标模型即可，无额外模型调用——这是 SkillOpt 的核心卖点。

---

## 6. 模型后端层 / Model backend layer（一图）

```
trainer.train() 启动时配置：
  configure_azure_openai(...)        trainer.py:634 → model/__init__.py:392
  set_optimizer_backend / set_target_backend   trainer.py:690-691
  set_optimizer_deployment / set_target_deployment  trainer.py:692-693
  set_reasoning_effort               trainer.py:748 → model/__init__.py:497

运行时：
  chat_optimizer / chat_target       model/__init__.py:80 / :119
        └─ router 按 active backend 分发  model/router.py:45
             ├─ azure_openai.py   (azure_openai / openai_chat)
             ├─ codex_backend.py + codex_harness.py  (codex_exec)
             ├─ claude_backend.py (claude_chat / claude_code_exec)
             ├─ qwen_backend.py   (qwen_chat, 本地 vLLM)
             └─ minimax_backend.py(minimax_chat)
  token 计费：get_token_summary / reset_token_tracker  model/__init__.py:333/385
```

- optimizer 和 target **可用不同后端/模型**（如 optimizer=gpt-5.5，target=claude_code_exec）。
- `stage=` 参数（"rollout"/"analyst"/"merge"/"ranking"）只用于 token 分类统计，不影响路由。

---

## 7. 环境适配器层 / EnvAdapter layer

所有 benchmark 通过实现 `EnvAdapter`（`skillopt/envs/base.py`）接入。trainer 只认这个接口，**环境无关**。

```
EnvAdapter（抽象）                                     envs/base.py
  ├─ setup(cfg)                          一次性初始化
  ├─ get_dataloader()                    返回批次规划器
  ├─ build_env_from_batch(BatchSpec)     批次→env 句柄   ALFWorld: adapter.py:297
  ├─ build_train_env / build_eval_env    便捷封装        adapter.py:312/316
  ├─ rollout(env, skill, out_dir)        ① 跑任务        adapter.py:320
  ├─ reflect(results, skill, out_dir)    ② 反思          adapter.py:428
  ├─ get_error/success_minibatch_prompt  analyst prompt  base.py:285/298
  └─ get_task_types()                    任务类型         adapter.py:458
```

prompt 加载优先级（`load_prompt(name, env)` `prompts/__init__.py:32`，注释见 `base.py:262-264`）：
1. `skillopt/envs/<env>/prompts/<name>.md`（env 专属）
2. `skillopt/prompts/<name>.md`（通用回退）

已内置 env（注册表 `scripts/train.py:_register_builtins`）：alfworld / searchqa / livemathematicianbench / spreadsheetbench / docvqa / officeqa / mmrb / mathverse / sealqa / swebench / babyvision（缺依赖则该 env 静默跳过）。

> 加新 benchmark：复制 `skillopt/envs/_template/`，详见官方 `docs/guide/new-benchmark.md` + 计划中的 `6_env_adapter_contract.md`。

---

## 8. 文件交叉引用索引 / Cross-reference index

按“我想改 X，去哪”组织：

| 我想… | 去这里 |
|---|---|
| 改训练主循环 / epoch 逻辑 | `engine/trainer.py:597`（+ 文档3） |
| 改 rollout 怎么跑任务 | `envs/alfworld/rollout.py:129`、`adapter.py:320`（+ 文档1=本文 §7） |
| 改 analyst 怎么分析轨迹 | `gradient/reflect.py:256/366`、prompt `envs/alfworld/prompts/analyst_*.md` |
| 改 patch 怎么合并/选择/应用 | `aggregate.py:143` / `clip.py:25` / `skill.py:165` |
| 改 gate 接受逻辑 | `evaluation/gate.py:76` |
| 改/加模型后端 | `model/router.py` + `model/<backend>.py`（+ 计划中的文档5） |
| 改配置项 | `config.py:185`（扁平化映射！见文档2 §10） |
| 看数据结构字段 | 文档2 + `types.py` |
| 查产物在哪 | 文档2 §9 |
| 查改动会不会出 bug | 文档4（红线汇总） |

---

## 9. 命名提醒 / Naming gotcha

代码里 **ReflACT** = SkillOpt 的训练循环内部代号（`ReflACTTrainer`、各 docstring）。二者是**同一套东西**，不是两个系统。环境变量也多用 `REFLACT_` 前缀（如 `REFLACT_MODEL_BACKEND` `router.py:11`）。

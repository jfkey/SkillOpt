# SkillOpt 内部文档 6 — 环境适配器契约（EnvAdapter Contract）

> **目的 / Purpose**：把 `EnvAdapter` 抽象接口逐个方法讲清楚，配 ALFWorld（模拟器型）和 template/officeqa（数据集型）两个对照，讲明白“加一个新 benchmark 要实现什么、trainer 在哪一步调你的哪个方法”。改 ALFWorld、加新 benchmark 时看这里。
>
> 配套：trainer 何时调这些方法见 [`3_pipeline_stages.md`](3_pipeline_stages.md)；数据结构见 [`2_data_contracts.md`](2_data_contracts.md)；官方手册 `docs/guide/new-benchmark.md`。
> 锚点 `file:line` 基于撰写时 HEAD。接口定义全在 `skillopt/envs/base.py`。

---

## 0. 一句话 / One-liner

> trainer 完全**环境无关**：它只认 `EnvAdapter`（`envs/base.py:37`）这一个接口。接新 benchmark = 实现一个 `EnvAdapter` 子类（5 个必选方法）+ 一个 `DataLoader`（或无）+ 注册 + 配置。rollout/reflect 的“怎么做任务、怎么分析”全在你的适配器里，6 阶段循环复用。

---

## 1. 两种环境原型 / Two archetypes

| 维度 | 模拟器型 Simulator-backed | 数据集型 Dataset-backed |
|---|---|---|
| 代表 | ALFWorld | SearchQA / OfficeQA / SpreadsheetBench / template |
| 有静态数据集? | 否（动态开模拟器） | 是（split_dir 下 train/val/test 的 json/jsonl） |
| dataloader | 自定义（`ALFWorldDataLoader`） | `SplitDataLoader` 子类 |
| `build_env_from_batch` 返回 | 惰性句柄 `ALFWorldBatchRun`（`adapter.py:297`） | 直接透传 item list（`env_template.py:81-83`） |
| rollout 输入 | 分块开/关模拟器 | 遍历 items 调 target |
| `get_dataloader()` | 返回自定义 loader | 返回 `SplitDataLoader` |

> 选型决定 `build_env_from_batch` / `rollout` 的写法，其余接口一致。

---

## 2. 必选方法 / Required (abstract) methods

5 个 `@abstractmethod`，不实现无法实例化：

| 方法 | 作用 | trainer 调用点 | 契约 |
|---|---|---|---|
| `build_train_env(batch_size, seed, **kw)` | 建训练 env 句柄 | `trainer.py:1103`（无 dataloader 路径） | 返回能传给 `rollout` 的对象 | `base.py:187` |
| `build_eval_env(env_num, split, seed, **kw)` | 建评估 env 句柄 | `trainer.py:614`（`_build_eval_env`） | split ∈ valid_seen/valid_unseen/… | `base.py:197` |
| `rollout(env, skill_content, out_dir, **kw)` | ① 跑任务 | `trainer.py:1122/1434` | 返回 `list[RolloutResult-dict]`，**必须含 `id`/`hard`/`soft`** | `base.py:216` |
| `reflect(results, skill_content, out_dir, **kw)` | ② 反思 | `trainer.py:1138` | 返回 `list[dict\|None]`（RawPatch），None 会被过滤 | `base.py:234` |
| `get_task_types()` | 任务类型清单 | 分桶统计 `trainer.py:447` | `list[str]` | `base.py:254` |

### rollout 契约（最关键）
- 返回 dict **至少** `id`(str)/`hard`(0/1)/`soft`(float[0,1])，其余进 `RolloutResult.extras`（见文档2 §1）。
- 🔴 `id` 必须与 `predictions/<id>/conversation.json` 目录名一致，否则 reflect 找不到轨迹。
- 🔴 自己实现 resume：out_dir 下 `results.jsonl` 存在就读回跳过（参考 `adapter.py:331-340`）。
- 把 skill 注入 target 的 system/user prompt（ALFWorld：`_build_skill_prompt` `rollout.py:50`；template：用作 system message）。

### reflect 契约
- 多数 env **直接委托** `run_minibatch_reflect`（`gradient/reflect.py:472`），传入 `get_error_minibatch_prompt()`/`get_success_minibatch_prompt()`（见 §4）。ALFWorld：`adapter.py:441-455`；template：`env_template.py:156-177`（注释里给了完整调用）。
- 返回的 dict 须含 `patch.{edits}` + `source_type`（`run_minibatch_reflect` 已自动处理 source_type 注入）。

---

## 3. 可选钩子 / Optional hooks（有默认实现）

| 方法 | 默认行为 | 何时重写 | `base.py` |
|---|---|---|---|
| `setup(cfg)` | 存 `self._cfg` | 需要一次性初始化（载数据、建 split）；记得 `super().setup(cfg)` + `dataloader.setup(cfg)` | `:46` |
| `get_dataloader()` | 返回 None | 有数据集时返回 loader（让 trainer 走 dataloader 路径） | `:54` |
| `requires_ray()` | False | 需要 Ray 并行 | `:58` |
| `build_env_from_batch(batch)` | 按 phase 路由到 build_train/eval_env | 想直接吃 BatchSpec（ALFWorld/template 都重写了） | `:171` |
| `build_reference_text(item)` | 取 `item["reference_text"]` | 有隐藏参考答案/计划（ALFWorld 重写成 PDDL 计划，`adapter.py:196`） | `:62` |
| `get_reference_metadata(item)` | reference 前 400 字 | 自定义参考预览 | `:66` |
| `attach_reference_context(results, items)` | 按 id 把 reference 贴到 result | 一般不用动 | `:76` |
| `select_representative_items(...)` | 按成败+类型分层抽样 | 一般不用动 | `:101` |
| `get_error/success_minibatch_prompt()` | 按 update_mode 加载 analyst prompt | 想完全自定义 analyst | `:285/298` |

---

## 4. Prompt 加载机制 / Prompt resolution

```
_env_name（从模块路径推导）  base.py:268
   "skillopt.envs.alfworld.adapter" → "alfworld"
_load_env_prompt(name)        base.py:278
   load_prompt(name, env=_env_name)  优先 envs/<env>/prompts/<name>.md
                                     回退 skillopt/prompts/<name>.md
get_error_minibatch_prompt()  base.py:285（按 skill_update_mode 选 analyst_error[_rewrite/_full_rewrite]）
get_success_minibatch_prompt() base.py:298
```

- 🔵 **零配置 fallback**：env 没提供专属 prompt 时自动用通用版（`skillopt/prompts/analyst_*.md`）。ALFWorld 提供了专属 `envs/alfworld/prompts/analyst_error.md` 所以用它。
- 🔵 `_env_name` 依赖**模块路径结构** `skillopt.envs.<name>.adapter`——目录命名必须规范，否则推导失败、加载不到 env 专属 prompt。
- update_mode 决定加载哪个变体（patch→`analyst_error`，rewrite→`analyst_error_rewrite`，full_rewrite→`analyst_error_full_rewrite`）。

---

## 5. DataLoader 契约 / Data loader

数据集型 env 用 `SplitDataLoader`（`datasets/base.py:225`）；模拟器型可自定义或返回 None。

### 层次
```
BaseDataLoader（抽象）                              datasets/base.py:72
   build_train_batch / build_eval_batch（abstract） :134/137
   make_base_seeds / shuffle_epoch_seeds / plan_train_epoch（确定性种子） :97/104/111
└─ SplitDataLoader（数据集型）                        datasets/base.py:225
      load_split_items(split_path)（你要实现）        loader_template.py:52
      load_raw_items(data_path)（仅 ratio 模式需要）   loader_template.py:86
```

### split 命名映射 ⚠️
trainer 用的 split 名 → 磁盘目录名（`_SPLIT_ALIAS` `datasets/base.py:154`）：
| trainer split | 磁盘目录 | 用途 |
|---|---|---|
| `train` | `train/` | 训练 rollout |
| `valid_seen` / `selection` | `val/` | **selection 集**（gate 验证） |
| `valid_unseen` / `test` | `test/` | 最终 held-out 测试 |

- 标准 split 目录：`train/` `val/` `test/`（`SPLIT_NAMES` `datasets/base.py:151`）。
- 两种 split_mode：`split_dir`（用现成目录）/ `ratio`（从 `data_path` 按 `2:1:7` 等比例确定性切分，`_compute_split_counts` `datasets/base.py:209`）。
- item 唯一硬要求：`id`(str)（`loader_template.py:21-34`），其余字段按你的 rollout 需要加。

---

## 6. 注册与选择 / Registration & selection

```
scripts/train.py:_register_builtins()        train.py:47
   try: from skillopt.envs.<name>.adapter import <Adapter>
        _ENV_REGISTRY["<name>"] = <Adapter>
   except ImportError: pass        # 缺依赖 → 静默跳过该 env
get_adapter(cfg)                              train.py:106
   env_name = cfg.get("env")  → _ENV_REGISTRY[env_name]
   inspect __init__ 签名，只传它接受的 kwargs  train.py:114+（防 cfg 多余键报错）
```

- 🔵 没有 `BENCHMARK_REGISTRY` 全局字典；**唯一注册表是 `_ENV_REGISTRY`（在 train.py 里）**，懒加载 + `try/except ImportError`。
- `cfg["env"]`（YAML `env.name`）选哪个 adapter。
- `get_adapter` 用 `inspect.signature` 过滤 kwargs，所以 adapter `__init__` 参数名要和扁平化后的 config 键对得上（如 `minibatch_size`/`edit_budget`/`analyst_workers`/`workers`/`max_completion_tokens`/`split_dir`…）。

---

## 7. 加新 benchmark 全流程 / Add a benchmark（照 README 走）

来自 `skillopt/envs/_template/README.md`：

1. **复制目录**：`cp -r skillopt/envs/_template skillopt/envs/your_benchmark`
2. **重命名文件+类**：`env_template.py→adapter.py`、`loader_template.py→loader.py`；`TemplateBenchmarkEnv→YourBenchmarkAdapter`、`TemplateBenchmarkLoader→YourBenchmarkLoader`；修 adapter 里的交叉 import。
3. **实现 TODO**：`adapter.py:rollout`（真实跑任务）+ `loader.py:_normalize_item`（item 规整）;想要真反思就解开 `reflect` 里的 `run_minibatch_reflect` 块。
4. **注册**：在 `scripts/train.py:_register_builtins()` 加 `try/except ImportError` 映射 `"your_benchmark" → YourBenchmarkAdapter`。
5. **配置**：建 `configs/your_benchmark/default.yaml`（从 `config_template.yaml` 起步）。`_base_` 是**字符串路径**不是 list。
6. **准备数据**：放成 `split_dir/{train,val,test}/*.json[l]`，或用 `ratio` 模式从单文件切（见文档 `0_start.md` 的物化套路）。
7. **冒烟测试**：用 `0_start.md` 的最小命令(`--train_size 1 --batch_size 1 ...`) 跑通 end-to-end。

> 已内置 env：alfworld / searchqa / livemathematicianbench / spreadsheetbench / docvqa / officeqa / mmrb / mathverse / sealqa / swebench / babyvision。**最佳抄作业对象**：数据集型看 `officeqa/`，模拟器型看 `alfworld/`。

---

## 8. ALFWorld 作为对照 / Worked example（模拟器型要点）

| 接口 | ALFWorld 实现 | 特别之处 |
|---|---|---|
| `build_env_from_batch` | 返回惰性 `ALFWorldBatchRun`（`adapter.py:297`） | 不开模拟器，只存清单（防 OOM） |
| `rollout` | `_run_batch` 分块开/关模拟器（`adapter.py:371`） | 每块 `build_alfworld_env`→`run_alfworld_batch`→`_close_env` |
| `reflect` | 委托 `run_minibatch_reflect`（`adapter.py:441`） | 用 env 专属 analyst prompt |
| `build_reference_text` | 从 `traj_data.json` 抽 PDDL 计划/人类步骤（`adapter.py:196`） | 给 analyst 隐藏参考 |
| `get_dataloader` | `ALFWorldDataLoader`（非 SplitDataLoader） | gamefile 路径而非 json 数据集 |
| `requires_ray` | False（`adapter.py:294`） | — |

---

## 附：加/改 env 自检 / Checklist
- [ ] 5 个必选方法都实现了？rollout 返回含 `id`/`hard`/`soft`？
- [ ] rollout 的 `id` 与 conversation 目录名一致？自己做了 results.jsonl resume？
- [ ] reflect 委托 `run_minibatch_reflect` 并传了 analyst prompt？
- [ ] 目录命名 `skillopt.envs.<name>.adapter`（否则 prompt 自动加载失效）？
- [ ] 在 `_register_builtins()` 注册（try/except ImportError）？
- [ ] config `env.name` 对、`_base_` 是字符串、adapter `__init__` 参数名与扁平 config 键对齐？
- [ ] split 目录 `train/val/test`、split 名映射（valid_seen→val, valid_unseen→test）理解正确？
- [ ] 跑通最小冒烟命令？

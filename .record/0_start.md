# SkillOpt 复现记录 — 0. 起步（环境 / GPT 模型服务 / 数据集）

> 总体架构提醒：SkillOpt 是「**optimizer 模型出编辑 + target 模型跑任务**」两个角色，
> 通过 OpenAI 兼容接口调用模型；它本身**不训练权重**，训练的是 skill 文档。
> 本方案用 **Azure OpenAI 上的 GPT 模型**（`gpt-4.1` / `gpt-4.1-mini` / `gpt-4o` / `gpt-4o-mini` …）
> 当 optimizer + target。模型在云端，所以**只需要一个 conda env（`skillopt`），不需要本地 GPU、不需要 vLLM**。
> SkillOpt 默认 backend 就是 `azure_openai`，调用走 HTTP，连不上时只需检查 endpoint / key / 部署名三件事。

---

## 1. 环境配置（单个 conda env）

```bash
conda create -n skillopt python=3.10
conda activate skillopt

cd /data3/liujunfeng/code/SkillOpt
pip install -e .                 # 核心依赖（pyproject.toml 的 [project].dependencies）
```

核心依赖里已经包含 `openai`、`azure-identity`、`azure-core`，**跑 GPT 模型不需要任何 extra**。

可选依赖（`pyproject.toml` 的 `[project.optional-dependencies]`，按需补装，pip 只补缺的、不会重装）：

```bash
pip install -e ".[docs,webui]"   # 文档站 + WebUI 面板（extra 之间用逗号、不要空格）
```

可选项 key：`alfworld` / `claude` / `qwen` / `docs` / `webui` / `dev` / `all`。

> ⚠️ 用 GPT 时**不要**装 `[qwen]`（那是给本地 vLLM 用的，依赖很重）；也不需要 `[claude]`。
> `requirements.txt` 只是给人看的镜像清单，**真正生效的是 `pyproject.toml`**。

---

## 2. 配置 Azure OpenAI（GPT 模型服务）

### 2.1 你给的 Python 片段 → SkillOpt 的对应关系

你提供的连通性测试脚本里，关键字段和 SkillOpt 的配置项一一对应：

| 你脚本里的变量 | 取值（示例） | SkillOpt 对应 | 说明 |
|---|---|---|---|
| `endpoint` | `https://hellotaoli.openai.azure.com/` | `AZURE_OPENAI_ENDPOINT` | 资源根地址，**末尾带 `/`** |
| `subscription_key` | `<your-api-key>` | `AZURE_OPENAI_API_KEY` | API Key |
| `api_version` | `2024-12-01-preview` | `AZURE_OPENAI_API_VERSION` | 保持和门户一致 |
| `deployment` | `gpt-4.1` | `--optimizer_model` / `--target_model` | **是「部署名」不是模型名**，要和 Azure 门户里建的 deployment 完全一致 |
| `model` / `model_name` | `gpt-4.1` | 同上 | 仅展示用，SkillOpt 只认 deployment 名 |

> 🔑 **最大的坑：认证方式。**
> SkillOpt 的 `azure_openai` backend 默认 `AUTH_MODE=azure_cli`（走 `az login` 拿 token）。
> 用 **API Key** 必须显式设 `AZURE_OPENAI_AUTH_MODE=api_key`，否则它会忽略你的 key 去找 Azure CLI 凭据而报错。

### 2.2 用 `.env` 配置（推荐）

仓库根目录有 `.env.example`，复制成 `.env` 填好即可：

```bash
cp .env.example .env
```

`.env` 内容（对应你的脚本）：

```bash
export AZURE_OPENAI_ENDPOINT=https://hellotaoli.openai.azure.com/
export AZURE_OPENAI_API_VERSION=2024-12-01-preview
export AZURE_OPENAI_API_KEY=<把-your-api-key-填这里>
export AZURE_OPENAI_AUTH_MODE=api_key      # ← 用 key 必加，否则默认走 azure_cli
```

加载到当前 shell：

```bash
set -a; source .env; set +a
```

> 其它认证方式（在 Azure VM 上更省事）：
> - `AZURE_OPENAI_AUTH_MODE=azure_cli`（默认）：先 `az login`，不用填 key；
> - `AZURE_OPENAI_AUTH_MODE=managed_identity` + `AZURE_OPENAI_MANAGED_IDENTITY_CLIENT_ID=...`。
> 强 optimizer + 弱 target 想用不同部署/资源时，用 per-role 前缀：
> `OPTIMIZER_AZURE_OPENAI_*` / `TARGET_AZURE_OPENAI_*`（endpoint / api_key / api_version / auth_mode 都可单独覆盖）。

### 2.3 先跑通连通性（基于你给的脚本，改成读环境变量）

把你给的片段改成从环境变量取值，**别把 key 硬编码进文件**，存成 `.record/check_azure.py`：

```python
import os
from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
)

deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")  # = 门户里的部署名

resp = client.chat.completions.create(
    model=deployment,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "I am going to Paris, what should I see?"},
    ],
    max_completion_tokens=512,   # SkillOpt 也用 max_completion_tokens（新参数）
    temperature=1.0,
)
print(resp.choices[0].message.content)
```

运行（确认能打印出回答，再去跑训练）：

```bash
conda activate skillopt
set -a; source .env; set +a
AZURE_OPENAI_DEPLOYMENT=gpt-4.1 python .record/check_azure.py
```

> 说明：你原片段里的 `frequency_penalty` / `presence_penalty` / `top_p` 都可去掉（默认即可）；
> `max_completion_tokens=13107` 调小到几百块测试更快。换 `gpt-4o` / `gpt-4o-mini` / `gpt-4.1-mini`
> 时，**只改 `deployment` 这一个值**（前提是 Azure 门户里建了同名部署）。

---

## 3. 数据集下载与处理

> ⚠️ **核心坑**：`data/` 目录里**只有 ID 清单（manifest），没有真实题目**。
> 而且各 config 里 `split_dir` 指向的目录名（如 `data/searchqa_split`）
> 和仓库里实际存在的 manifest 目录名（`data/searchqa_id_split`）**并不一致**，
> 需要你**物化（materialize）原始数据**后，把 `--split_dir` 指到物化结果。
> 详细字段对应见 `data/README.md`。

### 通用流程（每个 benchmark 都一样）

1. 从 manifest 目录（`data/<bench>_*_split/{train,val,test}/items.json`）读出 ID 列表；
2. 从对应的 HF 原始数据集下载原始样本（建议设 `export HF_ENDPOINT=https://hf-mirror.com` 走镜像）；
3. 按 lookup key 把 ID 匹配到原始样本，补齐环境需要的字段（question/answer/...）；
4. 把物化后的 train/val/test 写到 config 里 `split_dir` 指定的路径。

### 各 benchmark 速查（config 实际路径 vs manifest 目录）

| Benchmark | config 的 `split_dir` | 仓库 manifest 目录 | 原始数据源 | 必须物化的字段 / 额外资源 |
|---|---|---|---|---|
| **SearchQA** ✅ 推荐首选 | `data/searchqa_split` | `data/searchqa_id_split` | HF `lucadiliello/searchqa`（按 `id`=`key` 匹配） | `question` / `context` / `answers` |
| **LiveMathematicianBench** | `data/livemathematicianbench_split` | `data/livemathematicianbench_id_split` | HF `LiveMathematicianBench/...`，ID 格式 `<month>:<no>` | `question`/`choices`/`correct_choice`/`theorem_type`/`theorem`/`sketch`/`paper_link` |
| **DocVQA** | `data/docvqa/splits` | `data/docvqa_id_split` | HF `lmms-lab/DocVQA` validation 的 10% 子集 | `question`/`answer`(或`ground_truth`)/本地 `image_path` |
| **OfficeQA** | `data/officeqa_split`（+ `data_dirs: data/officeqa_docs_official`） | `data/officeqa_id_split` | HF `databricks/officeqa`（**gated，需授权**），`officeqa_full.csv` 按 `uid` 匹配 | `question`/`ground_truth`，并放好 source docs |
| **SpreadsheetBench** | `data/spreadsheetbench_split`（+ `data_root: data/spreadsheetbench_verified_400`） | `data/spreadsheetbench_id_split` | HF `KAKA22/SpreadsheetBench` Verified 400 | 按 `id`/`spreadsheet_path` 匹配，并把表格目录放到 `data/spreadsheetbench_verified_400` |
| **ALFWorld** ✅ 唯一可直接用 | `data/alfworld_path_split` | `data/alfworld_path_split`（同名） | `alfworld-download` 下 `json_2.1.1` | 无需物化，但要 `export ALFWORLD_DATA=<含 json_2.1.1 的根目录>` |

> 说明：
> - **SearchQA 最适合第一个跑通**：纯文本单轮 QA（`max_turns: 1`），物化最简单，只要三字段。
> - **ALFWorld** 的 manifest 能直接当 `--split_dir`，但要先 `pip install -e ".[alfworld]"` + `alfworld-download` + 设 `$ALFWORLD_DATA`；它是具身多步任务（`max_steps: 50`），最重。
> - **OfficeQA** 的原始 CSV 在 HF 上是 **gated（门控）**，需要先在 HF 上申请访问权限。
> - 物化脚本可以自己写：读 `items.json` → 拉 HF 数据 → 按 key join → 写出 `train/val/test`（每个含环境 `dataloader.py` 期望的字段）。

### SearchQA 物化最小示例（HF → 按 manifest join → 写 split_dir）

```bash
conda activate skillopt
export HF_ENDPOINT=https://hf-mirror.com      # 国内走镜像，可选
pip install datasets                          # 物化脚本需要，核心依赖里没带
```

```python
# .record/materialize_searchqa.py
import json, os
from pathlib import Path
from datasets import load_dataset

SRC = Path("data/searchqa_id_split")   # manifest（只有 id）
DST = Path("data/searchqa_split")      # ← config 里 split_dir 指向这里
ds = load_dataset("lucadiliello/searchqa", split="train")
by_id = {str(r["key"]): r for r in ds}   # 按 key 建索引

for split in ["train", "val", "test"]:
    items = json.loads((SRC / split / "items.json").read_text())
    out = []
    for it in items:
        rec = by_id.get(str(it["id"]))
        if rec is None:
            continue
        out.append({
            "id": it["id"],
            "question": rec["question"],
            "context": rec["context"],
            "answers": rec["answers"],
        })
    (DST / split).mkdir(parents=True, exist_ok=True)
    (DST / split / "items.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(split, len(out))
```

> 字段名以 `data/README.md` 与 `skillopt/envs/searchqa/dataloader.py` 实际期望为准，跑通后按报错微调。

---

## 4. 跑通的最小命令（SearchQA + GPT-4.1）

```bash
conda activate skillopt
set -a; source .env; set +a               # 载入 AZURE_OPENAI_* 与 AUTH_MODE=api_key

python scripts/train.py \
    --config configs/searchqa/default.yaml \
    --optimizer_model gpt-4.1 \
    --target_model gpt-4.1 \
    --reasoning_effort "" \                # ← 关键！见下方说明
    --split_dir data/searchqa_split        # ← 指向你物化好的目录
# 产物：outputs/skillopt_searchqa_<model>_<ts>/best_skill.md + 测试分数
```

> 也可以用封装脚本（内部就是上面这条命令，模型走环境变量）：
> ```bash
> OPTIMIZER_MODEL=gpt-4.1 TARGET_MODEL=gpt-4.1 \
>   bash scripts/run_searchqa.sh --reasoning_effort "" --split_dir data/searchqa_split
> ```

> ⚠️ **关键坑：`reasoning_effort`。**
> `configs/_base_/default.yaml` 和 SearchQA 配置默认 `reasoning_effort: medium`，
> 这是给 **推理型模型（gpt-5.x / o 系列）** 用的。
> `gpt-4.1` / `gpt-4.1-mini` / `gpt-4o` / `gpt-4o-mini` **不是推理模型**，
> 传 `reasoning_effort` 会被 API 以 400 拒绝。
> 所以用这些 GPT 模型时务必加 `--reasoning_effort ""` 关掉它（`rewrite_reasoning_effort` 基线已是 `""`，不用动）。

> 默认 backend 就是 `azure_openai`，**不需要 `--backend`**；
> `--optimizer_model` / `--target_model` 填的就是 Azure **部署名**。
> 想 optimizer 用强模型、target 用弱模型，分别填即可，例如
> `--optimizer_model gpt-4.1 --target_model gpt-4o-mini`。

---

## 5. ALFWorld 专项：安装 + 数据 + 排错

> ALFWorld 是唯一 manifest 能直接当 `--split_dir` 的 benchmark（loader 直接读 `gamefile`/`task_type`），
> 但实测会连撞 3 个坑。按下面顺序处理即可跑通。

### 5.1 安装：**不要装 `alfworld[full]`**

SkillOpt 的 ALFWorld 只用 **TextWorld 文本环境**（`vendor/config_tw.yaml` → `type: AlfredTWEnv`），
不需要 `[full]` 带的视觉栈（`ai2thor`/`visdom`/`torch`/`opencv`）。

- ❌ `pip install alfworld[full]` 会去编译 `visdom`，而 `visdom` 的 `setup.py` 顶部 `import pkg_resources`，
  在 **setuptools ≥ 81**（本机 82）下 `pkg_resources` 已被移除 → 编译沙箱报 `No module named 'pkg_resources'`。
- ✅ 正确做法：只装基础包 + 补一个漏掉的 `omegaconf`：
  ```bash
  pip install -e ".[alfworld]"     # = alfworld + gymnasium
  pip install omegaconf            # rollout.py 需要，核心依赖/extra 都没带 → 否则 ModuleNotFoundError
  ```
- （万一别处又报 `pkg_resources` 缺失，回退 setuptools 即可：`pip install "setuptools<81"`。）

### 5.2 下载数据并设 `$ALFWORLD_DATA`

```bash
export ALFWORLD_DATA="${HOME}/.cache/alfworld"
alfworld-download        # 拉 json_2.1.1 + pddl + tw-pddl，约几百 MB；它本身不需要 [full]
```

### 5.3 ⚠️ Bug C：`Unable to find game '.../SkillOpt/json_2.1.1/.../game.tw-pddl'`

**现象**：数据明明在 `$ALFWORLD_DATA/json_2.1.1/`，env 也成功 `Initializing AlfredTWEnv... 494 games`，
但一到 `reset()` 加载具体题目就报找不到 `game.tw-pddl`，且路径前缀是**项目根**而不是 `$ALFWORLD_DATA`。

**根因**：ALFWorld 内部有**两套路径**，`$ALFWORLD_DATA` 只覆盖其中一套：

| 阶段 | 用的路径 | 是否展开 `$ALFWORLD_DATA` |
|---|---|---|
| 建环境 / 扫描游戏列表 | `vendor/config_tw.yaml` 里的 `$ALFWORLD_DATA/json_2.1.1/...` | ✅ alfworld 库 `alfred_tw_env.py` 用 `os.path.expandvars` 展开 |
| `reset()` 加载某道题 | split 清单 `items.json` 的 `gamefile`（相对路径 `json_2.1.1/...`） | ❌ loader 原样丢给 `textworld.load()`，**不拼 `$ALFWORLD_DATA`、不 expandvars** |

第二套相对路径只能按**当前工作目录**解析。`run_alfworld.sh` 会 `cd` 到项目根，
于是去找 `<项目根>/json_2.1.1/...` → 而数据在 `~/.cache/alfworld/json_2.1.1/...` → 不一致 → 报错。
（`data/README.md` 也写了 "gamefile … must be expanded before use"，但 loader 从没做这个展开——仓库毛刺。）

**修复（三选一，推荐 ①）**：

```bash
# ① 软链接（最小改动、可逆）：让相对路径在 cwd 下解析得到
ln -s "$ALFWORLD_DATA/json_2.1.1" /data3/liujunfeng/code/SkillOpt/json_2.1.1
#   撤销：rm /data3/liujunfeng/code/SkillOpt/json_2.1.1
#   换 ALFWORLD_DATA 位置时记得重建软链

# ② 让 ALFWORLD_DATA 直接等于项目根，再下载（数据落到项目根，天然命中；缺点：几百 MB 进仓库目录）
# export ALFWORLD_DATA=/data3/liujunfeng/code/SkillOpt && alfworld-download

# ③ 物化 split（最"正"、要写代码）：把每个 gamefile 用 os.path.expandvars("$ALFWORLD_DATA/"+gamefile)
#    改成绝对路径写到新 split_dir，和 SearchQA 物化同一套路
```

### 5.4 跑通命令

```bash
cd /data3/liujunfeng/code/SkillOpt
set -a; source .env; set +a               # 载入 AZURE_OPENAI_* + AUTH_MODE=api_key
OPTIMIZER_MODEL=gpt-4.1 TARGET_MODEL=gpt-4.1 \
  bash scripts/run_alfworld.sh --reasoning_effort ""   # ← 仍要关 reasoning（gpt-4.1 非推理模型）
```

> 补充：`$ALFWORLD_DATA` **不经过 `train.py` 的命令行参数**，而是 `run_alfworld.sh` 里 `export` 后
> 由 python 子进程**继承环境变量**，最终在 alfworld 库内 `os.path.expandvars` 读 `os.environ['ALFWORLD_DATA']` 时才展开。

---

## 待办 / 下一步

- [ ] 跑 `.record/check_azure.py`，确认 gpt-4.1 部署能正常返回（endpoint / key / 部署名 / `AUTH_MODE=api_key` 四件事）
- [ ] 写并跑 SearchQA 物化脚本（HF 下载 + 按 manifest `id` join + 输出 `data/searchqa_split`）
- [ ] 跑通一次最小 SearchQA 训练（记得 `--reasoning_effort ""`），确认 6 阶段循环 + gate 正常
- [ ] 对比不同 target：`gpt-4o-mini` / `gpt-4.1-mini`（便宜）vs `gpt-4.1` / `gpt-4o`（强），看 skill 提升幅度
- [ ] ALFWorld：按 §5 装好（不装 `[full]` + `omegaconf` + `json_2.1.1` 软链），跑通基线 rollout
- [ ] 再切换到其它 benchmark（LiveMathematicianBench / DocVQA / SpreadsheetBench）

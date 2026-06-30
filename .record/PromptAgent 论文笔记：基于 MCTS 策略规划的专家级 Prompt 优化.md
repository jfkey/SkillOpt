# PromptAgent 论文笔记：基于 MCTS 策略规划的专家级 Prompt 优化

# PromptAgent: Strategic Planning with Language Models Enables Expert-level Prompt Optimization

> **会议**: ICLR 2024
> 

> **作者**: Xinyuan Wang, Zhen Wang, Chenxi Li, Fan Bai, Haotian Luo, Jiayou Zhang, Nebojsa Jojic, Eric Xing, Zhiting Hu
> 

> **机构**: UC San Diego, MBZUAI, Microsoft Research, CMU, Georgia Tech
> 

> **论文链接**: [https://arxiv.org/abs/2310.16427](https://arxiv.org/abs/2310.16427)
> 

> **代码仓库**: [https://github.com/maitrix-org/PromptAgent](https://github.com/maitrix-org/PromptAgent)
> 

---

## 一、TL;DR 一句话总结

**PromptAgent 把 prompt 优化建模成"策略规划问题"，用 MCTS（蒙特卡洛树搜索） + LLM 自我反思 (self-reflection) 自动迭代地把"普通人写的 prompt"打磨成"专家级 prompt"**，在 12 个任务上显著超越人工 prompt、CoT 和 APE 等 baseline。

![PromptAgent Header](https://github.com/maitrix-org/PromptAgent/raw/main/images/Header.png)

PromptAgent Header

---

## 二、问题背景与动机

### 2.1 什么是"专家级 prompt"？

在论文 Figure 1 中，作者用一个生物医学领域的疾病实体抽取（NCBI）任务作为对比：

| Prompt 类型 | 内容 | 输出 |
| --- | --- | --- |
| **普通用户 prompt** | "Extract the disease or condition from the sentence, if any is mentioned." | ❌ `c2 deficiency gene`（错误，把基因当成了疾病） |
| **采样法（APE）prompt** | "If any disease or condition is mentioned in the sentence, extract it." | ❌ 仍然错误 |
| **专家级 prompt** | 包含任务描述、领域知识、解决方案指导、异常处理、输出格式等多层结构 | ✅ `c2 deficiency` |

专家级 prompt 通常包含 6 类结构化元素：

- **Task Description** 任务描述
- **Domain Knowledge** 领域知识（例如 "avoid associated elements: inheritance patterns, genes or gene loci (like PAH)..."）
- **Solution Guidance** 解决方案指导
- **Exception Handling** 异常处理（例如 "the term 'locus' should be recognized as a genomic location, not a disease name"）
- **Priority & Emphasis** 优先级和强调
- **Output Formatting** 输出格式（例如 `{entity_1, entity_2, ...}`）

### 2.2 已有方法的不足

| 方法类别 | 代表 | 不足 |
| --- | --- | --- |
| Soft prompt tuning | Prompt Tuning, LoRA | 需要梯度，闭源 API LLM 用不了 |
| 离散 prompt 搜索 | AutoPrompt, RLPrompt, TEMPERA | 同上 |
| 迭代采样 | APE, GrIPS, ProTeGi | 启发式编辑/释义，**缺乏原则化的搜索策略**，容易陷入"普通 prompt 的局部变种" |
| LLM Agent | Auto-GPT 类 | 规划只用一次，**探索深度不够** |

> **核心痛点**：现有方法大多只在 prompt 空间做"漫无目的的局部采样"，没有像人类专家那样基于错误反馈"有针对性地迭代"，也没有"前瞻 + 回溯"的策略性规划能力。
> 

---

## 三、核心方法：把 Prompt 优化建模为 MDP

### 3.1 问题形式化

给定：

- 基座模型 $B$（base LLM，例如 GPT-3.5）
- 目标任务 $T$，训练样本 $(Q, A) = \{q_i, a_i\}_{i=1}^N$
- 初始 prompt $P_0$（例如 "Let's solve this problem step-by-step"）
- 评估指标 $R$（如准确率）

**优化目标**：

$P^* = \arg\max_{P \in S} \sum_i R(p_B(a_i | q_i, P))$

其中 $S$ 是自然语言 prompt 的空间（无限且不可枚举）。

### 3.2 MDP 四元组 $(S, A, T, r)$

PromptAgent 将这个搜索过程定义为马尔可夫决策过程：

| MDP 元素 | 在 PromptAgent 中的含义 |
| --- | --- |
| **State** $s_t$ | 当前版本的 prompt（自然语言文本） |
| **Action** $a_t$ | **Error feedback（错误反馈）**——而不是简单的"编辑/释义" |
| **Transition** $T$ | 由 optimizer LLM 根据 $(s_t, a_t)$ 生成新 prompt $s_{t+1}$ |
| **Reward** $r$ | 新 prompt 在 held-out 集上的性能（如准确率） |

**这里的关键创新**：action 不是"删词/换词"这种低层编辑操作，而是 **高层的错误诊断与改进建议**。这让搜索空间变得有意义、可解释。

### 3.3 State Transition：核心三步（论文 Figure 3b）

给定当前 prompt $s_t$，生成新 prompt $s_{t+1}$ 经历三步：

**Step 1 — Retrieve Errors from Base Model（错误收集）**

- 从训练集随机采样一个 batch（默认 batch size = 5）
- 用 base LLM (gpt-3.5-turbo) + 当前 prompt 做预测
- 收集预测错误的样本

**Step 2 — Generate Error Feedback as Action（生成错误反馈 = action）**

- 用 optimizer LLM (gpt-4) + **meta-prompt 1** 反思错误
- 输出错误反馈，例如：
    
    > "Ignoring Context and Detail — The model might be overlooking the details of the premise 'kids play in water coming up in streams out of a tiled floor', which directly implies the hypothesis."
    > 

**Step 3 — Update Prompt = New State（更新 prompt）**

- 用 optimizer LLM + **meta-prompt 2** + 之前的轨迹 + 当前 error feedback
- 生成新的 prompt $s_{t+1}$

> 💡 **本质**：这三步把"人类专家如何打磨 prompt"这一过程显式地形式化了 —— 收集错误 → 反思错误 → 修正 prompt。
> 

### 3.4 两个关键 Meta-Prompt（与代码逐字对齐）

> ⚠️ 复现提醒：以下为本仓库 `src/prompt_optim_agent/world_model/prompts/gradient_descent_prompts.py` 中的**实际模板原文**。代码里变量名是 `example_string`（不是 `error_string`）、`gradient`（不是 `error_feedback`）。

**Meta-prompt 1 `gradient_prompt_tempelate`** —— 生成 action（即 "gradient" / 错误反馈）：

```
I'm writing prompts for a language model designed for a task.

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{example_string}

For each wrong example, carefully examine each question and wrong answer step by step, provide comprehensive and different reasons why the prompt leads to the wrong answer. At last, based on all these reasons, summarize and list all the aspects that can improve the prompt.
```

**Meta-prompt 2 `optimize_prompt_tempelate(_single)`** —— 状态转移（生成新 prompt）：

```
I'm writing prompts for a language model designed for a task.

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{example_string}

Based on these errors, the problems with this prompt and the reasons are:
{gradient}

There are a list of former prompts including the current prompt, and each prompt is modified from its former prompts:
{trajectory_prompts}

Based on the above information, please write {steps_per_gradient} new prompts following these guidelines:
1. The new prompts should solve the current prompt's problems.
2. The new prompts should consider the list of prompts and evolve based on the current prompt.
3. Each new prompt should be wrapped with <START> and <END>.

The new prompts are:
```

> `num_new_prompts == 1` 时用 `_single` 版本（"write 1 new prompt"），否则用复数版本；新 prompt 用正则 `<START>(.*?)<END>` 抽取（`_clean_optim_response`）。

**喂给 base model 的前向 prompt**（`build_forward_prompts_completion`）：默认 `post_instruction=False` 时为
`{cur_prompt}\n{question}\n{answer_format_prompt}`；`post_instruction=True` 时为 `{question}\n{cur_prompt}`。
`answer_format_prompt` 由各任务自定义（如选择题用 `<answer>...</answer>`，NCBI 用 `{entity_1,entity_2,...}`）。

#### 3.4.1 全对例外：Ascend Gradient（论文未强调，但代码关键）

`gradient_descent.py` 中有一个重要分支：**当采样到的整个 batch 全部预测正确（`acc == 1`）时**，没有错误样本可反思，代码切换到 `_all_correct_exception`，改用 **ascend 模板**——对"正确样本"反思"为什么这个 prompt 答对了"，再据此继续强化 prompt（`ascend_gradient_prompt_tempelate` / `ascend_optimize_prompt_tempelate`）。复现时务必实现此分支，否则在简单 batch 上会崩溃或停滞。

---

## 四、MCTS 策略规划

![MCTS Tree](https://github.com/maitrix-org/PromptAgent/raw/main/images/mcts_00.jpg)

MCTS Tree

为什么用 MCTS？因为 prompt 空间巨大，纯贪心或 beam search 都做不到"前瞻 + 回溯"，而 MCTS 能在 **探索 (exploration)** 和 **利用 (exploitation)** 之间找平衡。

### 4.1 MCTS 四个核心操作

每次 MCTS 迭代依次执行：**Selection → Expansion → Simulation → Back-propagation**。

#### ① Selection（选择）

从根节点出发，每层用 **UCT 公式** 选择最有希望的子节点：

$a^*_t = \arg\max_{a'_t \in A(s_t)} \left( Q(s_t, a'_t) + c \cdot \sqrt{\frac{\ln N(s_t)}{N(\text{ch}(s_t, a'_t))}} \right)$

- 第一项 $Q(s_t, a'_t)$：利用（exploitation），即子节点的预期价值
- 第二项 $c sqrt{ln N / N_{text{child}}}$：探索（exploration），鼓励访问较少被访问的节点
- 论文中 $c = 2.5$

#### ② Expansion（扩展）

到达叶子节点后：

- 对训练集采样 `expand_width` 个 batch（默认 3 个）
- 每个 batch 跑一次"错误收集 + 反馈 + 更新 prompt"流程
- 每个 batch 生成 `num_samples` 个新 prompt（默认 1 个）
- 得到 `expand_width × num_samples` 个新子节点

#### ③ Simulation（模拟）

为了简化计算，论文里 **没有用传统 random rollout**，而是反复执行 expansion 操作，每次选择 reward 最高的子节点向下扩展，直到达到 `depth_limit` 或触发 early stopping。

**Early stopping 条件**（当深度 > 2 时）：

- $r < text{min_threshold}$（低于父节点和根节点 reward 的均值）→ 提前终止
- $r > text{max_threshold}$（已达到当前最大 reward）→ 鼓励缩短路径

#### ④ Back-propagation（反向传播）

从终止节点回溯到根节点，更新每个 state-action 对的 $Q$ 值：

$Q^*(s_t, a_t) = \frac{1}{M} \sum_{j=1}^{M} \left( \sum_{s' \in S^j_{s_t}, a' \in A^j_{a_t}} r(s', a') \right)$

即：**对所有从** $s_t$ **出发的未来轨迹的累积 reward 求平均**。

### 4.2 输出策略

12 次 MCTS 迭代后，从树里选择最终 prompt 的策略：

1. **找出"平均 reward 最高的路径"**
2. **在这条路径上选择"reward 最高的节点（prompt）"**

> 论文比较了多种输出策略，发现 (1)+(2) 这个组合在实验中效果最好。
> 

### 4.3 三种超参数组合

| 配置 | depth_limit | expand_width | num_samples |
| --- | --- | --- | --- |
| **Standard** | 8 | 3 | 1 |
| **Wide** | 6 | 3 | 2 |
| **Lite** | 4 | 3 | 1 |

每个任务都跑这三种配置然后选最优的。MCTS 迭代次数 $tau = 12$。

### 4.4 算法伪代码（Algorithm 1）

```python
def PromptAgent_MCTS(s_0, p_θ, r_θ, p_φ, d, L, τ, c):
    """
    s_0     : 初始 prompt (state)
    p_θ     : state transition function (LLM 优化器生成新 prompt)
    r_θ     : reward function (held-out 集上的性能)
    p_φ     : action generation function (LLM 生成 error feedback)
    d       : 每次扩展生成的 action 数
    L       : depth_limit
    τ       : 迭代次数
    c       : UCT 探索权重
    """
    for n in range(τ):
        # 1. SELECTION：从根开始，UCT 选择子节点直到叶子
        for t in range(L):
            if A(s_t) is not empty:
                a_t = argmax_{a in A(s_t)} [Q(s_t, a) + c * sqrt(ln N(s_t) / N(ch(s_t, a)))]
                s_{t+1} = ch(s_t, a_t)
                N(s_t) += 1
            else:
                # 2. EXPANSION + SIMULATION
                for i in range(d):
                    a_t^i ~ p_φ(a | s_t)              # 生成 error feedback
                    s_{t+1}^i ~ p_θ(s | s_t, a_t^i)   # 生成新 prompt
                    r_t^i = r_θ(s_t, a_t^i)           # 评估 reward
                # 选 reward 最高的扩展
                a_t = argmax r_t^i
                s_{t+1} = ch(s_t, a_t)
                
            if early_stopping(s_{t+1}):
                break
        
        # 3. BACK-PROPAGATION
        for t from T-1 down to 0:
            Q(s_t, a_t) ← update by Equation 2
    
    # 4. 输出：在 best avg reward path 上选 best reward 节点
    return best_node
```

---

## 五、代码结构与运行流程

### 5.1 仓库结构

```
PromptAgent/
├── datasets/              # 12 个任务的数据集
├── images/                # README 配图
├── src/
│   ├── main.py            # 入口，加载 yaml 配置
│   ├── search_algo/       # mcts.py, beam_search.py
│   ├── world_model/       # state transition + action generation
│   ├── language_model/    # openai / palm / hf_textgeneration 等
│   ├── tasks/             # base_task.py + 各任务实现
│   └── test.py            # 测试已优化 prompt 的性能
├── example_config.yaml    # 配置文件（重要！）
└── requirements.txt
```

### 5.2 快速运行

```bash
git clone https://github.com/XinyuanWangCS/PromptAgent.git   # 官方仓库（README 安装命令）
cd PromptAgent
conda create -n prompt_agent && conda activate prompt_agent
pip install -r requirements.txt

# 在 example_config.yaml 中填入 base/optim 两个模型的 OpenAI API key
python src/main.py --config_dir example_config.yaml
```

跑一个 `penguins_in_a_table` 任务约 **2 小时 / $5**（GPT-4 优化器 + GPT-3.5 base）。

### 5.3 配置文件关键字段（与 `example_config.yaml` 完全一致）

> ⚠️ 复现提醒：配置是**嵌套**的，且字段名与论文术语不同。常见错误：写成扁平的 `iteration_num` / `exploration_weight` / `num_samples` 都跑不起来。正确映射如下：
> - 论文 `num_samples`（每步生成的新 prompt 数）→ 代码 `world_model_setting.num_new_prompts`
> - 论文 UCT 探索权重 `c` → 代码 `search_setting.w_exp`
> - 搜索相关参数全部在 `search_setting` 下，前向/优化相关在 `world_model_setting` 下。

```yaml
# 基础设置
task_name: bigbench          # bigbench | ncbi | med_qa | cb | subj | trec | ...
search_algo: mcts            # mcts | beam_search
print_log: true
log_dir: ./logs/
init_prompt: |
  Answer questions about a table of penguins and their attributes.

task_setting:
  train_size: 70             # 错误采样用
  eval_size: 50              # reward（held-out）计算用
  test_size: 79              # >0 时优化结束后对选出的节点做测试
  seed: 42
  data_dir: ./datasets/penguins_in_a_table.json
  post_instruction: false    # false: prompt+question | true: question+prompt

base_model_setting:          # 被优化的目标模型（执行任务），温度 0
  model_type: openai         # openai | palm | hf_text2text | hf_textgeneration | ct_model | vllm
  model_name: gpt-4o-mini
  temperature: 0.0
  api_key: null              # openai/palm 必填，否则 main.py 校验报错
  device: null               # cuda | cpu | cuda:x（本地模型用）
  gpu_ids: null              # [0,1,...]（vllm 多卡用）
  model_path: null           # ct_model 需要

optim_model_setting:         # 优化器模型（生成 gradient + 新 prompt），温度 1.0
  model_type: openai
  model_name: gpt-4-turbo
  temperature: 1.0
  api_key: null

search_setting:
  iteration_num: 10          # MCTS 迭代次数（论文用 12）
  expand_width: 3            # 每个节点扩展的 batch 数（=分支数）
  depth_limit: 5             # MCTS 最大深度（论文 Standard 用 8）
  min_depth: 2               # 早停的最小深度
  w_exp: 2.5                 # UCT 探索权重 c
  beam_width: 3              # 仅 beam_search 用

world_model_setting:
  train_shuffle: true
  num_new_prompts: 1         # 每个优化步生成的新 prompt 数（beam search 建议 3）
  train_batch_size: 5        # 每个 batch 的样本数
```

> 注意 repo 默认值（`iteration_num=10`, `depth_limit=5`, base=`gpt-4o-mini`, optim=`gpt-4-turbo`）与论文实验（`iteration_num=12`, Standard 配置 `depth_limit=8`, base=`gpt-3.5-turbo`, optim=`gpt-4`）不同。**严格复现论文请改回论文超参**（见 4.3）。

### 5.4 自定义新任务的步骤（来自 README）

继承 `tasks/base_task.py` 中的基类，实现：

1. **`load_task_dataset` / `transform_format`** — 数据加载，建议拆分 train/test 存为 JSON
2. **`clean_labels` / `build_forward_prompts_completion`** — 把 question + options 拼接成模型输入
3. **`clean_response`** — 从模型回复中抽取最终答案
4. **`cal_correct`** — 比较预测与标签（用于 MCTS 错误采样）
5. **`cal_metric`** — 验证集指标（用于 MCTS reward）

```python
class CustomTask(BaseTask):
    def cal_correct(self, preds, labels):
        # 用于错误采样：返回每个样本对/错
        return [int(p == l) for p, l in zip(preds, labels)]
    
    def cal_metric(self, preds, labels):
        # 用于 MCTS reward：返回标量（如 accuracy）
        return sum(self.cal_correct(preds, labels)) / len(labels)
```

### 5.5 支持的模型类型

| `model_type` | 说明 |
| --- | --- |
| `openai` | OpenAI API（GPT-3.5/4） |
| `palm` | Google PaLM |
| `hf_textgeneration` | HuggingFace 文本生成模型（推荐 Mistral-7B-Instruct-v0.2） |
| `hf_text2text` | T5 类 |
| `ct_model` | 本地下载模型 |
| `vllm` | (2024.08 新增) 加速本地推理 |

---

## 六、实验与关键结果

### 6.1 任务覆盖（12 个）

| 域 | 任务 |
| --- | --- |
| **BIG-Bench Hard (BBH)** | Penguins, Geometry, Epistemic, Object Counting, Temporal, Causal Judgement |
| **领域专家任务** | NCBI（疾病实体抽取）, Biosses（生物医学句子相似度）, Med QA（医学问答） |
| **通用 NLU** | Subj（主观/客观）, TREC（问题分类）, CB（自然语言推理） |

### 6.2 与 Baselines 的对比

**BBH 平均提升**：相比 human prompt +28.9%，相比 CoT +9.5%，相比 APE +11.2%。

| 方法 | BBH 平均 | 领域专家平均 | 通用 NLU 平均 |
| --- | --- | --- | --- |
| Human (ZS) | 0.513 | 0.526 | 0.658 |
| CoT | 0.707 | 0.531 | 0.699 |
| GPT Agent | 0.561 | 0.406 | 0.543 |
| APE | 0.690 | 0.582 | 0.778 |
| **PromptAgent** | **0.802** | **0.655** | **0.868** |

### 6.3 跨模型迁移（Prompt Generalization）

**用 GPT-3.5 优化的 prompt 直接迁移到其他模型**：

| 目标模型 | 表现 |
| --- | --- |
| GPT-4（更强） | **进一步提升**，12/12 任务中 11 个超过 baselines |
| PaLM 2（更弱） | 整体下降，但 7/12 任务仍优于 baselines，尤其领域任务（如 NCBI）依然显著 |

> 💡 这说明：**专家级 prompt 的领域洞见可以无缝迁移到更强的 LLM 上**，复用价值高（每个任务只需优化一次）。
> 

### 6.4 搜索策略消融

| 搜索算法 | 平均准确率 |
| --- | --- |
| MC（一次性采样） | 0.635 |
| Beam Search | 0.697 |
| Greedy Search | 0.698 |
| **MCTS (PromptAgent)** | **0.754** |

**关键洞察**：贪心和 beam 都是"严格向前"的，无法回溯；MCTS 能"前瞻 + 回溯"，相对最强 baseline 提升 5.6%。

### 6.5 探索效率

PromptAgent 在 **更少的 prompt 探索数** 下达到更高准确率（论文 Figure 4a 中 PromptAgent 的点聚集在"左上角"）。

### 6.6 收敛性

论文 Figure 4b 展示了 Epistemic 任务的 reward 随 MCTS depth 增长的曲线：

- 前 2-3 层 reward 快速上升
- depth 3 之后趋于稳定
- 反映了 PromptAgent **学习过程稳定可控**

### 6.7 其他消融结论

| 超参数 | 最佳取值 | 结论 |
| --- | --- | --- |
| 迭代次数 | 12 | 8 太少 → 探索不足；16 太多 → 训练集过拟合 |
| 探索权重 c | 2.5 | c=1 只走一条路径；c=4 每次都开新路径，都不平衡 |
| Optimizer LLM | GPT-4 > GPT-3.5 | 但 GPT-3.5 当 optimizer 仍能显著超过初始 prompt，框架不强依赖最强 LLM |
| Base LLM | GPT-3.5 > PaLM 2 | PaLM 2 当 base 时性能更差，但 PromptAgent 仍能提升它 |

---

## 七、定性分析：NCBI 任务的优化轨迹

论文 Figure 5 展示了 PromptAgent 在 NCBI 任务上的完整优化轨迹，从人工 prompt 一步步进化为专家级 prompt：

**$s_0$（人工初始 prompt）** — F1 = 0.521

> "Extract the disease or condition from the sentence, if any is mentioned."
> 

↓ $a_0$: error feedback 指出"模型混淆了疾病和基因/缩写"

$s_1$ — F1 = 0.609

> 增加：区分疾病和相关因素（基因等），考虑缩写变体
> 

↓ $a_1$: error feedback 指出"locus 被错误识别为疾病"

$s_2$ — F1 = 0.622

> 增加：明确"locus 是基因组位置，不是疾病名称"
> 

↓ $a_2$: error feedback 指出"PAH 被错误识别为疾病（实为基因）"

**$s_3$（专家级 prompt）** — F1 = **0.645**

> 整合所有信息：明确排除遗传模式（autosomal dominant）、基因（PAH）、蛋白质、生物通路；考虑常见缩写；locus 是基因组位置；输出格式 `{entity_1, entity_2, ...}`
> 

> 🎯 **核心观察**：每一步都通过"错误反馈"显式地注入领域知识，最终 prompt 是所有 error feedback 沉淀下来的"知识晶体"。
> 

### 优化前后的 prompt 长度对比（NCBI）

```
Human (52字)：
Extract the disease or condition from the sentence, if any is mentioned.

PromptAgent (~120 词)：
You're tasked with extracting diseases or conditions from the given sentence, 
remember to be cautious and avoid incorporating any associated elements such 
as inheritance patterns (like autosomal dominant), genes or gene loci (like 
PAH), proteins, or biological pathways. The task does not entail making 
assumptions or inferences about the disease names based on other advanced 
biological terms in the context. Consider both specific diseases and broader 
categories, and remember diseases and conditions can also appear as common 
abbreviations or variations. Provide the identified diseases or conditions 
in this format: {entity_1, entity_2,....}. If there are no diseases or 
conditions present, output an empty list in this form: {}. Note that the 
term 'locus' should be recognized as a genomic location and not a disease name.
```

---

## 八、与 TEMPERA 等同类方法的对比

| 维度 | TEMPERA (ICLR 2023) | APE (ICLR 2023) | **PromptAgent (ICLR 2024)** |
| --- | --- | --- | --- |
| **优化时机** | Test-time（每个 query 实时优化） | 全局，一次性优化 | 全局，一次性优化 |
| **Action 空间** | 编辑 instruction / few-shot / verbalizer | Monte Carlo 采样新 prompt | **Error feedback（高层反思）** |
| **搜索算法** | RL（PPO） | MC 搜索 | **MCTS（带前瞻 + 回溯）** |
| **Reward 信号** | 编辑前后的 score 差 | 评估集 accuracy | 评估集 accuracy |
| **可解释性** | 中（自然语言 prompt） | 中 | **高（详细领域知识）** |
| **是否需要训练** | 是（RL agent） | 否 | 否（只用 LLM 推理） |
| **是否针对每个 query** | 是 | 否 | 否 |

### 与 TEMPERA 的本质差异

- **TEMPERA** 的核心是"测试时为每个输入动态编辑 prompt"，是一种**实时适配**思路
- **PromptAgent** 的核心是"训练时为整个任务优化一个高质量的全局 prompt"，是一种**离线提炼**思路
- TEMPERA 强调 **query 适应性**，PromptAgent 强调 **领域知识沉淀**
- 两者并不冲突，可以叠加使用

---

## 九、关键洞察与启发

### 9.1 核心创新

1. **新建模视角**：把 prompt 优化从"采样问题"重新定义为"策略规划问题"
2. **新 Action 空间**：error feedback 作为 action，比"删/替/换"更高层、更接近人类思维
3. **MCTS 引入 prompt 优化**：第一次把"前瞻 + 回溯"的能力引入这个领域
4. **自我反思（self-reflection）+ 规划（planning）的结合**：把这两条研究主线第一次合到 prompt 优化里

### 9.2 工程启发

- **手写 prompt 时不妨"做加法"**：不只是任务描述，还应明确写出 **领域知识、异常情况、输出格式**
- **专家级 prompt 在更强模型上效果更好**：值得为关键任务做一次"prompt 投资"，未来 LLM 升级时收益叠加
- **错误反馈驱动的迭代**比拍脑袋改 prompt 高效得多 —— 这个思路可以**人类手工模拟**：跑一批样例 → 找出错的 → 让 GPT-4 帮你诊断原因 → 改 prompt

### 9.3 局限性

- **强依赖 optimizer LLM 的领域知识**（GPT-4 不懂的领域，比如 HIPAA 限制下的医疗数据，可能优化效果差）
- **专家级 prompt 在弱 LLM 上效果可能下降**（PaLM 2 实验有验证）
- **优化成本不低**：单任务约 $5，时间 1~2 小时
- **训练集过小可能过拟合**（迭代 16 次反而下降）

### 9.4 后续发展

- 2024.06 PromptAgent 集成进 [LLM Reasoners](https://github.com/maitrix-org/llm-reasoners) 库
- 2024.08 支持 vLLM 加速本地推理
- 思路被后续工作（如 PRewrite, PromptBreeder, OPRO）扩展

---

## 十、附录：12 个任务的初始 vs PromptAgent 优化结果

| 任务 | Human | APE | **PromptAgent** | 提升 |
| --- | --- | --- | --- | --- |
| Penguins | 0.595 | 0.747 | **0.873** | +27.8% |
| Geometry | 0.227 | 0.490 | **0.670** | +44.3% |
| Epistemic | 0.452 | 0.708 | **0.806** | +35.4% |
| Object Counting | 0.612 | 0.716 | **0.860** | +24.8% |
| Temporal | 0.720 | 0.856 | **0.934** | +21.4% |
| Causal Judge | 0.470 | 0.570 | **0.670** | +20.0% |
| NCBI (F1) | 0.521 | 0.576 | **0.645** | +12.4% |
| Biosses | 0.550 | 0.700 | **0.750** | +20.0% |
| Med QA | 0.508 | 0.470 | **0.570** | +6.2% |
| Subj | 0.517 | 0.696 | **0.806** | +28.9% |
| TREC | 0.742 | 0.834 | **0.886** | +14.4% |
| CB | 0.714 | 0.804 | **0.911** | +19.7% |

---

## 十一、代码级复现指南（核心补充，便于自己重写）

> 本节把"论文概念"对齐到"本仓库的真实实现"，是复现 idea 时最容易踩坑的地方。逐条对应到源码文件。

### 11.1 模块装配关系（一张图看懂）

```
main.py (读 yaml)
  └─ BaseAgent (agent.py)            # 用 5 个配置 dict 组装一切
       ├─ get_task(task_name)        # tasks/__init__.py：动态 import tasks/<name>.py 的 CustomTask
       ├─ get_language_model(...)    # base_model + optim_model 两个 LLM
       ├─ get_world_model(algo)      # mcts→WorldModel, beam→BeamSearchWorldModel
       │     └─ GradientDescent      # 真正的"改 prompt 引擎"
       └─ get_search_algo(algo)      # mcts→MCTS, beam→BeamSearch
            └─ search(init_state)    # 跑 MCTS，输出 logs/<时间戳>-<task>-algo_<algo>/data.json
```

> 关键设计：每个子系统都用「字符串→类」注册表（各包的 `__init__.py`）。新增任务/模型/算法 = 写一个类 + 在对应 `__init__.py` 注册一行。`get_task` 没有注册字典——**文件名本身就是 key**（`importlib.import_module(f".{task_name}")` 取 `CustomTask`）。

### 11.2 三类数据集划分的真实语义（`base_task.py`）

复现时极易混淆：代码里有 **train / eval / test 三个 split**，与论文表述对应如下：

| split | 用途 | 对应代码 |
| --- | --- | --- |
| `train` | **错误采样**：每次扩展从这里取 batch，跑 base model 找错例 | `world_model.get_train_batch()`，无限循环采样 |
| `eval` | **reward 计算**：新 prompt 在这个 held-out 集上的指标 = 节点 reward | `evaluate_prompt()` |
| `test` | 最终测试：仅在 `test_size>0` 时对选出的若干节点跑一次 | `test_prompt()` |

- **train 与 eval 可能重叠**（见 `split_list_dataset`：eval 取 `dataset[-eval_size:]`，train 取 `dataset[:train_size]`，都来自去掉 test 后的同一份）。这是论文设定，不是 bug。
- `cal_correct`（逐样本对/错，用于找错例）和 `cal_metric`（标量/元组指标，用于 reward）是**两个不同函数**，务必都实现。NER 类任务 `cal_metric` 返回 `(f1, precision, recall)` 元组，代码统一用 `metric[0]` 当 reward（`_reward_type_helper`）。

### 11.3 MCTS 实现的关键细节（`search_algo/mcts.py`）

复现 MCTS 时，论文伪代码省略了下面这些**必须实现**的工程细节：

1. **节点 = prompt**：`MCTSNode.prompt` 是 state，`action` 存的是产生它的那次优化输出，`reward` = eval 集指标，`Q` = `mean(cum_rewards)`。
2. **UCT**（`_uct`）：`Q + w_exp * sqrt(ln(N_parent+1) / max(1, len(cum_rewards)))`。根节点 `N_parent=0`。
3. **动态终止阈值**（论文只一笔带过，代码是核心）：
   - 深度终止：`depth >= depth_limit`。
   - **min_threshold 早停**：`min_threshold` 初始化为根节点 reward；节点终止条件是 `reward < (min_threshold + parent.reward)/2 且 depth > min_depth`。
   - **early_stop / max threshold**：`reward > mcts_threshold 且 depth > min_depth` 时提前停止 simulation，并把全局 `mcts_threshold` 抬高到该 reward（鼓励短而强的路径）。
4. **Simulation 不是随机 rollout**：`_simulate` 反复 `_expand` 并每次走 reward 最高的子节点（`simulate_choice = argmax`），直到终止。
5. **Back-propagation**：从路径末端往根累加，`cum_reward = sum(rewards)`，追加进每个节点的 `cum_rewards` 列表（Q 是它的均值）。
6. **每次迭代 deepcopy 路径**存入 `trace_in_each_iter`，用于最后挑选输出路径。

### 11.4 输出节点的选择（`prepare_output` / `output_to_json`）

跑完 `iteration_num` 次迭代后，代码会保存 `data.json`，其中给出多种候选：

- `top_k_reward_nodes`：全局 reward 最高的 top-k 节点（代码 `k=1`）。
- `best_q_path`：平均 Q 最高的路径。
- `best_reward_path`：**平均 reward 最高的路径**。
- `best_reward_path_last_node`：该路径的最后一个节点。
- `best_reward_path_selected_node`：**该路径上 reward 最高的节点 ← 论文采用的最终选择策略**（对应笔记 4.2 的"先选最佳平均 reward 路径，再选路径上最佳 reward 节点"）。

> 仅当 `test_size>0` 时，才会对 `best_q_path ∪ best_reward_path ∪ top_k_reward_nodes` 这批节点跑 test 评估并记录 `test_metric`。

### 11.5 语言模型接口（`language_model/`）

自己重写时，任何模型类只需实现两个方法即可接入：

```python
class MyModel:
    def batch_forward_func(self, batch_prompts: list[str]) -> list[str]: ...
    def generate(self, input: str) -> str: ...
```

- base model 用 `batch_forward_func`（批量答题）；optim model 用 `generate`（单条生成 gradient / 新 prompt）。
- 约定：base 温度 0（稳定答题），optim 温度高（如 1.0，鼓励多样化改写）。
- 本地后端（`hf_*`/`ct_model`/`vllm`）用 `device`/`gpu_ids`/`model_path`；`vllm` 加速本地推理。

### 11.6 复现最小路线图

1. **先跑通官方**：填 `example_config.yaml` 的两个 `api_key`，`python src/main.py --config_dir example_config.yaml`，确认 `logs/.../data.json` 产出。
2. **对齐论文超参**：把 `iteration_num→12`、`depth_limit→8`(Standard)、base→`gpt-3.5-turbo`、optim→`gpt-4`，三套配置（Standard/Wide/Lite，见 4.3）各跑取最优。
3. **自己重写时按依赖顺序搭**：① 任务类（数据 + `cal_correct`/`cal_metric`/`clean_response`）→ ② 两个 LLM 包装 → ③ GradientDescent（含 **ascend 全对例外**，见 3.4.1）→ ④ WorldModel（train/eval/test 三 loader + reward）→ ⑤ MCTS（含 11.3 的动态阈值与非随机 simulation）→ ⑥ 输出选择（11.4）。
4. **验证用 `test.py`**：拿优化出的 prompt 在 test 集复测，确认相对初始 prompt 的提升。

---

## 十二、引用

```
@article{wang2023promptagent,
  title={PromptAgent: Strategic Planning with Language Models Enables 
         Expert-level Prompt Optimization},
  author={Wang, Xinyuan and Li, Chenxi and Wang, Zhen and Bai, Fan and 
          Luo, Haotian and Zhang, Jiayou and Jojic, Nebojsa and Xing, 
          Eric P and Hu, Zhiting},
  journal={arXiv preprint arXiv:2310.16427},
  year={2023}
}
```

**相关资源**：

- 📄 论文：[https://arxiv.org/abs/2310.16427](https://arxiv.org/abs/2310.16427)
- 💻 代码：[https://github.com/maitrix-org/PromptAgent](https://github.com/maitrix-org/PromptAgent)
- 📊 LLM Reasoners 集成版：[https://github.com/maitrix-org/llm-reasoners/tree/main/examples/PromptAgent](https://github.com/maitrix-org/llm-reasoners/tree/main/examples/PromptAgent)
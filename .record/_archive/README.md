# _archive — 已被取代的旧稿（保留历史，勿引用）

这里两份是 **MCTS-SkillOpt 最早期的单目标迁移草稿**，已被 `../mcts_00`–`../mcts_06` 系列**整体取代**。
保留它们仅为追溯设计演化；**当前正典是 `mcts_*` 系列**，工程/实验内容已抽取并翻译进
`../mcts_03_code_integration_plan.md`（工程）与 `../mcts_04_experiment_plan.md`（评测）。

| 旧稿 | 取代它的当前文档 | 关键差异（为什么过时） |
|---|---|---|
| `SkillOpt 迁移设计：把 MCTS 引入技能文档优化.md` | `mcts_03_code_integration_plan.md`（+ `mcts_01`/`mcts_05`） | 旧稿是**单目标 success-only**：reward 是标量、产物仍是单个 `best_skill.md`（"最佳平均 reward 路径上的最佳节点"）。当前设计是**二维价值 (success,cost) + Pareto 前沿**，backup 回传二维向量、selection 时才标量化（Option B）。 |
| `SkillOpt MCTS 迁移：开销与风险评估.md` | `mcts_04_experiment_plan.md`（+ `mcts_02`/`mcts_05`） | 开销模型（reward 调用 ≈ I×W×D）、缓解三件套、风险清单仍有效，已搬入 `mcts_04`；但其 reward=标量、输出=单 skill 的假设已被二维/前沿取代。 |

> ⚠️ 不要再基于这两份做改动或引用其 `file:line`；以 `mcts_*` 系列为准。

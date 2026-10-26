# Coding 任务包（作者侧）

这组小型 Python 题用于建立可信题面、参考实现和判分用例。当前共有 8 题：

| 目录 / 入口函数 | 类型 | 语义 |
| --- | --- | --- |
| `stable_unique` | 函数实现 | 整数去重，保留首次出现的顺序 |
| `chunk_list` | 函数实现 | 按正整数 k 分块，保留不足 k 的末块；k ≤ 0 抛出 `ValueError` |
| `merge_intervals` | 函数实现 | 合并重叠或共享端点的闭区间，不修改输入 |
| `deep_merge` | 函数实现 | 两侧值均为字典时递归合并，否则右值覆盖；列表整体替换 |
| `parse_config` | 函数实现 | 简单键值文本；首个等号分隔、重复键最后覆盖，无行内注释 |
| `max_false_run` | 函数实现 | 最长连续 `False` 片段长度 |
| `binary_search_first` | 修复错误 | 非降序数组中目标值的最左下标，找不到返回 -1 |
| `dedup_records` | 修复错误 | 同 id 取最大时间戳，并列取最后出现者，按 id 排序 |

所有题目均要求保持输入不变。`binary_search_first` 题面要求二分查找和 O(log n) 时间复杂度；当前普通输入输出用例不能证明复杂度，没有实现性能判分。

## 文件与视图

每个题目目录包含：

```text
task.json                 # 完整题面、签名、运行约定、公开/隐藏用例、错误版本清单
reference.py              # 可信参考实现，固定文件名
buggy.py                  # 仅 bug_fix 题目：供修复的初始实现
mutants/*.py              # 可调用但有语义错误的版本，用来检查隐藏用例辨别力
```

`task.json` 使用 `schema_version: "1.0"`，包含 `id`、`title`、`task_kind`、`prompt`、`entry_point`、`signature`、`runtime`、`preserve_inputs`、`cases` 和 `mutants`。`task_kind` 为 `function_implementation` 或 `bug_fix`；仅后者有 `starter_file: "buggy.py"`。

每个用例包含唯一 `id`、`visibility`、位置参数数组 `args`，以及二选一的 `expected` JSON 值或 `raises: "ValueError"`。每题至少 2 个公开、6 个隐藏用例，另有 3 个错误版本。判分检查返回值或预期异常，也检查调用前后输入是否相同。

视图约定如下；目前没有模型调用或视图导出命令：

- **模型可见内容**：题面、签名、运行约定、公开用例；修复题额外提供 `buggy.py`。
- **作者与判分内容**：完整 `task.json`、隐藏用例、`reference.py` 和 `mutants/`。隐藏用例用于检查参考实现并识别错误版本，不应随题目发送给模型。

`visibility` 是逻辑划分，不是访问控制：作者侧 `task.json` 当前存储全部用例，整个目录不能直接作为模型输入。这里的参考实现与错误版本是任务基础资产，不是模型生成的训练答案。

## 本地作者自检

在项目根目录运行：

```bash
uv run dataforge check-coding-pack examples/coding --output runs/coding_pack/report.json
```

自检要求参考实现通过全部用例、每个错误版本被至少一个隐藏用例识别，修复题的初始实现也须被隐藏用例识别。报告记录用例结果、错误版本检出情况和资产指纹。非通过结果以非零状态退出。

当前自检只检查返回值、异常和输入是否变化，不检查复杂度或禁止 I/O 等行为约束；stdout/stderr 仅被捕获以免污染报告，题面的禁止 I/O 要求仍是题目契约，不能由自检结果证明。

运行约定是 Python 3.13、`timeout_seconds: 2`、`memory_mb: 256`；本机已用 Python 3.13.12 验证。作者自检在独立进程中执行本地可信资产，对每个资产执行设置墙钟超时。`memory_mb` 目前仅是未来执行器的契约，自检没有强制内存隔离，也不隔离网络和文件系统。因此这不是执行不可信模型代码的沙箱。

`code_tests` 模型代码执行尚未接入，现有 `dataforge run` 不接受这些任务包。本机没有 Docker / Podman，本阶段没有进行 LLM 在线调用；后续接入候选生成、隔离执行和训练器导出时，需要单独实现对应流程。

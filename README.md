# DataForge

用于 LLM / VLM 训练数据构造的 Python 工具包与 CLI。支持从已有问题或多轮对话合成 SFT 回答，以及从可信标注种子构造、试解筛选普通 RLVR 任务，再完成检查、去重、分析和导出。

模型调用支持 OpenAI Responses API、Anthropic Messages API，以及无需密钥和网络的 `mock` 演示模式。RL 数据分三条路线：

- **RLHF 偏好数据**：已有 `preference` 结构与校验，尚未实现候选排序、偏好对生成或奖励模型训练。
- **普通 RLVR（当前优先）**：支持纯文本可信种子、教师试解、精确匹配/数值答案验证、筛选和任务导出。
- **Agentic RL**：工具、环境、轨迹与执行验证仍是 [规划文档](docs/rl_tool_data_guide.md) 中的设计，尚未接入流水线。

本项目当前构造数据和提供答案验证函数，不执行模型训练。

另提供 [8 题 coding 作者任务包](examples/coding/README.md)，包含可信参考实现、公开/隐藏用例和错误版本，供本地自检。模型代码的 `code_tests` 判分尚未接入。

## 快速开始

运行环境：Python 3.11+、macOS / Linux。当前执行锁使用 `fcntl`，暂不支持 Windows 原生运行。

在项目根目录执行：

```bash
uv sync --extra dev

# 完全离线的文本流程
uv run dataforge run configs/demo.yaml
uv run dataforge validate runs/demo/clean.jsonl
uv run dataforge analyze runs/demo/clean.jsonl

# 完全离线的图文流程，附带一张原创几何图形示例图
uv run dataforge run configs/demo_vlm.yaml

# 完全离线的 RLVR 试解筛选：固定答案演示 2 题通过、1 题不通过
uv run dataforge run configs/demo_rlvr.yaml
uv run dataforge validate runs/demo_rlvr/clean.jsonl
uv run dataforge analyze runs/demo_rlvr/clean.jsonl

# 重用已经落盘的结果，不重复生成
uv run dataforge run configs/demo.yaml --resume
```

如果对应任务目录已存在，给 `run` 命令加上 `--resume` 即可。也可以使用 `uv run python -m dataforge`。`mock` 只生成明确标记的确定性占位回答，用于验证流程和图片打包，不理解图片，也不能作为训练数据。`offline-rlvr-demo` 模式固定返回最终答案 `42`，不读取标签，也不实际求解问题。

真实 API 调用前，先编辑配置里的 `model.model: REPLACE_WITH_YOUR_MODEL`，填入当前账户可用的模型名。图文任务需要选择支持图像输入的模型，并设置 `vision: true`。

```bash
export OPENAI_API_KEY='your-openai-key'
uv run dataforge run configs/sft_text.yaml
# 普通 RLVR 使用同一组小型示例种子，可替换为自己的可信标注数据
uv run dataforge run configs/rlvr_text.yaml

export ANTHROPIC_API_KEY='your-anthropic-key'
uv run dataforge run configs/sft_vlm.yaml
```

密钥只通过环境变量交给官方 SDK；`.env.example` 是变量示例，CLI **不会自动读取 `.env`**。配置文件不接受 API 密钥字段。真实调用会消耗 API 用量。

## 项目结构

```text
configs/                    # 离线示例与真实模型配置
prompts/
  sft/generate.jinja2        # 文本教师回答指令
  vlm/qa.jinja2              # 图文教师回答指令
  rlvr/solve.jinja2          # RLVR 独立试解指令，不读取标签
examples/
  text_seeds.jsonl           # 原创文本种子
  vlm_seeds.jsonl
  rlvr_seeds.jsonl           # 三条可信标注的纯文本示例
  coding/                   # 八题可信 coding 任务包与作者侧验证资产
  assets/shapes.png
src/dataforge/
  cli.py                    # run / validate / analyze / check-coding-pack
  config.py                 # 严格配置校验与路径解析
  schemas.py                # 种子、SFT、preference、rl_task、rlvr、模型请求/结果
  runner.py                 # 并发、执行记录、恢复、流水线
  prompting.py              # Jinja2 严格变量渲染
  cleaning.py               # 精确去重，不改写正文
  providers/                # OpenAI、Anthropic、mock
  recipes/                  # SFT 回答合成、RLVR 任务构造
  verifiers.py              # 最终答案格式、精确匹配、数值答案验证
  datasets/                 # JSONL、图片、训练/验证集导出
  quality/                  # 规则检查、统计报告
tests/
data/                       # 自有输入数据，默认不提交 Git
runs/                       # 每个任务的结果，默认不提交 Git
```

依赖由 `pyproject.toml` 声明，`uv.lock` 固定本次验证的版本。当前接入已按 OpenAI 3.22.1、Anthropic 1.9.0 验证。

## 输入和内部数据格式

种子文件是 UTF-8 JSONL，每行一个对象。`id` 必须唯一；`prompt` 是训练任务本身的对话，最后一条必须是 `user`。系统消息只能位于第一条。图片只支持用户消息中的本地相对路径。

```json
{"id":"example_001","source_id":"lesson_001","group_id":"lesson_001","prompt":[{"role":"user","content":[{"type":"text","text":"为什么 6 是偶数？"}]}],"variables":{"style":"用一句话解释。"}}
```

图文内容使用 `{"type":"image","uri":"assets/shapes.png"}`。`uri` 相对于配置的 `media_root` 解析，不支持 URL、绝对路径或跳出该目录的路径；图片需可被 Pillow 解码，格式为 PNG、JPEG、WEBP 或 GIF。图片 Base64 仅在调用 API 时构建，数据文件保存路径引用。

内部样本共享 `schema_version: "1.0"`、`id`、`prompt`、`source_id`、`group_id`、`metadata`：

| `task_type` | 专属字段 | 当前状态 |
| --- | --- | --- |
| `sft` | `completion`：助手消息列表 | 支持生成、校验、分析、导出 |
| `preference` | `chosen`、`rejected`：助手消息列表 | 仅结构与校验 |
| `rl_task` | `reference`：对象；`verifier`：字符串标识 | 仅结构与校验；不执行验证器 |
| `rlvr` | `reference`：严格参考答案；`verifier`：`exact_match` / `math_answer` | 纯文本种子试解、验证、筛选和导出 |

SFT 样本中的 `completion` 和 `prompt` 使用相同的消息/内容块结构。偏好样本保存候选回答的比较，RLVR 样本保存任务与参考答案，教师试解文本只作为筛选依据。原有 `rl_task` 保留为通用数据结构；执行当前 RLVR 流水线需使用 `task_type: rlvr`。

`validate` 校验的是内部样本文件，不接受种子文件或导出的 `messages` 文件。种子校验在 `run` 的预检查中完成；非法种子、缺失模板变量或无效图片会在请求模型前报错。

同源样本必须提供相同的 `group_id`，例如同一篇文档或同一张图片派生的问题和改写。生成时的默认值是：

- `source_id = seed.source_id or seed.id`
- `group_id = seed.group_id or seed.source_id or seed.id`

项目不会自动推断语义来源。训练/验证划分使用 `group_id` 的稳定哈希，确保同组样本进入同一分区；`validation_fraction` 是按组哈希的划分阈值，小数据集的实际比例可能偏离设置，甚至出现空分区。

## 普通 RLVR

首版接收**已有可信题目和答案**，让教师模型独立试解，再保留答案验证通过的任务。不生成新题或参考答案；验证通过仅说明教师的最终答案与给定标签一致，不能证明标签正确，也不校验推理过程。

种子在通用字段之外增加 `verifier` 和 `reference`，仅支持纯文本 `prompt`：

```json
{"id":"multiply_001","group_id":"multiply_6_7","prompt":[{"role":"user","content":[{"type":"text","text":"计算 6 × 7。"}]}],"verifier":"math_answer","reference":{"answer":"42","tolerance":0}}
```

- `exact_match`：`answer` 必须是非空单行字符串；`strip` 默认 `true`，`ignore_case` 默认 `false`，二者必须是布尔值。
- `math_answer`：`answer` 必须是数值字符串；`tolerance` 是非负有限 JSON 数值，默认 `0`，按绝对误差比较。支持正负整数、小数、科学计数法、整数分数和严格千分位；数值长度最多 256 字符，指数绝对值最多 1000。不执行表达式，不支持单位或符号等价证明。

流水线将统一的最终答案约定加入问题，教师请求和导出任务使用相同约定：最后一个非空行必须为 `#### <答案>`，且回答中只能有一个答案标记。模板只接收种子的通用字段，不暴露 `reference` 或 `verifier`；标签不发送给教师模型。自有题目和 `variables` 也应避免包含额外的答案提示。

运行 `configs/demo_rlvr.yaml` 时，固定答案 `42` 通过乘法题与阅读题，故意无法通过答案为 `5` 的加法题。预期 `report.json` 中 `accepted: 2`，验证统计为 `evaluated: 3, passed: 2, failed: 1`。这个模式用于检查筛选与产物，不可用于训练。

RLVR 的 `quality.min_chars` / `max_chars` 检查教师试解回答的长度。`report.pass_rate` 以预期候选数为分母，统计去重后的候选产出率；`report.verification.pass_rate` 以实际执行验证的试解数为分母，统计答案验证通过率，包含通过验证后被去重的候选。当前输入使用 `reference`，不使用 Agentic 规划中的 `label` 字段。

训练侧可以复用相同验证函数，将样本的 `verifier`、`reference` 和模型回答传入：

```python
from dataforge.verifiers import verify_answer

result = verify_answer(
    "math_answer", {"answer": "1/2", "tolerance": 0}, "解题过程\n#### 0.5"
)
reward = float(result.passed)  # 1.0；失败原因见 result.reason
```

## Coding 作者任务包

[`examples/coding`](examples/coding/README.md) 包含 6 道函数实现题和 2 道错误修复题。每题提供完整题面、签名、公开/隐藏用例、可信参考实现和 3 个错误版本。作者侧自检命令：

```bash
uv run dataforge check-coding-pack examples/coding --output runs/coding_pack/report.json
```

自检运行本地可信资产，检查参考实现是否通过所有用例，以及隐藏用例能否识别错误版本。它使用独立进程和每个资产的墙钟超时；没有网络、文件系统或内存隔离，不能作为不可信模型代码沙箱。题目的 Python 3.13 约定已在本机 3.13.12 验证，`memory_mb: 256` 目前是未来执行器契约，未强制实施。二分题的 O(log n) 要求也没有性能判分支持。

当前 `dataforge run` 尚不接受这些任务包，`code_tests`、候选代码生成与模型代码执行仍待接入。本阶段未调用在线 LLM，本机没有 Docker / Podman；这些资产用于建立题目和判分依据，不是模型生成的训练答案。公开/隐藏视图及格式见任务包说明。

## 配置与 Prompt

配置中的路径相对于 **YAML 文件所在目录** 解析。执行结果写入 `output_dir/name`。可复制 `configs/demo.yaml` 后修改：

```yaml
name: my_sft_run
task_type: sft
input_path: ../data/seeds.jsonl
media_root: ../data
prompt_template: ../prompts/sft/generate.jinja2
output_dir: ../runs
model:
  provider: openai
  model: REPLACE_WITH_YOUR_MODEL
  vision: false
  max_output_tokens: 2048
  timeout_seconds: 60
  max_retries: 2
concurrency: 4
samples_per_seed: 1
quality:
  min_chars: 10
  max_chars: 20000
  deduplicate: true
validation_fraction: 0.1
split_seed: 42
```

`samples_per_seed` 决定每个种子的调用次数，默认 1。重复候选可能被去重。重试仅由官方 SDK 处理，外层不叠加自动重试；`max_retries` 是首次请求之外的最大重试次数。

`temperature` 默认省略。OpenAI 可按目标模型的支持情况显式配置。当前支持的 Anthropic SDK 不接受该参数，因此 `provider: anthropic` 配置非空 `temperature` 会直接报错。

Jinja2 模板仅暴露 `seed` 的通用字段，例如 `{{ seed.variables.style }}`。可选变量使用 `seed.variables.get('style')`；引用缺失的必需变量会报错。生成指令作为模型调用的独立 instructions/system 输入，原始对话保留自己的系统消息。**合成指令不会被拼进训练样本的用户问题。** RLVR 的最终答案格式约定属于任务本身，因此随问题保留到训练数据中。

SFT 直接将模型文本作为回答；RLVR 从试解的最后答案行执行验证，保留原始任务与参考答案。两种流程均不要求模型再输出一层 JSON。原始响应、调用状态、用量和请求 ID 一并保存，方便检查截断和失败。

## 质量检查与运行产物

流程为：预检查 → 渲染指令 → 调用模型并保存响应 → 构建样本 → 规则过滤与精确去重 → 统计 → 分组划分并导出。

规则检查覆盖空回答、拒绝、未完成/截断响应和字符长度。工具调用不作为完整回答。SFT 去重对 `prompt + completion` 的规范 JSON 计算哈希，图像引用使用内容哈希；RLVR 按任务问题、参考答案和验证器去重，多个通过的试解不会产生重复任务。不压缩空白、不删除 Markdown、不改写代码缩进。RLVR 另检查答案格式和标签一致性；数据质量检查不判断标签的事实正确性，不包含模型评审、模糊/语义去重或自动修复。

每次执行保留：

```text
runs/<name>/
  config.yaml               # 实际路径和参数，不含 API 密钥
  manifest.json             # 状态、版本、配置/输入/模板/图片指纹
  raw.jsonl                 # 请求任务标识、标准结果、原始 API 响应
  errors.jsonl              # 失败请求历史，恢复后仍保留
  samples.jsonl             # SFT 回答或 RLVR 任务构成的内部样本
  quality.jsonl             # sample_id、是否通过、过滤原因
  clean.jsonl               # 通过规则和去重后的内部样本
  rejected.jsonl            # 被过滤样本或失败响应的原因
  report.json               # 数量、通过率、原因、Token 用量；RLVR 另有验证统计
  exports/
    train.jsonl
    validation.jsonl
    assets/                 # 图文数据引用的图片副本
```

运行报告的 `pass_rate` 以预期候选总数为分母；Token 用量来自已经收到并落盘的响应，不是账单或价格估算。API 请求失败会记录到 `errors.jsonl`，命令以非零状态退出；质量不合格会记入 `rejected.jsonl`，不等同于请求异常。

`analyze` 对内部文件输出基础统计；SFT 提供回答字符统计，preference/RL 任务提供通用数量与来源信息，RLVR 另有验证器分布。RLVR 的试解验证统计保存在运行级 `report.json` 中。

## 恢复执行和导出

相同 `name` 的目录已存在时，默认拒绝覆盖。使用 `--resume` 会跳过已记录到 `raw.jsonl` 的响应，重新尝试尚未成功收到响应的请求，并重建清洗结果、报告和导出。拒绝或截断等已经落盘的响应会被复用并过滤，不会在恢复时自动再次付费生成。

恢复要求配置、输入、模板和图片内容指纹一致。修改参数或数据后，请使用新的 `name`。任务锁阻止相同任务并发写入。恢复只能避免重复已经落盘的调用：如果进程在 API 已处理请求、但本地尚未记录结果时崩溃，恢复可能重复调用并产生费用。

文本导出每行包含 `id`、`group_id` 和 `messages`，消息形如 `{"role":"user","content":"问题"}`。对接训练器时按其要求映射 `messages` 字段。

RLVR 导出包含 `id`、`task_type: "rlvr"`、`group_id`、`messages`、`reference` 和 `verifier`。`messages` 只包含任务问题及其最终答案约定，不追加教师试解回答；试解文本保存在 `raw.jsonl`。这是通用交换格式，尚未提供 verl / slime 的现成适配器。

图文导出保留内容块 `{"type":"image","uri":"assets/<hash>.png"}`，并提供顶层 `images` 列表；图片复制到 `exports/assets`，路径相对于 `exports` 根目录。**这是本项目的可搬运交换格式，接入具体 VLM 训练框架仍需对应的导出适配器。**

## 开发验证

```bash
uv run pytest -q
uv run ruff check src tests
```

测试使用官方 SDK 加本地模拟 HTTP 响应，覆盖请求转换、图像内容、结果状态、质量过滤与恢复执行，无需真实密钥或 API 请求。当前未执行真实账户的在线生成验证，实际模型权限与参数支持仍需由所选模型确认。本版不包含数据库、网页界面、分布式任务队列或大规模流式处理。

相关官方接口文档：[OpenAI Python SDK](https://developers.openai.com/api/docs/libraries)、[OpenAI 图像输入](https://developers.openai.com/api/docs/guides/images-vision)、[Anthropic Python SDK](https://platform.claude.com/docs/en/cli-sdks-libraries/sdks/python)、[Anthropic 图像输入](https://platform.claude.com/docs/en/build-with-claude/vision)。

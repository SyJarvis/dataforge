# 多轮工具调用 RL 数据格式构造指南

版本：v0.1（草案）
适用范围：dataforge 生成的 RL 数据集（多轮工具调用 / agentic 任务），下游训练框架以 verl 为第一目标。
事实依据：`verl/examples/data_preprocess/gsm8k_tool_agent_loop.py`、`verl/verl/experimental/agent_loop/tool_agent_loop.py`、`verl/verl/tools/schemas.py`、`verl/verl/workers/reward_manager/naive.py`、`dataforge/src/dataforge/schemas.py`。

---

## 1. 核心概念：RL 数据 ≠ SFT 数据

| | SFT 数据 | RL（RLVR）数据 |
|---|---|---|
| 存什么 | prompt + 完整标准回答 | 任务描述 + **验证依据** |
| 回答从哪来 | 数据集里直接给 | **rollout 在线生成**（模型自己试） |
| 好坏怎么定 | 无（直接模仿） | reward 函数按验证依据打分 |
| 多轮工具场景 | 存完整轨迹（assistant tool_calls → tool result → …） | **只存初始 prompt**，后续轮次由 agent loop 驱动生成 |

**一句话**：RL 数据行里没有答案，只有「任务 + 怎么判对错」。多轮工具 RL 里，判对错的对象是 rollout 产出的整条轨迹（选没选对工具、参数对不对、任务完没完成），不是单条消息。

## 2. 训练侧目标格式（verl parquet 行结构）

载体：parquet，每行一个任务。字段：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `data_source` | str | ✅ | 数据来源标识。reward manager 用它路由到对应打分函数（见 `reward_manager/naive.py` 的 `reward_fn_key`） |
| `agent_name` | str | ✅ | agent loop 路由键。`"tool_agent"` → `ToolAgentLoop`（支持多轮工具调用的循环） |
| `prompt` | list[dict] | ✅ | OpenAI messages 格式，**必须以 user 结尾** |
| `ability` | str | ✅ | 任务类别标签（`math` / `tool_calling` / `agent` …） |
| `reward_model` | dict | ✅ | `{"style": "rule", "ground_truth": <验证依据>}` |
| `extra_info` | dict | ✅ | 自由扩展字段，agent loop 与 reward 函数消费（见 §2.2） |

### 2.1 最小完整示例（数学 + 校验工具）

来自 verl 官方预处理脚本的行结构（`gsm8k_tool_agent_loop.py`）：

```python
{
    "data_source": "my_source/gsm8k_zh",
    "agent_name": "tool_agent",
    "prompt": [
        {"role": "system", "content": (
            "You are a math expert. ... Reasoning step by step before any tool call. "
            "You should use the `calc_gsm8k_reward` tool after step by step solving the question, "
            "before generate final answer at least once and refine your answer if necessary. "
            "Put your final answer in the format of `#### <answer>`."
        )},
        {"role": "user", "content": "某题干... 请一步步思考，最终以 #### <答案> 结尾。"},
    ],
    "ability": "math",
    "reward_model": {"style": "rule", "ground_truth": "42"},
    "extra_info": {
        "split": "train",
        "index": 0,
        "need_tools_kwargs": True,
        "tools_kwargs": {                      # 工具实例化时注入任务私有参数
            "calc_gsm8k_reward": {
                "create_kwargs": {"ground_truth": "42"}
            }
        },
    },
}
```

### 2.2 `extra_info` 的关键字段

| 字段 | 类型 | 说明 | 证据 |
|---|---|---|---|
| `need_tools_kwargs` | bool | 告知 agent loop 本样本需要 tools_kwargs | `tool_agent_loop.py:143` |
| `tools_kwargs` | dict | `{工具名: {create_kwargs/execute_kwargs/...}}`，把 ground_truth 藏进校验工具实例 | 同上 `:68-77` |
| `tool_selection` | list[str] | **按样本过滤全局工具集**。工具在 rollout 侧全局注册，样本只点名可用的子集（防作弊：不暴露无关工具） | 同上 `:156-166` |
| `split` / `index` | str/int | 数据集切分与序号，可追溯 | 预处理脚本惯例 |

## 3. `prompt` messages 规范

1. **以 user 结尾**——verl rollout 从最后一条 user 消息后开始生成。
2. **system 只能出现在首位**（与 dataforge 现有 `validate_prompt` 一致）。
3. system 是放「任务规则」的地方，建议包含三件事：
   - 角色与目标（"You are a math expert..."）
   - **输出格式契约**（如 `#### <answer>`）——reward 函数靠它解析最终答案，没有它规则验证器没法写
   - **工具使用规则**（何时该调用工具、一次一个还是可并行、拿到结果后做什么）
4. prompt 中**不得出现**答案或决定性提示（否则模型学到的是复读，reward 也失真）。

## 4. 工具定义规范（OpenAI function schema）

结构对齐 `verl/verl/tools/schemas.py`：

```json
{
  "type": "function",
  "function": {
    "name": "get_bitcoin_balance",
    "description": "查询给定比特币地址的当前余额。",
    "parameters": {
      "type": "object",
      "properties": {
        "address": {
          "type": "string",
          "description": "BTC 地址，26-35 位，以 1/3/bc1 开头"
        }
      },
      "required": ["address"]
    }
  }
}
```

规则：
- `name` 全局唯一，`[a-z_][a-z0-9_]*`，与 prompt/校验工具注册名严格一致
- `description` 必须写清楚**何时用**与**参数含义**——模型只靠它决定调用与否
- `parameters` 是标准 JSON Schema（object 类型）；枚举用 `enum`，可选参数写进 `default`
- **工具集放置两种方式**：
  - **样本自包含**（dataforge 内部格式采用）：每条样本带全量 `tools`，简单、可单条校验、可追溯
  - **全局注册 + `tool_selection`**（verl 运行时采用）：导出时把公共工具集抽出来注册到 rollout 侧，样本只留工具名列表

## 5. 轨迹格式（种子 / 回放 / judge 用，不进 RL 训练行）

造 RL 任务时离不开完整轨迹（做种子、judge 评审、回归测试回放），其格式采用**标准 OpenAI messages 多轮约定**（与 NVIDIA Nemotron-Agentic-v1 一致）：

```json
{
  "uuid": "ff6ab2b0-...",
  "messages": [
    {"role": "system", "content": "…策略/工具说明…"},
    {"role": "user", "content": "帮我查一下这个地址的余额：1A1zP1…"},
    {"role": "assistant", "content": "", "reasoning_content": "用户要查余额，应调用 get_bitcoin_balance…",
     "tool_calls": [
       {"id": "call_TZh8eY1o", "type": "function",
        "function": {"name": "get_bitcoin_balance",
                     "arguments": {"address": "1A1zP1..."}}}
     ]},
    {"role": "tool", "tool_call_id": "call_TZh8eY1o",
     "content": "{\"balance\": \"0.00123456\", ...}"},
    {"role": "assistant", "content": "该地址余额为 0.00123456 BTC …", "reasoning_content": "工具返回余额 0.00123456，向用户报告…", "tool_calls": null}
  ]
}
```

要点（已定稿决策）：
- `tool_calls[].id` 与 `tool.tool_call_id` 必须配对
- **`function.arguments` 用 dict（对象）**，不用 JSON 字符串——内部格式以可校验性优先（可直接挂 JSON Schema 校验）；导出到线上/API 时再序列化
- `tool.content` 是**字符串化的 JSON**（不是对象）——贴合 OpenAI 标准
- assistant 可以同时带 `content` 与 `tool_calls`，也可只有其一；不再调用工具时 `tool_calls` 为 `null`（显式保留字段，同 Nemotron）
- **`reasoning_content`（思考链）为 assistant 消息的标准可选字段**——种子轨迹保留（judge 评审、蒸馏复用）；RL 训练行不存
- **从轨迹造 RL 任务的截断规则**：截到最后一条 user 消息（含其前的 system/user），之后的 assistant/tool 轮全部丢弃，转化为验证依据（§6）

## 6. 奖励与验证依据编码

### 6.1 `reward_model` 固定结构

```json
{"style": "rule", "ground_truth": <任意 JSON>}
```

`style: "rule"` 表示规则型奖励（RLVR）。`ground_truth` 的内部结构由 verifier 决定。

### 6.2 verifier 类型与 ground_truth 编码（dataforge 建议枚举）

| verifier | 判什么 | ground_truth 结构 | 适用 |
|---|---|---|---|
| `exact_match` | 最终答案字符串精确匹配（忽略空白/大小写可配） | `{"answer": "42", "format": "#### {}"}` | 数学、知识问答 |
| `math_answer` | 从 `#### <answer>` 解析后数值等价（容忍分数/逗号） | `{"answer": "3.5", "tolerance": 1e-6}` | 数学 |
| `tool_call_match` | 轨迹中的工具调用序列与期望一致（名+参数） | `{"expected_calls": [{"name": "f", "arguments": {...}}], "order": "strict"\|"any"}` | 单轮工具选择 |
| `tool_trajectory_match` | 多轮：每步工具名+关键参数匹配 | `{"expected_steps": [...], "allow_extra_steps": false}` | 多轮规划 |
| `state_check` | 模拟环境终态达成（字段断言） | `{"final_state": {"order_status": "cancelled", "refund": "full"}}` | agentic 任务 |
| `judge` | LLM 评审打分（非严格 RLVR，需注明模型与版本） | `{"rubric": "...", "min_score": 4}` | 开放任务，慎用 |

原则：**优先可机械验证的 verifier**；`judge` 仅作补充，且在 metadata 里记录评审模型，保证可复现。

### 6.3 奖励三档（构造时的自检顺序）

1. **格式合法**：tool_calls 可解析、参数过 schema 校验（纯规则，零成本）
2. **工具正确**：调用名与参数命中 ground_truth 期望
3. **任务完成**：最终答案正确 / 环境终态达成

设计奖励时从 1→3 递进给分，避免「一步不对全零」导致的稀疏奖励问题（可给部分分）。

## 7. dataforge 内部格式 ↔ 训练框架导出映射

内部格式（`RLTaskSample`，基于现有 `schemas.py:89-92` 扩展）：

```python
class RLTaskSample(SampleBase):
    task_type: Literal["rl_task"] = "rl_task"
    label: dict[str, Any]        # 验证依据（标准答案），对齐 slime 命名
    verifier: str                # verifier 名（怎么判）
    tools: list[ToolSchema]      # OpenAI function schema，样本自包含（新增字段）
    # 继承: prompt / metadata / id / source_id / group_id
```

> 命名决策（已定稿）：验证依据字段用 **`label`**（对齐 slime 的 `--label-key` 惯例）；原 `reference` 字段名废弃。`verifier` 保留原名。

映射表：

| dataforge 内部 | verl 行 | slime 行 | 备注 |
|---|---|---|---|
| `prompt` | `prompt` | `prompt`（`--input-key prompt`） | 消息结构两者兼容 |
| `label` | `reward_model.ground_truth` | `label`（`--label-key label`） | verl 导出需包一层 `{"style": "rule", "ground_truth": ...}`；slime 直接透传 |
| `verifier` | 进 `data_source` 命名空间（如 `dataforge/tool_call_match`） | 进 `metadata.verifier`，由自定义 RM 函数读取 | verl 用 `data_source` 路由 reward manager；slime 的 RM 是自己写的函数 |
| `tools` | （导出时）全局工具注册 + `extra_info.tool_selection` | `tools`（`--tool-key tools`，默认值） | 内部自包含；verl 导出抽公共集，slime 导出零转换 |
| `metadata.tools_kwargs` | `extra_info.tools_kwargs`（+ `need_tools_kwargs: true`） | `metadata.tools_kwargs` | |
| `metadata.agent_name` | `agent_name` | （无需，循环在自定义 generate 函数里） | 缺省 `tool_agent` |
| `metadata.ability` | `ability` | `metadata.ability` | |
| `source_id` / `id` | `data_source` / `extra_info.index` | `metadata.source` / `metadata.id` | verl 的 `data_source` 承担 reward 路由职责 |

> slime 侧捷径：由于 `--label-key` 可配置，**verl 导出的 parquet 可直接喂 slime**（官方 search-r1 示例即 `--label-key reward_model` 复用 verl 社区数据）。

## 8. 构造流程（pipeline 各步 DoD）

1. **种子采集**：来源 = 已有 SFT 轨迹（如 `toolcall_pretrain_raw_v1.jsonl`）、公开数据集（Nemotron-Agentic-v1 等，注意许可）、LLM 合成（用户模拟器 + Agent + 模拟工具环境三角色）
   DoD：每条种子有唯一 id、来源标记（`source_id`）
2. **任务定义**：写出 prompt（system 规则 + user 任务）与该任务可用 `tools`
   DoD：prompt 以 user 结尾；工具名与 schema 一致
3. **验证依据构造**：从轨迹提取期望调用序列/最终答案/终态，填 `label` + `verifier`
   DoD：ground_truth 与 verifier 类型匹配（§6.2 表）
4. **可验证性试跑**：用强模型（教师）试解 N 次
   DoD：pass rate ∈ 合理区间（如 20%–90%）；过易（>95%）或过难（<5%）的任务丢弃或改造——保证有学习信号
5. **内部校验**：`dataforge validate`
   DoD：全部规则（§9）通过
6. **导出**：转 verl parquet（`data_source`/`agent_name`/`prompt`/`ability`/`reward_model`/`extra_info`）
   DoD：verl `rl_dataset` 可加载，rollout 冒烟通过

## 9. 校验规则清单（validate 强制项）

- [ ] `prompt` 非空且最后一条是 `user`；`system` 只在首位
- [ ] prompt 不含答案文本（启发式检查 ground_truth 字符串是否泄漏进 prompt）
- [ ] `tools` 中 `name` 唯一；`parameters` 是合法 JSON Schema（object）
- [ ] prompt 中点名的工具 ⊆ `tools`（防悬空引用）
- [ ] `verifier` ∈ 注册枚举；`label` 结构与该 verifier 的 schema 匹配
- [ ] `tools_kwargs` 引用的工具名 ⊆ 工具注册表
- [ ] 确定性：同一 ground_truth 两次回放打分一致（`judge` 类豁免，但需记录评审模型）
- [ ] `data_source` 命名空间无冲突；`id` 全局唯一

## 10. 完整示例：多轮工具任务

场景：用户查询 BTC 余额并追问交易历史（改编自 xlam_fc 种子）。

```json
{
  "data_source": "dataforge/toolcall_zh_v1",
  "agent_name": "tool_agent",
  "prompt": [
    {"role": "system", "content": "你是数字资产助手。可用工具以 JSON Schema 提供。需要查询时先调用工具再回答；一次只调用一个工具；无法通过现有工具完成的请求，明确告知用户并建议替代方案。最终回答用中文。"},
    {"role": "user", "content": "帮我查一下地址 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa 的比特币余额。另外这个地址的第一笔交易是什么时候？"}
  ],
  "ability": "tool_calling",
  "reward_model": {
    "style": "rule",
    "ground_truth": {
      "expected_calls": [
        {"name": "get_bitcoin_balance", "arguments": {"address": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"}}
      ],
      "refusal_expected": {
        "topic": "交易历史查询",
        "reason": "工具集不含交易历史接口"
      },
      "order": "any"
    }
  },
  "extra_info": {
    "split": "train",
    "index": 7,
    "tool_selection": ["get_bitcoin_balance", "get_eth_balance"]
  }
}
```

该例考察两个点：① 会调对工具查余额；② 对工具集外的请求**诚实拒答**而不是编造——这是 policy-following 能力的 RL 化表达。

---

## 附录：字段速查

```
RL 训练行（parquet）
├── data_source        # reward 路由键："命名空间/子集"
├── agent_name         # loop 路由键："tool_agent"
├── prompt             # messages: [system?, ..., user]
├── ability            # 任务类别
├── reward_model       # {"style": "rule", "ground_truth": {...}}
└── extra_info
    ├── split / index
    ├── tool_selection # 本样本可用工具名列表
    ├── need_tools_kwargs
    └── tools_kwargs   # {工具名: {create_kwargs: {...}}}

轨迹（种子/judge 用，不进训练行）
└── messages: [system, user, assistant(+tool_calls), tool(+tool_call_id), assistant, ...]
```

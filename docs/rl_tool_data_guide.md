# 多轮工具调用 RL 数据格式构造指南

版本：v0.3
适用范围：dataforge 生成的 RL 数据集（多轮工具调用 / agentic 任务）。
数据规范基线：**OpenAI Chat Completions 消息与工具约定**（messages / tool_calls / function schema）。
参考实现（仅作证据来源，非规范来源）：verl、slime、NVIDIA Nemotron-Agentic-v1、xlam_fc。

---

## 1. 核心概念：RL 数据 ≠ SFT 数据

| | SFT 数据 | RL（RLVR）数据 |
|---|---|---|
| 存什么 | prompt + 完整标准回答 | 任务描述 + **验证依据** |
| 回答从哪来 | 数据集里直接给 | **rollout 在线生成**（模型自己试） |
| 好坏怎么定 | 无（直接模仿） | reward 函数按验证依据打分 |
| 多轮工具场景 | 存完整轨迹（assistant tool_calls → tool result → …） | **只存初始 prompt**，后续轮次由训练时的 agent loop 驱动生成 |

**一句话**：RL 数据行里没有答案，只有「任务 + 怎么判对错」。多轮工具 RL 里，判对错的对象是 rollout 产出的整条轨迹（选没选对工具、参数对不对、任务完没完成），不是单条消息。

## 2. RL 任务行格式（dataforge 通用格式）

规范主体。**基于 OpenAI 约定构建，框架无关**——verl、slime 等训练框架通过导出适配（§8），不影响本格式。

```json
{
  "id": "rl-0001",
  "task_type": "rl_task",
  "messages": [
    {"role": "system", "content": "…任务规则…"},
    {"role": "user", "content": "…任务内容…"}
  ],
  "tools": [
    {"type": "function", "function": {"name": "f", "description": "…", "parameters": {…}}}
  ],
  "verifier": "tool_call_match",
  "label": { "expected_calls": [ … ] },
  "metadata": { "source": "…", "split": "train", "index": 0 }
}
```

字段定义：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `id` | str | ✅ | 全局唯一样本 id |
| `task_type` | str | ✅ | 恒为 `rl_task`（区分于 `sft` / `preference`） |
| `messages` | list | ✅ | **标准 OpenAI messages**，以 user 结尾（§3） |
| `tools` | list | ✅ | **标准 OpenAI function schema 列表，内联自包含**（§4） |
| `verifier` | str | ✅ | 验证方法名，指向注册的校验函数（§7.2） |
| `label` | dict | ✅ | 验证依据（标准答案），结构由 verifier 决定（§7.2） |
| `metadata` | dict | ➖ | 自由扩展：来源、切分、难度、统计等 |

设计原则：

1. **样本自包含**：一行数据 = 完整任务（prompt + 工具 + 验证依据），不依赖外部注册表即可校验、回放、评测。任何框架的适配都发生在导出时（§8），而非生成时。
2. **OpenAI 约定为唯一消息标准**：不发明私有消息结构；`tools` 即 OpenAI function calling 的 `tools` 参数原样。
3. **验证依据与方法分离**：`verifier`（怎么判）+ `label`（拿什么判），二者独立演进——同一份 label 可被多种 verifier 消费。

## 3. `messages` 规范（标准 OpenAI 约定）

1. **以 user 结尾**——训练 rollout 从最后一条 user 消息后开始生成。
2. **system 只能出现在首位**。
3. system 是放「任务规则」的地方，建议包含三件事：
   - 角色与目标
   - **输出格式契约**（如 `#### <answer>`）——reward 函数靠它解析最终答案
   - **工具使用规则**（何时调用、能否并行、拿到结果后做什么）
4. prompt 中**不得出现**答案或决定性提示（否则模型学到的是复读，reward 也失真）。

## 4. `tools` 规范（标准 OpenAI function schema）

```json
{
  "type": "function",
  "function": {
    "name": "get_bitcoin_balance",
    "description": "查询给定比特币地址的当前余额。",
    "parameters": {
      "type": "object",
      "properties": {
        "address": {"type": "string", "description": "BTC 地址，26-35 位，以 1/3/bc1 开头"}
      },
      "required": ["address"]
    }
  }
}
```

规则：
- `name` 全局唯一，`[a-z_][a-z0-9_]*`，与消息中实际调用的函数名严格一致
- `description` 必须写清楚**何时用**与**参数含义**——模型只靠它决定调用与否
- `parameters` 是标准 JSON Schema（object 类型）；枚举用 `enum`，可选参数写进 `default`
- **内联自包含**：每条样本携带该任务可用的全量工具集。干扰项（不该用的工具）也是任务设计的一部分——考察模型在 N 个工具里选对 1 个。

## 5. 轨迹格式（种子 / 回放 / judge 用，不进 RL 任务行）

完整多轮轨迹同样采用**标准 OpenAI messages 约定**：

```json
{
  "id": "traj-0001",
  "task_type": "trajectory",
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
    {"role": "assistant", "content": "该地址余额为 0.00123456 BTC …",
     "reasoning_content": "工具返回余额 0.00123456，向用户报告…", "tool_calls": null}
  ],
  "tools": [ …同 §4… ],
  "metadata": { "source": "…", "judge_score": 4.5 }
}
```

要点（已定稿决策）：
- `tool_calls[].id` 与 `tool.tool_call_id` 必须配对
- **`function.arguments` 用 dict（对象）**，不用 JSON 字符串——内部以可校验性优先（可直接挂 JSON Schema 校验）；对接线上 API 时再序列化
- `tool.content` 是**字符串化的 JSON**（不是对象）——贴合 OpenAI 标准
- assistant 可以同时带 `content` 与 `tool_calls`，也可只有其一；不再调用工具时 `tool_calls` 为 `null`（显式保留字段）
- **`reasoning_content`（思考链）为 assistant 消息的标准可选字段**——种子轨迹保留（judge 评审、蒸馏复用）；RL 任务行不存
- **从轨迹造 RL 任务的截断规则**：截到最后一条 user 消息（含其前的 system/user），之后的 assistant/tool 轮全部丢弃，转化为 `label` + `verifier`（§7）

## 6. 模拟工具环境（Simulated Tool Environment）

### 6.1 定位：不是训练时的 agent runtime

两者常被混淆，职责与归属完全不同：

| | 训练时 agent runtime | 造数据的模拟工具环境 |
|---|---|---|
| 归属 | **训练框架**（verl 的 `ToolAgentLoop` / slime 的 custom generate 函数） | **dataforge**（数据构造管线组件） |
| 职责 | RL rollout 时驱动多轮循环：模型发 tool_call → 执行工具 → 结果回填 → 继续生成 | 生成种子轨迹时扮演「工具执行环境」角色，对 tool_call 返回结果 |
| 运行时机 | 训练中，每 step 大量并发 | 离线，造数据时执行 |
| dataforge 是否实现 | ❌ 不实现，复用训练框架现成的 | ✅ 实现，是核心组件之一 |

dataforge **不做训练时 runtime**。本节只讨论后者。

### 6.2 三个用途（对应 §9 流程）

1. **种子合成**（§9.1）：LLM 合成轨迹时，Agent 角色发出 tool_call，模拟环境返回工具结果（Nemotron-Agentic-v1 即此范式）
2. **可验证性试跑**（§9.4）：教师模型试解 N 次测 pass rate，过程中工具必须有响应可返回
3. **回放验证**（§9.5）：「同一 label 两次回放打分一致」要求工具响应可复现

### 6.3 两条实现路线

| 路线 | 机制 | 优点 | 缺点 |
|---|---|---|---|
| **A. 确定性 mock** | 预先录制/生成好工具响应，调用时查表返回 | 严格可复现；回放零成本 | 覆盖有限，长尾调用无响应可用 |
| **B. LLM 模拟器** | 用 LLM 扮演环境，根据工具 schema + 调用参数生成逼真响应 | 覆盖广、响应自然 | 不可复现（同参数两次响应可能不同）；有 API 成本 |

### 6.4 混合策略（定稿）

**生成期用 B，验证期用 A**：

```
种子合成（一次性）：
  Agent(LLM) 发 tool_call
      → LLM 模拟器（路线 B）生成工具响应
      → 响应固化进轨迹的 tool 消息（§5）
              ↓
轨迹落盘 = 响应已冻结（此后不再调用 LLM 模拟器）

回放 / 试跑 / 校验（可多次）：
  读轨迹中已固化的 tool 响应（路线 A 语义，查表）
      → 确定性回放，打分可复现
```

要点：
- **LLM 模拟器只在生成期运行一次**，其产出固化在轨迹里；轨迹一旦落盘即冻结
- 回放/试跑/校验**永远从固化响应读取**，保证 §9.5「两次回放打分一致」可成立
- `state_check` 类 verifier（断言环境终态）依赖此确定性：环境终态在生成期由模拟器产出后，同样固化进 `label.final_state`
- LLM 模拟器的模型与版本记入轨迹 `metadata.env_model`，保证生成过程可追溯（区别于可复现）

## 7. 验证依据编码（`verifier` + `label`）

### 7.1 职责划分

- `verifier`：**怎么判**——验证方法名，指向注册的校验函数（枚举，见 7.2）
- `label`：**拿什么判**——标准答案/期望行为，结构由 verifier 类型决定

### 7.2 verifier 枚举与 label 结构

| verifier | 判什么 | label 结构 | 适用 |
|---|---|---|---|
| `exact_match` | 最终答案字符串精确匹配（忽略空白/大小写可配） | `{"answer": "42", "format": "#### {}"}` | 知识问答 |
| `math_answer` | 数值等价（容忍分数/逗号/精度） | `{"answer": "3.5", "tolerance": 1e-6}` | 数学 |
| `tool_call_match` | 轨迹中的工具调用与期望一致（名+参数） | `{"expected_calls": [{"name": "f", "arguments": {...}}], "order": "strict"\|"any"}` | 单轮工具选择 |
| `tool_trajectory_match` | 多轮：每步工具名+关键参数匹配 | `{"expected_steps": [...], "allow_extra_steps": false}` | 多轮规划 |
| `state_check` | 模拟环境终态达成（字段断言，终态由 §6.4 固化） | `{"final_state": {"order_status": "cancelled", "refund": "full"}}` | agentic 任务 |
| `judge` | LLM 评审打分（非严格可验证，需注明模型与版本） | `{"rubric": "...", "min_score": 4}` | 开放任务，慎用 |

原则：**优先可机械验证的 verifier**；`judge` 仅作补充，且必须在 metadata 记录评审模型，保证可复现。

### 7.3 奖励三档（构造时的自检顺序）

1. **格式合法**：tool_calls 可解析、参数过 schema 校验（纯规则，零成本）
2. **工具正确**：调用名与参数命中 label 期望
3. **任务完成**：最终答案正确 / 环境终态达成

设计奖励时从 1→3 递进给分，避免「一步不对全零」导致的稀疏奖励问题（可给部分分）。

## 8. 训练框架对接（导出适配，附录）

本节仅为交付参考，不影响 §2 的规范主体。各框架消费方式不同，适配发生在**导出器**：

### 8.1 导出 → slime（近零转换）

slime 的数据键名可配置（`--input-key` / `--label-key` / `--tool-key`），且工具内联在数据行、加载时传入 `apply_chat_template(tools=...)`——与 §2 的自包含设计天然一致：

| dataforge | slime 侧 |
|---|---|
| `messages` | `prompt`（`--input-key prompt`） |
| `label` | `label`（`--label-key label`） |
| `tools` | `tools`（`--tool-key tools`，即默认值） |
| `verifier` | `metadata.verifier`，由自定义 RM 函数（`--custom-rm-path`）读取 |
| `metadata.*` | `metadata` |

### 8.2 导出 → verl（需转换）

verl 的工具在 rollout 侧全局注册、样本按 `extra_info.tool_selection` 过滤；验证依据包在 `reward_model` 里，按 `data_source` 路由 reward manager：

| dataforge | verl 侧 |
|---|---|
| `messages` | `prompt` |
| `label` + `verifier` | `reward_model: {"style": "rule", "ground_truth": label}`；verifier 并入 `data_source` 命名空间（如 `dataforge/tool_call_match`） |
| `tools` | 全局工具注册 + `extra_info.tool_selection`（导出时抽公共工具集） |
| `id` / `source_id` | `extra_info.index` / `data_source` |

> 两框架互操作捷径：slime 键名可配，verl 格式数据可 `--label-key reward_model` 直接喂 slime（slime 官方 search-r1 示例即如此复用 verl 社区数据）。

## 9. 构造流程（pipeline 各步 DoD）

1. **种子采集**：来源 = 已有 SFT 轨迹、公开数据集（注意许可）、LLM 合成（用户模拟器 + Agent + 模拟工具环境，见 §6）
   DoD：每条种子有唯一 id、来源标记（`metadata.source`）
2. **任务定义**：写出 messages（system 规则 + user 任务）与该任务可用 `tools`
   DoD：messages 以 user 结尾；工具名与 schema 一致
3. **验证依据构造**：从轨迹提取期望调用序列/最终答案/终态，填 `label` + `verifier`
   DoD：label 结构与 verifier 类型匹配（§7.2 表）
4. **可验证性试跑**：用强模型（教师）在模拟环境（§6.4 回放模式）试解 N 次
   DoD：pass rate ∈ 合理区间（如 20%–90%）；过易（>95%）或过难（<5%）的任务丢弃或改造——保证有学习信号
5. **内部校验**：`dataforge validate`
   DoD：全部规则（§10）通过
6. **导出**：按训练框架选择导出器（slime 近零转换 / verl 需转换，§8）
   DoD：目标框架可加载数据，rollout 冒烟通过

## 10. 校验规则清单（validate 强制项）

- [ ] `messages` 非空且最后一条是 `user`；`system` 只在首位
- [ ] messages 不含答案文本（启发式检查 label 字符串是否泄漏进 messages）
- [ ] `tools` 中 `name` 唯一；`parameters` 是合法 JSON Schema（object）
- [ ] 消息中实际调用的工具名 ⊆ `tools`（防悬空引用）
- [ ] `verifier` ∈ 注册枚举；`label` 结构与该 verifier 的 schema 匹配
- [ ] 轨迹中 `tool_calls[].id` 与 `tool.tool_call_id` 一一配对（轨迹行）
- [ ] 确定性：同一 label 两次回放打分一致（`judge` 类豁免，但需记录评审模型）
- [ ] `id` 全局唯一

## 11. 完整示例：多轮工具任务

场景：用户查询 BTC 余额并追问交易历史（改编自 xlam_fc 种子）。

```json
{
  "id": "rl-toolcall-zh-0007",
  "task_type": "rl_task",
  "messages": [
    {"role": "system", "content": "你是数字资产助手。可用工具以 JSON Schema 提供。需要查询时先调用工具再回答；一次只调用一个工具；无法通过现有工具完成的请求，明确告知用户并建议替代方案。最终回答用中文。"},
    {"role": "user", "content": "帮我查一下地址 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa 的比特币余额。另外这个地址的第一笔交易是什么时候？"}
  ],
  "tools": [
    {"type": "function", "function": {
      "name": "get_bitcoin_balance",
      "description": "查询给定比特币地址的当前余额。",
      "parameters": {
        "type": "object",
        "properties": {
          "address": {"type": "string", "description": "BTC 地址"}
        },
        "required": ["address"]
      }
    }},
    {"type": "function", "function": {
      "name": "get_eth_balance",
      "description": "查询给定以太坊地址的 ETH 余额。",
      "parameters": {
        "type": "object",
        "properties": {
          "address": {"type": "string", "description": "ETH 地址"}
        },
        "required": ["address"]
      }
    }}
  ],
  "verifier": "tool_call_match",
  "label": {
    "expected_calls": [
      {"name": "get_bitcoin_balance", "arguments": {"address": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"}}
    ],
    "refusal_expected": {
      "topic": "交易历史查询",
      "reason": "工具集不含交易历史接口"
    },
    "order": "any"
  },
  "metadata": {
    "source": "xlam_fc",
    "split": "train",
    "index": 7
  }
}
```

该例考察两个点：① 会调对工具查余额；② 对工具集外的请求**诚实拒答**而不是编造。注意 `get_eth_balance` 是刻意保留的干扰项。

---

## 附录：字段速查

```
RL 任务行（本规范主体）
├── id / task_type("rl_task")
├── messages        # 标准 OpenAI messages: [system?, ..., user]
├── tools           # 标准 OpenAI function schema 列表（内联自包含）
├── verifier        # 验证方法名（枚举）
├── label           # 验证依据（结构由 verifier 决定）
└── metadata        # source / split / index / …自由扩展

轨迹行（种子/judge/回放用）
├── id / task_type("trajectory")
├── messages        # 完整多轮：assistant(tool_calls, reasoning_content?) ↔ tool(tool_call_id)
├── tools           # 同上
└── metadata        # judge_score / env_model 等

模拟工具环境（§6，dataforge 组件）
├── 生成期：LLM 模拟器 → 响应固化进轨迹
└── 验证期：读固化响应 → 确定性回放

导出适配（附录，不进规范）
├── → slime：键名映射（prompt/label/tools），近零转换
└── → verl：reward_model 包装 + 工具抽全局注册 + tool_selection
```

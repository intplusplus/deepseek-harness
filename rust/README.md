# dsh — Rust foundation

> 本目录是 DeepSeek Harness 的 **Rust 重构起点**，与现有 TypeScript monorepo **并存**、互不干扰。
> 目标不是一次性重写 4000+ TS 文件，而是先把**领域骨架**（领域模型 + 不变量 + 线格式）
> 用 Rust 的类型系统重新锚定一遍，再逐块把实现搬过来。

当前状态：**骨架 + 工具/权限 + 异步驱动（重试恢复与 post-execute）+ system-prompt 宿主
（change 事件 / assemble waterfall / scope 链父级遍历 / 整层销毁）+ compaction + context
（含 file/session reference 独立测试）+ todo/plan + session 投影 + skill 完成，626 个测试全绿，
`cargo clippy --workspace --all-targets -D warnings` 与 `cargo fmt --check` 干净**。
`cargo run -p dsh` 跑一条由真实异步驱动跑出来的 turn（2 个 step、1 次真实工具执行）。

---

## 1. 为什么这样切

TS 版依赖三件 Rust 没有的东西，重构的绝大部分设计决策都围绕替代它们：

| TS 机制 | Rust 里的替代方案 |
| --- | --- |
| 声明合并（`interface SessionEventMap { 'compaction/x': X }`） | **封闭 Core 联合 + Extension 槽**：`enum SessionEvent { Core(CoreEvent), Extension(ExtensionEvent) }`，Extension 原样保留 `serde_json::Map` |
| 结构化类型 / 鸭子类型 | trait + newtype，跨 crate 只通过具体类型交互 |
| 可逆 effect 的 `ctx.effect()` 闭包 | **RAII guard**：`ServiceGuard` / `ObserverGuard` 持有恢复逻辑，`Drop` 时回滚 |

其余部分是直接搬运：品牌化 ID → `#[serde(transparent)]` newtype + 验证型构造器；
append-only 日志 → `Vec<SessionEvent>` + 三重校验；派生状态 → 纯函数 fold。

---

## 2. Crate 划分与 TS 映射

```
rust/
├─ Cargo.toml                workspace（resolver 2，共享 serde/serde_json/thiserror/uuid/tokio/async-trait）
├─ fixtures/                 TS v4 线格式 golden fixture 与生成脚本
└─ crates/
   ├─ dsh-plugin/            Cordis 式插件内核
   ├─ dsh-llm/               provider 中立的 LLM 消息词汇
   ├─ dsh-session/           append-only session 事件日志 + 持久化
   ├─ dsh-agent/             agent 身份与 inbox 折叠
   ├─ dsh-tools/             工具契约、注册表、权限 gate、模型序调度
   ├─ dsh-provider/          模型调用边界：异步 provider trait
   ├─ dsh-agent-loop/        turn/step 异步驱动
   ├─ dsh-retry/             provider 路由的请求重试：策略、退避、持久化事件、可取消等待
   ├─ dsh-system-prompt/     system prompt 注册表与组装规则
   ├─ dsh-compaction/        compaction 事件、surface/prune 规则、预算与触发策略
   ├─ dsh-context/           请求时钟、文件引用、session 引用
   ├─ dsh-todo/              todo_write 状态模型与 fold
   ├─ dsh-plan/              plan mode 状态机与 exit_plan_mode
   ├─ dsh-session-projection/ session 投影注册表与 drive
   ├─ dsh-session-stats/     token / turn 统计 fold
   ├─ dsh-session-turn-outline/ turn 大纲 fold
   ├─ dsh-skill/             skill 词表、校验、排序合并、模型侧渲染
   └─ dsh/                   headless 演示二进制（端到端冒烟）
```

| TS 包 | Rust crate | 已移植 |
| --- | --- | --- |
| `cordis`（外部依赖） | `dsh-plugin` | 服务注册/解析、observer 与 waterfall 事件、`Scope` 可逆 effect、插件加载失败回滚 |
| `packages/llm/llm` | `dsh-llm` | `ContentBlockMap`（7 种）、`Message`（5 角色）、`MessageSource`、`StreamChunk`、`AssistantStreamRecord`、`LlmCallConfig`、`ToolSchema`、`TokenUsage`、`FinishReason`、`LlmFailure` |
| `packages/core/session` | `dsh-session` | `SessionEventMap`（14 核心事件）、`SessionHeader`/`EpochHeader`/`RequestContext`、seq 与安全整数校验、canonical header 校验、`SESSION_FORMAT_VERSION = 4`、4 个 fold、JSONL 读写与代次选择 |
| `packages/core/agent` | `dsh-agent` | `Agent`、`InboxTarget`、`agent/inbox/spliced` 折叠 |
| `packages/llm/llm`（provider 边界） | `dsh-provider` | `LlmRequest` / `LlmResponse`、异步 `LlmProvider` trait（`async_trait`，保持 `&dyn` 可用）、`ScriptedProvider` / `EchoProvider` / `FnProvider` |
| `packages/core/tools` | `dsh-tools` | 工具契约、`ToolRegistry`、`CancelSignal`、`parse_arguments`、权限链（`PreToolDecision` / `PostToolDecision`、waterfall 收敛、单调 guard、`ToolRestriction`） |
| `packages/core/agent-loop`（tool-calls 部分） | `dsh-tools::schedule` | 模型序提交、exclusive 屏障、bounded parallel 组真正并发、取消补记合成结果、post-execute 接线 |
| `packages/core/agent-loop`（turn 推进） | `dsh-agent-loop` | 异步 turn/step 边界、request header 与 route 首次落盘、工具结果上下文回收、step/attempt 上限、`agent/request-error` 恢复点、可取消等待 |
| `packages/llm/llm-retry` | `dsh-retry` | 策略解析与校验、有界指数退避 + 对称抖动、`llm/retry` 与 `llm/retry-started` 持久化事件、重试状态 fold、纯决策函数 `plan_retry`、异步可取消 `RetryDelay` |
| `packages/core/system-prompt` | `dsh-system-prompt` | section/context/tool/variable 注册表（RAII `Registration`）、中心化 order 表、严格 `{{variable}}` 插值、canonical 排序、suppression、`refreshContext`、Config 校验 |
| `packages/compaction/compaction` | `dsh-compaction` | 4 个 `compaction/*` 事件、checkpoint source、tool pairing、surface fold |
| `packages/compaction/compaction-tool-result-pruner` | `dsh-compaction::prune` | 确定性 head/middle/tail 剪枝与预算 |
| `packages/compaction/compaction-basic`（纯策略部分） | `dsh-compaction::policy` | 容量预算、pressure 阈值、retention 校验、attempt 上限、overflow 恢复 |
| `packages/context/time-context` | `dsh-context::time_context` | 请求时钟、时区解析、时间戳/时长渲染、注入调度、browser zone 策略 |
| `packages/context/file-reference` | `dsh-context::file_reference` | `@path` 语法、候选 token、模型 guidance |
| `packages/context/session-reference` | `dsh-context::session_reference` | canonical URI、mention 解析、surface 投影、byte 预算预览、spill 规则 |
| `packages/todo/tool-todo` | `dsh-todo` | `todo_write` 契约、整表替换、`todo/write` 事件、last-write-wins fold |
| `packages/plan/plan-mode` | `dsh-plan` | 持久化模式开关、选择/flush 状态机、`/plan` 命令、`exit_plan_mode` 工具与审阅流程 |
| `packages/session/session-projection` | `dsh-session-projection` | 投影注册表、eager drive、每 session 水位、变更订阅、checkpoint/restore/hydrate |
| `packages/session/session-stats` | `dsh-session-stats` | token / turn / tool 统计 fold 与客户端视图 |
| `packages/session/session-turn-outline` | `dsh-session-turn-outline` | turn 大纲 fold、prompt/response 预览、结束状态 |
| `packages/skill/skill` | `dsh-skill` | 名称语法、词表与线格式、候选/定义校验、层内 rank 合并与跨层 shadowing、`<skill_content>` 渲染 |
| `packages/core/scope` | — | 语义并入 `dsh-plugin::Scope` |

---

## 3. 关键设计决策

### 3.1 声明合并 → Core + Extension

TS 里插件靠 `declare module` 往事件表、内容块表、消息源表里加成员。Rust 没有这个能力，
所以每个可扩展的词汇表都拆成两层：

```rust
// 封闭的核心联合：已知 tag 严格反序列化，畸形 payload 直接报错
enum CoreEvent { TurnStart { .. }, AssistantMessage { .. }, .. }

// 开放槽位：未知 tag 原样保留，读不回的类型可安全忽略
struct ExtensionEvent { tag: String, payload: Map<String, Value> }

enum SessionEvent { Core(CoreEvent), Extension(ExtensionEvent) }
```

**手写 serde 路由**是必须的：内部标签 enum（`#[serde(tag = "type")]`）无法携带任意未知 payload，
所以序列化按 tag 分发、反序列化先试 `CoreEvent` 再降级 `Extension`。同一个模式用在
`ContentBlock`、`MessageSource`、`InboxTarget` 上。
关键不变量：**已知 tag 绝不静默降级为 Extension** —— 否则一个字段笔误会变成"未知插件事件"被忽略。

### 3.2 可逆 effect = RAII

```rust
let guard = ctx.services.insert(Arc::new(MyService));  // 返回 ServiceGuard
drop(guard);                                            // 恢复上一个提供者或清空槽位
```

`Scope` 以 LIFO 顺序持有 guard，`Drop` 时倒序 unwind。插件 `load()` 返回 `Err` 时
`PluginHost` 直接 drop 整个 scope —— 已注册的服务、observer、defer 全部回滚（demo 第 7 段验证）。

### 3.3 Observer vs Waterfall

- **observer**：`catch_unwind` 包容 panic，监听器返回值被丢弃，**不能否决**事件。
- **waterfall**：监听器必须调用 `next(value)` 委托；不调用即短路。`run_waterfall` 用递归链实现，
  保证"每个监听器都恰好跑一次、顺序确定"。

### 3.4 事件布局：小事件不该为大事件付账

日志是 append-only 且无上界的，所以**事件的 inline 体积由每个事件共同承担**——
一个只有 `turn: 1` 的 `turn/start` 不该和一个带完整消息 + 流记录 + usage 的
`assistant/message` 一样大。所有超过几个指针的载荷都装箱：

```rust
RequestHeader { header: Box<EpochHeader>, .. }          // 160B -> 8B
AssistantMessage { message: Box<AssistantMessage>,      // 128B -> 8B
                   usage: Option<Box<TokenUsage>>, .. } //  80B -> 8B
FinishReason::Error { failure: Box<LlmFailure> }        // 104B -> 8B
AssistantStreamRecord::Chunk { chunk: Box<StreamChunk> }
```

结果：`SessionEvent` **280B → 88B**，`AssistantStreamRecord` **176B → 112B**，
`TurnEndReason` **104B → 24B**。流记录尤其关键——一条长响应会存成百上千条。
`Box<T>` 的序列化与 `T` 完全一致，所以线格式不受影响（有测试锁定）。
`tests/log.rs::an_event_stays_small_regardless_of_its_largest_variant` 防止回归。
`dsh-compaction::CompactionEvent::Summary` 同样装箱：它是解码视图，不是落盘结构。

### 3.5 工具调度：顺序是唯一不能让步的事

`dsh-tools::schedule` 只保证两件事，其余都可以后补：

1. **结果按模型序提交。** 派发可以并发，但策略、结果、结果上下文绝不重排。
   日志里每条 `tool/result` 都与发起它的 `tool/call` 配对。
2. **每个模型调用都必须落定。** 一个从未启动的调用也会拿到一对
   `tool/call` + `tool/result`，携带合成的 `ABORTED_BEFORE_DISPATCH` 错误——
   否则 `derive_messages()` 会看到一个没有回答的调用。

当前并行组把 body 派发到 Tokio blocking pool，最多 `max_parallel_tool_calls` 个同时在跑；
提交仍在调度器自己的任务上按模型序完成。exclusive 调用是屏障。并发只改变"body 什么时候跑"，
不改变可观测日志——`parallel_bodies_overlap_up_to_the_configured_bound` 用 barrier 证明真正重叠，
`barriers_do_not_change_the_committed_log` 证明日志与串行一致。

### 3.6 工具权限：waterfall 收敛 + 单调 guard

`dsh-tools::gate` 移植 `packages/core/tools` 的权限链。纯规则全部可离线测试：

| 规则 | 函数 | 语义 |
| --- | --- | --- |
| waterfall 收敛 | `converge_pre_tool(votes, default)` | 第一个不委托的 vote 胜出；全委托则收敛到注册表的 `allow` 默认 |
| 单调 guard | `apply_guard(decision, guard_reason)` | 只能把 allow 变成 deny，**永不重议** waterfall 已拒绝的决策 |
| ask 降级 | `resolve_ask(name, reason, outcome)` | 无审批 seam 时 ask→deny；四种审批结果一一映射到 allow/deny |
| 决策→动作 | `evaluate_pre_tool` / `evaluate_pre_dispatch` | deny 物化成带 reason 的失败，cancel 选出规范的 before-dispatch 中止 |
| 后置应用 | `apply_post_tool` | accept 保留或替换**一个**投影；block 把纠正反馈变成无值失败 |
| 可见性过滤 | `ToolRestriction::new/admits` | allow/deny 集合快照编译，链上求交 |

三个刻意的建模决定：

1. **`PostToolDecision::Accept` 的 content/value 互斥用 enum 变体表达，不是两个 `Option`。**
   payload 是 `AcceptPayload::{Plain, Content(Vec<ContentBlock>), Value(Value)}`。
   它的 serde 是**手写**的：untagged 两变体联合在 `{"kind":"accept","value":1}` 下会因未知字段
   被忽略而误配到 content 变体，手写才能同时保住 tag 保真与 TS 的运行时校验。
2. **gate 走独立入口函数，不给 `ToolSchedule` 加字段。**
   `execute_tool_calls_gated(..., gate)` 是独立入口，原 `execute_tool_calls` 保持签名并委托
   `AllowAllGate`。
3. **gate 位置在 `tool/call` 落盘后、查注册表前**，对齐 TS 的 pre-execute。
   post-execute 在**提交前**运行：dispatched body、unknown tool、gate deny/cancel 都会过
   `PostToolGate` + finalize；只有"取消从未启动"的合成 abort 对直接落盘，对齐
   `appendSkippedToolCall`。

### 3.7 品牌化 ID 与 seq

`branded_id!` / `branded_number!` 宏生成 newtype，构造器做验证：
`SessionId::new("")` 报错；`SessionSeq::new(n)` 拒绝负数、非整数、NaN、> 2^53-1。
`Session::append` 额外强制 **seq 严格连续**，从源头掐断"日志出现空洞"。

### 3.8 驱动层：turn 与 step 的边界

`dsh-agent-loop` 把"一轮对话"拆成 **turn 套 step** 两层，两层都是日志里的事件，
不是内存里的游标：

```
run_turn()  (async)
├─ turn/start(turn)
├─ user/message × n          ← 排队的输入先落盘，成为派生历史的一部分
├─ run_steps()
│   └─ 循环直到出现结束原因
│       ├─ step/start(turn, step)
│       ├─ request/header + request/context   ← 仅当日志里还没有（首次）
│       ├─ derive_messages(events) → LlmRequest
│       ├─ provider.generate().await → assistant/message
│       │     └─ 失败？→ recover_request()          ← §3.10
│       ├─ 无 tool call → 结束（按 finish_reason 映射）
│       ├─ execute_tool_calls().await → tool/call … tool/result
│       ├─ 延迟上下文 → user/message（下一步的输入）
│       └─ step/end(turn, step)
└─ turn/end(turn, reason)
```

三条约定：

1. **驱动不持有历史。** 每步都从 `derive_messages(session.events())` 重新派生，
   `call_config()` 也从 `fold_request_header` 读回来 —— 内存里重复的只有 turn/step 计数器。
2. **request header 只写一次。** 之后的 step 复用 fold 出来的 config；route context 同理。
3. **`DEFAULT_MAX_STEPS = 32` 是 Rust 侧的边界。** TS 的循环靠模型自己停下；
   离线驱动必须有个上界，超界以 `TurnEndReason::Error`（code `STEP_LIMIT`）结束而不是挂住。

### 3.9 重试：把决策从等待里拆出来

`dsh-retry` 是第一个真正走 Extension 槽的插件事件。TS 里 `llm/retry` 靠
`declare module` 并入 `SessionEventMap`；Rust 里它是 `dsh-retry` 的普通类型，通过
`encode_retry` / `decode_retry` 进出 `SessionEvent::Extension`。

**决策与等待分离。** `plan_retry(provider, failure, policy, state, random)` 返回
`RetryPlan`（延迟多少、第几次、链 ID），**等待由调用方去做**。规则——预算耗尽、
`Retry-After` 超上限、非可重试码、策略切换重置链——都可以在没有时钟的情况下断言。

本轮把等待也异步化了：`RetryDelay` 是 `async_trait`，driver 用
`tokio::time::timeout`/`select` 把它与 `RetryConfig::cancel` 竞争；取消返回
`TurnEndReason::Aborted { User }`，且 `llm/retry` 已经在等待前落盘。

三条刻意保留的 Rust 化差异：

1. **`delayMs` 与 `policyKey` 里的数字按 JS 写。** `JSON.stringify(500)` 写 `500`，
   `serde_json` 对同样的 `f64` 写 `500.0`；`json::number()` 把整数值 `f64` 写成整数。
   测试 `the_policy_key_matches_the_typescript_bytes` 直接对拍 TS 的字节。
2. **已知 tag 坏了是错误，不是跳过。** `decode_retry` 对别的 tag 返回 `None`，
   对 tag 正确但 payload 畸形返回 `Some(Err)`。
3. **重试状态用 `BTreeMap`**，键是 `retry_state_key(provider, policyKey)` 的 JSON 字符串；
   fold 被重放，必须确定。

### 3.10 恢复点：`recover_request`

驱动层在 `provider.generate().await` 失败的地方开一个恢复点，对应 TS 的 `agent/request-error`。
`LoopConfig.retry: Option<RetryConfig>` 为 `None` 时行为与接入重试前完全一致。

配了策略时：

```
generate() 失败
├─ 没有该 provider 的策略 ────────────────→ 以 failure 结束 turn
├─ attempts >= max_attempts ──────────────→ 以 code "ATTEMPT_LIMIT" 结束
├─ fold_retry_state(events) + plan_retry()
│     └─ 决策为「不重试」──────────────────→ 以 failure 结束 turn
└─ 决策为「重试」
    ├─ append llm/retry          ← 先落盘，再等待
    ├─ delay.wait(delay_ms).await  raced with cancel
    │     ├─ 取消 ────────────────────────→ TurnEndReason::Aborted { User }
    │     └─ 完成
    └─ append llm/retry-started → 同 step 再来一次 attempt
```

**`llm/retry` 先落盘，`llm/retry-started` 后落盘。** 进程在等待期间死掉，日志里仍然留着
"曾经排定过一次重试"。**重试不新开 step**：预算花在同一个 step 的 attempt 上，
每 attempt 都重新 `derive_messages`，失败诊断文本不进入上下文。
`DEFAULT_MAX_ATTEMPTS = 8` 是 Rust 侧新增的边界。

### 3.11 system-prompt：注册表 + 组装规则

`dsh-system-prompt` 移植 `packages/core/system-prompt`。TS 侧那是一个 **Cordis 服务**：
section / context / tool provider / variable 通过 `ctx.effect()` 的 disposer 注册，
每次注册与销毁都发 `system-prompt/change`，`assemble()` 跑一条协作式 waterfall。
这些是**生命周期**，需要插件宿主；Rust 侧搬走一切"有数据、有规则、有线格式"的东西。

**注册是可逆 effect 的 RAII 形状。** 每个注册返回一个 `Registration` guard，
drop 时精确移除自己加的东西。Provider 是注入的闭包（`Fn(&AssembleContext) -> ..`），
所以注册表内部**从不**调用时钟、文件系统或随机源。

两套刻意区分的集合语义：

| 对象 | 语义 | 原因 |
| --- | --- | --- |
| variables | **活迭代** | 复刻 TS 的 live `Map` 迭代器：早注册的 provider 在当次组装中再注册的 provider，当次就可见；含"表被抽干即停止看见新插入"的 detach 规则 |
| tool providers | **快照** | 成员在任何 provider 运行前冻结，组装途中再注册不改变本次组装 |

**顺序是最不能漂移的部分**：section 按 `(order, 码元名称)` 排序、context 按 order、
tool 按配置的顺序（未列出的按字典序排在 `TOOL_ORDER_REST` 之后）。

**这一层抓到的缺陷**：`Registration::drop` 一开始会调 `Rc::upgrade()`，
而它自身就持有一个 `Weak` —— 在 `Scope` LIFO unwind 下后 drop 的 guard 会触发
`already borrowed: BorrowMutError` panic。修法是**不在 drop 里 upgrade**。

### 3.12 线格式保真

- 字段一律 `#[serde(rename_all = "camelCase")]`，事件 tag 显式命名（`"turn/start"` 等）。
- `EpochHeader` **没有** `system` 字段 —— TS 侧它在类型上存在但线格式上不可表示。
- `offloaded` 这类 `field?: true` 语义用 `#[serde(skip_serializing_if = "Not::not")]` 表达。
- `Session::append` 做**无损 JSON 往返校验**：`to_value` → `from_value` → 相等。
- canonical header 校验拒绝 `tools: []` 和空 `adapterDefaults`。
- **信封是扁平的，判别标签在最前。** 当前（v4）格式没有 v0 时代的嵌套信封；
  TS 的 `encodeCurrentEvent` 一路恒等传下来，所以事件就是
  `{"type":"turn/start","turn":1}` 这样的扁平对象。

#### 字段顺序：`preserve_order` 打开之后才成为可观测属性

`serde_json` 默认把 `Map` 当 `BTreeMap`（按键排序），插件事件的声明顺序在落盘时丢失。
工作区根 `Cargo.toml` 已打开 `preserve_order`。两处因此修掉：

1. **`ExtensionEvent::new` 把 `type` 前置。** TS 每一行都以 `type:` 开头。
2. **`LlmRetryEvent` 手写 `Serialize`。** TS 声明序是
   `retryId, turn, step, provider, mode, policyKey, retry, maxRetries, delayMs, failure`
   —— `mode` 是**第五**个键，而 `#[serde(tag = "mode")]` 会把标签顶到最前。

四个字节级测试钉住这两条；`dsh-compaction` 也补了 summary/prune 的精确 JSON 断言。

### 3.13 异步化：provider、调度与取消

`LlmProvider::generate` 是异步的：模型调用在等网络，阻塞 trait 会逼驱动为每个请求开线程。
驱动持有 `&dyn LlmProvider`，所以 trait 必须 object-safe；原生 `async fn` in trait 不满足
dyn-compatibility，于是用 `async_trait`，把每个方法统一成 boxed `Send` future。
`FnProvider` 仍接收普通闭包，测试脚本不必变成 future。

调度器的 body 是同步 `ToolHandler`，所以跑在 Tokio blocking pool 上；`JoinSet` 负责
bounded group 的开始/汇合。三条不变量（模型序提交、每个调用落定、exclusive 屏障）
在并发下保持不变。取消是 `CancelSignal` 加上 `RetryConfig::cancel` 的 `watch` 通道，
等待与调度都能被同一个取消源打断。

### 3.14 compaction：规则与锁生命周期分开

`dsh-compaction` 搬走所有可离线判定的东西：

- `events`：4 个 `compaction/*` 扩展事件的类型与**精确字段序**（含 `type` 前置）。
- `surface`：按契约邻接应用 summary/prune 替换的纯 fold。
- `pairing`：surface 位置上的 tool-call/result 平衡。
- `prune`：确定性 head/middle/tail 工具结果剪枝与预算。
- `state`：lock 检查、range 校验、重放投影。
- `policy`：容量预算、pressure 阈值、retention 校验、attempt 上限、overflow 恢复。

**未搬**：异步 backend（真正调用模型写 summary）、durable lock 生命周期、手动命令入口、
token meter / attachment 服务、session append 边界。`policy` 里 `thresholdTokens` 与
`retainTokens` 的关系是可单测的纯函数；本轮修掉了两个因测试窗口取值不当而误报的用例
（`contextWindow` 必须大于 headroom）。

### 3.15 context：环境边界全部注入

`dsh-context` 不读进程时钟、文件系统、环境或随机源：

- `time_context`：`TimeZoneResolver` 注入时区表，`should_inject(interval, now, last)` 是纯函数；
  渲染文本（含 browser zone 策略、elapsed）按 TS 逐字对齐。
- `file_reference`：`@path` 语法、cursor 位置的 active token、模型 guidance。
- `session_reference`：canonical URI、mention 解析、surface 投影、byte 预算预览；
  真正的 spill 写入在 `TextSpillSink` 后面。

**已知缺口**：`time_context` 有 21 个测试；`file_reference` / `session_reference`
的实现已落地并通过编译，但**没有独立测试**（子 agent 在补测试前被限流中断）。

### 3.16 todo 与 plan mode：日志驱动的状态机

- `dsh-todo`：`todo_write` 是整表替换；每次写入落一条 `todo/write` 扩展事件；
  fold 是 last-write-wins，`turn/start` 清空、`turn/end` 保留。模型可见的确认文本与
  参数/输出 schema 按 TS 对齐。宿主只负责把返回的事件 append 进日志。
- `dsh-plan`：`plan/mode` 是持久化开关；`/plan` 命令和 `exit_plan_mode` 的两阶段
  （set→commit、decide→commit）状态机是纯函数。plan mode **不通过 ToolRestriction 限制工具
  catalog**——TS 侧同样只靠 guidance 约束，本 crate 的 `tool_restriction()` 明确返回"无限制"。
  审阅问题、答案分类、退出决定都可离线测试；用户问题通道本身留给 interaction 服务。

两个 crate 都**不**调用模型、不执行工具、不 append 日志。

### 3.17 session 投影：类型擦除 + 水位

`dsh-session-projection` 是 GUI/cache 层的读模型：

- 键是 `&'static str`，状态是 `ProjectionState`（`Serialize + DeserializeOwned + Send + Sync`），
  定义持有 `init(header, inheritedEventCount)`、`apply(state, event, seq) -> Option<S>`、
  可选的 `view -> Arc<W>` 和缓存版本。
- `apply` 返回 `None` 表示"不是我的事件"，注册表复用原 `Arc`，不算视图、不发通知；
  `Some` 总是新引用。客户端视图用 `Arc::ptr_eq` 比较两次分配。
- 每 session/key 一个水位；`drive_event` 先静默折叠缺失前缀，只有目标事件可能触发通知；
  所有单元完成后再统一发 `ProjectionChange`。
- `checkpoint`/`restore`/`hydrate` 是持久化缓存的契约：版本不匹配、坏行、水位超出日志末尾
  的处理按"全量读丢弃、尾读拒绝"区分。
- `dsh-session-stats` 与 `dsh-session-turn-outline` 是两个具体 fold；状态用
  `deny_unknown_fields` + 显式 `validate()` 代替 Zod。

**刻意差异**：TS 用 `WeakMap<Session, ...>` 按对象身份隔离，Rust 用 `SessionId` 键；
TS stats 的墙钟字段（`llmMs`/`ttftMs` 等）没有对应值，因为核心事件不带时间戳；
outline 额外记录 `endSeq`/步骤/工具计数。`stateVersion` 分别用 2/3，避免与 TS 缓存行混用。

### 3.18 skill：名字即类型

`dsh-skill` 把 kebab-case 语法做成 `SkillName` newtype：非法名字不可表示，反序列化也校验。
词表（summary / resourceBase / invocationSource）按 TS 声明序序列化；合并规则是
层内 `(rank, providerOrder, localOrder)` 排序去重、跨层"最近层直接胜出"，
runtime rank 250、bundled 600。`render_skill_content` 与 TS 逐字节一致，
`escape_text` 先转义 `&` 再转 `<`/`>`。

**未搬**：Cordis 注册表、provider 发现/加载、collect cache、scope 链、fs provider。
这些是宿主生命周期；本 crate 只有纯词表、校验、合并与渲染。

### 3.19 golden fixture：TS 真实 codec 产出

`rust/fixtures/` 的 v4 物理行由**真实 TS codec** 生成：`generate.mjs` 用 Node 22 的
`--experimental-transform-types` 直接加载
`session-format-v3-to-v4`、`session-format-v1-to-v2` 的源码，再由 shipped decoder
逐行回读校验 `type` 与 `data` 相等。`resolve-hook.mjs` 把 workspace 包名映射到 `src`，
所以不需要 `pnpm install`。

证据等级写在 `fixtures/README.md`：这是 **encoder-level** 证据，不是一次端到端
`Session.append` 运行；payload projection 是按 producer 插入序记录，不是 schema 序。
Rust 侧 `tests/fixtures.rs` 做 TS 写→Rust 读→逐字节写回的比较；仍未覆盖真实 TS
运行时对 Rust 产物的回读（需要可运行的 TS 工作区）。

---

## 4. 构建与验证

```bash
cd rust
cargo build --workspace
cargo test  --workspace     # 544 tests
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
cargo run   -p dsh          # 端到端冒烟：turn → fold → JSONL → 回滚 → unload
```

测试分布：

| crate | tests | crate | tests |
| --- | ---: | --- | ---: |
| `dsh-agent` | 4 | `dsh-llm` | 12 |
| `dsh-plugin` | 8 | `dsh-session` | 30 |
| `dsh-tools` | 55 | `dsh-system-prompt` | 76 |
| `dsh-agent-loop` | 24 | `dsh-retry` | 58 |
| `dsh-compaction` | 123 | `dsh-context` | 81 |
| `dsh-todo` | 21 | `dsh-plan` | 44 |
| `dsh-session-projection` | 36 | `dsh-session-stats` | 13 |
| `dsh-session-turn-outline` | 16 | `dsh-skill` | 21 |
| `dsh-provider` | 4 | **合计** | **626** |

demo 二进制依次演示：插件加载 → **由 `AgentLoop` 真实驱动的一条 turn**（`ScriptedProvider`
先回一个 `echo` 工具调用、再回一句收尾文本，`EchoTool` 真实执行，2 个 step）→
派生消息与 JSONL 发布/重载/相等校验 → 插件声明的扩展事件（compaction + inbox splice）→
waterfall 决策 → 其余 fold → 失败加载回滚 → unload → agent 身份。

---

## 5. 尚未移植（有意留白）

| 项 | 状态 / 阻塞原因 |
| --- | --- |
| 真实 provider 实现（DeepSeek HTTP / 流式） | trait 已异步化；仍缺 HTTP/SSE 装配与真实 wire 类型 |
| 重试的生命周期 | 可取消等待已完成；插件卸载前排空、always 模式先问下游仍缺宿主 |
| compaction 异步 backend | 事件、surface、prune、budget 策略已完成；模型调用写 summary、durable lock、手动命令未搬 |
| 工具权限的 async 部分 | 纯规则与 post-execute 已完成；挂载的 `ApprovalService`、审批往返、取消追平在途审批未搬 |
| system-prompt 宿主（Cordis 部分） | ✅ 已完成：注册表、change 事件、assemble waterfall、scope 链父级遍历、整层销毁（76 tests） |
| context 的 file/session reference 测试 | ✅ 已完成：实现 + 独立测试（81 tests） |
| PTC（programmatic tool calling） | 需要嵌套 dispatch 与 `tool/ptc-dispatch*` 事件 |
| session format 迁移链（v0→v4） | Rust 目前只读写当前 v4；旧代次读取与迁移未实现 |
| TUI / IPC / 桌面壳 | 与语言无关，优先级最低 |
| TS ↔ Rust 端到端互操作 | encoder-level fixture 已建立；真实 TS 运行时回读 Rust 产物、`Session.append` 端到端仍待补 |
| 投影的缓存 I/O / 客户端 UI | registry 只用内存与抽象 checkpoint；数据库/文件淘汰、React 展示层不在 crate 内 |
| skill 的 provider/collect cache/scope 链 | 纯词表与合并规则已完成；宿主注册与发现未搬 |

**注意**：JSONL 线格式已从"按类型定义复刻"升级为 **encoder-level 字节对拍**（§3.19），
但真正替换 TS 侧之前仍需一次端到端 golden run（TS 写 → Rust 读 → Rust 写 → TS 读）。

---

## 6. 下一步建议顺序

1. ~~**补 context 的 file/session reference 测试**~~ ✅ 已完成（81 tests；本轮续做）。
2. **session format 迁移链**：把 v0→v1→v2→v3→v4 的读取/迁移搬进来，
   用 `packages/session/session-persistence-jsonl/tests/fixtures/released-v0-real-shapes.jsonl`
   做第一个真实旧代次 fixture。
3. **真实 provider**：DeepSeek HTTP + 流式，接到异步 `LlmProvider`；
   用 `test-support/llm-replay` 的录制作 replay，避免测试依赖网络。
4. **compaction backend**：把 §3.14 的纯策略接到真实 summary 调用、lock、append；
   触发点接进 `dsh-agent-loop` 的 step 边界。
5. **审批服务**：把 `resolve_ask` / post-execute 接到 async waterfall 与
   `ApprovalService`，补取消追平。
6. ~~**system-prompt 宿主**~~ ✅ 已完成（76 tests；本轮续做）。
7. **PTC** 与 **TUI/IPC** 按产品优先级推进。
8. 把 `rust/README.md` 作为回归基准，逐个替换 TS 包的实现入口。

---

## 7. 本轮并行移植记录（2026-10-21）

本轮开了 6 个子 agent 并行，另外由主 agent 补了 `dsh-skill`：

| 工作流 | 结果 |
| --- | --- |
| 异步核心（provider/tools/agent-loop/retry） | 完成；async provider、Tokio 并发调度、post-execute 接线、可取消等待 |
| TS 线格式 golden fixture | 完成；真实 TS codec 生成 4 个场景，Rust 双向字节比较 |
| compaction | 完成纯规则层（123 tests）；backend/lock 未搬 |
| context | time-context 完成（21 tests）；file/session reference 实现落地、测试缺失 |
| todo + plan | 完成（21 + 44 tests）；宿主注册与 interaction 通道未搬 |
| session projection/stats/outline | 完成（65 tests） |
| skill（主 agent） | 完成（21 tests）；宿主层未搬 |

5 个子 agent 在收尾阶段被 Provider 限流中断，留下的是**已经通过编译与测试**的代码；
本轮已统一修复中断点（`dsh-plan` 的借用错误、compaction 两个测试窗口、context 的
`should_inject` 边界断言与未用 import、clippy 的 enum 体积告警），并完成全量验证。
后续接手时先跑 §4 的四条命令；若继续并行，建议按 crate 边界分工，并在根 `Cargo.toml`
与 `rust/README.md` 上保持单写者。

---

## 8. 续做记录（收尾中断点）

上一轮（2026-10-21）在收尾阶段被限流中断，留下两处只落地实现、缺独立测试的缺口。
本轮由主 agent 续做，补齐并刷新基线：

| 工作流 | 结果 |
| --- | --- |
| system-prompt 宿主 | ✅ 完成 change 事件 / assemble waterfall / scope 链父级遍历 / 整层销毁；补 22 测试（76 tests） |
| context file/session reference | ✅ 补 60 测试（file_reference 19 + session_reference 41，共 81 tests） |
| 全量验证 | 626 tests 通过；clippy `-D warnings` 与 `fmt --check` 干净 |

剩余 §6 的 #2–#5（session 迁移链 / 真实 provider / compaction backend / 审批服务）
尚未启动；本轮继续按 crate 边界分工推进，根 `Cargo.toml` 与 `README.md` 保持单写者。

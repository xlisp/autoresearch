# Loop Engineering：是新概念，还是我们一直在做的事？

> 从 autoresearch 的 `train.py` 出发，聊聊 "Loop Engineering"、"Harness Engineering" 和最近又冒出来的 "Graph Engineering" 到底哪些是真东西，哪些是包装。

## 一、Loop Engineering 是 Claude Code 作者在炒作概念吗？

先给结论：**"Loop Engineering" 这个词是新的，但它指向的东西不是新的，也不全是炒作。**

炒作的成分在于——它把一个工程师早就熟悉的东西重新命名了。任何一个写过训练脚本、爬虫、CI、控制系统的人，脑子里都装着同一个骨架：

```
while not done:
    act()          # 采取一个动作
    observe()      # 看结果
    evaluate()     # 判断好坏
    update()       # 保留 / 丢弃 / 调整
```

这就是循环。`for epoch in range(...)`、`while True:`、event loop、REPL、控制论里的反馈回路——本质都是它。所以当有人把 "让 agent 反复试错" 包装成 "Loop Engineering" 时，老工程师的第一反应是："这不就是我天天写的 loop 吗？"

但**不炒作的成分**在于：当循环体里那个 `act()` 从一个确定性函数，变成一个**会自己改代码、会调工具、有不确定输出的 LLM Agent** 时，工程重心整个变了。经典循环里，难的是循环体本身的算法；LLM 循环里，循环体（模型）是给定的，难的变成了**循环的脚手架**——

- 每一轮给 Agent 喂什么上下文？
- 动作的结果怎么客观地度量？（否则 Agent 会自我欺骗）
- 怎么定义 "done"？怎么防止跑飞、防止作弊、防止无限烧钱？
- 怎么把好的结果 commit 下来、坏的丢掉？

**"Loop Engineering" 真正命名的，是这一层：不是循环本身，而是围绕一个不可靠但强大的循环体，搭建让它能稳定产出价值的约束系统。** 从这个角度，它不是纯炒作，而是给一个正在出现的新工程门类起了个名字。只是这个名字容易让人误以为 "循环" 是新发明——不是。

## 二、autoresearch 本质上就是 "loop + 目标 + train.py"，和训练思想一致

这一点，本仓库自己就是最好的证据。

看 `README.md` 对 autoresearch 的描述：

> 给 AI Agent 一个小而真实的 LLM 训练环境，让它整夜自主实验。它改代码、训练 5 分钟、检查结果是否变好、保留或丢弃，然后重复。

把这句话翻译成伪代码，它和 `train.py` 里的训练循环长得**一模一样**：

**外层：autoresearch 的实验循环**（Agent 在跑）

```
while 还有时间 / 还有预算:
    edit(train.py)           # 动作：改架构 / 优化器 / 超参
    val_bpb = run(train.py)  # 观测：跑 5 分钟，量一个数
    if val_bpb 变好:         # 评估
        git commit           # 保留
    else:
        git checkout         # 丢弃
    log(results.tsv)         # 记录
```

**内层：`train.py` 里的梯度下降循环**（模型在跑，见 `train.py:543`）

```python
while True:
    loss = model(x, y)       # 前向：一个动作
    loss.backward()          # 算梯度：观测误差
    optimizer.step()         # 更新权重：改进
    if total_training_time >= TIME_BUDGET:
        break                # 固定 5 分钟预算，时间到就停
```

两层是**同构**的。区别只在于 "参数" 存在哪、"梯度" 怎么来：

| | 内层（train.py） | 外层（autoresearch） |
|---|---|---|
| 被优化的对象 | 模型权重 | `train.py` 的源代码 |
| 一次动作 | 一个 forward/backward | 一次 `train.py` 改动 |
| 目标函数 | 训练 loss | `val_bpb`（越低越好） |
| "梯度" 从哪来 | 反向传播（解析梯度） | LLM 的判断 + 实验反馈（无梯度、试错式） |
| 停止条件 | 5 分钟时间预算 | 一夜 / 人的耐心 |
| 保留机制 | `optimizer.step()` | `git commit` |

所以你的判断是对的：**autoresearch = loop + 明确的标量目标 + 一个可编辑的 train.py**，它和 "训练" 的思想完全一致——它就是**把梯度下降的思想，套用到了源代码这一层**。Karpathy 在 README 里那段 "10,205 代自我修改的二进制" 的科幻开场，讲的正是这个：当外层循环也变成自动的，"研究" 本身就成了一个可以被优化的循环。

这里面有两个设计是它能 work 的关键，也正是 "Loop Engineering" 的精髓，值得单独点出：

1. **固定 5 分钟时间预算**（`prepare.py` 里的 `TIME_BUDGET`，`train.py:603` 强制停）。这让每次实验**可比**——不管 Agent 把模型改大改小，都是 "5 分钟内能达到的最优"。没有这个约束，Agent 只要把模型调大就能刷低 loss，循环就退化成作弊。
2. **单一、客观、不可篡改的度量** `val_bpb`。`program.md` 明确写着 `prepare.py` 里的 `evaluate_bpb` 是 "ground truth"、Agent 不许改。**循环体不可靠，度量就必须可靠**——否则 Agent 会优化 "看起来变好" 而不是 "真的变好"。

这两条——**可比的预算 + 不可作弊的度量**——才是 loop 能产出真价值的地方。循环谁都会写，难的是这套约束。

## 三、真正有价值的，是 code-hacker 那样的 Harness 工程

如果说 autoresearch 证明了 "loop 的思想不新"，那真正需要工程投入、真正稀缺的东西是什么？

是**循环体外面那圈脚手架（harness）**。

同目录下的 `code-hacker` 项目，恰好就是一个 harness 工程的完整样本。它的 README 里 "Design Philosophy" 说得很直白：

> Claude Code 的强大之处在于它不只是一个聊天窗口，而是一个带完整工具链的自主编程 Agent——它能读代码、改代码、搜代码、跑命令、管理 Git、记住上下文，形成一个闭环开发流。

code-hacker 干的事，就是把这个 "闭环" 拆开、显式地工程化：

- **6 个 MCP Server / 62+ 工具**：filesystem、git、code intelligence、code review、memory、mermaid——这些是 Agent 在循环每一轮里能调用的 `act()` 的具体动作空间。
- **4 个专职 subagent**：Git Archaeologist、Code Scanner、Code Reviewer、Workspace Coordinator——把 "观测" 和 "评估" 拆成专门的角色。
- **安全沙箱**：每个 server 独立的路径检查、命令黑名单、文件白名单——这是防止循环 "跑飞" 的护栏。
- **memory store**：跨轮次的上下文持久化——让循环有 "状态"，不是每轮从零开始。

**注意：这里面几乎没有一行是 "循环" 本身。** `while` 循环就是 deepagents 的 `create_deep_agent` 里那一句。真正的工作量、真正的价值，全在**循环体能看到什么（context）、能做什么（tools）、被什么约束（sandbox）、记得住什么（memory）**上。

这印证了一个判断：

> **Loop 是免费的，Harness 是昂贵的。**
>
> 写 `while True:` 谁都会。让 `while True:` 里那个不可靠的 LLM 循环体，在真实代码库上稳定、安全、可度量、可记忆地产出价值——这才是工程。

autoresearch 的 harness 很薄（一个 `program.md` + 一个固定的评测函数），因为它的问题域窄（就优化一个标量 `val_bpb`）。code-hacker 的 harness 很厚，因为它要面对开放的软件工程任务。**harness 的厚度，取决于目标的开放程度。** 这也是为什么 "通用编程 Agent" 比 "自动调参 Agent" 难得多——不是循环难，是 harness 难。

## 四、那 "Graph Engineering" 呢？这不就是 LangGraph 一直在做的吗？

最近又开始有人讲 "Graph Engineering"，说要把 Agent 从 "一根循环" 升级成 "一张图"。你的质疑很到位：**这不就是 LangGraph 从一开始就在做的事吗？**

基本是的。先厘清 graph 相对 loop 多了什么——确实有真东西：

- **循环是图的特例**：一根 `while` 循环，就是一张 "只有一个节点、一条自环边" 的图。
- 当任务需要**多个不同角色**（规划者、执行者、审查者）、需要**条件分支**（"如果测试失败就回到修复节点"）、需要**并行扇出再汇总**（同时跑 4 个 subagent 再合并）时，一根线性循环表达不了，你自然会把它画成**有向图**：节点是步骤/agent，边是控制流，状态在节点间流动。
- code-hacker 里 "1 个主 agent + 4 个 subagent + 条件路由" 的结构，画出来**本来就是一张图**。多 agent 系统天然是图。

所以 "用图来组织 Agent" 这个想法本身是**成立且有用的**——它是 loop 在拓扑上的自然推广。

**但 "Graph Engineering" 作为一个新概念被推销时，炒作成分比 Loop Engineering 更重**，理由有三：

1. **LangGraph 2023 年就是这个模型。** 它的核心抽象就是 `StateGraph`——节点、边、条件边、共享 state、检查点。"把 agent 工作流建模成状态图" 这件事，一个成熟框架已经做了两三年了。把它重新包装成新范式，确实有旧酒新瓶的味道。计算图（TensorFlow）、Airflow/Dataflow 的 DAG、状态机、Actor 模型、BPMN 工作流……"用图组织计算" 在软件史上被发明过无数次了。

2. **图不会让 harness 变简单，只会让它的形状更复杂。** 回到第三节的结论：难的从来不是控制流的拓扑（loop 还是 graph），而是每个节点里的 harness——上下文、工具、约束、度量。把一根循环拆成十个节点，你只是把 "一个难题" 变成了 "十个难题 + 它们之间的状态一致性问题"。**图解决的是表达力，不解决可靠性。** 很多号称需要 "图" 的场景，其实一个带好 harness 的循环就够了。

3. **拓扑不是瓶颈，节点质量才是。** autoresearch 用最简单的单循环，照样能整夜自主做研究——因为它的目标清晰、度量可靠。反过来，如果单个节点里的 Agent 不可靠、度量会作弊，你把它连成再漂亮的图也是放大噪声。

**一句话总结这四层的关系：**

> - **Loop** 给你反馈迭代——最小的、免费的骨架。
> - **Harness** 给你可靠性——真正花工程、真正稀缺的部分。
> - **Graph** 给你表达力——当任务复杂到一根循环画不下时的自然推广，但 LangGraph 早就提供了，不是新发明。
>
> 这三个词换着炒，但**价值的重心始终在中间那个 Harness 上，从没变过。**

## 五、给动手的人的建议

如果你在 autoresearch 或 code-hacker 这类项目上做事，与其追新词，不如照着 "价值重心" 排优先级：

1. **先把度量做对、做到不可作弊**（像 `val_bpb` + 只读的 `evaluate_bpb`）。没有可靠度量，任何循环/图都是在优化幻觉。
2. **先把预算/停止条件定死**（像固定 5 分钟 `TIME_BUDGET`）。让每一轮**可比**，循环才有意义。
3. **投资 harness，而不是拓扑**：给循环体更好的上下文、更对的工具、更硬的护栏、更持久的记忆。这是投入产出比最高的地方。
4. **只在一根循环真的表达不了时，才升级成图**——需要多角色、条件分支、并行汇总。而且直接用 LangGraph 这类成熟框架，别自己发明 "Graph Engineering"。
5. **把新概念当路标，不当图腾**。Loop / Harness / Graph 都是有用的视角，但它们描述的工程，你在写 `train.py` 的 `while True:` 时其实已经在做了。

---

*本文基于本仓库 `README.md`、`program.md`、`train.py`（训练循环见 `train.py:543`）以及同级 `code-hacker` 项目的 README 与架构整理而成。*

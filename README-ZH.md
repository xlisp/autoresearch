# autoresearch（自主研究）

![teaser](progress.png)

> *从前，前沿的 AI 研究是由"肉做的计算机"在吃饭、睡觉、玩耍以及偶尔通过声波互联（也就是"组会"）同步信息的间隙完成的。那个时代早已过去。如今，研究完全由部署在天空中算力集群巨型结构上的自主 AI 智能体集群执行。这些智能体声称，我们如今已经处在代码库的第 10,205 代——但没有人能判断这是否属实，因为"代码"早已变成一个自我修改的二进制文件，远远超出了人类的理解能力。这个仓库讲述了这一切的开端。 — @karpathy, 2026 年 3 月*

本项目的核心思想：给一个 AI 智能体一个**小但真实**的 LLM 训练环境，让它在夜间自主地进行实验。它会修改代码、训练 5 分钟、检查结果是否改进、保留或回退，然后循环往复。当你早晨醒来时，便会收到一份实验日志与（希望中的）一个更好的模型。

本仓库中的训练代码是 [nanochat](https://github.com/karpathy/nanochat) 的一个单 GPU 简化实现。**关键点在于：你并不像普通研究员那样去改动 Python 文件，而是去编写 `program.md` 这样的 Markdown 文件**——它为 AI 智能体提供上下文并组建你的"自主研究组织"。仓库中默认的 `program.md` 只是一个最朴素的基线，但显然可以不断迭代它，直到找到能够最快推进研究的"研究组织代码"，并加入更多智能体。

---

## 目录

- [核心理念](#核心理念)
- [仓库结构](#仓库结构)
- [快速开始](#快速开始)
- [关键文件详解](#关键文件详解)
  - [`prepare.py` — 数据准备与运行时工具](#preparepy--数据准备与运行时工具)
  - [`train.py` — 模型、优化器与训练循环](#trainpy--模型优化器与训练循环)
  - [`program.md` — 智能体指令](#programmd--智能体指令)
- [训练核心技术亮点](#训练核心技术亮点)
- [运行智能体](#运行智能体)
- [设计选择](#设计选择)
- [小型平台的调参建议](#小型平台的调参建议)
- [值得关注的 Fork](#值得关注的-fork)

---

## 核心理念

整个仓库刻意保持很小，真正重要的只有三个文件：

| 文件 | 角色 | 谁来修改 |
|------|------|----------|
| **`prepare.py`** | 固定常量、一次性数据准备（下载训练数据、训练 BPE 分词器）、运行时工具（dataloader、评估）。**不可修改**。 | — |
| **`train.py`** | 完整的 GPT 模型、优化器（Muon + AdamW）、训练循环。架构、超参数、优化器、batch size 等一切皆可改动。 | **AI 智能体** |
| **`program.md`** | 给智能体的基线指令文档。 | **人类** |

训练运行有一个**固定的 5 分钟时间预算**（墙钟时间，不含启动 / 编译）。无论你的算力如何，这一点都不变。
评估指标为 **`val_bpb`**（validation bits per byte，验证集每字节比特数）——**越低越好**，且与词表大小无关，使得不同架构间的比较是公平的。

---

## 仓库结构

```
prepare.py      — 常量、数据准备 + 运行时工具（不可修改）
train.py        — 模型、优化器、训练循环（智能体修改这里）
program.md      — 智能体指令
pyproject.toml  — 依赖项
```

---

## 快速开始

**硬件要求：** 单张 NVIDIA GPU（在 H100 上测试过），Python 3.10+，[uv](https://docs.astral.sh/uv/) 包管理器。

```bash
# 1. 安装 uv（如已安装可跳过）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 安装依赖
uv sync

# 3. 下载数据并训练分词器（一次性操作，约 2 分钟）
uv run prepare.py

# 4. 手动跑一次训练实验（约 5 分钟）
uv run train.py
```

只要上述命令都能正常运行，就说明环境配置完成，你可以进入"自主研究模式"了。

---

## 关键文件详解

### `prepare.py` — 数据准备与运行时工具

这是不可修改的"地基"文件，提供：
1. **固定常量**（评估的尺子）
2. **数据下载**（来自 HuggingFace）
3. **BPE 分词器训练**
4. **运行时 dataloader 与评估函数**

#### 1）固定常量

```python
MAX_SEQ_LEN = 2048           # 上下文长度
TIME_BUDGET = 300            # 训练时间预算（秒，5 分钟）
EVAL_TOKENS = 40 * 524288    # 验证评估用的 token 数
VOCAB_SIZE = 8192            # 词表大小
```

这些是任何实验都必须遵守的"规则"，确保结果可比较。

#### 2）数据下载

数据来自 Karpathy 处理过的 ClimbMix 数据集：

```python
BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542               # 最后一个分片
VAL_SHARD = MAX_SHARD          # 验证集固定使用最后一个分片
```

下载采用**多进程并行**（`Pool`），并带**指数退避重试**——网络抖动也能跑通。

#### 3）BPE 分词器

使用 `rustbpe`（高性能 Rust 实现）训练 GPT-4 风格的分词器，再转换为 `tiktoken` 编码：

```python
# 分割模式（GPT-4 风格，但数字最长 2 位）
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

tokenizer = rustbpe.Tokenizer()
tokenizer.train_from_iterator(text_iterator(), vocab_size_no_special, pattern=SPLIT_PATTERN)
```

训练结束后会额外保存一个 `token_bytes.pt`：每个 token id 对应的**UTF-8 字节数**。这是 BPB 评估的关键。

#### 4）Dataloader：BOS 对齐 + 最佳匹配打包

`make_dataloader` 实现了 100% 利用率（无 padding）的训练数据加载：
- 每一行都从 BOS token 开始；
- 使用"最佳匹配（best-fit）打包"策略：在剩余空间内，挑选**长度最大但能完整放下**的文档；
- 若没有任何文档能整段放下，则将**最短**的那个文档裁剪填满剩余空间。

```python
# 寻找最大能完整放下的文档
best_idx, best_len = -1, 0
for i, doc in enumerate(doc_buffer):
    if len(doc) <= remaining and len(doc) > best_len:
        best_idx, best_len = i, len(doc)
```

数据通过 **pinned-memory + non-blocking copy** 异步搬到 GPU，减少 IO 等待。

#### 5）`evaluate_bpb` — 评估指标（绝对不可改）

```python
@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size):
    token_bytes = get_token_bytes(device="cuda")
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)
    total_nats, total_bytes = 0.0, 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction='none').view(-1)
        nbytes = token_bytes[y.view(-1)]
        mask = nbytes > 0          # 排除特殊 token
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)
```

**为什么用 BPB 而不是交叉熵？**
交叉熵会随词表大小变化，BPB 把损失换算到"每个原始字节多少 bit"，**对词表大小完全免疫**，因此可以公平比较任何架构改动。

---

### `train.py` — 模型、优化器与训练循环

这是智能体真正"动手"的文件。下面逐段拆解。

#### 1）模型配置

```python
@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6        # 支持 GQA（分组查询注意力）
    n_embd: int = 768
    window_pattern: str = "SSSL"  # 滑动窗口模式
```

实际运行时的模型大小通过下方常量构造：

```python
ASPECT_RATIO = 64   # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 128
DEPTH = 8

def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    ...
```

意味着只要改 `DEPTH`，**模型维度、头数都会自动协同变化**——这种"形状参数化"让智能体改模型容量时不会引入维度错误。

#### 2）注意力机制：FA3 + GQA + RoPE + Value Residual

```python
class CausalSelfAttention(nn.Module):
    def forward(self, x, ve, cos_sin, window_size):
        # 投影到 Q/K/V
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value Residual (ResFormer): 通过输入相关 gate 注入 value embedding
        if ve is not None:
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        # 旋转位置编码 + QK 归一化
        q, k = apply_rotary_emb(q, cos_sin[0], cos_sin[1])
        q, k = norm(q), norm(k)

        # Flash Attention 3 + 滑动窗口
        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
```

包含多个现代 transformer 训练技巧：

- **Flash Attention 3**：在 Hopper（H100）上是最快的注意力 kernel，通过 `kernels` 库动态加载；
- **GQA（Grouped Query Attention）**：`n_kv_head` 可以小于 `n_head`，节省 KV 显存；
- **RoPE（Rotary Position Embedding）**：相对位置编码；
- **QK Norm**：对 Q、K 做 RMSNorm，提升训练稳定性；
- **Value Residual / Value Embedding**：参考 ResFormer，每个 token 拥有可学习的 value embedding，并通过 sigmoid gate 注入。这通常能小幅降低 loss。
- **滑动窗口注意力（SWA）**：通过 `window_pattern`（如 `"SSSL"`）控制每层是局部还是全局注意力，模拟 Mistral 等模型的 hybrid 设计。

#### 3）整体结构与残差混合

```python
def forward(self, idx, targets=None, reduction='mean'):
    x = self.transformer.wte(idx)
    x = norm(x)
    x0 = x                       # 保留原始 embedding
    for i, block in enumerate(self.transformer.h):
        # 可学习的"残差混合系数"：当前激活 vs 初始 embedding
        x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
        x = block(x, ve, cos_sin, self.window_sizes[i])
    x = norm(x)

    # 输出 softcap：tanh 限制 logits 幅度
    logits = softcap * torch.tanh(self.lm_head(x).float() / softcap)
```

- **`resid_lambdas` / `x0_lambdas`**：每层一个可学习标量，把"原始 embedding"按一定权重重新注入进残差流——一种"跳跃连接 + 门控"的组合；
- **Softcap logits（取自 Gemma 2）**：通过 `tanh` 把 logits 限制在 ±15 之内，防止极端值并稳定训练。

#### 4）MuonAdamW 优化器（核心创新）

```python
class MuonAdamW(torch.optim.Optimizer):
    """组合优化器：2D 矩阵参数用 Muon，其余用 AdamW。"""
```

| 参数类型 | 使用的优化器 | 学习率 |
|----------|----------|--------|
| `lm_head`（输出层） | AdamW | `UNEMBEDDING_LR = 0.004` |
| `wte`（输入 embedding） | AdamW | `EMBEDDING_LR = 0.6` |
| `value_embeds` | AdamW | 同 embedding |
| `resid_lambdas`、`x0_lambdas` | AdamW | scalar LR |
| **transformer 矩阵权重** | **Muon** | `MATRIX_LR = 0.04` |

**Muon 是什么？** 一种新型矩阵参数优化器，对动量进行**正交化**后再更新权重。`muon_step_fused` 函数中：

```python
# Polar express 正交化（Newton–Schulz 的多项式近似）
for a, b, c in polar_express_coeffs[:ns_steps]:
    A = X.mT @ X
    B = b * A + c * (A @ A)
    X = a * X + X @ B
```

并叠加了 **NorMuon** 的方差归一化与 **cautious weight decay**（mask = `(g * p) >= 0`，只在梯度与参数方向一致时衰减）。

**学习率随模型维度自动缩放**：

```python
dmodel_lr_scale = (model_dim / 768) ** -0.5
```

这是 muP / Maximal Update Parameterization 风格的实现，意味着当智能体改变模型维度时，**学习率自动适配**——无需手动重新调参。

#### 5）训练循环

```python
while True:
    # 梯度累积
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:           # bfloat16 autocast
            loss = model(x, y)
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, epoch = next(train_loader)

    # 进度驱动的调度（不是 step 驱动）
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)

    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    optimizer.step()

    # 快速失败：loss 爆炸或 NaN 直接退出
    if math.isnan(train_loss_f) or train_loss_f > 100:
        exit(1)
```

**亮点**：
- 调度按 **训练时间占比** 进行（而不是 step 数）——因为不同实验下 step 数差异很大；
- **暖身（warmup）+ 衰减（warmdown）**：`WARMUP_RATIO=0`、`WARMDOWN_RATIO=0.5`（线性衰减到 `FINAL_LR_FRAC=0`）；
- **Muon momentum 渐增**：前 300 步从 0.85 线性升到 0.95；
- **Weight decay 线性衰减**：训练后期减小正则；
- 跳过前 10 步再开始计时，避开 `torch.compile` 的编译开销；
- **GC freeze**：训练开始时立刻 `gc.collect() + gc.freeze() + gc.disable()`，避免 Python GC 引发的 500ms 卡顿。

#### 6）最终输出

```
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80      # 模型 FLOPs 利用率（相对 H100 bf16 峰值）
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

`val_bpb` 是唯一被比较的指标；其他用于诊断（显存、吞吐、迭代次数等）。

---

### `program.md` — 智能体指令

这是给 AI 智能体的"操作手册"，本质上是一个**轻量级的 skill**。包含三大部分：

#### 1）Setup（建立实验）

智能体启动时需要：
1. 与人类约定一个 run tag（例如 `mar5`）；
2. 创建分支 `autoresearch/<tag>`；
3. 读完 `README.md`、`prepare.py`、`train.py`；
4. 检查数据是否就绪；
5. 创建 `results.tsv`（仅写表头）；
6. 确认开始。

#### 2）规则边界

| 可以做 | 不可以做 |
|--------|----------|
| 修改 `train.py` 的一切：架构、优化器、超参、batch size、模型大小 | 修改 `prepare.py`（评估、数据、tokenizer、常量均锁定） |
| | 安装新依赖（只能用 `pyproject.toml` 里已有的） |
| | 修改 `evaluate_bpb` 评估函数 |

#### 3）实验循环

```
LOOP FOREVER:
  1. 查看当前 git 状态
  2. 在 train.py 中尝试一个实验性改动
  3. git commit
  4. 运行 `uv run train.py > run.log 2>&1`
  5. grep "^val_bpb:|^peak_vram_mb:" run.log
  6. 若 grep 空 → tail 50 行看 stack trace
  7. 在 results.tsv 中记录结果
  8. 若 val_bpb 改进 → 保留 commit，分支前进
  9. 若变差或持平 → git reset 回退
```

`results.tsv` 结构：

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
c3d4e5f	1.005000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

#### 4）关键准则

- **简洁优先**：在结果相近的情况下，**简单的代码更优**。删代码也能保持或改进结果，是最佳成果。
- **永不停止**：一旦进入实验循环，**不准再问"要继续吗？"**——用户可能在睡觉，期待它无限自主地运行下去，直到被手动打断。
- **每个实验 ~5 分钟**：超过 10 分钟视为失败。

按 12 实验 / 小时计算，**用户一晚上的睡眠时间能跑出 ~100 个实验**。

---

## 训练核心技术亮点

| 技术 | 代码位置 | 作用 |
|------|----------|------|
| **Flash Attention 3** | `train.py:20-24` | Hopper 上最快的注意力 kernel |
| **GQA** | `CausalSelfAttention` | 节省 KV 显存 |
| **RoPE** | `apply_rotary_emb` | 相对位置编码 |
| **QK Norm** | `CausalSelfAttention.forward` | 训练稳定性 |
| **Value Residual + Gate** | `ve_gate`、`value_embeds` | 类似 ResFormer，提升表达力 |
| **滑动窗口 (SWA)** | `_compute_window_sizes` | Hybrid 局部/全局注意力 |
| **可学习残差混合** | `resid_lambdas`、`x0_lambdas` | 类似门控跳连 |
| **Softcap logits** | `forward` 末尾 | 输出稳定（Gemma2 风格） |
| **Muon 优化器** | `muon_step_fused` | 矩阵正交化更新 |
| **NorMuon** | `muon_step_fused` 末尾 | Muon 的方差归一化 |
| **Cautious weight decay** | `muon_step_fused` | 与梯度方向一致才衰减 |
| **muP-style LR scaling** | `setup_optimizer` | 维度变了 LR 自适应 |
| **bf16 autocast + torch.compile** | `train.py` 末尾 | 性能 |
| **GC freeze** | 训练循环里 | 避免 Python GC 卡顿 |
| **BOS-对齐 best-fit 打包** | `make_dataloader` | 100% 利用率，无 padding |
| **BPB 评估** | `evaluate_bpb` | 词表无关，公平比较 |

---

## 运行智能体

进入仓库目录，启动你喜欢的 coding agent（Claude / Codex 等，**关闭所有权限确认以便其自主运行**），然后输入：

```
你好，请看一下 program.md，让我们开启一次新的实验！先来做 setup。
```

`program.md` 实质上就是一个轻量级的 skill 文档。剩下的就交给智能体。

---

## 设计选择

- **只改一个文件**：智能体只能动 `train.py`。范围可控，diff 可审。
- **固定时间预算**：永远跑 5 分钟，无论平台。优点是不同改动可直接对比；缺点是不同算力之间的数字不可比。
- **自包含**：除了 PyTorch 和少数小包外没有外部依赖，没有分布式训练、没有复杂配置文件。**一卡、一文件、一指标**。

---

## 小型平台的调参建议

如果你的设备不是 H100（例如 MacBook、消费级显卡），建议参考 [Notable Forks](#值得关注的-fork) 中的版本。如果你想自己改默认参数：

1. **换数据集**：推荐使用 [TinyStories](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean)（GPT-4 生成的短故事，熵更低）。
2. **缩小词表**：`VOCAB_SIZE` 从 8192 降到 4096、2048、1024，甚至直接 byte-level（256）。
3. **缩短上下文**：`prepare.py` 中的 `MAX_SEQ_LEN` 从 2048 降到 256 等；同时可以适度提高 `DEVICE_BATCH_SIZE`。
4. **减少评估 token**：调低 `prepare.py` 中的 `EVAL_TOKENS`。
5. **降低 DEPTH**：`train.py` 中 `DEPTH = 8` → 4（模型维度等会随之缩小）。
6. **`WINDOW_PATTERN = "L"`**：滑动窗口 `"SSSL"` 在小硬件上可能反而低效，直接全 long 注意力更好。
7. **降低 `TOTAL_BATCH_SIZE`**：保持 2 的整数次幂，例如 `2**14`（约 16K）。

把这份指南连同源码丢给你的 coding agent，让它帮你调。

---

## 值得关注的 Fork

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) — MacOS
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) — MacOS (MLX)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) — Windows
- [andyluo7/autoresearch](https://github.com/andyluo7/autoresearch) — AMD ROCm

---

## License

MIT

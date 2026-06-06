# FSD L2 归一化余弦距离 + 可学习 Scale 实施文档（Codex 版 v1.4）

> 目标读者：Codex / 代码实现 Agent  
> 目标任务：在原始 FSD baseline 上，实现 **L2 归一化 + 余弦相似度 + 可学习正数 scale** 的原型距离改进。  
> 当前实验分支：`exp-prototype-distance-v1`  
> 对齐代码：`teheperinko541/Few-Shot-AIGI-Detector/main` 的真实代码结构。  
> 实验设置：对齐用户已跑过的 **1/10 GenImage mini 数据集实验设置**，不是原论文全量数据训练设置。  
> v1.4 修订重点：继续使用 1/10 GenImage mini 实验设置与 TensorBoard，但修复 TensorBoard 相关的两个细节：  
> 1. 不在 `train.py` 顶部无条件导入 `SummaryWriter`，改为在 `args.use_tensorboard and args.rank == 0` 时惰性导入，避免未安装 tensorboard 时影响 baseline；  
> 2. `train/loss` 只写入一次，避免 cosine 模式下同一 step 重复写 TensorBoard。  
> 同时建议在 `requirements.txt` 中补充 `tensorboard`，便于复现实验。

---

## 0. 一句话目标

原始 FSD 使用平方欧氏距离进行原型分类：

```python
distance = ((query - prototype) ** 2).sum(dim=-1)
logits = -distance
loss = cross_entropy(logits, labels)
```

本次改成：

```python
support_norm = L2_normalize(support)
query_norm = L2_normalize(query)
prototype = mean(support_norm)
prototype_norm = L2_normalize(prototype)
cosine_score = dot(query_norm, prototype_norm)
scale = exp(log_scale)
scale = clamp(scale, max=max_scale)
logits = cosine_score * scale
loss = cross_entropy(logits, labels)
```

直观理解：

```text
原始 FSD：
query 和哪个 prototype 的“位置距离”更近，就更像哪个类。

本方案：
先把 query 和 prototype 都放到单位球面上，
再看 query 和哪个 prototype 的“方向”更像。
最后用可学习正数 scale 调整 cosine logits 的放大程度。
```

---

## 1. 分支与实验边界

### 1.1 Git 分支

当前实验分支：

```text
exp-prototype-distance-v1
```

该分支应从原始 FSD baseline 的 `main` 分支创建。

不要从：

```text
exp-dual-branch-dwt
```

继续修改。

原因：本分支只验证“距离度量改动”本身，不能和 DWT 双分支混在一起，否则后续无法判断效果变化来自距离，还是来自双分支。

### 1.2 本分支只做的事

只实现：

```text
L2-normalized cosine prototypical metric
learnable positive scale
TensorBoard 记录 cosine metric 相关指标
```

保持不变：

```text
数据集结构不变
dataloader 不变
episodic training 不变
ResNet50 encoder 主体不变
support/query 切分方式不变
labels 生成顺序不变
train/eval/test 的 episode 布局不变
```

### 1.3 本分支不要做的事

不要加入：

```text
DWT frequency branch
dual branch model
support-set reliability fusion
prototype separation loss
prototype margin loss
Mahalanobis distance
Gaussian prototype
multi-prototype
graph / Isomap distance
fixed scale 消融作为默认主方案
独立验证集
best checkpoint
early stopping
```

---

## 2. 必须保留的原始 FSD 数据布局

这是本次实现最容易出“静默错误”的地方。Codex 不要重新设计 episode 布局。

真实原始代码中的训练流程是：

```python
batch_data = torch.stack([next(train_iters[c])[0] for c in selected_classes], dim=0)
# [num_class, batch_size * task_size, C, H, W]

labels = torch.arange(0, args.num_class_train, device=args.device).repeat(
    args.batch_size * args.num_query_train
)

batch_data = rearrange(batch_data, 'n b c h w -> (n b) c h w')

with autocast(enabled=args.use_fp16, device_type="cuda"):
    outputs = model(batch_data)

outputs = rearrange(
    outputs,
    '(n b t) l -> b t n l',
    n=args.num_class_train,
    b=args.batch_size,
)
```

本次只替换 loss / metric，不改变上面的排列方式。

### 2.1 loss 输入形状

沿用原始 `compute_prototypical_loss` 的输入布局：

```python
inputs.shape == [B, T, Nc, D]
```

其中：

```text
B  = episode batch size，即 args.batch_size
T  = support_num + query_num
Nc = episode 内类别数
D  = embedding dimension，原始 FSD 中为 1024
```

labels：

```python
labels.shape == [B * Nq * Nc]
```

labels 顺序：

```python
labels = torch.arange(Nc, device=device).repeat(B * Nq)
```

不要改成 `repeat_interleave`。

### 2.2 support / query 切分

保持原始方式：

```python
support_set = inputs[:, :support_num, ...]   # [B, Ns, Nc, D]
query_set = inputs[:, support_num:, ...]     # [B, Nq, Nc, D]
```

### 2.3 query flatten 顺序

保持：

```python
query_flat = rearrange(query_set, "b q n l -> b (q n) l").unsqueeze(2)
```

含义：

```text
query 展平顺序是：
先固定 b，再固定 q，然后 n 从 0 到 Nc-1 变化。
```

因此 labels 必须是：

```python
torch.arange(Nc).repeat(B * Nq)
```

---

## 3. 方法定义

### 3.1 L2 normalize 顺序

第一版固定使用下面的顺序：

```text
1. support feature 先 L2 normalize
2. query feature 先 L2 normalize
3. normalized support feature 求均值得到 prototype
4. prototype 再 L2 normalize
5. query 和 prototype 点乘，得到 cosine similarity
```

代码：

```python
support_set = F.normalize(support_set, p=2, dim=-1, eps=eps)
query_set = F.normalize(query_set, p=2, dim=-1, eps=eps)

prototypes = support_set.mean(dim=1, keepdim=True)
prototypes = F.normalize(prototypes, p=2, dim=-1, eps=eps)
```

说明：

```text
support 先 normalize：
避免某个 support 样本因为特征范数大而主导 prototype。

prototype 均值后再 normalize：
多个单位向量的均值不一定还是单位长度，所以要再归一化一次。
```

### 3.2 cosine score

```python
query_flat = rearrange(query_set, "b q n l -> b (q n) l").unsqueeze(2)
cosine_scores = (query_flat * prototypes).sum(dim=-1)
# [B, Nq*Nc, Nc]
```

### 3.3 learnable positive scale

本方案不直接学习 temperature，也不直接学习 scale，而是学习：

```text
log_scale
```

前向传播：

```python
scale = exp(log_scale)
scale = clamp(scale, max=max_scale)
logits = cosine_scores * scale
```

默认：

```yaml
init_scale: 10.0
max_scale: 100.0
scale_eps: 1.0e-12
```

初始化：

```python
log_scale = log(init_scale)
```

对应关系：

```text
scale = 1 / temperature
temperature = 1 / scale
```

实现中建议只使用：

```text
log_scale
scale
```

不要在代码里把 temperature 作为可学习参数。

---

## 4. 1/10 GenImage mini 数据集实验设置

### 4.1 当前文档使用 mini setting，不使用原论文全量训练设置

本分支实验应对齐用户之前已经跑过的 1/10 GenImage mini 数据集实验设置。不要默认使用原论文全量数据的训练步数，也不要直接照搬原仓库 parser 默认的 `200000 steps`。

推荐配置：

```yaml
dataset: GenImage 1/10 mini
seed: 42

total_training_steps: 15000
save_interval: 2500
eval_interval: 2500
log_interval: 200

lr: 1.0e-4
lr_scheduler_step: 5000
lr_scheduler_gamma: 0.5

batch_size: 16
num_class_train: 3
num_support_train: 5
num_query_train: 5

num_class_val: 2
num_support_val: 5
num_query_val: 15

num_class_test: 2
num_support_test: 5
num_query_test: 15

use_fp16: true
```

说明：

```text
1. total_training_steps=15000：对齐用户 mini baseline / DWT 实验。
2. eval_interval=2500、save_interval=2500：每 2500 step 评估并保存。
3. log_interval=200：每 200 step 输出常规 train/loss 和 train/lr。
4. lr_scheduler_step=5000、gamma=0.5：15000 step 内发生多次学习率衰减。
5. baseline 和 cosine 实验必须使用同一套训练参数。
```

### 4.2 学习率衰减含义

若：

```yaml
lr: 1e-4
lr_scheduler_step: 5000
lr_scheduler_gamma: 0.5
total_training_steps: 15000
```

则学习率大致为：

```text
step 0 ~ 4999:       1e-4
step 5000 ~ 9999:    5e-5
step 10000 ~ 14999:  2.5e-5
step 15000 附近:     1.25e-5
```

注意：如果后续确认历史 mini baseline 使用了不同训练参数，应以历史 mini baseline 为准。不要让 Codex 自行根据原论文或原仓库脚本猜测。

---

## 5. 配置项要求

### 5.1 parser 默认值必须保持 baseline 安全

parser 默认值必须是：

```yaml
metric: squared_euclidean
```

原因：不传任何新参数时，原始 FSD baseline 行为必须完全不变。

### 5.2 cosine 实验通过命令行显式开启

新实验推荐命令行配置：

```bash
--metric cosine \
--init_scale 10.0 \
--max_scale 100.0 \
--scale_eps 1e-12
```

注意：

```text
“主方案是 cosine”指实验配置，不是 parser 默认值。
parser 默认值仍然必须是 squared_euclidean。
```

### 5.3 新增参数位置

真实仓库中 `TrainParser` 和 `TestParser` 都继承 `ModelParser`。

因此以下训练和测试都要用到的参数必须加到 `ModelParser`：

```python
self.parser.add_argument(
    "--metric",
    type=str,
    default="squared_euclidean",
    choices=["squared_euclidean", "cosine"],
    help="Prototype metric: original squared Euclidean or L2-normalized cosine.",
)

self.parser.add_argument(
    "--init_scale",
    type=float,
    default=10.0,
    help="Initial positive scale for cosine logits. Equivalent to temperature=1/init_scale.",
)

self.parser.add_argument(
    "--max_scale",
    type=float,
    default=100.0,
    help="Maximum positive scale after exp(log_scale).",
)

self.parser.add_argument(
    "--scale_eps",
    type=float,
    default=1e-12,
    help="Epsilon for L2 normalization and safe reciprocal logging.",
)
```

训练阶段使用的 TensorBoard 参数加到 `TrainParser`：

```python
self.parser.add_argument(
    "--use_tensorboard",
    type=self._str2bool,
    default=False,
    help="Whether to write TensorBoard logs. Default False keeps baseline runnable without tensorboard.",
)

self.parser.add_argument(
    "--tb_log_interval",
    type=int,
    default=20,
    help="Interval for TensorBoard metric logging.",
)
```

---

### 5.4 requirements.txt 要求

因为本分支正式支持 TensorBoard，建议在 `requirements.txt` 中新增：

```text
tensorboard
```

同时，`train.py` 中仍然必须采用惰性导入方式。原因是：

```text
1. requirements.txt 加 tensorboard：方便新环境复现实验。
2. 惰性导入 SummaryWriter：保证当用户设置 --use_tensorboard False 时，baseline 路径不依赖 tensorboard。
```

也就是说，二者都建议做：

```text
requirements.txt 补 tensorboard
train.py 中惰性导入 SummaryWriter
```

不要只做顶层无条件导入。

---

## 6. 新增文件：`model/cosine_metric_fsd.py`

新增文件：

```text
model/cosine_metric_fsd.py
```

功能：

```text
封装原始 ResNet50 encoder
新增可学习参数 log_scale
提供 get_scale()
```

参考实现：

```python
import math

import torch
import torch.nn as nn
import timm


class CosineMetricFSD(nn.Module):
    """Single-branch FSD model with a learnable positive cosine logit scale."""

    def __init__(
        self,
        pretrained: bool = True,
        embedding_dim: int = 1024,
        init_scale: float = 10.0,
        max_scale: float = 100.0,
    ):
        super().__init__()

        if init_scale <= 0:
            raise ValueError(f"init_scale must be positive, got {init_scale}.")
        if max_scale <= 0:
            raise ValueError(f"max_scale must be positive, got {max_scale}.")
        if init_scale > max_scale:
            raise ValueError(
                f"init_scale should not be larger than max_scale, got "
                f"init_scale={init_scale}, max_scale={max_scale}."
            )

        self.encoder = timm.create_model(
            "resnet50",
            pretrained=pretrained,
            num_classes=embedding_dim,
        )

        self.log_scale = nn.Parameter(
            torch.tensor(math.log(init_scale), dtype=torch.float32)
        )
        self.max_scale = float(max_scale)

    def forward(self, x):
        return self.encoder(x)

    def get_scale(self):
        scale = self.log_scale.exp()
        scale = torch.clamp(scale, max=self.max_scale)
        return scale
```

说明：

```text
1. 真正可学习的参数只有 log_scale。
2. scale = exp(log_scale)，只是前向传播中计算出的正数，不是第二个可学习参数。
3. clamp 用于避免 scale 无限增大。
4. 因为 log_scale 在 model.parameters() 里，所以 optimizer 会自动更新它。
5. 当前真实 train.py 没有 DDP(model) 包裹，因此可以直接 model.get_scale()，不需要 model.module.get_scale()。
```

---

## 7. 修改 `model/prototypical_utils.py`

### 7.1 必须新增 `import torch`

真实原始文件顶部只有：

```python
from einops import rearrange
import torch.nn.functional as F
```

本次新增函数会用到：

```python
torch.long
torch.is_tensor
torch.tensor
```

因此必须在文件顶部新增裸导入：

```python
import torch
```

最终文件顶部应类似：

```python
from einops import rearrange
import torch
import torch.nn.functional as F
```

如果不加 `import torch`，第一次走 cosine 分支时会报：

```text
NameError: name 'torch' is not defined
```

### 7.2 保留原函数

必须保留原始函数：

```python
compute_prototypical_loss(inputs, labels, support_num)
```

不要修改其行为。

当：

```bash
--metric squared_euclidean
```

时，必须继续走原函数，保证 baseline 不变。

### 7.3 新增 `compute_cosine_prototypical_loss`

新增函数，和原函数并列。

参考实现：

```python
def compute_cosine_prototypical_loss(
    inputs,
    labels,
    support_num,
    scale,
    eps=1e-12,
):
    """Compute L2-normalized cosine prototypical loss.

    Args:
        inputs: [B, T, Nc, D], same layout as original FSD.
        labels: [B * Nq * Nc], labels in 0..Nc-1.
        support_num: number of support samples per class.
        scale: positive scalar, usually model.get_scale().
        eps: epsilon for L2 normalization and safe reciprocal logging.

    Returns:
        loss: scalar cross entropy loss.
        scores: [B * Nq * Nc, Nc], cosine logits after scale.
        debug_dict: auxiliary metrics for logging.
    """

    if inputs.dim() != 4:
        raise ValueError(
            f"Expected inputs with shape [B, T, Nc, D], got {tuple(inputs.shape)}."
        )

    # Important for AMP/fp16:
    # The encoder output may be fp16 because model forward runs under autocast.
    # Cosine normalization and dot product should be computed in fp32 for stability.
    inputs = inputs.float()

    labels = labels.to(device=inputs.device, dtype=torch.long)

    if not torch.is_tensor(scale):
        scale = torch.tensor(scale, device=inputs.device, dtype=torch.float32)
    else:
        scale = scale.to(device=inputs.device)
    scale = scale.float()

    support_set = inputs[:, :support_num, ...]
    query_set = inputs[:, support_num:, ...]

    # L2 normalize support and query features first.
    support_set = F.normalize(support_set, p=2, dim=-1, eps=eps)
    query_set = F.normalize(query_set, p=2, dim=-1, eps=eps)

    # Compute prototypes from normalized support features.
    prototypes = support_set.mean(dim=1, keepdim=True)

    # Re-normalize prototypes because the mean of unit vectors is not necessarily a unit vector.
    prototypes = F.normalize(prototypes, p=2, dim=-1, eps=eps)

    query_flat = rearrange(query_set, "b q n l -> b (q n) l").unsqueeze(2)
    cosine_scores = (query_flat * prototypes).sum(dim=-1)

    scores = cosine_scores * scale
    scores = rearrange(scores, "b q c -> (b q) c")

    loss = F.cross_entropy(scores, labels)

    scale_detached = scale.detach().float()
    debug_dict = {
        "scale": scale_detached,
        "temperature": 1.0 / scale_detached.clamp_min(eps),
        "cosine_mean": cosine_scores.detach().float().mean(),
        "cosine_std": cosine_scores.detach().float().std(unbiased=False),
        "cosine_min": cosine_scores.detach().float().min(),
        "cosine_max": cosine_scores.detach().float().max(),
    }

    return loss, scores, debug_dict
```

### 7.4 重要注意事项

不要写：

```python
scores = -cosine_scores
```

因为 cosine similarity 越大越好，不需要取负号。

不要直接学习：

```python
temperature = nn.Parameter(...)
```

不要直接学习：

```python
scale = nn.Parameter(...)
```

只学习：

```python
log_scale = nn.Parameter(...)
scale = exp(log_scale)
```

---

## 8. 修改 `train.py`

### 8.1 import

原始 import：

```python
import timm
from model.prototypical_utils import compute_prototypical_loss
```

新增：

```python
from model.cosine_metric_fsd import CosineMetricFSD
from model.prototypical_utils import (
    compute_prototypical_loss,
    compute_cosine_prototypical_loss,
)
```

注意：不要在 `train.py` 文件顶部无条件写：

```python
from torch.utils.tensorboard import SummaryWriter
```

原因：如果环境没有安装 `tensorboard`，顶层无条件导入会让 `train.py` 启动时直接报错。即使用户运行的是 `--metric squared_euclidean --use_tensorboard False`，也会因为 import 已经发生而崩溃。v1.4 要求使用惰性导入，见 8.2 节。

### 8.2 创建 TensorBoard writer：必须惰性导入

在：

```python
args = TrainParser().args
setup_dist(args)
logger.setup(log_dir=args.output_dir, device=args.device)
```

之后新增：

```python
tb_writer = None
if args.use_tensorboard and args.rank == 0:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorBoard logging was requested, but the 'tensorboard' package is not installed. "
            "Please install it with `pip install tensorboard`, add `tensorboard` to requirements.txt, "
            "or rerun with `--use_tensorboard False`."
        ) from exc

    tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tb"))
```

说明：

```text
1. 训练由 torchrun 启动，setup_dist(args) 会设置 args.rank。
2. 只允许 rank 0 写 TensorBoard，避免多进程重复写日志。
3. TensorBoard 不替代原 logger；原 logger 继续保留。
4. 惰性导入可以保证：当 --use_tensorboard False 时，即使环境没有安装 tensorboard，baseline 训练也不会因为 import 报错。
```

训练结束前关闭：

```python
if tb_writer is not None:
    tb_writer.close()
```

### 8.3 创建模型

原始代码：

```python
logger.info("Creating model 'resnet50'... ")
model = timm.create_model("resnet50", pretrained=True, num_classes=1024)
```

改成：

```python
if args.metric == "cosine":
    logger.info("Creating CosineMetricFSD with learnable log_scale... ")
    model = CosineMetricFSD(
        pretrained=True,
        embedding_dim=1024,
        init_scale=args.init_scale,
        max_scale=args.max_scale,
    )
elif args.metric == "squared_euclidean":
    logger.info("Creating model 'resnet50' with squared Euclidean metric... ")
    model = timm.create_model("resnet50", pretrained=True, num_classes=1024)
else:
    raise ValueError(f"Unsupported metric: {args.metric}")
```

然后保持：

```python
model = model.to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
```

注意：

```text
CosineMetricFSD 内部包含 log_scale。
因此必须用 model.parameters() 创建 optimizer，不能把 log_scale 写在 loss 函数外面但忘记加入 optimizer。
```

### 8.4 训练 loss 选择

原始代码：

```python
loss, _ = compute_prototypical_loss(outputs, labels, args.num_support_train)
```

改成：

```python
if args.metric == "cosine":
    scale = model.get_scale()
    loss, scores, metric_debug = compute_cosine_prototypical_loss(
        outputs,
        labels,
        args.num_support_train,
        scale=scale,
        eps=args.scale_eps,
    )
else:
    loss, scores = compute_prototypical_loss(
        outputs,
        labels,
        args.num_support_train,
    )
    metric_debug = None
```

### 8.5 TensorBoard 训练日志

在 loss 计算之后、`backward()` 之前或之后均可记录。建议放在 loss 计算之后。

v1.4 要求：`train/loss` 只写一次，不要在 cosine 模式下重复写两次。推荐写法是先统一记录 `train/loss`，再在 cosine 模式下额外记录 `metric/*`：

```python
if tb_writer is not None and step % args.tb_log_interval == 0:
    with torch.no_grad():
        tb_writer.add_scalar("train/loss", float(loss.detach().cpu()), step)

        if args.metric == "cosine":
            scale_for_log = model.get_scale().detach().float()
            tb_writer.add_scalar("metric/scale", float(scale_for_log.cpu()), step)
            tb_writer.add_scalar(
                "metric/log_scale",
                float(model.log_scale.detach().float().cpu()),
                step,
            )
            tb_writer.add_scalar(
                "metric/temperature",
                float((1.0 / scale_for_log.clamp_min(args.scale_eps)).cpu()),
                step,
            )
            tb_writer.add_scalar(
                "metric/cosine_mean",
                float(metric_debug["cosine_mean"].cpu()),
                step,
            )
            tb_writer.add_scalar(
                "metric/cosine_std",
                float(metric_debug["cosine_std"].cpu()),
                step,
            )
            tb_writer.add_scalar(
                "metric/cosine_min",
                float(metric_debug["cosine_min"].cpu()),
                step,
            )
            tb_writer.add_scalar(
                "metric/cosine_max",
                float(metric_debug["cosine_max"].cpu()),
                step,
            )
```

不要写成两段彼此独立的：

```python
# 第一段 cosine 写 train/loss
if args.metric == "cosine":
    tb_writer.add_scalar("train/loss", ...)

# 第二段又无条件写 train/loss
if tb_writer is not None:
    tb_writer.add_scalar("train/loss", ...)
```

否则 cosine 模式下同一个 step 会向 TensorBoard 写入两条 `train/loss`，导致曲线重复或日志变脏。

### 8.6 logger 日志仍然保留

保留原始：

```python
logger.logkv_mean("loss", loss.item())
```

保留原始：

```python
if step % args.log_interval == 0:
    logger.logkv("step", step)
    logger.logkv("effective_step", effective_step)
    logger.logkv("lr", scheduler.get_last_lr()[0] if scheduler is not None else args.lr)
    logger.dumpkvs()
```

可以额外在 cosine 模式下给 logger 加：

```python
if args.metric == "cosine":
    with torch.no_grad():
        scale_for_log = model.get_scale().detach().float()
        logger.logkv_mean("metric_scale", float(scale_for_log.cpu()))
        logger.logkv_mean("metric_log_scale", float(model.log_scale.detach().float().cpu()))
        logger.logkv_mean(
            "metric_temperature",
            float((1.0 / scale_for_log.clamp_min(args.scale_eps)).cpu()),
        )
        logger.logkv_mean("metric_cosine_mean", float(metric_debug["cosine_mean"].cpu()))
        logger.logkv_mean("metric_cosine_std", float(metric_debug["cosine_std"].cpu()))
```

### 8.7 AMP 与梯度累积保持不变

保留原始逻辑：

```python
scaler.scale(loss / args.accumulation_steps).backward()
```

不要改成：

```python
loss.backward()
```

不要改变：

```text
GradScaler
accumulation_steps
scheduler.step() 时机
```

### 8.8 TensorBoard 记录 train/lr

原始代码在 `log_interval` 时记录 lr 到 logger。TensorBoard 也建议在同一位置记录：

```python
if tb_writer is not None and step % args.log_interval == 0:
    current_lr = scheduler.get_last_lr()[0] if scheduler is not None else args.lr
    tb_writer.add_scalar("train/lr", current_lr, step)
```

---

## 9. 修改 train.py 中途 eval

训练中途 eval 必须使用同一个 metric。

原始 eval 中：

```python
_, scores = compute_prototypical_loss(outputs, labels, args.num_support_val)
```

改成：

```python
if args.metric == "cosine":
    scale = model.get_scale()
    _, scores, _ = compute_cosine_prototypical_loss(
        outputs,
        labels,
        args.num_support_val,
        scale=scale,
        eps=args.scale_eps,
    )
else:
    _, scores = compute_prototypical_loss(
        outputs,
        labels,
        args.num_support_val,
    )
```

注意：

```text
训练用 cosine，eval 也必须用 cosine。
训练用 squared_euclidean，eval 也必须用 squared_euclidean。
```

不要改变 eval 阶段的二分类逻辑：

```python
labels = torch.arange(0, 2, device=args.device).repeat(args.num_query_val)
outputs = rearrange(outputs, '(n b) l -> 1 b n l', n=2)
```

### 9.1 TensorBoard 记录 eval 指标

原始 eval 中会得到：

```python
acc
ap
```

建议在每个 fake class 评估完成后记录：

```python
if tb_writer is not None:
    split_tag = "val_unseen" if VAL_FOLDERS[i] == args.exclude_class else "val_seen"
    tb_writer.add_scalar(f"{split_tag}/acc_{VAL_FOLDERS[i]}", acc.item(), step)
    tb_writer.add_scalar(f"{split_tag}/ap_{VAL_FOLDERS[i]}", ap.item(), step)
```

这样可以保留用户之前实验中使用过的 TensorBoard 风格：

```text
val_seen/acc_*
val_seen/ap_*
val_unseen/acc_*
val_unseen/ap_*
```

---

## 10. 修改 `test.py`

### 10.1 import

新增：

```python
from model.cosine_metric_fsd import CosineMetricFSD
from model.prototypical_utils import (
    compute_prototypical_loss,
    compute_cosine_prototypical_loss,
)
```

### 10.2 创建模型

原始测试代码：

```python
model = timm.create_model("resnet50", pretrained=True, num_classes=1024)
load_model(args.ckpt_path, model=model)
```

修改为：

```python
if args.metric == "cosine":
    logger.info("Creating CosineMetricFSD for testing... ")
    model = CosineMetricFSD(
        pretrained=False,
        embedding_dim=1024,
        init_scale=args.init_scale,
        max_scale=args.max_scale,
    )
elif args.metric == "squared_euclidean":
    logger.info("Creating baseline ResNet50 for testing... ")
    model = timm.create_model("resnet50", pretrained=False, num_classes=1024)
else:
    raise ValueError(f"Unsupported metric: {args.metric}")

load_model(args.ckpt_path, model=model)
model = model.to(args.device)
```

重要：

```text
test.py 加载 checkpoint 时 pretrained=False。
因为 checkpoint 会覆盖模型权重，不需要先下载 ImageNet 权重。
这样也避免无网络服务器卡住或报错。
```

### 10.3 测试 scores 选择

原始：

```python
_, scores = compute_prototypical_loss(outputs, labels, args.num_support_test)
```

修改为：

```python
if args.metric == "cosine":
    scale = model.get_scale()
    _, scores, _ = compute_cosine_prototypical_loss(
        outputs,
        labels,
        args.num_support_test,
        scale=scale,
        eps=args.scale_eps,
    )
else:
    _, scores = compute_prototypical_loss(
        outputs,
        labels,
        args.num_support_test,
    )
```

保留：

```python
prob = scores.softmax(dim=-1).cpu()
```

### 10.4 checkpoint 与 metric 必须匹配

第一版不支持 baseline checkpoint 和 cosine checkpoint 互相加载。

```text
metric=squared_euclidean:
    必须加载原始 ResNet50 checkpoint。

metric=cosine:
    必须加载 CosineMetricFSD checkpoint。
```

如果加载时报 `state_dict` key 不匹配，这是预期行为，说明 checkpoint 和 metric 不匹配。

不要在第一版中加入：

```python
strict=False
```

也不要自动做 key mapping。

### 10.5 train/test 的 max_scale 必须一致

`get_scale()` 中有：

```python
scale = torch.clamp(scale, max=self.max_scale)
```

因此测试时的 `max_scale` 必须与训练时一致。

第一版推荐统一使用：

```bash
--max_scale 100.0
```

不要训练时用：

```bash
--max_scale 100.0
```

测试时改成：

```bash
--max_scale 10.0
```

否则如果训练后学到的 scale 大于测试时 max_scale，测试阶段的 scale 会被额外夹小，softmax 分布会变平，AP 可能被无意改变。

---

## 11. 修改运行脚本

### 11.1 训练仍然使用 torchrun

真实仓库训练依赖 `setup_dist(args)`，会读取：

```text
LOCAL_RANK
RANK
WORLD_SIZE
MASTER_ADDR
MASTER_PORT
```

因此训练必须继续使用 `torchrun` 或原 `scripts/train.sh` 启动。

不要直接运行：

```bash
python train.py
```

### 11.2 cosine mini 训练脚本参数

在训练命令中加入 mini setting 和 cosine 参数：

```bash
--total_training_steps 15000 \
--save_interval 2500 \
--eval_interval 2500 \
--log_interval 200 \
--lr_scheduler_step 5000 \
--lr_scheduler_gamma 0.5 \
--metric cosine \
--init_scale 10.0 \
--max_scale 100.0 \
--scale_eps 1e-12 \
--use_tensorboard True \
--tb_log_interval 20
```

完整示例：

```bash
OMP_NUM_THREADS=1 torchrun $DISTRIBUTED_ARGS train.py \
    --data_root "$data_root" \
    --output_dir $OUTPUT_PATH \
    --num_workers $NUM_WORKERS \
    --seed $SEED \
    --batch_size 16 \
    --lr 1e-4 \
    --lr_scheduler_step 5000 \
    --lr_scheduler_gamma 0.5 \
    --exclude_class $EXCLUDE_CLASS \
    --total_training_steps 15000 \
    --save_interval 2500 \
    --eval_interval 2500 \
    --log_interval 200 \
    --accumulation_steps 1 \
    --use_fp16 True \
    --metric cosine \
    --init_scale 10.0 \
    --max_scale 100.0 \
    --scale_eps 1e-12 \
    --use_tensorboard True \
    --tb_log_interval 20
```

### 11.3 baseline mini 训练脚本参数

baseline 对照应使用相同 mini setting，只是：

```bash
--metric squared_euclidean
```

示例：

```bash
OMP_NUM_THREADS=1 torchrun $DISTRIBUTED_ARGS train.py \
    --data_root "$data_root" \
    --output_dir $OUTPUT_PATH \
    --num_workers $NUM_WORKERS \
    --seed $SEED \
    --batch_size 16 \
    --lr 1e-4 \
    --lr_scheduler_step 5000 \
    --lr_scheduler_gamma 0.5 \
    --exclude_class $EXCLUDE_CLASS \
    --total_training_steps 15000 \
    --save_interval 2500 \
    --eval_interval 2500 \
    --log_interval 200 \
    --accumulation_steps 1 \
    --use_fp16 True \
    --metric squared_euclidean \
    --use_tensorboard True \
    --tb_log_interval 20
```

### 11.4 测试脚本参数

测试命令中必须加入与 checkpoint 对应的 metric：

```bash
python test.py \
    --data_root "$data_root" \
    --test_class $TEST_CLASS \
    --ckpt_path $CKPT_PATH \
    --num_workers 8 \
    --seed $SEED \
    --use_fp16 True \
    --metric cosine \
    --init_scale 10.0 \
    --max_scale 100.0 \
    --scale_eps 1e-12
```

如果测试 baseline checkpoint，则用：

```bash
--metric squared_euclidean
```

---

## 12. Checkpoint 保存与加载

### 12.1 保存

真实 `save_model` 会保存 `model.state_dict()`。

CosineMetricFSD 的 checkpoint 中应包含：

```text
encoder.*
log_scale
```

原始 baseline checkpoint 中不会有：

```text
encoder.*
log_scale
```

### 12.2 加载

真实 `load_model` 使用默认严格匹配的：

```python
load_state_dict(checkpoint["model"])
```

因此：

```text
CosineMetricFSD checkpoint 只能用 CosineMetricFSD 加载。
baseline checkpoint 只能用原始 timm ResNet50 加载。
```

第一版不要做 checkpoint 兼容转换。

### 12.3 args 保存

原始训练保存 checkpoint 时已经保存 `args`。

建议保持：

```python
kwargs = {
    "step": step,
    "effective_step": effective_step,
    "model": model,
    "optimizer": optimizer,
    "scheduler": scheduler,
    "scaler": scaler,
    "args": args,
}
```

这样 checkpoint 中会记录：

```text
metric
init_scale
max_scale
scale_eps
use_tensorboard
tb_log_interval
mini setting 相关参数
```

---

## 13. Smoke Test

实现完成后至少做以下检查。

### 13.1 parser 检查

```bash
python test.py --help
```

应能看到：

```text
--metric
--init_scale
--max_scale
--scale_eps
```

训练参数中还应有：

```text
--use_tensorboard
--tb_log_interval
```

### 13.2 CosineMetricFSD forward 检查

```python
model = CosineMetricFSD(pretrained=False, init_scale=10.0, max_scale=100.0)
x = torch.randn(4, 3, 224, 224)

z = model(x)
scale = model.get_scale()

assert z.shape == (4, 1024)
assert torch.isfinite(z).all()
assert torch.isfinite(scale).all()
assert scale.item() > 0
assert scale.item() <= 100.0
assert abs(scale.item() - 10.0) < 1e-4
```

### 13.3 Cosine loss shape 检查

```python
B, Ns, Nq, Nc, D = 2, 5, 5, 3, 1024
T = Ns + Nq

inputs = torch.randn(B, T, Nc, D, requires_grad=True)
labels = torch.arange(Nc).repeat(B * Nq)
scale = torch.tensor(10.0)

loss, scores, debug = compute_cosine_prototypical_loss(
    inputs,
    labels,
    support_num=Ns,
    scale=scale,
)

assert scores.shape == (B * Nq * Nc, Nc)
assert torch.isfinite(loss).all()
assert torch.isfinite(scores).all()
assert debug["cosine_min"] >= -1.0001
assert debug["cosine_max"] <= 1.0001
```

### 13.4 梯度检查

注意：每次 backward 都要重新构造计算图，不要对同一个 loss 连续 backward 两次。

```python
loss.backward()
assert inputs.grad is not None
assert torch.isfinite(inputs.grad).all()
```

完整模型检查：

```python
model = CosineMetricFSD(pretrained=False)
x = torch.randn(6, 3, 224, 224)
z = model(x)

assert model.log_scale.requires_grad
```

训练一两个 step 后应满足：

```text
encoder 参数有梯度
model.log_scale 有梯度
loss finite
scale finite
TensorBoard event file 正常生成
```

### 13.5 baseline 路径检查

当：

```bash
--metric squared_euclidean
```

时：

```text
1. 使用原始 timm ResNet50。
2. 使用原始 compute_prototypical_loss。
3. 不创建 CosineMetricFSD。
4. 不访问 model.get_scale()。
5. 不访问 model.log_scale。
6. 原始训练和测试逻辑应保持不变。
7. TensorBoard 可以记录 train/loss、train/lr、val_seen/val_unseen 指标，但不应记录 cosine 专属 metric。
```

---

## 14. TensorBoard 预期指标

### 14.1 baseline 和 cosine 都应记录

```text
train/loss
train/lr

val_seen/acc_*
val_seen/ap_*
val_unseen/acc_*
val_unseen/ap_*
```

### 14.2 仅 cosine 模式记录

```text
metric/scale
metric/log_scale
metric/temperature
metric/cosine_mean
metric/cosine_std
metric/cosine_min
metric/cosine_max
```

观察建议：

```text
metric/scale 不应出现 NaN 或 inf。
metric/scale 如果长期贴着 max_scale=100，说明 scale 被上限卡住，后续需要分析是否过度自信。
metric/temperature = 1 / scale，应该始终为正。
metric/cosine_min 和 metric/cosine_max 理论上应大致位于 [-1, 1] 附近。
```

---

## 15. 实验对照建议

第一阶段先跑两组：

```text
1. 原始 FSD baseline
   --metric squared_euclidean

2. L2 cosine + learnable scale
   --metric cosine
   --init_scale 10.0
   --max_scale 100.0
```

所有条件必须一致：

```text
seed
exclude_class
data_root
batch_size
lr
lr_scheduler_step
lr_scheduler_gamma
total_training_steps
num_support_train
num_query_train
num_support_val/test
num_query_val/test
eval_interval
save_interval
log_interval
use_fp16
use_tensorboard
tb_log_interval
```

后续建议补充消融：

```text
L2 cosine + fixed scale=10
```

目的：

```text
判断提升主要来自 L2 cosine metric，
还是 learnable scale 也带来了额外贡献。
```

但第一版不要求 Codex 实现 fixed scale。

---

## 16. 推荐实验命名

```text
exp-l2cos-learnscale-init10-max100-mini15000
```

含义：

```text
l2cos: L2-normalized cosine metric
learnscale: learnable positive scale
init10: init_scale = 10
max100: max_scale = 100
mini15000: 1/10 GenImage mini 数据集，15000 training steps
```

---

## 17. 论文表述建议

可以这样描述：

```text
原始 FSD 直接使用平方欧氏距离进行原型分类，容易受到特征范数和尺度差异的影响。为缓解该问题，我们将 query 特征与类别原型投影到单位超球面上，并采用余弦相似度作为原型匹配分数。同时，引入可学习的正数 logit scale 对相似度 logits 进行自适应缩放，使模型能够自动调节类别分布的 sharpness，从而提升少样本原型分类的稳定性。
```

白话理解：

```text
以前是看 query 离哪个类中心位置更近；
现在是先把所有向量长度统一，再看 query 和哪个类中心方向更像；
最后让模型自己学习这个相似度分数应该放大多少。
```

---

## 18. Codex 执行优先级

请按以下顺序实现：

```text
1. 确认当前分支是 exp-prototype-distance-v1。
2. 确认当前代码来自原始 main baseline，不是 DWT 双分支分支。
3. 修改 requirements.txt：
   - 建议新增 tensorboard
4. 修改 util/parser.py：
   - ModelParser 加 metric/init_scale/max_scale/scale_eps
   - TrainParser 加 use_tensorboard/tb_log_interval，默认 False
5. 新增 model/cosine_metric_fsd.py。
6. 修改 model/prototypical_utils.py：
   - 文件顶部新增 import torch
   - 保留 compute_prototypical_loss
   - 新增 compute_cosine_prototypical_loss
   - cosine loss 内部强制 inputs.float() / scale.float()
7. 修改 train.py：
   - 不要顶层 import SummaryWriter
   - 在 args.use_tensorboard and args.rank == 0 时惰性导入 SummaryWriter
   - rank 0 创建 tb_writer
   - 根据 args.metric 创建模型
   - 根据 args.metric 选择 loss
   - eval 阶段使用同一个 metric
   - 保留 logger
   - 增加 TensorBoard 日志
8. 修改 test.py：
   - 根据 args.metric 创建模型
   - 加载 checkpoint 时 pretrained=False
   - 根据 args.metric 选择 scores
   - 确保 max_scale 与训练时一致
9. 修改 scripts/train.sh 和 scripts/eval.sh，加入 mini setting、cosine 参数和 TensorBoard 参数。
10. 运行 smoke test。
11. 确认 --metric squared_euclidean baseline 路径不受影响。
12. 确认 --use_tensorboard False 时不依赖 tensorboard 包。
```

---

## 19. 最终验收标准

实现完成后应满足：

```text
1. metric=squared_euclidean 时，原始 FSD baseline 行为不变。
2. metric=cosine 时，模型使用 CosineMetricFSD。
3. CosineMetricFSD 中只有一个新增可学习参数 log_scale。
4. scale = clamp(exp(log_scale), max=max_scale)。
5. model/prototypical_utils.py 顶部有 import torch。
6. cosine loss 内部使用 fp32 计算。
7. support/query 先 L2 normalize。
8. support prototype 由 normalized support 求均值得到。
9. prototype 求均值后再次 L2 normalize。
10. logits = cosine_scores * scale。
11. train / eval / test 使用同一个 metric。
12. test.py 加载 checkpoint 时不再使用 pretrained=True。
13. baseline checkpoint 和 cosine checkpoint 不互相混用。
14. train/test 使用相同的 max_scale。
15. mini setting 对齐：15000 steps、2500 eval/save、200 log、lr_scheduler_step=5000、gamma=0.5。
16. TensorBoard event file 正常生成。
17. TensorBoard 能看到 train/loss、train/lr、metric/scale、metric/temperature、cosine 分布和 val 指标。
18. smoke test 中 loss 可以 backward，encoder 和 log_scale 都有梯度。
```

---

## 20. 最终总结

本次实现的是一个独立的 FSD 原型距离改进版本：

```text
不改变数据集；
不改变 episodic training；
不改变 ResNet50 encoder 主体；
不加入双分支；
不加入额外 margin loss；
不加入 Mahalanobis / 多原型 / 图距离；
只把原型匹配从 squared Euclidean 改为 L2-normalized cosine similarity，
并引入 learnable positive scale 校准 cosine logits。
```

主方案：

```text
metric = cosine
support/query L2 normalize
prototype mean 后再次 normalize
scale = clamp(exp(log_scale), max=100)
logits = cosine * scale
loss = cross entropy
```

实验运行：

```text
1/10 GenImage mini 数据集
15000 training steps
2500 eval/save interval
200 log interval
lr=1e-4
lr_scheduler_step=5000
lr_scheduler_gamma=0.5
TensorBoard enabled
tb_log_interval=20
```

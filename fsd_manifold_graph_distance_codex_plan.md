# 给 Codex 的开发说明：在 FSD 中加入“流形感知图距离”并直接进入训练

> 目标仓库：`https://github.com/teheperinko541/Few-Shot-AIGI-Detector`  
> 目标文件优先级：  
> 1. `model/prototypical_utils.py`  
> 2. `util/parser.py`  
> 3. `train.py`  
> 4. `test.py`  
>
> 任务目标：在不推翻原始 FSD 原型网络框架的前提下，把原来的 query-prototype 平方欧氏距离，扩展为可选的 **基于 kNN 图最短路径的流形感知距离**。  
> 用户希望 **直接开始训练**，所以需要实现训练路径，而不是只做测试阶段替换。

---

## 0. 当前 FSD 原始逻辑概括

原始 `model/prototypical_utils.py` 里核心函数是：

```python
compute_prototypical_loss(inputs, labels, support_num)
```

输入张量约定：

```text
inputs.shape = (batch_size, task_samples_num, class_num, feature_dim)
```

其中：

```text
task_samples_num = support_num + query_num
```

训练默认参数大致是：

```text
batch_size = 16
num_class_train = 3
num_support_train = 5
num_query_train = 5
feature_dim = 1024
```

所以训练时：

```text
inputs.shape = (16, 10, 3, 1024)
support_set.shape = (16, 5, 3, 1024)
query_set.shape   = (16, 5, 3, 1024)
```

原始 FSD 做法：

```python
support_set = inputs[:, :support_num, ...]
query_set = inputs[:, support_num:, ...]

prototypes = support_set.mean(dim=1, keepdim=True)
scores = - ((rearrange(query_set, 'b q n l -> b (q n) 1 l') - prototypes) ** 2).sum(dim=-1)
scores = rearrange(scores, 'b n c -> (b n) c')
loss = F.cross_entropy(scores, labels)
```

原始输出：

```text
scores.shape = (batch_size * query_num * class_num, class_num)
```

例如训练时：

```text
scores.shape = (16 * 5 * 3, 3) = (240, 3)
```

每一张 query 都必须得到对所有类别 prototype 的分数。

---

## 1. 本次修改的核心思想

原始 FSD：

```text
query q 直接和每个 prototype c_k 算平方欧氏距离
score_k = -||q - c_k||²
```

修改后：

```text
query q、support 特征、prototype 特征构成一张局部 kNN 图
在图上计算 q 到每个 prototype 的最短路径距离
score_k = -D_graph(q, c_k)
```

核心目标不是做完整 Isomap 降维，而是借鉴 Isomap 的前两步：

```text
kNN 局部邻接图 + 图上最短路径距离
```

不做 MDS 降维。

---

## 2. 重要原则：不要改坏 FSD 的基本结构

必须保留：

```text
1. backbone 不改，仍然由 train.py / test.py 中 timm.create_model("resnet50", ..., num_classes=1024) 提特征。
2. support/query 的切分逻辑不改。
3. prototype 默认仍然是 support set 特征均值。
4. labels 的构造顺序不改。
5. cross_entropy 的使用方式不改。
6. 新的 graph distance 必须返回和原始 scores 完全一致的 shape。
7. 必须保留原始 euclidean 距离作为开关，方便 baseline 对比和排错。
```

---

## 3. 强烈要求：新增 distance_type 开关

请把 `compute_prototypical_loss` 改成类似：

```python
def compute_prototypical_loss(
    inputs,
    labels,
    support_num,
    distance_type="euclidean",
    graph_k=3,
    graph_mode="label_aware_global",
    graph_edge_weight="euclidean",
    graph_query_k_global=3,
    graph_query_min_per_class=1,
    graph_proto_connect="all_own_support",
    graph_cross_class_policy="forbid_support_support",
    graph_fallback="euclidean",
    graph_alpha=1.0,
    graph_warmup_alpha=None,
    current_step=None,
):
    ...
```

其中最少必须支持：

```text
distance_type = "euclidean"  # 原始方法
distance_type = "graph"      # 新方法
```

如果参数太多，也可以用 `**kwargs` 或 dataclass，但推荐先简单实现，保证训练能跑。

---

## 4. 推荐的主方法：Label-aware Global Graph Distance

我们之前讨论过多种图构造方式。推荐主方法使用：

```text
一张大图 + 禁止不同类别 support-support 直接连边 + query 使用全局近邻竞争 + 每类至少保证可达
```

可以命名为：

```text
label_aware_global
```

### 4.1 为什么推荐它

它比“每个类别独立建图”更接近原始 FSD 的逻辑。

原始 FSD 是：

```text
同一个 query 同时和所有类别 prototype 比较
```

推荐方法也保持：

```text
同一个 query 在同一张图里同时面对所有类别结构
```

但它避免了普通全图 kNN 的问题：

```text
real support 和 ADM support 之间直接连边，可能形成跨类别捷径
```

因此推荐主方法是：

```text
support-support：只允许同类 support 内部连边
prototype-support：prototype 只强制连接本类 support
query-support：query 可以连接所有类别 support，以保留类别竞争
query-prototype：默认不要直接连接，避免退化成欧氏距离
```

---

## 5. 我们讨论过的多种图构造方案

### 方案 0：原始欧氏距离 baseline

```text
不建图。
prototype = support 平均值。
score_k = -||q - c_k||²。
```

必须保留，用于对比。

推荐用途：

```text
baseline / debug / 检查 graph 方法有没有改坏整体框架
```

---

### 方案 1：每个类别独立建图，Class-wise Graph

计算 q 到第 k 类 prototype 时，只使用：

```text
q + 第 k 类 support + 第 k 类 prototype
```

例如三类：

```text
D(q, c_real): q + real_support + c_real
D(q, c_ADM):  q + ADM_support  + c_ADM
D(q, c_SD):   q + SD_support   + c_SD
```

优点：

```text
1. 解释最干净。
2. 不会出现跨类别路径。
3. 每个类别天然都能得到一个分数。
```

缺点：

```text
1. 每类 support 很少，例如 5-shot 时只有 5 个 support + 1 prototype + 1 query。
2. 图太小，所谓流形结构可能不稳定。
3. 类别之间缺乏同图竞争，和原始 FSD “一个 query 同时比较所有 prototype” 的感觉稍弱。
```

建议：

```text
作为消融实验实现，不作为第一推荐主方法。
```

可命名：

```text
graph_mode="classwise"
```

---

### 方案 2：普通全图，不做类别限制

节点：

```text
所有 support + 所有 prototype + 当前 query
```

边：

```text
直接按 kNN 建图，不限制类别。
```

优点：

```text
1. 最像传统 Isomap 的局部图。
2. 实现简单。
```

缺点：

```text
1. real support 和 ADM support 可能直接连边。
2. q 到 real prototype 的最短路径可能经过 ADM support。
3. 容易出现跨类别捷径，分类边界被破坏。
```

建议：

```text
可以作为消融，不推荐作为主方法。
```

可命名：

```text
graph_mode="plain_global"
```

---

### 方案 3：推荐主方法，一张大图但禁止 support-support 跨类别边

节点：

```text
所有 support + 所有 prototype + 当前 query
```

边规则：

```text
1. 同类 support-support 可以连边。
2. 不同类 support-support 禁止连边。
3. prototype 只连接本类 support。
4. query 可以连接任意类别 support。
5. 默认不连接 query-prototype。
```

优点：

```text
1. 保留一个 query 同时面对所有类别的竞争逻辑。
2. 避免不同类别 support 之间形成捷径。
3. 比 classwise 图拥有更统一的分类结构。
4. 比 plain global 更符合分类任务。
```

缺点：

```text
1. 如果 query 的全局 kNN 只连到某一个类别，其他类别 prototype 可能不可达。
2. 需要增加每类可达保护或 fallback。
```

推荐加两个保护：

```text
A. query 先连接全局最近的 graph_query_k_global 个 support。
B. 如果某个类别没有被 query 连接到，则额外连接该类别最近的 graph_query_min_per_class 个 support。
```

这样既保留全局竞争，又保证每个类别都有分数。

推荐作为主方法：

```text
graph_mode="label_aware_global"
```

---

### 方案 4：全图 + 跨类别边惩罚

允许不同类别 support-support 连边，但跨类别边乘惩罚系数：

```text
same-class edge: w
cross-class edge: lambda * w
```

例如：

```text
lambda = 2.0
```

优点：

```text
1. 不完全禁止跨类关系。
2. 可以表达边界样本的模糊性。
3. 比完全禁止更柔和。
```

缺点：

```text
1. 多了 lambda 超参数。
2. 不如方案 3 简单清楚。
3. 审稿时需要解释为什么设这个惩罚。
```

建议：

```text
先不作为主方法，后续作为拓展实验。
```

可命名：

```text
graph_mode="penalized_cross_class"
graph_cross_class_penalty=2.0
```

---

## 6. 推荐主方法的详细 pipeline

对每个 episode、每个 query 单独构图。

### 6.1 输入

```text
support_set.shape = (B, S, C, L)
query_set.shape   = (B, Q, C, L)
prototypes.shape  = (B, 1, C, L)
```

其中：

```text
B = batch_size，任务数量
S = support_num
Q = query_num
C = class_num
L = feature_dim
```

### 6.2 处理顺序

对每个 batch index `b`：

```text
support_b = support_set[b]      # (S, C, L)
query_b   = query_set[b]        # (Q, C, L)
proto_b   = prototypes[b, 0]    # (C, L)
```

注意 query 的 flatten 顺序必须和原始代码一致：

```text
query_set 原始维度为 (Q, C, L)
flatten 后顺序应为：
q=0,class=0
q=0,class=1
q=0,class=2
q=1,class=0
q=1,class=1
q=1,class=2
...
```

也就是和：

```python
rearrange(query_set, 'b q n l -> b (q n) 1 l')
```

保持一致。

### 6.3 对每张 query 构图

对每个 `q_idx in range(Q)`，每个 `true_class_placeholder in range(C)` 实际上不需要知道标签，只是遍历所有 query 点：

```text
current_query = query_b[q_idx, class_idx]  # (L,)
```

对于一个 episode 内，总共有：

```text
Q * C
```

张 query。

对每张 query 建一个图。

### 6.4 图节点

推荐节点顺序固定为：

```text
support nodes:   C * S 个
prototype nodes: C 个
query node:      1 个
```

总节点数：

```text
N = C * S + C + 1
```

节点顺序建议：

```text
support index:
class 0 support 0
class 0 support 1
...
class 0 support S-1
class 1 support 0
...
class C-1 support S-1

prototype index:
proto class 0
proto class 1
...
proto class C-1

query index:
current query
```

这样后续能清楚知道每个节点的类别归属。

---

## 7. 建图细节：推荐主方法 label_aware_global

### 7.1 两两距离

先把所有节点拼成：

```python
nodes.shape = (N, L)
```

为了避免 AMP 半精度导致图距离不稳定，图距离内部建议强制 float32：

```python
nodes = nodes.float()
```

计算两两距离矩阵：

```python
dist_matrix.shape = (N, N)
```

可选边权：

```text
graph_edge_weight="euclidean"
graph_edge_weight="squared_euclidean"
```

推荐默认：

```text
euclidean
```

理由：

```text
图路径长度是多段边相加，普通欧氏距离更符合路径长度直觉。
```

但要保留平方欧氏距离选项，因为原始 FSD 使用平方欧氏距离，后续可做消融。

---

### 7.2 初始化邻接矩阵

```python
adj = torch.full((N, N), inf, device=nodes.device, dtype=torch.float32)
adj.fill_diagonal_(0)
```

---

### 7.3 support-support 边

推荐规则：

```text
只在同一类别 support 内部建 kNN。
禁止不同类别 support-support 直接连边。
```

对每个类 `c`：

```text
取该类 S 个 support 节点。
在这 S 个节点内部计算 kNN。
连接每个 support 到同类最近的 graph_k 个 support。
```

如果 `S=5`，`graph_k=3`，则每个 support 最多连 3 个同类 support。

注意：

```text
如果 graph_k >= S，则使用 S-1，因为不能连自己。
```

边要无向化：

```python
adj[i, j] = dist_matrix[i, j]
adj[j, i] = dist_matrix[i, j]
```

---

### 7.4 prototype-support 边

强制每个 prototype 连接本类 support。

推荐默认：

```text
graph_proto_connect="all_own_support"
```

也就是：

```text
c_real 连接所有 real support
c_ADM  连接所有 ADM support
c_SD   连接所有 SD support
```

理由：

```text
prototype 是本类 support 均值，理论上应与本类 support 强绑定。
否则 prototype 可能在 kNN 图中孤立，导致 query 到 prototype 不可达。
```

不要让 prototype 连接其他类别 support。

---

### 7.5 query-support 边

推荐分两步。

#### 第一步：全局最近邻竞争

让当前 query 在所有 support 里找最近的 `graph_query_k_global` 个点：

```text
query -> top-k nearest support among all classes
```

例如：

```text
graph_query_k_global = 3
```

这保留了：

```text
query 同时面对所有类别，谁近谁先接住 query
```

#### 第二步：每类可达保护

检查每个类别是否至少有 `graph_query_min_per_class` 条 query-support 边。

如果某个类别没有被 query 连到，则额外连接该类别最近的 support。

推荐默认：

```text
graph_query_min_per_class = 1
```

理由：

```text
FSD 要求每个 query 对每个类别都输出分数。
如果 q 只连到 ADM 分支，real 和 SD prototype 可能不可达。
每类至少一条 query-support 边可以保证每个类别都有图路径。
```

注意：

```text
这种额外边不是作弊，因为只使用 support 标签和特征距离，不使用 query 标签。
```

#### 是否允许 query-query 边

第一版训练不要使用 transductive。

```text
不把其他 query 放进图。
当前图只包含当前 query 一个 query 节点。
```

理由：

```text
避免 query 之间相互影响，保持 inductive 设定，和原始 FSD 更公平。
```

之后可做 transductive 版本。

---

### 7.6 是否允许 query-prototype 直接边

默认不允许。

理由：

```text
如果 q 直接连 prototype，最短路径可能直接走 q -> prototype，退化接近欧氏距离。
```

除非做消融：

```text
graph_allow_query_proto_edge=True
```

默认：

```text
False
```

---

## 8. 最短路径算法选择

我们讨论过 Dijkstra 和 Floyd。

结论：

```text
在同一张图、同一套边权、没有负权边时，Dijkstra 和 Floyd 得到的最短路径距离理论上相同。
它们只是计算方法不同，不是距离定义不同。
```

### 推荐训练第一版用 Floyd-Warshall

理由：

```text
1. 每张 query 的图很小，例如 C=3,S=5 时 N=3*5+3+1=19。
2. Floyd 实现简单，得到完整 N×N 最短路径矩阵。
3. 直接取 query 节点到各 prototype 节点的距离即可。
```

实现上请用 PyTorch，不要用 scipy/networkx。

伪代码：

```python
def floyd_warshall_torch(adj):
    # adj: (N, N), float32
    dist = adj.clone()
    N = dist.shape[0]
    for k in range(N):
        dist = torch.minimum(dist, dist[:, k:k+1] + dist[k:k+1, :])
    return dist
```

注意：

```text
torch.minimum 是分段可导的，但 kNN/topk/最短路径本身是非光滑的。
这不是理想可微流形学习，但可以作为第一版 hard graph distance 训练。
```

---

## 9. 训练阶段的关键风险：不要让计算图被 detach

用户希望直接开始训练，所以必须支持训练。

### 9.1 不要使用这些库做训练图距离

不要在训练 loss 中使用：

```text
networkx
scipy.sparse.csgraph
numpy shortest path
sklearn Isomap
```

理由：

```text
这些会把 Tensor 转到 CPU / numpy，导致 PyTorch 计算图断开。
loss 可能能算，但 backbone 参数无法通过 graph distance 正常更新。
```

### 9.2 使用纯 PyTorch

所有距离矩阵、邻接矩阵、Floyd 更新都尽量用 torch 张量完成。

允许 Python for-loop，但张量运算要在 PyTorch 里。

### 9.3 AMP 注意事项

在 `train.py` 里有 autocast/fp16。

图距离内部建议强制 float32：

```python
with torch.cuda.amp.autocast(enabled=False):
    graph_scores = compute_graph_scores(...features.float()...)
```

或者在函数内部：

```python
inputs_for_graph = inputs.float()
```

输出的 `scores` 可以是 float32，cross entropy 没问题。

---

## 10. 训练策略：直接训练 + 稳定性选项

用户希望直接开始训练，因此实现时允许：

```text
distance_type="graph"
graph_alpha=1.0
graph_start_step=0
```

但建议保留稳定选项：

### 10.1 纯 graph 训练

```text
scores = graph_scores
```

命令行可设：

```bash
--distance_type graph --graph_alpha 1.0 --graph_warmup_steps 0
```

优点：

```text
符合“直接训练”的目标。
```

缺点：

```text
hard kNN + hard shortest path 非光滑，训练可能不稳定或较慢。
```

---

### 10.2 推荐保留：欧氏 + 图距离混合

实现：

```python
scores = (1 - alpha) * euclidean_scores + alpha * graph_scores
```

其中：

```text
alpha = graph_alpha
```

如果 `graph_alpha=1.0`，就是纯 graph。

如果 `graph_alpha=0.5`，就是混合。

理由：

```text
1. 保留原始 FSD 稳定信号。
2. 减少图断开、kNN 抖动带来的不稳定。
3. 方便消融实验。
```

---

### 10.3 推荐保留：warm-up

虽然用户希望直接训练，但为了实验稳定，建议支持：

```text
前 graph_warmup_steps 使用 euclidean
之后逐渐增加 graph_alpha
```

例如：

```text
graph_warmup_steps = 5000
graph_alpha 从 0 线性升到 1
```

但默认可以设为：

```text
graph_warmup_steps = 0
```

---

## 11. 图距离不可达的处理

即使有每类可达保护，仍建议保留 fallback。

### 推荐 fallback

```text
graph_fallback="euclidean"
```

规则：

```python
if D_graph(q, c_k) is inf or nan:
    D_graph(q, c_k) = D_euclidean(q, c_k)
```

不推荐默认使用：

```text
1e6 大数
```

理由：

```text
大数容易造成 loss 极端不稳定。
```

---

## 12. 新增 parser 参数

在 `util/parser.py` 中给 TrainParser 和 TestParser 都增加类似参数：

```python
self.parser.add_argument('--distance_type', type=str, default='euclidean',
                         choices=['euclidean', 'graph'])

self.parser.add_argument('--graph_mode', type=str, default='label_aware_global',
                         choices=['classwise', 'plain_global', 'label_aware_global', 'penalized_cross_class'])

self.parser.add_argument('--graph_k', type=int, default=3)

self.parser.add_argument('--graph_edge_weight', type=str, default='euclidean',
                         choices=['euclidean', 'squared_euclidean'])

self.parser.add_argument('--graph_query_k_global', type=int, default=3)

self.parser.add_argument('--graph_query_min_per_class', type=int, default=1)

self.parser.add_argument('--graph_cross_class_penalty', type=float, default=2.0)

self.parser.add_argument('--graph_alpha', type=float, default=1.0)

self.parser.add_argument('--graph_warmup_steps', type=int, default=0)

self.parser.add_argument('--graph_fallback', type=str, default='euclidean',
                         choices=['euclidean', 'large'])

self.parser.add_argument('--graph_large_distance', type=float, default=1e6)
```

如果不想一次实现所有方案，最少必须实现：

```text
distance_type
graph_mode = label_aware_global
graph_k
graph_query_k_global
graph_query_min_per_class
graph_edge_weight
graph_alpha
graph_fallback
```

---

## 13. 修改 train.py

当前调用：

```python
loss, _ = compute_prototypical_loss(outputs, labels, args.num_support_train)
```

需要改成：

```python
loss, _ = compute_prototypical_loss(
    outputs,
    labels,
    args.num_support_train,
    distance_type=args.distance_type,
    graph_mode=args.graph_mode,
    graph_k=args.graph_k,
    graph_edge_weight=args.graph_edge_weight,
    graph_query_k_global=args.graph_query_k_global,
    graph_query_min_per_class=args.graph_query_min_per_class,
    graph_cross_class_penalty=args.graph_cross_class_penalty,
    graph_alpha=args.graph_alpha,
    graph_warmup_steps=args.graph_warmup_steps,
    current_step=step,
    graph_fallback=args.graph_fallback,
    graph_large_distance=args.graph_large_distance,
)
```

验证阶段也要传入相同参数，或者为了公平可单独支持：

```text
--eval_distance_type
```

第一版可直接用同一套参数。

---

## 14. 修改 test.py

当前调用：

```python
_, scores = compute_prototypical_loss(outputs, labels, args.num_support_test)
```

也需要传参数：

```python
_, scores = compute_prototypical_loss(
    outputs,
    labels,
    args.num_support_test,
    distance_type=args.distance_type,
    graph_mode=args.graph_mode,
    graph_k=args.graph_k,
    graph_edge_weight=args.graph_edge_weight,
    graph_query_k_global=args.graph_query_k_global,
    graph_query_min_per_class=args.graph_query_min_per_class,
    graph_cross_class_penalty=args.graph_cross_class_penalty,
    graph_alpha=args.graph_alpha,
    graph_warmup_steps=0,
    current_step=None,
    graph_fallback=args.graph_fallback,
    graph_large_distance=args.graph_large_distance,
)
```

---

## 15. 实现函数建议

在 `model/prototypical_utils.py` 中新增辅助函数。

建议拆成：

```python
def _pairwise_distance(x, squared=False, eps=1e-12):
    ...

def _build_label_aware_global_adj(
    support,
    prototypes,
    query,
    graph_k,
    graph_edge_weight,
    graph_query_k_global,
    graph_query_min_per_class,
    graph_fallback,
    ...
):
    ...

def _floyd_warshall_torch(adj):
    ...

def _compute_graph_scores_label_aware_global(
    support_set,
    query_set,
    prototypes,
    ...
):
    ...
```

### 15.1 支持集 shape 说明

`support_set` 是：

```text
(B, S, C, L)
```

为了方便构图，单个 episode 内可以转成：

```text
support_by_class = support_set[b].permute(1, 0, 2)
support_by_class.shape = (C, S, L)
```

即：

```text
class, support, feature
```

### 15.2 query flatten 顺序

单个 episode：

```text
query_set[b].shape = (Q, C, L)
```

遍历顺序必须是：

```python
for q_idx in range(Q):
    for class_idx in range(C):
        query = query_set[b, q_idx, class_idx]
```

这样输出顺序才对应 labels：

```text
[0, 1, 2, 0, 1, 2, ...]
```

---

## 16. 推荐 graph scores 输出形状

对于每个 episode：

```text
episode_scores.shape = (Q * C, C)
```

所有 batch 拼接：

```text
scores.shape = (B * Q * C, C)
```

必须和原始代码一致。

---

## 17. classwise 模式实现说明

如果实现 `graph_mode="classwise"`：

对每个 query 和每个 candidate class k：

```text
nodes = q + support[k] + prototype[k]
```

算 q 到 prototype[k] 的图距离。

得到：

```text
D(q, c_0), D(q, c_1), ..., D(q, c_{C-1})
```

再组成：

```text
scores_q = -distances
```

注意：

```text
classwise 模式每个 candidate class 都单独建图。
```

它天然能得到每个类分数。

---

## 18. plain_global 模式实现说明

如果实现 `graph_mode="plain_global"`：

```text
所有 support + prototypes + q 一起建图。
所有节点按全局 kNN 连接。
不禁止跨类别 support-support。
prototype 可强制连本类 support。
```

这个模式主要用于消融，不推荐主用。

---

## 19. penalized_cross_class 模式实现说明

如果实现 `graph_mode="penalized_cross_class"`：

先全局 kNN，再根据节点类别调整边权：

```text
support-support same class: weight = d
support-support different class: weight = lambda * d
prototype-own support: weight = d
prototype-other support: 禁止或 lambda * d，推荐禁止
query-support: weight = d
```

这个模式作为后续增强即可。

---

## 20. 单元测试要求

Codex 必须添加或至少临时写一个小测试脚本，确认：

### 20.1 原始 euclidean 模式输出一致

用随机 tensor：

```python
inputs = torch.randn(B, S+Q, C, L, device='cuda', requires_grad=True)
labels = torch.arange(C, device='cuda').repeat(B * Q)
```

比较：

```text
新函数 distance_type="euclidean"
```

和原始公式结果是否一致。

### 20.2 graph 模式 shape 正确

确认：

```text
loss 是 scalar
scores.shape == (B * Q * C, C)
```

### 20.3 graph 模式 backward 不报错

```python
loss.backward()
assert inputs.grad is not None
assert torch.isfinite(inputs.grad).all()
```

如果 `inputs.grad` 为 None，说明实现中有 detach/numpy/networkx/scipy 导致断图。

### 20.4 训练一个 mini step

在真实 train.py 里跑 1-2 step，确认：

```text
1. loss 正常
2. 不出现 nan/inf
3. GPU 显存可接受
4. 速度虽然变慢但能运行
```

---

## 21. 推荐训练命令

### 21.1 直接 graph 训练

```bash
bash scripts/train.sh \
  --distance_type graph \
  --graph_mode label_aware_global \
  --graph_k 3 \
  --graph_edge_weight euclidean \
  --graph_query_k_global 3 \
  --graph_query_min_per_class 1 \
  --graph_alpha 1.0 \
  --graph_warmup_steps 0 \
  --graph_fallback euclidean
```

如果原脚本不支持追加参数，请修改 `scripts/train.sh`。

---

### 21.2 更稳的混合训练

```bash
bash scripts/train.sh \
  --distance_type graph \
  --graph_mode label_aware_global \
  --graph_k 3 \
  --graph_edge_weight euclidean \
  --graph_query_k_global 3 \
  --graph_query_min_per_class 1 \
  --graph_alpha 0.5 \
  --graph_warmup_steps 0 \
  --graph_fallback euclidean
```

---

### 21.3 warm-up 训练

```bash
bash scripts/train.sh \
  --distance_type graph \
  --graph_mode label_aware_global \
  --graph_k 3 \
  --graph_edge_weight euclidean \
  --graph_query_k_global 3 \
  --graph_query_min_per_class 1 \
  --graph_alpha 1.0 \
  --graph_warmup_steps 5000 \
  --graph_fallback euclidean
```

---

## 22. 需要记录的实验

至少跑这些：

### 22.1 baseline

```text
FSD original euclidean
```

### 22.2 主方法

```text
FSD + label_aware_global graph distance
k=3
edge_weight=euclidean
```

### 22.3 k 值消融

```text
k = 1, 2, 3, 5
```

注意：

```text
k=1 可能导致图结构过稀。
k 太大可能接近欧氏距离。
```

### 22.4 图构造消融

```text
classwise
plain_global
label_aware_global
penalized_cross_class
```

如果时间不够，至少做：

```text
euclidean baseline
classwise
label_aware_global
```

### 22.5 边权消融

```text
euclidean edge weight
squared_euclidean edge weight
```

### 22.6 query 连边策略消融

```text
只有 global top-k
global top-k + min_per_class
只用 min_per_class
```

推荐主设置：

```text
global top-k + min_per_class
```

---

## 23. 评价指标

原仓库已经用：

```text
Accuracy
AveragePrecision
```

继续保留。

注意：

```text
Accuracy 看最终分类对不对。
AP 看 fake 分数排序能力，分数变化即使不改变最终预测，也可能影响 AP。
```

建议额外日志记录：

```text
1. graph unreachable count
2. fallback count
3. mean graph distance
4. mean euclidean distance
5. margin：正确类距离和最近错误类距离的差
```

---

## 24. 不要做的事情

第一版不要做：

```text
1. 不做完整 Isomap。
2. 不做 MDS 降维。
3. 不引入 sklearn Isomap。
4. 不把所有 query 一起放进图做 transductive。
5. 不删除原始 euclidean 代码。
6. 不改变数据集读取逻辑。
7. 不改变 backbone。
8. 不改变 labels 顺序。
9. 不用 numpy/scipy/networkx 做训练图距离。
```

---

## 25. transductive 版本说明：暂不作为第一版

我们讨论过 transductive：

```text
同一个 episode 中所有 query 一起参与建图，q1 可以通过 q2/q3 到 support/prototype。
```

它的优点：

```text
利用 query 之间的无标签分布结构，可能提升效果。
```

缺点：

```text
1. 它比原始 inductive 设置更强。
2. q1 的预测会受 q2/q3 影响。
3. 审稿时需要单独说明。
```

因此第一版不要做 transductive。

后续可增加：

```text
graph_inference_mode="transductive"
```

但必须单独实验，不要和 inductive 方法混在一起比较。

---

## 26. 最终推荐实现版本

如果只能实现一个版本，请实现：

```text
distance_type="graph"
graph_mode="label_aware_global"
graph_edge_weight="euclidean"
graph_k=3
graph_query_k_global=3
graph_query_min_per_class=1
graph_proto_connect="all_own_support"
graph_fallback="euclidean"
graph_alpha=1.0
graph_warmup_steps=0
```

一句话：

> 保留原型均值；每张 query 单独建一张类别感知大图；support-support 只允许同类连接；prototype 强制连本类 support；query 先做全局近邻竞争，再保证每类至少可达；用 PyTorch Floyd 算 query 到所有 prototype 的最短路径距离；scores = -graph distance；直接进入训练。

---

## 27. 预期代码结构示意

```python
def compute_prototypical_loss(...):
    support_set = inputs[:, :support_num, ...]
    query_set = inputs[:, support_num:, ...]
    prototypes = support_set.mean(dim=1, keepdim=True)

    euclidean_scores = compute_original_scores(query_set, prototypes)

    if distance_type == "euclidean":
        scores = euclidean_scores
    elif distance_type == "graph":
        graph_scores = compute_graph_scores(
            support_set=support_set,
            query_set=query_set,
            prototypes=prototypes.squeeze(1),
            ...
        )
        alpha = resolve_alpha(graph_alpha, graph_warmup_steps, current_step)
        scores = (1 - alpha) * euclidean_scores + alpha * graph_scores
    else:
        raise ValueError

    loss = F.cross_entropy(scores, labels)
    return loss, scores
```

其中：

```text
compute_original_scores 输出 shape: (B*Q*C, C)
compute_graph_scores 输出 shape:    (B*Q*C, C)
```

---

## 28. 成功标准

代码修改完成后，必须满足：

```text
1. distance_type=euclidean 时，结果和原始 FSD 基本一致。
2. distance_type=graph 时，loss、scores shape 正确。
3. graph 模式可以跑通一次 forward/backward。
4. graph 模式训练时模型参数有梯度。
5. train.py 可以正常跑至少几个 step。
6. test.py 可以用 graph 模式正常输出 acc/AP。
7. 代码里有足够注释，说明每种 graph_mode 的区别。
8. 不引入破坏训练计算图的 numpy/scipy/networkx。
```

---

## 29. 给 Codex 的执行顺序

请按这个顺序改：

```text
1. 备份原始 prototypical_utils.py。
2. 在 prototypical_utils.py 中拆出 compute_original_scores。
3. 实现 pairwise distance。
4. 实现 label-aware global graph adjacency。
5. 实现 torch Floyd-Warshall。
6. 实现 compute_graph_scores。
7. 在 compute_prototypical_loss 中加入 distance_type 分支。
8. 给 parser.py 添加参数。
9. 修改 train.py 调用。
10. 修改 test.py 调用。
11. 写一个 quick_test_graph_distance.py 或在临时代码里验证 shape/backward。
12. 确认 euclidean 模式和原始一致。
13. 用 graph 模式跑 1-2 个训练 step。
14. 如果 graph 训练太慢，先不要优化算法，先确保正确性。
```

---

## 30. 论文写法提示

这个方法不要写成“完整 Isomap”。

建议写成：

```text
受 Isomap 中局部邻接图和测地距离估计思想启发，我们提出一种流形感知原型距离。
该方法在每个 few-shot episode 内构建类别感知 kNN 图，用图上最短路径距离替代原始查询样本与类别原型之间的欧氏距离。
```

推荐方法名：

```text
Manifold-aware Prototype Distance
Local Manifold Prototypical Distance
Graph Geodesic Prototype Distance
```

中文名：

```text
流形感知原型距离
局部流形原型距离
图测地原型距离
```

---

## 31. 一句话总结给 Codex

请实现一个可训练的 FSD 图距离版本：保持 FSD 原型网络整体结构不变，只把 query-prototype 欧氏距离替换成 PyTorch 实现的 label-aware kNN graph shortest-path distance；主方法采用一张大图、禁止不同类别 support-support 连边、prototype 强制连接本类 support、query 全局近邻竞争并保证每类可达；输出 scores shape 必须和原始一致，并保留 euclidean baseline 开关。

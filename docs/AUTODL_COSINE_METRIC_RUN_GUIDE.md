# AutoDL Cosine Metric 实验执行说明

本文档用于在 AutoDL + Cursor 环境中运行 `exp-prototype-distance-v1` 分支的 L2 cosine + learnable scale 实验。

核心原则：

- Windows 本机只改代码、提交、推送到 GitHub。
- AutoDL 从 GitHub 拉取最新代码。
- 训练、评估、TensorBoard、结果表都在 AutoDL 上完成。
- 实验输出统一放在 `/root/autodl-tmp/runs/` 下，不提交到 Git。

---

## 1. 当前是否可以直接拉取运行

可以。当前 GitHub 上的 `origin/exp-prototype-distance-v1` 已包含：

- cosine metric 实现
- `METRIC=cosine` 脚本开关
- TensorBoard 记录
- `requirements.txt` 中的 `tensorboard`

AutoDL 上需要满足：

- 仓库目录存在：`/root/autodl-tmp/Few-Shot-AIGI-Detector-main`
- 远程 `origin` 指向 GitHub：`https://github.com/zenghao0718/MC-MPD.git`
- 数据目录存在：`/root/autodl-tmp/data`
- `/root/autodl-tmp/data` 下直接包含：
  - `ADM`
  - `BigGAN`
  - `glide`
  - `Midjourney`
  - `SD`
  - `VQDM`
  - `real`

如果数据不在 `/root/autodl-tmp/data`，运行时用 `DATA_ROOT=你的数据目录` 覆盖。

---

## 2. 在 AutoDL 上拉取最新代码

在 AutoDL 的 Cursor 终端里执行：

```bash
cd /root/autodl-tmp/Few-Shot-AIGI-Detector-main
git switch exp-prototype-distance-v1
source /etc/network_turbo
git pull --ff-only
git status
```

正常情况：

- 当前分支是 `exp-prototype-distance-v1`
- `git pull --ff-only` 成功
- `git status` 显示工作区干净

如果 GitHub 连接超时，先确认已经执行：

```bash
source /etc/network_turbo
```

---

## 3. 确认环境和数据

确认数据目录：

```bash
ls /root/autodl-tmp/data
```

确认至少能看到：

```text
ADM  BigGAN  glide  Midjourney  SD  VQDM  real
```

如果当前环境还没安装新依赖，执行：

```bash
pip install -r requirements.txt
```

`tensorboard` 已经写入 `requirements.txt`。如果不安装它，训练仍可能启动，但开启 TensorBoard 时会报缺包。

---

## 4. 跑一个单类 cosine 训练

下面以 `ADM` 作为 held-out unseen class 为例。

输出目录沿用 `runs/` 规范：

```bash
cd /root/autodl-tmp/Few-Shot-AIGI-Detector-main
source /etc/network_turbo

RUN_ROOT=/root/autodl-tmp/runs/exp-prototype-distance-v1/l2cos_learnscale_init10_max100_seed42_steps15000/exclude_ADM
mkdir -p "$RUN_ROOT"

METRIC=cosine \
EXCLUDE_CLASS=ADM \
OUTPUT_PATH="$RUN_ROOT" \
SEED=42 \
TOTAL_STEPS=15000 \
USE_TENSORBOARD=True \
bash scripts/train.sh 2>&1 | tee "$RUN_ROOT/train.log"
```

这条命令做的事：

- `METRIC=cosine`：开启 L2 cosine + learnable scale。
- `EXCLUDE_CLASS=ADM`：训练时排除 ADM，后面用 ADM 做 unseen 测试。
- `OUTPUT_PATH="$RUN_ROOT"`：checkpoint、日志、TensorBoard 都放进本次任务目录。
- `USE_TENSORBOARD=True`：开启 TensorBoard。

训练期间 checkpoint 默认保存到：

```text
/root/autodl-tmp/runs/exp-prototype-distance-v1/l2cos_learnscale_init10_max100_seed42_steps15000/exclude_ADM/ckpt/
```

TensorBoard event 默认保存到：

```text
/root/autodl-tmp/runs/exp-prototype-distance-v1/l2cos_learnscale_init10_max100_seed42_steps15000/exclude_ADM/tb/
```

---

## 5. 跑六类 leave-one-out cosine 训练和评估

如果要按 6 个 fake generator 全部跑一遍：

```bash
cd /root/autodl-tmp/Few-Shot-AIGI-Detector-main
source /etc/network_turbo

RUNS_ROOT=/root/autodl-tmp/runs/exp-prototype-distance-v1/l2cos_learnscale_init10_max100_seed42_steps15000

METRIC=cosine \
RUNS_ROOT="$RUNS_ROOT" \
MODE=all \
TOTAL_STEPS=15000 \
bash scripts/run_all_excludes.sh
```

输出结构会是：

```text
/root/autodl-tmp/runs/exp-prototype-distance-v1/l2cos_learnscale_init10_max100_seed42_steps15000/
|-- exclude_ADM/
|-- exclude_BigGAN/
|-- exclude_glide/
|-- exclude_Midjourney/
|-- exclude_SD/
|-- exclude_VQDM/
```

如果只想先跑一个类别，可以用：

```bash
CLASSES="ADM" \
METRIC=cosine \
RUNS_ROOT="$RUNS_ROOT" \
MODE=all \
TOTAL_STEPS=15000 \
bash scripts/run_all_excludes.sh
```

---

## 6. 单独评估一个 cosine checkpoint

如果训练已经完成，只想单独评估 ADM：

```bash
cd /root/autodl-tmp/Few-Shot-AIGI-Detector-main
source /etc/network_turbo

RUN_ROOT=/root/autodl-tmp/runs/exp-prototype-distance-v1/l2cos_learnscale_init10_max100_seed42_steps15000/exclude_ADM
CKPT_PATH="$RUN_ROOT/ckpt/resnet50_step[15000].pth"

METRIC=cosine \
TEST_CLASS=ADM \
OUTPUT_PATH="$RUN_ROOT" \
CKPT_PATH="$CKPT_PATH" \
TOTAL_STEPS=15000 \
bash scripts/eval.sh 2>&1 | tee "$RUN_ROOT/eval.log"
```

注意：

- cosine checkpoint 必须配 `METRIC=cosine`。
- baseline checkpoint 必须配 `METRIC=squared_euclidean`。
- checkpoint 文件名里有方括号：`resnet50_step[15000].pth`，所以路径一定要加引号。

---

## 7. 启动 TensorBoard

单类 ADM 的 TensorBoard 启动命令：

```bash
tensorboard \
  --logdir /root/autodl-tmp/runs/exp-prototype-distance-v1/l2cos_learnscale_init10_max100_seed42_steps15000/exclude_ADM/tb \
  --host 0.0.0.0 \
  --port 6006
```

六类一起查看时，可以直接把 logdir 指到总目录：

```bash
tensorboard \
  --logdir /root/autodl-tmp/runs/exp-prototype-distance-v1/l2cos_learnscale_init10_max100_seed42_steps15000 \
  --host 0.0.0.0 \
  --port 6006
```

然后在 AutoDL 或 Cursor 的端口转发/服务列表里找到 `6006` 端口，复制可以打开的网址。

需要给用户的链接格式应该类似：

```text
TensorBoard 地址：<AutoDL 或 Cursor 给出的 6006 端口访问链接>
```

不要只说“打开 TensorBoard”。如果暂时看不到链接，就告诉用户：

```text
请在 AutoDL 的端口转发或服务列表中找到 6006 端口，复制对应访问地址。
```

建议重点看这些曲线：

- `train/loss`
- `train/lr`
- `metric/scale`
- `metric/log_scale`
- `metric/temperature`
- `metric/cosine_mean`
- `metric/cosine_std`
- `metric/cosine_min`
- `metric/cosine_max`
- `val_seen/acc_*`
- `val_seen/ap_*`
- `val_unseen/acc_*`
- `val_unseen/ap_*`

如果 `metric/scale` 长期贴着 `100.0`，说明 learnable scale 被上限卡住，后续分析结果时要记录这一点。

---

## 8. baseline 对照怎么跑

`bash scripts/train.sh` 默认是 baseline，因为 `METRIC` 默认值是 `squared_euclidean`。

推荐 baseline 复现输出目录：

```bash
RUN_ROOT=/root/autodl-tmp/runs/baseline_fsd_paper/exclude_ADM
mkdir -p "$RUN_ROOT"

EXCLUDE_CLASS=ADM \
OUTPUT_PATH="$RUN_ROOT" \
SEED=42 \
TOTAL_STEPS=15000 \
USE_TENSORBOARD=True \
bash scripts/train.sh 2>&1 | tee "$RUN_ROOT/train.log"
```

如果显式写 baseline，也可以：

```bash
METRIC=squared_euclidean \
EXCLUDE_CLASS=ADM \
OUTPUT_PATH="$RUN_ROOT" \
USE_TENSORBOARD=True \
bash scripts/train.sh 2>&1 | tee "$RUN_ROOT/train.log"
```

注意：

- baseline 不会记录 `metric/scale`、`metric/cosine_*`。
- cosine 才会记录这些 cosine 专属 TensorBoard 指标。

---

## 9. 任务结束后的结果对比表

每次任务结束后，需要在对应 `runs/` 目录下保存结果对比表。

推荐 Markdown 文件：

```text
/root/autodl-tmp/runs/exp-prototype-distance-v1/l2cos_learnscale_init10_max100_seed42_steps15000/paper_vs_reproduce_comparison.md
```

如需要机器可读，也可以额外保存：

```text
/root/autodl-tmp/runs/exp-prototype-distance-v1/l2cos_learnscale_init10_max100_seed42_steps15000/paper_vs_reproduce_comparison.csv
```

Markdown 表格模板：

```markdown
# Paper vs Reproduce vs Cosine Metric Comparison

| 方法/分支 | 训练设置 | 测试类别 | Accuracy | AP | Checkpoint | 日志目录 | 备注 |
|---|---|---:|---:|---:|---|---|---|
| 原论文 FSD | 论文报告设置 | ADM | 待填 | 待填 | 论文数据 | 论文 PDF / 表格 | 原论文结果 |
| baseline_fsd_paper | mini 15000 steps, seed 42 | ADM | 待填 | 待填 | `/root/autodl-tmp/runs/baseline_fsd_paper/exclude_ADM/ckpt/resnet50_step[15000].pth` | `/root/autodl-tmp/runs/baseline_fsd_paper/exclude_ADM` | 复现 baseline |
| exp-prototype-distance-v1 cosine | mini 15000 steps, seed 42, init_scale=10, max_scale=100 | ADM | 待填 | 待填 | `/root/autodl-tmp/runs/exp-prototype-distance-v1/l2cos_learnscale_init10_max100_seed42_steps15000/exclude_ADM/ckpt/resnet50_step[15000].pth` | `/root/autodl-tmp/runs/exp-prototype-distance-v1/l2cos_learnscale_init10_max100_seed42_steps15000/exclude_ADM` | L2 cosine + learnable scale |
```

如果 6 类全部跑完，建议每个测试类别一行。

---

## 10. 常见问题

### 10.1 我直接运行 `bash scripts/train.sh`，为什么不是 cosine？

因为脚本默认：

```bash
METRIC=${METRIC:-squared_euclidean}
```

也就是说，不写 `METRIC=cosine` 就是 baseline。

cosine 必须这样开启：

```bash
METRIC=cosine bash scripts/train.sh
```

### 10.2 TensorBoard 没有 event 文件怎么办？

先确认训练命令里有：

```bash
USE_TENSORBOARD=True
```

再确认输出目录下是否有：

```bash
ls <OUTPUT_PATH>/tb
```

如果报缺少 TensorBoard，执行：

```bash
pip install -r requirements.txt
```

### 10.3 checkpoint 加载报 key 不匹配怎么办？

先确认 metric 和 checkpoint 类型一致：

- baseline checkpoint 用 `METRIC=squared_euclidean`
- cosine checkpoint 用 `METRIC=cosine`

第一版不支持二者互相加载，不要用 `strict=False` 绕过。

### 10.4 数据路径不在 `/root/autodl-tmp/data` 怎么办？

用 `DATA_ROOT` 覆盖：

```bash
DATA_ROOT=/你的/GenImage/路径 METRIC=cosine bash scripts/train.sh
```

这个路径下面要直接包含 `ADM`、`BigGAN`、`glide`、`Midjourney`、`SD`、`VQDM`、`real`。

### 10.5 想先只确认命令能启动，不想跑完整 15000 steps 怎么办？

可以临时跑很短步数做启动检查：

```bash
METRIC=cosine \
EXCLUDE_CLASS=ADM \
OUTPUT_PATH=/root/autodl-tmp/runs/debug/cosine_startup_check/exclude_ADM \
TOTAL_STEPS=2 \
SAVE_INTERVAL=2 \
EVAL_INTERVAL=2 \
LOG_INTERVAL=1 \
USE_TENSORBOARD=True \
bash scripts/train.sh
```

这个只用于启动检查，不作为正式实验结果。

---

## 11. 正式 cosine 实验推荐命令汇总

单类 ADM：

```bash
cd /root/autodl-tmp/Few-Shot-AIGI-Detector-main
source /etc/network_turbo

RUN_ROOT=/root/autodl-tmp/runs/exp-prototype-distance-v1/l2cos_learnscale_init10_max100_seed42_steps15000/exclude_ADM
mkdir -p "$RUN_ROOT"

METRIC=cosine \
EXCLUDE_CLASS=ADM \
OUTPUT_PATH="$RUN_ROOT" \
SEED=42 \
TOTAL_STEPS=15000 \
USE_TENSORBOARD=True \
bash scripts/train.sh 2>&1 | tee "$RUN_ROOT/train.log"
```

六类 leave-one-out：

```bash
cd /root/autodl-tmp/Few-Shot-AIGI-Detector-main
source /etc/network_turbo

RUNS_ROOT=/root/autodl-tmp/runs/exp-prototype-distance-v1/l2cos_learnscale_init10_max100_seed42_steps15000

METRIC=cosine \
RUNS_ROOT="$RUNS_ROOT" \
MODE=all \
TOTAL_STEPS=15000 \
bash scripts/run_all_excludes.sh
```

TensorBoard：

```bash
tensorboard \
  --logdir /root/autodl-tmp/runs/exp-prototype-distance-v1/l2cos_learnscale_init10_max100_seed42_steps15000 \
  --host 0.0.0.0 \
  --port 6006
```


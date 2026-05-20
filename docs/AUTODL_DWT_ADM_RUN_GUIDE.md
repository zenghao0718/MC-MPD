# AutoDL DWT 双分支 ADM 试跑实施指南

本文档用于在 AutoDL 上试跑一次 `exclude_ADM` 的 DWT 双分支实验。

目标流程：

```text
进入 screen
  -> 拉取 exp-dual-branch-dwt 分支
  -> 设置 runs/ 下的实验目录
  -> 生成 DWT stats
  -> 跑 smoke test
  -> 正式训练 ADM
  -> 训练完成后测试 ADM
  -> 用 TensorBoard 观察
```

## 0. 粘贴命令时怎么选

Cursor / VS Code 终端提示：

```text
是否确实要将 N 行文本粘贴到终端？
```

如果是本文档里的多行命令块，请选择：

```text
粘贴(P)
```

不要选择：

```text
粘贴为一行(O)
```

原因：

```text
粘贴(P) 会保留换行，适合多行命令。
粘贴为一行(O) 会把多行压成一行，容易把 mkdir、python、torchrun 拼坏。
```

如果只复制一条单行命令，两种方式通常都可以。

## 1. 本次实验目录

本次 ADM 试跑结果建议放在：

```text
/root/autodl-tmp/runs/exp-dual-branch-dwt/dwt_r50r50_varnorm_tau1_aux02_seed42_steps15000/exclude_ADM
```

目录结构预期是：

```text
runs/
└── exp-dual-branch-dwt/
    └── dwt_r50r50_varnorm_tau1_aux02_seed42_steps15000/
        └── exclude_ADM/
            ├── ckpt/
            ├── tb/
            ├── dwt_stats/
            │   └── exclude_ADM_freq_stats.json
            ├── dwt_stats.log
            ├── smoke.log
            ├── train.log
            └── eval.log
```

这些文件不需要提交 Git。

## 2. 进入 screen

建议先开一个 screen，防止 SSH 断开后训练中断。

```bash
screen -S dwt_adm
```

退出但保持任务后台运行：

```text
Ctrl + A，然后按 D
```

重新进入：

```bash
screen -r dwt_adm
```

如果提示已经 attached，用：

```bash
screen -d -r dwt_adm
```

## 3. 进入项目并同步代码

```bash
cd /root/autodl-tmp/Few-Shot-AIGI-Detector-main
source /etc/network_turbo
git fetch origin
git switch exp-dual-branch-dwt || git switch -c exp-dual-branch-dwt --track origin/exp-dual-branch-dwt
git pull --ff-only
git status
```

正常结果应该包含：

```text
On branch exp-dual-branch-dwt
nothing to commit, working tree clean
```

如果工作区不干净，先停止，不要继续训练。

## 4. 设置实验目录变量

```bash
RUN_ROOT=/root/autodl-tmp/runs/exp-dual-branch-dwt/dwt_r50r50_varnorm_tau1_aux02_seed42_steps15000/exclude_ADM
mkdir -p "$RUN_ROOT"
mkdir -p "$RUN_ROOT/dwt_stats"
echo "$RUN_ROOT"
ls -lh "$RUN_ROOT"
```

如果看到 `dwt_stats` 目录，说明这一步成功。

## 5. DWT stats 是什么

DWT stats 是 DWT 高频图的训练集统计量，也就是：

```text
mean: DWT 高频图每个通道的平均值
std:  DWT 高频图每个通道的标准差
```

原始 RGB 分支看的是 `ToTensor()` 后的图片。

DWT 分支看的是另一种输入：

```text
ToTensor 后的 RGB 图片
  -> DWT 小波分解
  -> 取 LH / HL / HH 高频部分
  -> abs 取强度
  -> 对 RGB 三个颜色通道求平均
  -> log1p 压缩数值
  -> 得到 3 通道 DWT 高频图
```

这个 DWT 高频图的数值范围和普通 RGB 图片不一样，所以需要用训练集统计 mean/std，让输入更稳定：

```python
x_freq = (x_freq - mean) / std
```

注意：

```text
exclude_ADM 实验训练时不使用 ADM。
所以 DWT stats 也必须排除 ADM。
不能用 val/test 统计，否则会数据泄露。
```

## 6. 生成 ADM 对应的 DWT stats

如果还没有 stats 文件，执行：

```bash
python tools/compute_dwt_stats.py --data_root /root/autodl-tmp/data --exclude_class ADM --output "$RUN_ROOT/dwt_stats/exclude_ADM_freq_stats.json" --batch_size 64 --num_workers 8 2>&1 | tee "$RUN_ROOT/dwt_stats.log"
```

生成后检查：

```bash
ls -lh "$RUN_ROOT/dwt_stats/exclude_ADM_freq_stats.json"
```

正常时会看到 JSON 文件存在。

也可以简单看一下内容：

```bash
cat "$RUN_ROOT/dwt_stats/exclude_ADM_freq_stats.json"
```

重点确认：

```text
per_folder_counts 里没有 ADM
split 是 train only
preprocess 是 Resize -> CenterCrop -> ToTensor -> DWT -> abs -> RGB mean over subbands -> log1p
```

## 7. 跑 smoke test

smoke test 不读数据集，不是正式训练。它检查：

```text
DWT 输出 shape
双分支 forward
prototype / distance / alpha
loss backward
RGB encoder 和 frequency encoder 是否都有梯度
baseline 单分支函数是否仍可用
```

执行：

```bash
PYTHONPATH=. python scripts/check_dual_dwt_smoke.py --device cuda 2>&1 | tee "$RUN_ROOT/smoke.log"
```

正常结果：

```text
All DWT dual-branch smoke checks passed.
```

如果报 `No module named 'model'`，说明没有加 `PYTHONPATH=.`，请用上面的完整命令重跑。

## 8. 正式训练 ADM

训练命令如下。复制多行命令时选择 `粘贴(P)`。

```bash
export HF_ENDPOINT=${HF_ENDPOINT:-"https://hf-mirror.com"}
OMP_NUM_THREADS=1 torchrun \
  --nproc_per_node 1 \
  --nnodes 1 \
  train.py \
  --data_root /root/autodl-tmp/data \
  --output_dir "$RUN_ROOT" \
  --num_workers 8 \
  --seed 42 \
  --batch_size 16 \
  --lr 1e-4 \
  --lr_scheduler_gamma 0.5 \
  --lr_scheduler_step 5000 \
  --exclude_class ADM \
  --total_training_steps 15000 \
  --accumulation_steps 1 \
  --save_interval 2500 \
  --eval_interval 2500 \
  --log_interval 200 \
  --num_class_train 3 \
  --num_support_train 5 \
  --num_query_train 5 \
  --num_class_val 2 \
  --num_support_val 5 \
  --num_query_val 15 \
  --use_fp16 True \
  --use_dual_branch \
  --freq_input_type dwt \
  --freq_stats_path "$RUN_ROOT/dwt_stats/exclude_ADM_freq_stats.json" \
  2>&1 | tee "$RUN_ROOT/train.log"
```

这一步会生成：

```text
$RUN_ROOT/train.log
$RUN_ROOT/ckpt/
$RUN_ROOT/tb/
```

如果中途 OOM，不要马上改参数。先保存报错并查看最后 50 行：

```bash
tail -n 50 "$RUN_ROOT/train.log"
```

## 9. 训练完成后测试 ADM

如果训练跑满 15000 step，checkpoint 预期是：

```text
$RUN_ROOT/ckpt/resnet50_step[15000].pth
```

先设置：

```bash
CKPT="$RUN_ROOT/ckpt/resnet50_step[15000].pth"
ls -lh "$CKPT"
```

然后测试：

```bash
python test.py \
  --data_root /root/autodl-tmp/data \
  --output_dir "$RUN_ROOT" \
  --test_class ADM \
  --ckpt_path "$CKPT" \
  --num_workers 8 \
  --seed 42 \
  --num_class_test 2 \
  --num_support_test 5 \
  --num_query_test 15 \
  --use_fp16 True \
  --use_dual_branch \
  --freq_input_type dwt \
  --freq_stats_path "$RUN_ROOT/dwt_stats/exclude_ADM_freq_stats.json" \
  2>&1 | tee "$RUN_ROOT/eval.log"
```

测试日志会保存到：

```text
$RUN_ROOT/eval.log
```

## 10. 如果想让 Cursor 自动执行训练后测试

可以让 Cursor 按本指南执行：

```text
先运行第 8 节训练命令。
如果训练命令正常结束，并且第 9 节的 CKPT 文件存在，再运行第 9 节测试命令。
```

不要让 Cursor 把第 8 节和第 9 节强行压成一行执行。

## 11. 查看训练日志

实时查看：

```bash
tail -f "$RUN_ROOT/train.log"
```

每 5 秒刷新最后 50 行：

```bash
watch -n 5 'tail -n 50 /root/autodl-tmp/runs/exp-dual-branch-dwt/dwt_r50r50_varnorm_tau1_aux02_seed42_steps15000/exclude_ADM/train.log'
```

退出：

```text
Ctrl + C
```

## 12. 查看 GPU / 内存

看 GPU：

```bash
watch -n 2 nvidia-smi
```

看系统内存：

```bash
watch -n 5 free -h
```

退出都是：

```text
Ctrl + C
```

## 13. 开 TensorBoard

建议新开一个终端或 screen：

```bash
screen -S tb_dwt_adm
```

启动 TensorBoard：

```bash
tensorboard --logdir /root/autodl-tmp/runs/exp-dual-branch-dwt/dwt_r50r50_varnorm_tau1_aux02_seed42_steps15000/exclude_ADM/tb --host 0.0.0.0 --port 6006
```

AutoDL 页面里映射 `6006` 端口后打开。

重点看：

```text
train/loss
loss/total
loss/dual
loss/rgb
loss/freq
acc/dual
acc/rgb
acc/freq
alpha/rgb_mean
alpha/rgb_std
alpha/freq_mean
alpha/freq_std
dist/ratio_mean
val_seen/acc_xxx
val_unseen/acc_ADM
```

## 14. 判断是否正常

正常现象：

```text
DWT stats 能生成
smoke test 通过
训练不出现 NaN / Inf
checkpoint 按 2500 step 保存
eval_interval 能输出验证结果
TensorBoard 能看到 loss 和 alpha 曲线
```

需要警惕：

```text
CUDA out of memory
loss 变成 nan
loss/freq 完全不降
alpha/rgb_std 长期接近 0
dist/ratio_mean 长期大于 10 或小于 0.1
eval/test 仍然只走 RGB 分支
```

如果失败，优先把对应日志最后 50 行发出来：

```bash
tail -n 50 "$RUN_ROOT/train.log"
tail -n 50 "$RUN_ROOT/eval.log"
tail -n 50 "$RUN_ROOT/smoke.log"
```

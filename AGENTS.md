# AGENTS.md

## 文档存放规则

本项目的说明文档、实验指南、提示词文档、论文 PDF、方案记录等文档类文件，统一放在仓库根目录的 `docs/` 文件夹下。

包括但不限于：

- 论文 PDF
- 实验实施指南
- Cursor / Codex 提示词文档
- 方案设计文档
- AutoDL 操作文档
- 后续实验记录和说明

不要把文档文件散放在项目根目录。

## Git 规则

`docs/` 文件夹以及其中的文档文件需要同步到 GitHub，并在 AutoDL 上使用。

因此：

- 不要把 `docs/` 加入 `.gitignore`。
- 不要把 `docs/*.md` 加入 `.gitignore`。
- 不要把 `docs/*.pdf` 加入 `.gitignore`。
- 移动或新增文档后，应正常加入 Git 提交。

## 本机与 AutoDL 实验流程

本机没有配置完整训练环境和显卡，因此不要在本机跑训练实验或长时间 GPU 任务。

标准流程是：

- 在本机修改代码，并只做不依赖显卡的轻量检查。
- 将代码提交到 Git，并推送到 GitHub。
- 在 AutoDL 上从 GitHub 拉取最新代码。
- 使用 AutoDL 上的显卡运行训练、评估和实验。

## 实验输出与结果记录规则

所有实验输出统一放在 AutoDL 的 `/root/autodl-tmp/runs/` 下，不放进 Git，也不要放在项目根目录。

路径格式沿用已有实验结果的组织方式：

- 原论文或 baseline 复现结果放在 `/root/autodl-tmp/runs/baseline_fsd_paper/exclude_ADM` 这类目录下。
- 实验分支结果放在 `/root/autodl-tmp/runs/<分支名>/<实验配置名>/exclude_<类别>` 这类目录下。
- 多类别 leave-one-out 任务中，每个类别继续使用 `exclude_ADM`、`exclude_BigGAN`、`exclude_glide`、`exclude_Midjourney`、`exclude_SD`、`exclude_VQDM`。

每次训练任务都必须启用 TensorBoard。若脚本支持 `USE_TENSORBOARD`，默认保持 `USE_TENSORBOARD=True`，不要随意关闭。

每次任务启动 TensorBoard 后，必须给出用户可以直接打开并复制的网址。

- TensorBoard 标准启动命令为：`tensorboard --logdir <本次任务tb目录或总目录> --host 0.0.0.0 --port 6006`。
- 如果 AutoDL 或 Cursor 提供端口转发链接，必须把实际可打开的链接写出来。
- 如果暂时拿不到链接，必须明确告诉用户去 AutoDL 端口转发或服务列表复制 `6006` 的访问地址，不能只说“打开 TensorBoard”。

每次任务结束后，都必须整理一张结果对比表，比较：

- 原论文报告数据
- 当前复现 baseline 数据
- 当前实验数据

对比表放在本次任务对应的 `runs/` 目录下，例如：

- `/root/autodl-tmp/runs/<任务目录>/paper_vs_reproduce_comparison.md`
- 如需要机器可读格式，同时保存 `/root/autodl-tmp/runs/<任务目录>/paper_vs_reproduce_comparison.csv`

对比表至少包含：方法/分支、训练设置、测试类别、Accuracy、AP、checkpoint、日志目录、备注。


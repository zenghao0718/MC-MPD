# AutoDL 上构建 FSD 全量 GenImage 数据集：Codex 实施文档

## 0. 任务定位

本任务只新增数据准备工具脚本，不做训练、不做测试、不改模型、不改现有训练逻辑。

当前仓库：

```bash
/root/autodl-tmp/Few-Shot-AIGI-Detector-main
```

当前目标分支应为：

```bash
exp-ddfsd-dual-domain-margin-v1
```

已有原始 GenImage 压缩包目录：

```bash
/root/autodl-tmp/GenImage
```

旧的 1/10 数据集目录：

```bash
/root/autodl-tmp/data
```

`/root/autodl-tmp/data` 是旧实验数据，禁止覆盖、删除、移动、清空或修改。

新的完整 FSD 格式数据集输出目录：

```bash
/root/autodl-tmp/data_fsd_full/GenImage
```

后续训练时应使用：

```bash
--data_root /root/autodl-tmp/data_fsd_full/GenImage
```

注意：`--data_root` 要指向直接包含 `real/ADM/BigGAN/glide/Midjourney/SD/VQDM` 七个类目录的那一层。

---

## 1. 实施前检查要求

开始写代码前先做以下检查：

```bash
cd /root/autodl-tmp/Few-Shot-AIGI-Detector-main
git branch --show-current
git status --short
```

要求：

1. 当前分支应为 `exp-ddfsd-dual-domain-margin-v1`。
2. 不要新建其他仓库目录。
3. 只在当前仓库内新增工具脚本和必要文档。
4. 不要修改模型、训练、测试主流程代码，除非只是为了读取已有代码确认数据路径逻辑。
5. 如仓库有 `AGENTS.md` 或类似项目说明，先阅读并遵守。

---

## 2. 最终需要新增的脚本

在仓库 `tools/` 下新增 4 个脚本：

```text
tools/extract_genimage_archives.py
tools/check_raw_genimage_for_fsd.py
tools/build_fsd_genimage_layout.py
tools/check_fsd_genimage_layout.py
```

这 4 个脚本分别负责：

1. 自动顺序解压 GenImage 压缩包。
2. 检查解压后的原始 GenImage 目录。
3. 用 hardlink 构建 FSD 格式的 7 类数据集。
4. 检查最终 FSD 数据集是否正确。

禁止新增训练脚本，禁止启动训练，禁止启动测试。

---

## 3. 最终目标数据集结构

构建完成后应得到：

```text
/root/autodl-tmp/data_fsd_full/
└── GenImage/
    ├── real/
    │   ├── train/
    │   │   └── nature/
    │   └── val/
    │       └── nature/
    ├── ADM/
    │   ├── train/
    │   │   └── ai/
    │   └── val/
    │       └── ai/
    ├── BigGAN/
    │   ├── train/
    │   │   └── ai/
    │   └── val/
    │       └── ai/
    ├── glide/
    │   ├── train/
    │   │   └── ai/
    │   └── val/
    │       └── ai/
    ├── Midjourney/
    │   ├── train/
    │   │   └── ai/
    │   └── val/
    │       └── ai/
    ├── SD/
    │   ├── train/
    │   │   └── ai/
    │   └── val/
    │       └── ai/
    └── VQDM/
        ├── train/
        │   └── ai/
        └── val/
            └── ai/
```

最终类别固定为：

```text
real
ADM
BigGAN
glide
Midjourney
SD
VQDM
```

大小写必须保持一致：

```text
real  小写
glide 小写
SD    大写
```

---

## 4. 数据来源规则

### 4.1 real 类

用户已确定：

```text
real 类 = SD1.4 nature + SD1.5 nature
```

也就是：

```text
real/train/nature =
  stable_diffusion_v_1_4/train/nature
+ stable_diffusion_v_1_5/train/nature

real/val/nature =
  stable_diffusion_v_1_4/val/nature
+ stable_diffusion_v_1_5/val/nature
```

禁止使用其他来源的 `nature` 构造 real 类。

### 4.2 SD 假类

用户已确定：

```text
SD 假类 = SD1.4 ai + SD1.5 ai + Wukong ai
```

也就是：

```text
SD/train/ai =
  stable_diffusion_v_1_4/train/ai
+ stable_diffusion_v_1_5/train/ai
+ wukong/train/ai

SD/val/ai =
  stable_diffusion_v_1_4/val/ai
+ stable_diffusion_v_1_5/val/ai
+ wukong/val/ai
```

### 4.3 其他 fake 类

其他假类只使用各自来源的 `ai`：

```text
ADM/train/ai         ← ADM/train/ai
ADM/val/ai           ← ADM/val/ai

BigGAN/train/ai      ← BigGAN/train/ai
BigGAN/val/ai        ← BigGAN/val/ai

glide/train/ai       ← glide 或 GLIDE 的 train/ai
glide/val/ai         ← glide 或 GLIDE 的 val/ai

Midjourney/train/ai  ← Midjourney/train/ai
Midjourney/val/ai    ← Midjourney/val/ai

VQDM/train/ai        ← VQDM/train/ai
VQDM/val/ai          ← VQDM/val/ai
```

禁止把这些类别的 `nature` 放入最终 FSD 数据集。

---

## 5. 文件链接方式

最终构建方式固定为：

```text
hardlink
```

含义：最终 FSD 数据集中的文件和原始 GenImage 解压文件指向同一份磁盘数据。

要求：

1. 默认使用 hardlink。
2. hardlink 失败时直接报错。
3. 不允许自动退回 copy。
4. 不允许静默复制大文件。
5. 不允许 symlink 作为默认行为。
6. 不删除、不移动、不修改原始 GenImage 文件。
7. 不覆盖 `/root/autodl-tmp/data`。

在 Linux 中使用 `os.link(src, dst)` 创建 hardlink。

---

## 6. 文件命名策略

合并多个来源到同一个目标目录时，统一加来源前缀，避免重名：

```text
目标文件名 = source_tag__原文件名
```

建议 source tag：

```text
stable_diffusion_v_1_4 → sdv14
stable_diffusion_v_1_5 → sdv15
wukong                 → wukong
ADM                    → adm
BigGAN                 → biggan
glide / GLIDE          → glide
Midjourney             → midjourney
VQDM                   → vqdm
```

示例：

```text
stable_diffusion_v_1_4/train/ai/000_sdv4_00000.png
→ SD/train/ai/sdv14__000_sdv4_00000.png

stable_diffusion_v_1_5/train/ai/000_sdv5_00000.png
→ SD/train/ai/sdv15__000_sdv5_00000.png

wukong/train/ai/0_wukong_image0.png
→ SD/train/ai/wukong__0_wukong_image0.png

stable_diffusion_v_1_4/train/nature/xxx.png
→ real/train/nature/sdv14__xxx.png

stable_diffusion_v_1_5/train/nature/xxx.png
→ real/train/nature/sdv15__xxx.png
```

如果加前缀后仍然发现目标文件重名，说明数据或路径异常，脚本必须报错停止，禁止覆盖。

图片后缀支持：

```text
.jpg
.jpeg
.png
.bmp
.webp
```

匹配时大小写不敏感。

---

# 7. 脚本一：extract_genimage_archives.py

## 7.1 作用

自动顺序解压 `/root/autodl-tmp/GenImage` 下的 GenImage 压缩包。

此脚本只负责解压，不负责构建 FSD 数据集。

## 7.2 默认命令

```bash
python tools/extract_genimage_archives.py \
  --root /root/autodl-tmp/GenImage
```

## 7.3 参数建议

至少支持：

```text
--root              原始 GenImage 压缩包根目录，默认 /root/autodl-tmp/GenImage
--log_dir           日志目录，默认 {root}/_extract_logs
--marker_dir        完成标记目录，默认 {root}/_extract_done
--install_if_missing 如果 7zz/7z/7za 都不存在，尝试安装
--dry_run           只打印计划，不实际解压
--force             忽略完成标记，重新尝试解压；默认关闭
```

`--install_if_missing` 可以默认开启，也可以默认关闭但在文档中建议开启。更推荐默认开启，但必须打印将执行的安装命令。

## 7.4 解压工具优先级

优先使用命令行工具：

```text
7zz > 7z > 7za
```

如果都没有，先尝试安装：

```bash
apt-get update
apt-get install -y 7zip p7zip-full unzip
```

如果 apt 失败，再尝试：

```bash
conda install -c conda-forge p7zip unzip -y
```

安装后重新检测：

```bash
command -v 7zz
command -v 7z
command -v 7za
```

如果仍然找不到 7z 系工具，脚本停止并提示用户手动安装，不要退回 Python zipfile 慢速解压。

## 7.5 解压命令形式

使用：

```bash
7z x xxx.zip -o输出目录 -mmt=on
```

注意：

1. 使用 `x`，保留压缩包内部目录结构。
2. 使用 `-o` 指定输出目录。
3. 使用 `-mmt=on` 启用多线程。
4. 每次只解压一个 `.zip`。
5. 一个 `.zip` 成功完成后，再解下一个。

## 7.6 分卷 zip 处理规则

GenImage 常见分卷格式：

```text
imagenet_ai_0508_adm.z01
imagenet_ai_0508_adm.z02
...
imagenet_ai_0508_adm.zip
```

脚本必须：

1. 只对 `.zip` 主文件发起解压。
2. 跳过 `.z01/.z02/.z03/...`。
3. 跳过 `.tmp` 文件。
4. 如果同目录中存在 `.tmp` 文件，应认为下载可能未完成。默认报错或跳过该目录，不要强行解压。
5. 不要把 `.z01/.z02` 当成独立压缩包。

## 7.7 断点续跑

每个 `.zip` 成功解压后，在 marker 目录写完成标记。

标记名可以用 zip 文件绝对路径的哈希，例如：

```text
_extract_done/<sha1>.done
```

标记内容建议包含：

```text
zip_path
output_dir
tool
return_code
finish_time
```

下次运行时，如果对应 `.done` 存在，默认跳过该 zip。

## 7.8 安全要求

1. 不删除压缩包。
2. 不移动压缩包。
3. 不覆盖已有完成标记对应的解压结果。
4. 不清空任何目录。
5. 不并行解压多个 zip。
6. 解压 stdout/stderr 写入日志。

---

# 8. 脚本二：check_raw_genimage_for_fsd.py

## 8.1 作用

检查 `/root/autodl-tmp/GenImage` 是否已经解压到可构建 FSD 数据集的状态。

此脚本只读，不修改任何文件。

## 8.2 默认命令

```bash
python tools/check_raw_genimage_for_fsd.py \
  --root /root/autodl-tmp/GenImage
```

## 8.3 参数建议

至少支持：

```text
--root              原始 GenImage 根目录，默认 /root/autodl-tmp/GenImage
--json_out          可选，输出 JSON 统计报告
--max_bad_examples  最多打印多少个异常路径，默认 20
```

## 8.4 需要检查的来源

必须检查：

```text
ADM
BigGAN
glide 或 GLIDE
Midjourney
stable_diffusion_v_1_4
stable_diffusion_v_1_5
wukong
VQDM
```

注意：

1. `glide` 可能是小写，也可能是 `GLIDE`，脚本要兼容。
2. 如果出现多个候选目录，要报出实际使用哪个目录。
3. 不能凭空创建缺失目录。

## 8.5 检查内容

对每个来源统计：

```text
train/ai
val/ai
train/nature
val/nature
```

输出：

1. 来源目录是否存在。
2. split 目录是否存在。
3. ai/nature 子目录是否存在。
4. 图片数量。
5. 是否有 `.tmp` 未完成下载文件。
6. 是否存在空目录。
7. 是否存在 zip/z01 仍未解压的情况。
8. 可选：抽样检查少量图片能否用 PIL 打开。

注意：即使 ADM、BigGAN、glide、Midjourney、VQDM 的 `nature` 不参与最终构建，也可以统计出来，但不要把它们写入最终数据集。

---

# 9. 脚本三：build_fsd_genimage_layout.py

## 9.1 作用

把解压后的原始 GenImage 目录 hardlink 成 FSD 需要的 7 类目录结构。

## 9.2 默认命令

```bash
python tools/build_fsd_genimage_layout.py \
  --raw_root /root/autodl-tmp/GenImage \
  --out_root /root/autodl-tmp/data_fsd_full/GenImage \
  --link_mode hardlink \
  --name_policy source_prefix
```

## 9.3 参数建议

至少支持：

```text
--raw_root          原始 GenImage 解压根目录，默认 /root/autodl-tmp/GenImage
--out_root          输出 FSD 数据集根目录，默认 /root/autodl-tmp/data_fsd_full/GenImage
--link_mode         只要求支持 hardlink；如实现枚举，只允许 hardlink
--name_policy       默认 source_prefix
--dry_run           只打印计划，不创建链接
--resume            允许在已有 out_root 上继续构建，但不能覆盖已有文件
--json_out          可选，输出构建统计
```

不建议实现 `--overwrite`。如果实现，默认必须关闭，并且必须要求用户明确输入类似 `--overwrite --i_know_this_may_delete_outputs` 才能启用。但本任务不需要实现 overwrite。

## 9.4 输出目录保护

如果 `out_root` 已存在：

1. 默认报错停止。
2. 如果传入 `--resume`，允许继续，但不能覆盖已有文件。
3. 发现目标文件已存在且不是同一个 hardlink，必须报错。
4. 禁止删除 `out_root`。
5. 禁止动 `/root/autodl-tmp/data`。

## 9.5 hardlink 创建逻辑

使用：

```python
os.link(src, dst)
```

要求：

1. 创建前确保父目录存在。
2. 如果 dst 已存在，检查是否已是同一文件。
3. 如果不是同一文件，报错。
4. hardlink 失败时报错并输出源路径、目标路径、错误原因。
5. 不自动复制。

判断是否同一文件可用：

```python
os.stat(src).st_ino == os.stat(dst).st_ino
```

同时最好也比较 `st_dev`。

## 9.6 构建映射

### real

```text
stable_diffusion_v_1_4/train/nature → real/train/nature，前缀 sdv14__
stable_diffusion_v_1_5/train/nature → real/train/nature，前缀 sdv15__

stable_diffusion_v_1_4/val/nature → real/val/nature，前缀 sdv14__
stable_diffusion_v_1_5/val/nature → real/val/nature，前缀 sdv15__
```

### SD

```text
stable_diffusion_v_1_4/train/ai → SD/train/ai，前缀 sdv14__
stable_diffusion_v_1_5/train/ai → SD/train/ai，前缀 sdv15__
wukong/train/ai                 → SD/train/ai，前缀 wukong__

stable_diffusion_v_1_4/val/ai → SD/val/ai，前缀 sdv14__
stable_diffusion_v_1_5/val/ai → SD/val/ai，前缀 sdv15__
wukong/val/ai                 → SD/val/ai，前缀 wukong__
```

### ADM

```text
ADM/train/ai → ADM/train/ai，前缀 adm__
ADM/val/ai   → ADM/val/ai，前缀 adm__
```

### BigGAN

```text
BigGAN/train/ai → BigGAN/train/ai，前缀 biggan__
BigGAN/val/ai   → BigGAN/val/ai，前缀 biggan__
```

### glide

```text
glide 或 GLIDE/train/ai → glide/train/ai，前缀 glide__
glide 或 GLIDE/val/ai   → glide/val/ai，前缀 glide__
```

最终目标目录名必须是小写 `glide`。

### Midjourney

```text
Midjourney/train/ai → Midjourney/train/ai，前缀 midjourney__
Midjourney/val/ai   → Midjourney/val/ai，前缀 midjourney__
```

### VQDM

```text
VQDM/train/ai → VQDM/train/ai，前缀 vqdm__
VQDM/val/ai   → VQDM/val/ai，前缀 vqdm__
```

## 9.7 构建完成输出

控制台打印总表：

```text
class        train_count        val_count        sources
real         xxx                xxx              sdv14 nature + sdv15 nature
ADM          xxx                xxx              ADM ai
BigGAN       xxx                xxx              BigGAN ai
glide        xxx                xxx              glide ai
Midjourney   xxx                xxx              Midjourney ai
SD           xxx                xxx              sdv14 ai + sdv15 ai + wukong ai
VQDM         xxx                xxx              VQDM ai
```

如提供 `--json_out`，保存同样信息为 JSON。

---

# 10. 脚本四：check_fsd_genimage_layout.py

## 10.1 作用

检查最终 FSD 数据集是否能被当前代码读取。

## 10.2 默认命令

```bash
python tools/check_fsd_genimage_layout.py \
  --root /root/autodl-tmp/data_fsd_full/GenImage
```

## 10.3 参数建议

至少支持：

```text
--root              FSD 数据集根目录，默认 /root/autodl-tmp/data_fsd_full/GenImage
--json_out          可选，输出 JSON 检查报告
--sample_open       每个目录抽样打开多少张图片，默认 5
```

## 10.4 检查内容

必须检查：

1. 是否存在 7 个类别目录：

```text
real
ADM
BigGAN
glide
Midjourney
SD
VQDM
```

2. fake 类是否存在：

```text
train/ai
val/ai
```

3. real 类是否存在：

```text
train/nature
val/nature
```

4. 每个目录图片数量是否大于 0。
5. 每个类别 train/val 数量统计。
6. 是否存在坏链接。
7. 是否存在非图片文件混入。
8. 抽样用 PIL 打开图片，确认不是坏图。
9. 可选：确认文件名中包含 source tag 前缀。

输出总表：

```text
class        train_count        val_count        status
real         xxx                xxx              OK
ADM          xxx                xxx              OK
BigGAN       xxx                xxx              OK
glide        xxx                xxx              OK
Midjourney   xxx                xxx              OK
SD           xxx                xxx              OK
VQDM         xxx                xxx              OK
```

如果有任何缺失或坏图，返回非 0 exit code。

---

## 11. 推荐执行顺序

从仓库根目录执行：

```bash
cd /root/autodl-tmp/Few-Shot-AIGI-Detector-main
```

第一步：自动解压。

```bash
python tools/extract_genimage_archives.py \
  --root /root/autodl-tmp/GenImage
```

第二步：检查原始 GenImage。

```bash
python tools/check_raw_genimage_for_fsd.py \
  --root /root/autodl-tmp/GenImage
```

第三步：构建 FSD 全量数据集。

```bash
python tools/build_fsd_genimage_layout.py \
  --raw_root /root/autodl-tmp/GenImage \
  --out_root /root/autodl-tmp/data_fsd_full/GenImage \
  --link_mode hardlink \
  --name_policy source_prefix
```

第四步：检查最终数据集。

```bash
python tools/check_fsd_genimage_layout.py \
  --root /root/autodl-tmp/data_fsd_full/GenImage
```

后续训练时显式使用：

```bash
--data_root /root/autodl-tmp/data_fsd_full/GenImage
```

不要默认读取旧的：

```bash
/root/autodl-tmp/data
```

---

## 12. 明确禁止事项

本任务禁止：

1. 不要下载 GenImage。
2. 不要删除原始压缩包。
3. 不要删除原始解压目录。
4. 不要移动原始 GenImage。
5. 不要覆盖 `/root/autodl-tmp/data`。
6. 不要修改旧 1/10 数据集。
7. 不要启动训练。
8. 不要启动测试。
9. 不要改模型代码。
10. 不要把 hardlink 失败自动换成 copy。
11. 不要静默覆盖目标文件。
12. 不要把 `.z01/.z02` 当成独立压缩包解压。
13. 不要在 `.tmp` 未完成下载时强行解压。
14. 不要并行解压多个 zip。
15. 不要在仓库外新建第二份代码仓库。

---

## 13. 验收标准

完成后请给出：

1. 新增/修改文件列表。
2. 每个脚本的用途说明。
3. 每个脚本的命令行参数说明。
4. 是否已通过 `python -m py_compile tools/*.py`。
5. 是否做过 `--dry_run`。
6. 如果实际运行了检查脚本，请给出检查结果表。
7. 明确说明没有修改 `/root/autodl-tmp/data`。
8. 明确说明没有启动训练或测试。
9. 明确说明 hardlink 构建目标目录为：

```bash
/root/autodl-tmp/data_fsd_full/GenImage
```

---

## 14. 最终一句话

请在 AutoDL/Linux 上为当前 `exp-ddfsd-dual-domain-margin-v1` 仓库新增 4 个工具脚本：先用 `7zz/7z/7za` 多线程顺序解压 `/root/autodl-tmp/GenImage` 下的 GenImage 分卷 zip；再检查原始数据；然后用 `hardlink` 构建新的完整 FSD 格式数据集到 `/root/autodl-tmp/data_fsd_full/GenImage`；最后检查最终 7 类目录。旧的 `/root/autodl-tmp/data` 是 1/10 数据集，不允许动。real 类只用 SD1.4 和 SD1.5 的 nature；SD 假类用 SD1.4、SD1.5、Wukong 的 ai；其他假类只用各自 ai；合并文件统一加 `source_tag__` 前缀；后续训练通过 `--data_root /root/autodl-tmp/data_fsd_full/GenImage` 使用新数据集。

---

## 15. 本次落地后的实际使用命令

在 AutoDL/Linux 上从仓库根目录依次执行：

```bash
python tools/extract_genimage_archives.py --root /root/autodl-tmp/GenImage
python tools/check_raw_genimage_for_fsd.py --root /root/autodl-tmp/GenImage
python tools/build_fsd_genimage_layout.py --raw_root /root/autodl-tmp/GenImage --out_root /root/autodl-tmp/data_fsd_full/GenImage --link_mode hardlink --name_policy source_prefix
python tools/check_fsd_genimage_layout.py --root /root/autodl-tmp/data_fsd_full/GenImage
```

解压脚本是批处理脚本：同一条命令会递归扫描 `--root` 下所有完整 `.zip` 主文件，跳过 `.z01/.z02/.tmp`，成功项写入 done 标记；中断后再次运行同一条命令会跳过已完成项并继续剩余 zip。

本机阶段只做语法检查和小型临时目录验证，不解压真实数据、不构建真实 GenImage、不启动训练或测试。

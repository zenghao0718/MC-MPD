# Dual-Domain Few-Shot AIGI Detector

This repository contains the final full-data, 15,000-step main-experiment pipeline for Dual-Domain Few-Shot AIGI Detection (DDFSD). The method combines spatial RGB evidence and frequency-domain artifacts in a prototype-based detector for leave-one-generator-out evaluation on GenImage.

## Method

DDFSD uses an ImageNet-pretrained ResNet-50 RGB encoder and a ResNet-18 frequency encoder. The frequency input consists of the signed LH, HL, and HH bands from a Haar DWT of the image Y channel. Both encoders project to normalized embeddings, from which episode prototypes are built. Adaptive alpha weights fuse RGB and frequency distances according to support-set dispersion, while a prototype separation loss encourages real/fake and fake/fake margins. Main training uses branch dropout but always evaluates the full dual-branch model.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Training and evaluation require a CUDA-capable PyTorch environment. The scripts use a single process and one GPU.

## Data layout

`DATA_ROOT` must point directly to the directory containing these seven class directories:

```text
GenImage/
├── real/
│   ├── train/
│   └── val/
├── ADM/
│   ├── train/
│   └── val/
├── BigGAN/
│   ├── train/
│   └── val/
├── glide/
│   ├── train/
│   └── val/
├── Midjourney/
│   ├── train/
│   └── val/
├── SD/
│   ├── train/
│   └── val/
└── VQDM/
    ├── train/
    └── val/
```

The repository intentionally does not include dataset construction or extraction utilities.

## Main experiment

Train one leave-one-out task:

```bash
DATA_ROOT=/root/autodl-tmp/data_fsd_full/GenImage \
EXCLUDE_CLASS=ADM \
bash scripts/train_main.sh
```

Evaluate its step-15000 checkpoint:

```bash
DATA_ROOT=/root/autodl-tmp/data_fsd_full/GenImage \
EXCLUDE_CLASS=ADM \
bash scripts/eval_main.sh
```

Run all six tasks in order (`ADM BigGAN glide Midjourney SD VQDM`):

```bash
MODE=all bash scripts/run_main_experiment.sh
```

Use `MODE=train` or `MODE=eval` to run only that phase. A class failure stops the overall driver and reports its exit code.

### Fixed configuration

| Item | Main setting |
| --- | --- |
| Data | Full GenImage |
| Protocol | Six-class leave-one-out |
| Training | Single GPU, 15,000 steps, batch size 16 |
| Episode | 3-way (real + 2 fake), 5 support and 5 query per class |
| Scheduler | StepLR, step size 5,000, gamma 0.5 |
| Learning rates | RGB/frequency backbones `3e-5`; heads `1e-4` |
| Margins | `m_rf=1.2`, `m_ff=0.6` |
| Separation | `lambda_ff=0.5`, target `0.03`, warmup 2,500–7,500 |
| Branch dropout | dual 0.90, RGB-only 0.05, frequency-only 0.05 |
| Evaluation | step 15,000; 10-shot; seeds 42, 101, 102, 103, 104 |
| Inference | dual branch, adaptive alpha, complete validation query set |
| Metrics | Accuracy, AP, AUC |

## Checkpoints and outputs

The default run root is:

```text
/root/autodl-tmp/runs/Dual-Domain-Few-Shot-AIGI-Detector/main_full_steps15000/
└── exclude_<class>/
    ├── ckpt/
    ├── tb/
    ├── logs/
    ├── train.log
    ├── freq_stats.pt
    └── formal_eval_step15000/
        ├── ddfsd_eval_per_seed.csv
        ├── ddfsd_eval_summary.csv
        ├── config.json
        └── eval.log
```

`eval_main.sh` loads `ckpt/ddfsd_step[15000].pth` and the matching `freq_stats.pt`. It refuses to recompute frequency statistics during evaluation. Override `RUN_ROOT`, `RUN_DIR`, `CKPT_PATH`, or `FREQ_STATS_PATH` when files are stored elsewhere. Training refuses to reuse an output directory that already contains checkpoints.

Pretrained checkpoint download: [Baidu Netdisk](https://pan.baidu.com/s/1fXonbCSiWqMYksSyCAoN0A?pwd=vwq6).

TensorBoard events are written below each class run's `tb/` directory. For example:

```bash
tensorboard --logdir /root/autodl-tmp/runs/Dual-Domain-Few-Shot-AIGI-Detector/main_full_steps15000 --host 0.0.0.0 --port 6006
```

## Citation

```bibtex
@article{ddfsd,
  title   = {Dual-Domain Few-Shot AIGI Detector},
  author  = {Anonymous},
  journal = {To appear},
  year    = {2026}
}
```

## Acknowledgements

We thank the authors of the FSD paper and project, on which this work builds.

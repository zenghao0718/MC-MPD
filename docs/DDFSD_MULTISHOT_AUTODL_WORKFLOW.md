# DDFSD multi-shot AutoDL workflow

The batch evaluator keeps the formal resources separate:

- checkpoints: `ddfsd_full_main_reusable_ckpts_step15000`
- frequency statistics: `shared_full_freq_stats`
- existing evaluations and new `exclude_<class>/multishot` outputs:
  `ddfsd_full_main_reusable_eval_step15000`
- dataset: `/root/autodl-tmp/data_fsd_full/GenImage`

Run long evaluations in `screen`. The scripts evaluate only; they do not train models.

## ADM smoke test

```bash
cd /root/autodl-tmp/Few-Shot-AIGI-Detector-main
git checkout exp-ddfsd-dual-domain-margin-v1
git pull --ff-only origin exp-ddfsd-dual-domain-margin-v1
screen -S ddfsd_multishot_ADM_smoke

CLASSES="ADM" \
AGGREGATE=0 \
MAX_EVAL_QUERY_PER_CLASS=128 \
DATA_ROOT="/root/autodl-tmp/data_fsd_full/GenImage" \
CKPT_ROOT="/root/autodl-tmp/runs/exp-ddfsd-dual-domain-margin-v1/ddfsd_full_main_reusable_ckpts_step15000" \
FREQ_STATS_ROOT="/root/autodl-tmp/runs/exp-ddfsd-dual-domain-margin-v1/shared_full_freq_stats" \
OUTPUT_ROOT="/root/autodl-tmp/runs/exp-ddfsd-dual-domain-margin-v1/ddfsd_full_main_reusable_eval_step15000" \
CKPT_STEP=15000 \
bash scripts/run_ddfsd_multishot_all.sh
```

## Six-class formal evaluation

```bash
screen -S ddfsd_multishot_all6_eval

CLASSES="ADM BigGAN glide Midjourney SD VQDM" \
AGGREGATE=auto \
MAX_EVAL_QUERY_PER_CLASS=0 \
SKIP_COMPLETED=1 \
DATA_ROOT="/root/autodl-tmp/data_fsd_full/GenImage" \
CKPT_ROOT="/root/autodl-tmp/runs/exp-ddfsd-dual-domain-margin-v1/ddfsd_full_main_reusable_ckpts_step15000" \
FREQ_STATS_ROOT="/root/autodl-tmp/runs/exp-ddfsd-dual-domain-margin-v1/shared_full_freq_stats" \
OUTPUT_ROOT="/root/autodl-tmp/runs/exp-ddfsd-dual-domain-margin-v1/ddfsd_full_main_reusable_eval_step15000" \
CKPT_STEP=15000 \
bash scripts/run_ddfsd_multishot_all.sh
```

Detach with `Ctrl+A`, then `D`; resume with `screen -r <session-name>`.

If the internal layout differs, override templates without changing Python code:

```bash
OUTPUT_ROOT='/absolute/eval' \
CKPT_PATH_TEMPLATE='/absolute/checkpoints/exclude_{class}/ckpt/ddfsd_step[{step}].pth' \
FREQ_STATS_PATH_TEMPLATE='/absolute/stats/exclude_{class}/freq_stats.pt' \
MULTISHOT_OUTPUT_TEMPLATE='/absolute/eval/exclude_{class}/multishot' \
bash scripts/run_ddfsd_multishot_all.sh
```

Complete categories are skipped by default. An incomplete directory stops the run. Use a new
output template, or set `FORCE_RERUN=1`; forced cleanup is limited to the resolved class-specific
directory beneath `OUTPUT_ROOT`.

# Few-Shot AI-Generated Image Detector

Official Pytorch implementation of paper:

> [Few-Shot Learner Generalizes Across AI-Generated Image Detection](https://arxiv.org/abs/2501.08763)
>
> Shiyu Wu, Jing Liu, Jing Li, Yequan Wang

Novel AI-generated image detector which is able to effectively distinguish unseen fake images by utilizing very few new samples. 

## Requirements

You can setup the environment as follows:

```python
# create conda environment
conda create -n FSD -y python=3.12
conda activate FSD

# install dependencies
pip install -r requirements.txt
```

## Getting Data & Directory structure

To download [GenImage](https://arxiv.org/abs/2306.08571) dataset, please refer to [this repository](https://github.com/GenImage-Dataset/GenImage) or download from [Baidu Yunpan](https://pan.baidu.com/share/init?surl=i0OFqYN5i6oFAxeK6bIwRQ) with code ztf1. 

<details>
<summary> Please organize the above data as follows: </summary>

```
data/
|-- GenImage/
|   |-- ADM
|   |   |--train/ai/
|   |   |   |--0_adm_0.PNG
|   |   |   |......
|   |   |--val/ai/
|   |   |   |--0_adm_7.PNG
|   |   |   |......
|   |-- BigGAN
|   |-- glide
|   |-- Midjourney
|   |-- SD
|   |-- VQDM
|   |-- real
|   |   |--train/nature/
|   |   |   |......
```

Real data are those nature images from stable_diffusion_v_1_4 and stable_diffusion_v_1_5. 
</details>


## Training

```
bash scripts/train.sh
```

This script enables training with 4 GPUs, you can specify the number of GPUs by setting `GPU_NUM`.

### DWT dual-branch experiment

The DWT dual-branch path is disabled by default. Before enabling it, compute
train-split frequency statistics on the target dataset:

```
python tools/compute_dwt_stats.py \
    --data_root /root/autodl-tmp/data \
    --exclude_class ADM \
    --output /root/autodl-tmp/dwt_stats/exclude_ADM_freq_stats.json
```

Then add these options to the usual training/evaluation command:

```
--use_dual_branch \
--freq_input_type dwt \
--freq_stats_path /root/autodl-tmp/dwt_stats/exclude_ADM_freq_stats.json
```

The RGB branch keeps the original baseline `ToTensor()` input. DWT is computed
from the same tensor before the model forward pass.

## Inference

```
bash scripts/eval.sh
```

Please specify the checkpoint directroy in the script. 

## Checkpoints
We provide our checkpoints trained on each test part for our cross-generator evaluation at [Baidu Yunpan](https://pan.baidu.com/s/1zNxDKtFJ_5KXcMceNtrRqA?pwd=icml) with code icml. 

## Citing
If you find this repository useful for your work, please consider citing it as follows:
```
@article{wu2025fsd,
  title={Few-Shot Learner Generalizes Across AI-Generated Image Detection},
  author={Shiyu Wu and Jing Liu and Jing Li and Yequan Wang},
  eprint={2501.08763},
  year={2025},
  journal={arXiv preprint arXiv:2501.08763},
  url={https://arxiv.org/abs/2501.08763}
}
```

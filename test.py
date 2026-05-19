# -*- coding: utf-8 -*-
""" A simple test script. 
"""

import os
import torch
from tqdm import tqdm
from torch.amp import autocast
import timm
from einops import rearrange
from torchmetrics.classification import Accuracy, AveragePrecision

from model.dual_branch_fsd import DualBranchFSD
from model.prototypical_utils import compute_dual_branch_logits_and_losses, compute_prototypical_loss
from datasets import setup_val_dataloader
from util.frequency import haar_dwt_highfreq_rgb_mean, load_dwt_stats
from util.parser import TestParser
from util.utils import load_model
import util.logger as logger


def validate_dual_branch_args(args):
    if not args.use_dual_branch:
        return
    if args.freq_input_type != "dwt":
        raise NotImplementedError("First DWT dual-branch version only implements freq_input_type='dwt'.")
    if args.reliability_norm_mode != "episode_mean":
        raise NotImplementedError("First DWT dual-branch version only supports reliability_norm_mode='episode_mean'.")


def build_frequency_batch(batch_data, args, freq_mean, freq_std):
    return haar_dwt_highfreq_rgb_mean(
        batch_data,
        use_abs=args.dwt_use_abs,
        use_log1p=args.dwt_use_log1p,
        resize_to=224,
        mean=freq_mean,
        std=freq_std,
        input_scale=args.dwt_input_scale,
        eps=args.freq_norm_eps,
    )


# no ddp setting
def main(): 
    #################### prepare device ####################
    args = TestParser().args
    validate_dual_branch_args(args)
    args.device = torch.device("cuda")
    # terminal writer and file writer
    logger.setup(log_dir=args.output_dir, device=None)
    ############################################################


    #################### setup dataset and dataloader ####################
    logger.info("Creating test data loader...")

    # data we use in GenImage
    if args.test_class.upper() == "NONE": # test on all classes
        TEST_FOLDERS = ["real", "Midjourney", "SD", "ADM", "glide", "VQDM", "BigGAN"]
    else: # test on single class
        TEST_FOLDERS = ["real", args.test_class]
    
    test_dataloaders = {
        folder: setup_val_dataloader(
            folder_path=os.path.join(args.data_root, folder, "val"), 
            batch_size=args.num_support_test + args.num_query_test, 
            num_workers=args.num_workers, 
        ) for folder in TEST_FOLDERS
    }
    ############################################################

    
    #################### setup model ####################
    freq_mean = None
    freq_std = None
    if args.use_dual_branch:
        logger.info("Creating DWT dual-branch model 'resnet50 + resnet50'... ")
        freq_mean, freq_std = load_dwt_stats(args.freq_stats_path, device=args.device, dtype=torch.float32)
        model = DualBranchFSD()
    else:
        logger.info("Creating model 'resnet50'... ")
        model = timm.create_model("resnet50", pretrained=True, num_classes=1024)
    load_model(args.ckpt_path, model=model)

    # deployment
    model = model.to(args.device)
    ############################################################
    

    #################### testing ####################
    logger.info(f"Start testing with checkpoint {args.ckpt_path}. ")
    model.eval()

    acc_calculator = Accuracy(task="multiclass", num_classes=2)
    ap_calculator = AveragePrecision(task="multiclass", num_classes=2, thresholds=10)

    with torch.no_grad(): 
        for i in range(1, len(TEST_FOLDERS)): 
            # reset so per-class metrics are not contaminated by earlier classes
            acc_calculator.reset()
            ap_calculator.reset()

            prob_list = []
            label_list = []
            for (neg_batch, _), (pos_batch, _) in tqdm(zip(test_dataloaders["real"], test_dataloaders[TEST_FOLDERS[i]])): 
                batch_data = torch.stack([neg_batch, pos_batch], dim=0) # (2, task_size, c, h, w)
                batch_data = batch_data.to(args.device)

                labels = torch.arange(0, args.num_class_test, device=args.device).repeat(args.num_query_test)

                batch_data = rearrange(batch_data, 'n b c h w -> (n b) c h w')
                if args.use_dual_branch:
                    batch_rgb = batch_data
                    batch_freq = build_frequency_batch(batch_data, args, freq_mean, freq_std)
                    with autocast(enabled=args.use_fp16, device_type="cuda"):
                        outputs_rgb, outputs_freq = model(batch_rgb, batch_freq)
                        outputs_rgb = rearrange(outputs_rgb, '(n b) l -> 1 b n l', n=args.num_class_test)
                        outputs_freq = rearrange(outputs_freq, '(n b) l -> 1 b n l', n=args.num_class_test)
                        _, logit_dict, _ = compute_dual_branch_logits_and_losses(
                            outputs_rgb,
                            outputs_freq,
                            labels,
                            args.num_support_test,
                            use_aux_loss=args.use_aux_loss,
                            lambda_rgb=args.lambda_rgb,
                            lambda_freq=args.lambda_freq,
                            normalize_reliability_variance=args.normalize_reliability_variance,
                            reliability_norm_mode=args.reliability_norm_mode,
                            reliability_temperature=args.reliability_temperature,
                            detach_reliability=args.detach_reliability,
                            clip_reliability_weight=args.clip_reliability_weight,
                            alpha_min=args.alpha_min,
                            alpha_max=args.alpha_max,
                            reliability_eps=args.reliability_eps,
                        )
                        scores = logit_dict["logits_dual"]
                else:
                    with autocast(enabled=args.use_fp16, device_type="cuda"):
                        outputs = model(batch_data)
                    outputs = rearrange(outputs, '(n b) l -> 1 b n l', n=args.num_class_test) # we change the subscript sequence

                    _, scores = compute_prototypical_loss(outputs, labels, args.num_support_test)

                prob = scores.softmax(dim=-1).cpu()
                labels = labels.cpu()

                prob_list.append(prob)
                label_list.append(labels)
            
            total_prob = torch.cat(prob_list, dim=0)
            total_label = torch.cat(label_list, dim=0)
            acc = acc_calculator(total_prob, total_label)
            ap = ap_calculator(total_prob, total_label)

            logger.info(f'Evaluation on {TEST_FOLDERS[i]} done. evaluating num: {len(total_prob)}, accuracy: {acc}, average precision: {ap}. ')
    ############################################################


if __name__ == '__main__': 
    main()


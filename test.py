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

from model.cosine_metric_fsd import CosineMetricFSD
from model.prototypical_utils import (
    compute_prototypical_loss,
    compute_cosine_prototypical_loss,
)
from datasets import setup_val_dataloader
from util.parser import TestParser
from util.utils import load_model
import util.logger as logger


# no ddp setting
def main(): 
    #################### prepare device ####################
    args = TestParser().args
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

                batch_data = rearrange(batch_data, 'n b c h w -> (n b) c h w')
                labels = torch.arange(0, args.num_class_test, device=args.device).repeat(args.num_query_test)

                with autocast(enabled=args.use_fp16, device_type="cuda"):
                    outputs = model(batch_data)
                outputs = rearrange(outputs, '(n b) l -> 1 b n l', n=args.num_class_test) # we change the subscript sequence

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


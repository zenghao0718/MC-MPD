# -*- coding: utf-8 -*-
""" Train ResNet50 for classification. 
"""

import os
import random
import torch
import torch.distributed as dist
from tqdm import tqdm

from torch.amp import autocast, GradScaler
import timm
from einops import rearrange
from torchmetrics.classification import Accuracy, AveragePrecision

from torch.utils.tensorboard import SummaryWriter

from model.dual_branch_fsd import DualBranchFSD
from model.prototypical_utils import compute_dual_branch_logits_and_losses, compute_prototypical_loss
from datasets import setup_infinity_train_dataloader, setup_val_dataloader
from util.frequency import haar_dwt_highfreq_rgb_mean, load_dwt_stats
from util.parser import TrainParser
from util.utils import save_model, setup_dist
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


def log_dual_branch_scalars(tb_writer, step, loss_dict, logit_dict, debug_dict, labels):
    with torch.no_grad():
        dist_rgb_mean = debug_dict["dist_rgb"].detach().float().mean()
        dist_freq_mean = debug_dict["dist_freq"].detach().float().mean()
        dist_ratio = dist_rgb_mean / dist_freq_mean.clamp_min(1e-12)
        labels = labels.detach()
        scalars = {
            "loss/total": loss_dict["loss"],
            "loss/dual": loss_dict["loss_dual"],
            "loss/rgb": loss_dict["loss_rgb"],
            "loss/freq": loss_dict["loss_freq"],
            "acc/dual": (logit_dict["logits_dual"].detach().argmax(dim=-1) == labels).float().mean(),
            "acc/rgb": (logit_dict["logits_rgb"].detach().argmax(dim=-1) == labels).float().mean(),
            "acc/freq": (logit_dict["logits_freq"].detach().argmax(dim=-1) == labels).float().mean(),
            "alpha/rgb_mean": debug_dict["alpha_rgb"].detach().float().mean(),
            "alpha/rgb_std": debug_dict["alpha_rgb"].detach().float().std(unbiased=False),
            "alpha/rgb_min": debug_dict["alpha_rgb"].detach().float().min(),
            "alpha/rgb_max": debug_dict["alpha_rgb"].detach().float().max(),
            "alpha/freq_mean": debug_dict["alpha_freq"].detach().float().mean(),
            "alpha/freq_std": debug_dict["alpha_freq"].detach().float().std(unbiased=False),
            "alpha/freq_min": debug_dict["alpha_freq"].detach().float().min(),
            "alpha/freq_max": debug_dict["alpha_freq"].detach().float().max(),
            "dist/rgb_mean": dist_rgb_mean,
            "dist/freq_mean": dist_freq_mean,
            "dist/ratio_mean": dist_ratio,
            "variance/rgb_mean": debug_dict["var_rgb"].detach().float().mean(),
            "variance/freq_mean": debug_dict["var_freq"].detach().float().mean(),
            "variance_norm/rgb_mean": debug_dict["var_rgb_norm"].detach().float().mean(),
            "variance_norm/freq_mean": debug_dict["var_freq_norm"].detach().float().mean(),
        }
        for name, value in scalars.items():
            tb_writer.add_scalar(name, float(value.detach().cpu()), step)


def main(): 
    #################### prepare ####################
    args = TrainParser().args
    validate_dual_branch_args(args)
    setup_dist(args) # ddp setup

    # terminal writer and file writer
    logger.setup(log_dir=args.output_dir, device=args.device)

    # TensorBoard writer (only on main process)
    tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tb"))
    #################################################

    
    ########## setup dataset and dataloader #########
    logger.info("Creating training data loader...")

    # data we use in GenImage, real is nature from SDv14 & SDv15
    IMAGE_FOLDERS = ["real", "ADM", "BigGAN", "glide", "Midjourney", "SD", "VQDM"]
    IMAGE_FOLDERS.remove(args.exclude_class)
    logger.info(f"Exclude class: {args.exclude_class}")

    train_iters = {
        folder: setup_infinity_train_dataloader(
            folder_path=os.path.join(args.data_root, folder, "train"), 
            batch_size=(args.num_support_train + args.num_query_train) * args.batch_size, # batch_size * task_size
            num_workers=args.num_workers
        ) for folder in IMAGE_FOLDERS
    }

    VAL_FOLDERS = IMAGE_FOLDERS + [args.exclude_class] # put at last
    val_dataloaders = {
        folder: setup_val_dataloader(
            folder_path=os.path.join(args.data_root, folder, "val"), 
            batch_size=args.num_support_val + args.num_query_val, 
            num_workers=args.num_workers, 
        ) for folder in VAL_FOLDERS
    }
    #################################################

    
    ################## create model #################
    freq_mean = None
    freq_std = None
    if args.use_dual_branch:
        logger.info("Creating DWT dual-branch model 'resnet50 + resnet50'... ")
        freq_mean, freq_std = load_dwt_stats(args.freq_stats_path, device=args.device, dtype=torch.float32)
        model = DualBranchFSD()
    else:
        logger.info("Creating model 'resnet50'... ")
        model = timm.create_model("resnet50", pretrained=True, num_classes=1024)
    print(model)

    model = model.to(args.device)
    #################################################
    

    ######### create optimizer and criterion ########
    logger.info("Creating optimizer and scheduler... ")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(optimizer)

    # scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        gamma=args.lr_scheduler_gamma,
        step_size=args.lr_scheduler_step
    )
    print(scheduler)
    #################################################


    #################### training ###################
    logger.info("Start training for %d steps. " % args.total_training_steps)

    scaler = GradScaler(enabled=args.use_fp16)

    effective_step = 0
    # starts looping
    for step in range(1, args.total_training_steps + 1): 
        model.train()

        optimizer.zero_grad()

        # select classes for single prototypical task
        selected_classes = random.sample(IMAGE_FOLDERS, args.num_class_train)

        # get data
        batch_data = torch.stack([next(train_iters[c])[0] for c in selected_classes], dim=0) # (num_class, batch * task_size, c, h, w)
        batch_data = batch_data.to(args.device)

        # make labels
        labels = torch.arange(0, args.num_class_train, device=args.device).repeat(args.batch_size * args.num_query_train)

        batch_data = rearrange(batch_data, 'n b c h w -> (n b) c h w')
        if args.use_dual_branch:
            batch_rgb = batch_data
            batch_freq = build_frequency_batch(batch_data, args, freq_mean, freq_std)
            with autocast(enabled=args.use_fp16, device_type="cuda"):
                outputs_rgb, outputs_freq = model(batch_rgb, batch_freq)
                outputs_rgb = rearrange(
                    outputs_rgb,
                    '(n b t) l -> b t n l',
                    n=args.num_class_train,
                    b=args.batch_size,
                )
                outputs_freq = rearrange(
                    outputs_freq,
                    '(n b t) l -> b t n l',
                    n=args.num_class_train,
                    b=args.batch_size,
                )
                loss_dict, logit_dict, debug_dict = compute_dual_branch_logits_and_losses(
                    outputs_rgb,
                    outputs_freq,
                    labels,
                    args.num_support_train,
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
                loss = loss_dict["loss"]
        else:
            with autocast(enabled=args.use_fp16, device_type="cuda"):
                outputs = model(batch_data)
            outputs = rearrange(outputs, '(n b t) l -> b t n l', n=args.num_class_train, b=args.batch_size) # we change the subscript sequence

            loss, _ = compute_prototypical_loss(outputs, labels, args.num_support_train)

        logger.logkv_mean("loss", loss.item())
        if args.use_dual_branch:
            logger.logkv_mean("loss_dual", loss_dict["loss_dual"].item())
            logger.logkv_mean("loss_rgb", loss_dict["loss_rgb"].item())
            logger.logkv_mean("loss_freq", loss_dict["loss_freq"].item())
            if step % args.tb_log_interval == 0:
                log_dual_branch_scalars(tb_writer, step, loss_dict, logit_dict, debug_dict, labels)
        scaler.scale(loss / args.accumulation_steps).backward()
        
        # accumulate
        if step % args.accumulation_steps == 0:
            effective_step += 1

            scaler.unscale_(optimizer)
            # other options if you want

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None: 
                scheduler.step() # per effective iter
        
        # logger info
        if step % args.log_interval == 0:
            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else args.lr
            logger.logkv("step", step)
            logger.logkv("effective_step", effective_step)
            logger.logkv("lr", current_lr)
            kvs = logger.dumpkvs()
            tb_writer.add_scalar("train/loss", kvs.get("loss", 0.0), step)
            tb_writer.add_scalar("train/lr", current_lr, step)
        
        # save checkpoint
        if step % args.save_interval == 0: 
            logger.info('Save checkpoint at step: %d', step)
            
            kwargs = {
                'step': step, 
                'effective_step': effective_step, 
                'model': model, 
                'optimizer': optimizer, 
                'scheduler': scheduler, 
                'scaler': scaler, 
                'args': args
            }
            if args.use_dual_branch:
                kwargs["freq_mean"] = freq_mean.detach().cpu()
                kwargs["freq_std"] = freq_std.detach().cpu()

            save_model(os.path.join(args.output_dir, "ckpt"), args.model, **kwargs)
            torch.cuda.empty_cache()
        
        
        ##### evaluation #####
        if step % args.eval_interval == 0: 
            logger.info('Evaluating at step: %d', step)
            model.eval()

            acc_calculator = Accuracy(task="multiclass", num_classes=2)
            ap_calculator = AveragePrecision(task="multiclass", num_classes=2, thresholds=10)

            # build dataset
            with torch.no_grad(): 
                for i in range(1, len(VAL_FOLDERS)): 
                    # reset so per-class metrics are not contaminated by earlier classes
                    acc_calculator.reset()
                    ap_calculator.reset()

                    prob_list = []
                    label_list = []
                    for (real_batch, _), (fake_batch, _) in tqdm(zip(val_dataloaders["real"], val_dataloaders[VAL_FOLDERS[i]])): 
                        batch_data = torch.stack([real_batch, fake_batch], dim=0) # (2, task_size, c, h, w)
                        batch_data = batch_data.to(args.device)

                        labels = torch.arange(0, 2, device=args.device).repeat(args.num_query_val)

                        batch_data = rearrange(batch_data, 'n b c h w -> (n b) c h w')
                        if args.use_dual_branch:
                            batch_rgb = batch_data
                            batch_freq = build_frequency_batch(batch_data, args, freq_mean, freq_std)
                            with autocast(enabled=args.use_fp16, device_type="cuda"):
                                outputs_rgb, outputs_freq = model(batch_rgb, batch_freq)
                                outputs_rgb = rearrange(outputs_rgb, '(n b) l -> 1 b n l', n=2)
                                outputs_freq = rearrange(outputs_freq, '(n b) l -> 1 b n l', n=2)
                                _, logit_dict, _ = compute_dual_branch_logits_and_losses(
                                    outputs_rgb,
                                    outputs_freq,
                                    labels,
                                    args.num_support_val,
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
                            outputs = rearrange(outputs, '(n b) l -> 1 b n l', n=2) # we change the subscript sequence

                            _, scores = compute_prototypical_loss(outputs, labels, args.num_support_val)
                        
                        prob = scores.softmax(dim=-1).cpu()
                        labels = labels.cpu()

                        prob_list.append(prob)
                        label_list.append(labels)
                    
                    total_prob = torch.cat(prob_list, dim=0)
                    total_label = torch.cat(label_list, dim=0)
                    acc = acc_calculator(total_prob, total_label)
                    ap = ap_calculator(total_prob, total_label)

                    logger.info(f'Evaluation on {VAL_FOLDERS[i]} done. evaluating num: {len(total_prob)}, accuracy: {acc}, average precision: {ap}. ')
                    split_tag = "val_unseen" if VAL_FOLDERS[i] == args.exclude_class else "val_seen"
                    tb_writer.add_scalar(f"{split_tag}/acc_{VAL_FOLDERS[i]}", acc.item(), step)
                    tb_writer.add_scalar(f"{split_tag}/ap_{VAL_FOLDERS[i]}", ap.item(), step)
        ##### evaluation done #####
    
    tb_writer.close()
    #################################################


if __name__ == '__main__': 
    try:
        main()
    finally:
        # Explicitly tear down the NCCL process group so torchrun can exit
        # promptly between consecutive leave-one-out runs. Without this, the
        # next torchrun tends to stall on rendezvous for 10~25 minutes.
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


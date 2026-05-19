"""Smoke checks for the DWT dual-branch implementation.

This script is intended for AutoDL or another environment with torch/timm
available. It does not read datasets or run training.
"""

import argparse

import torch
from einops import rearrange

from model.dual_branch_fsd import DualBranchFSD
from model.prototypical_utils import (
    compute_dual_branch_logits_and_losses,
    compute_prototypical_loss,
)
from util.frequency import haar_dwt_highfreq_rgb_mean


def parse_args():
    parser = argparse.ArgumentParser(description="Run DWT dual-branch smoke checks.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def assert_finite(tensor, name):
    if not torch.isfinite(tensor).all():
        raise AssertionError(f"{name} contains NaN or Inf.")


def check_dwt(device):
    x = torch.rand(4, 3, 224, 224, device=device)
    freq = haar_dwt_highfreq_rgb_mean(x)
    assert freq.shape == (4, 3, 224, 224), f"Unexpected DWT shape: {freq.shape}"
    assert_finite(freq, "DWT output")


def check_dual_forward(device):
    model = DualBranchFSD(rgb_pretrained=False, freq_pretrained=False).to(device)
    model.eval()
    x_rgb = torch.rand(4, 3, 224, 224, device=device)
    x_freq = torch.rand(4, 3, 224, 224, device=device)
    with torch.no_grad():
        z_rgb, z_freq = model(x_rgb, x_freq)
    assert z_rgb.shape == (4, 1024), f"Unexpected RGB feature shape: {z_rgb.shape}"
    assert z_freq.shape == (4, 1024), f"Unexpected frequency feature shape: {z_freq.shape}"
    assert model.rgb_encoder is not model.freq_encoder


def check_prototype_distance(device):
    batch_size, support_num, query_num, class_num, feature_dim = 2, 5, 5, 3, 1024
    task_size = support_num + query_num
    inputs_rgb = torch.randn(batch_size, task_size, class_num, feature_dim, device=device, requires_grad=True)
    inputs_freq = torch.randn(batch_size, task_size, class_num, feature_dim, device=device, requires_grad=True)
    labels = torch.arange(class_num, device=device).repeat(batch_size * query_num)

    loss_dict, logit_dict, debug_dict = compute_dual_branch_logits_and_losses(
        inputs_rgb,
        inputs_freq,
        labels,
        support_num=support_num,
    )
    expected_logits = (batch_size * query_num * class_num, class_num)
    assert logit_dict["logits_dual"].shape == expected_logits
    assert logit_dict["logits_rgb"].shape == expected_logits
    assert logit_dict["logits_freq"].shape == expected_logits

    assert debug_dict["alpha_rgb"].shape == (batch_size, class_num)
    assert debug_dict["alpha_freq"].shape == (batch_size, class_num)
    assert (debug_dict["alpha_rgb"] >= 0.1 - 1e-6).all()
    assert (debug_dict["alpha_rgb"] <= 0.9 + 1e-6).all()
    assert (debug_dict["alpha_freq"] >= 0.1 - 1e-6).all()
    assert (debug_dict["alpha_freq"] <= 0.9 + 1e-6).all()
    assert torch.allclose(
        debug_dict["alpha_rgb"] + debug_dict["alpha_freq"],
        torch.ones_like(debug_dict["alpha_rgb"]),
        atol=1e-6,
    )

    assert debug_dict["var_rgb"].shape == (batch_size, class_num)
    assert debug_dict["var_freq"].shape == (batch_size, class_num)
    assert debug_dict["var_rgb_norm"].shape == (batch_size, class_num)
    assert debug_dict["var_freq_norm"].shape == (batch_size, class_num)
    assert debug_dict["dist_rgb"].shape == (batch_size, query_num * class_num, class_num)
    assert debug_dict["dist_freq"].shape == (batch_size, query_num * class_num, class_num)

    for name, value in loss_dict.items():
        assert_finite(value, name)
    loss_dict["loss"].backward()
    assert inputs_rgb.grad is not None and inputs_rgb.grad.abs().sum() > 0
    assert inputs_freq.grad is not None and inputs_freq.grad.abs().sum() > 0


def check_gradient_flow(device):
    batch_size, support_num, query_num, class_num = 1, 1, 1, 2
    task_size = support_num + query_num
    model = DualBranchFSD(rgb_pretrained=False, freq_pretrained=False).to(device)
    model.train()

    batch_rgb = torch.rand(class_num, batch_size * task_size, 3, 224, 224, device=device)
    batch_freq = torch.rand(class_num, batch_size * task_size, 3, 224, 224, device=device)
    batch_rgb = rearrange(batch_rgb, "n b c h w -> (n b) c h w")
    batch_freq = rearrange(batch_freq, "n b c h w -> (n b) c h w")
    labels = torch.arange(class_num, device=device).repeat(batch_size * query_num)

    outputs_rgb, outputs_freq = model(batch_rgb, batch_freq)
    outputs_rgb = rearrange(outputs_rgb, "(n b t) l -> b t n l", n=class_num, b=batch_size)
    outputs_freq = rearrange(outputs_freq, "(n b t) l -> b t n l", n=class_num, b=batch_size)
    loss_dict, _, _ = compute_dual_branch_logits_and_losses(
        outputs_rgb,
        outputs_freq,
        labels,
        support_num=support_num,
    )
    loss_dict["loss"].backward()

    rgb_grad_count = sum(
        1
        for p in model.rgb_encoder.parameters()
        if p.grad is not None and p.grad.detach().abs().sum() > 0
    )
    freq_grad_count = sum(
        1
        for p in model.freq_encoder.parameters()
        if p.grad is not None and p.grad.detach().abs().sum() > 0
    )
    assert rgb_grad_count > 0, "RGB encoder did not receive gradients."
    assert freq_grad_count > 0, "Frequency encoder did not receive gradients."


def check_baseline_function(device):
    batch_size, support_num, query_num, class_num, feature_dim = 2, 5, 5, 3, 1024
    task_size = support_num + query_num
    inputs = torch.randn(batch_size, task_size, class_num, feature_dim, device=device)
    labels = torch.arange(class_num, device=device).repeat(batch_size * query_num)
    loss, scores = compute_prototypical_loss(inputs, labels, support_num)
    assert scores.shape == (batch_size * query_num * class_num, class_num)
    assert_finite(loss, "baseline loss")
    assert_finite(scores, "baseline scores")


def main():
    args = parse_args()
    device = torch.device(args.device)
    check_dwt(device)
    check_dual_forward(device)
    check_prototype_distance(device)
    check_gradient_flow(device)
    check_baseline_function(device)
    print("All DWT dual-branch smoke checks passed.")


if __name__ == "__main__":
    main()

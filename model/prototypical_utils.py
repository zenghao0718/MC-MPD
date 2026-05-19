""" 
    Prototypical loss from https://arxiv.org/abs/1703.05175. 
"""

from einops import rearrange
import torch
import torch.nn.functional as F


def compute_prototypical_loss(inputs, labels, support_num): 
    """ Args:
            inputs: tensors with shape (batch_size, task_samples_num, class_num, features)
            labels: tensors with shape (N, ) or (N, C), where N = batch_size * query_num * class_num
            support_num: capacity of support set used to split inputs data
        
        Returns:
            loss: loss tensor with shape (1, )
            scores: score matrix for calculating cross entropy, which shape (N, C)
    """

    support_set = inputs[:, :support_num, ...]
    query_set = inputs[:, support_num:, ...]

    # compute the barycentres
    prototypes = support_set.mean(dim=1, keepdim=True) # (batch_size, 1, class_num, hidden_dim)

    # compute the distance between each query point and each barycentre, use negative value as scores (so the larger equals to the better)
    scores = - ((rearrange(query_set, 'b q n l -> b (q n) 1 l') - prototypes) ** 2).sum(dim=-1) # (batch_size, query_num * class_num, class_num)
    
    scores = rearrange(scores, 'b n c -> (b n) c')
    loss = F.cross_entropy(scores, labels)

    return loss, scores


def compute_dual_branch_logits_and_losses(
    inputs_rgb,
    inputs_freq,
    labels,
    support_num,
    use_aux_loss=True,
    lambda_rgb=0.2,
    lambda_freq=0.2,
    normalize_reliability_variance=True,
    reliability_norm_mode="episode_mean",
    reliability_temperature=1.0,
    detach_reliability=True,
    clip_reliability_weight=True,
    alpha_min=0.1,
    alpha_max=0.9,
    reliability_eps=1e-8,
):
    """Compute dual-branch prototype logits, reliability fusion, and losses.

    inputs_rgb and inputs_freq use the same layout as compute_prototypical_loss:
    [B, T, Nc, D]. Labels must follow the flattened query order (b, q, n).
    """

    if inputs_rgb.shape != inputs_freq.shape:
        raise ValueError(
            "inputs_rgb and inputs_freq must have the same shape, got "
            f"{tuple(inputs_rgb.shape)} and {tuple(inputs_freq.shape)}."
        )
    if inputs_rgb.dim() != 4:
        raise ValueError(f"Expected inputs with shape [B, T, Nc, D], got {tuple(inputs_rgb.shape)}.")
    if reliability_norm_mode != "episode_mean":
        raise NotImplementedError("First DWT dual-branch version only supports reliability_norm_mode='episode_mean'.")
    if reliability_temperature <= 0:
        raise ValueError("reliability_temperature must be positive.")
    if not 0.0 <= alpha_min <= alpha_max <= 1.0:
        raise ValueError("alpha_min and alpha_max must satisfy 0 <= alpha_min <= alpha_max <= 1.")

    labels = labels.to(device=inputs_rgb.device, dtype=torch.long)

    support_rgb = inputs_rgb[:, :support_num, ...]
    query_rgb = inputs_rgb[:, support_num:, ...]
    support_freq = inputs_freq[:, :support_num, ...]
    query_freq = inputs_freq[:, support_num:, ...]

    proto_rgb = support_rgb.mean(dim=1, keepdim=True)
    proto_freq = support_freq.mean(dim=1, keepdim=True)

    query_rgb_flat = rearrange(query_rgb, "b q n l -> b (q n) l").unsqueeze(2)
    query_freq_flat = rearrange(query_freq, "b q n l -> b (q n) l").unsqueeze(2)

    dist_rgb = ((query_rgb_flat - proto_rgb) ** 2).sum(dim=-1)
    dist_freq = ((query_freq_flat - proto_freq) ** 2).sum(dim=-1)

    var_rgb = ((support_rgb - proto_rgb) ** 2).sum(dim=-1).mean(dim=1)
    var_freq = ((support_freq - proto_freq) ** 2).sum(dim=-1).mean(dim=1)

    if normalize_reliability_variance:
        var_rgb_norm = var_rgb / (var_rgb.mean(dim=1, keepdim=True) + reliability_eps)
        var_freq_norm = var_freq / (var_freq.mean(dim=1, keepdim=True) + reliability_eps)
    else:
        var_rgb_norm = var_rgb
        var_freq_norm = var_freq

    reliability_logits = torch.stack(
        [
            -var_rgb_norm / reliability_temperature,
            -var_freq_norm / reliability_temperature,
        ],
        dim=-1,
    )
    alpha = torch.softmax(reliability_logits, dim=-1)
    alpha_rgb = alpha[..., 0]
    alpha_freq = alpha[..., 1]

    if detach_reliability:
        alpha_rgb = alpha_rgb.detach()
        alpha_freq = alpha_freq.detach()

    if clip_reliability_weight:
        alpha_rgb = torch.clamp(alpha_rgb, min=alpha_min, max=alpha_max)
        alpha_freq = 1.0 - alpha_rgb

    dist_dual = alpha_rgb[:, None, :] * dist_rgb + alpha_freq[:, None, :] * dist_freq

    logits_dual = -rearrange(dist_dual, "b q c -> (b q) c")
    logits_rgb = -rearrange(dist_rgb, "b q c -> (b q) c")
    logits_freq = -rearrange(dist_freq, "b q c -> (b q) c")

    loss_dual = F.cross_entropy(logits_dual, labels)
    loss_rgb = F.cross_entropy(logits_rgb, labels)
    loss_freq = F.cross_entropy(logits_freq, labels)
    if use_aux_loss:
        loss = loss_dual + lambda_rgb * loss_rgb + lambda_freq * loss_freq
    else:
        loss = loss_dual

    loss_dict = {
        "loss": loss,
        "loss_dual": loss_dual,
        "loss_rgb": loss_rgb,
        "loss_freq": loss_freq,
    }
    logit_dict = {
        "logits_dual": logits_dual,
        "logits_rgb": logits_rgb,
        "logits_freq": logits_freq,
    }
    debug_dict = {
        "alpha_rgb": alpha_rgb,
        "alpha_freq": alpha_freq,
        "var_rgb": var_rgb,
        "var_freq": var_freq,
        "var_rgb_norm": var_rgb_norm,
        "var_freq_norm": var_freq_norm,
        "dist_rgb": dist_rgb,
        "dist_freq": dist_freq,
        "dist_dual": dist_dual,
    }
    return loss_dict, logit_dict, debug_dict


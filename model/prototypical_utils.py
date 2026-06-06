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


def compute_cosine_prototypical_loss(
    inputs,
    labels,
    support_num,
    scale,
    eps=1e-12,
):
    """Compute L2-normalized cosine prototypical loss.

    Args:
        inputs: tensor with shape (batch_size, task_samples_num, class_num, features).
        labels: tensor with shape (batch_size * query_num * class_num, ).
        support_num: capacity of support set used to split inputs data.
        scale: positive scalar, usually model.get_scale().
        eps: epsilon for L2 normalization and safe reciprocal logging.

    Returns:
        loss: loss tensor with shape (1, ).
        scores: scaled cosine logits with shape (N, C).
        debug_dict: auxiliary tensors for logging.
    """

    if inputs.dim() != 4:
        raise ValueError(
            f"Expected inputs with shape [B, T, Nc, D], got {tuple(inputs.shape)}."
        )

    inputs = inputs.float()
    labels = labels.to(device=inputs.device, dtype=torch.long)

    if not torch.is_tensor(scale):
        scale = torch.tensor(scale, device=inputs.device, dtype=torch.float32)
    else:
        scale = scale.to(device=inputs.device)
    scale = scale.float()

    support_set = inputs[:, :support_num, ...]
    query_set = inputs[:, support_num:, ...]

    support_set = F.normalize(support_set, p=2, dim=-1, eps=eps)
    query_set = F.normalize(query_set, p=2, dim=-1, eps=eps)

    prototypes = support_set.mean(dim=1, keepdim=True)
    prototypes = F.normalize(prototypes, p=2, dim=-1, eps=eps)

    query_flat = rearrange(query_set, 'b q n l -> b (q n) l').unsqueeze(2)
    cosine_scores = (query_flat * prototypes).sum(dim=-1)

    scores = cosine_scores * scale
    scores = rearrange(scores, 'b n c -> (b n) c')
    loss = F.cross_entropy(scores, labels)

    scale_detached = scale.detach().float()
    cosine_detached = cosine_scores.detach().float()
    debug_dict = {
        "scale": scale_detached,
        "temperature": 1.0 / scale_detached.clamp_min(eps),
        "cosine_mean": cosine_detached.mean(),
        "cosine_std": cosine_detached.std(unbiased=False),
        "cosine_min": cosine_detached.min(),
        "cosine_max": cosine_detached.max(),
    }

    return loss, scores, debug_dict


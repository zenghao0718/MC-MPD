"""Quick checks for batched graph distance.

This script does not read images or run ResNet. It only creates random feature
tensors and compares the old loop graph implementation with the new batched
implementation.
"""

import argparse
import os
import sys

import torch
from einops import rearrange


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model.prototypical_utils import (  # noqa: E402
    compute_graph_prototypical_scores,
    compute_graph_prototypical_scores_loop,
    compute_prototypical_loss,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--support_num", type=int, default=5)
    parser.add_argument("--query_num", type=int, default=3)
    parser.add_argument("--class_num", type=int, default=3)
    parser.add_argument("--feature_dim", type=int, default=16)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(42)

    inputs = torch.randn(
        args.batch_size,
        args.support_num + args.query_num,
        args.class_num,
        args.feature_dim,
        device=args.device,
        requires_grad=True,
    )
    labels = torch.arange(args.class_num, device=args.device).repeat(args.batch_size * args.query_num)

    support_set = inputs[:, :args.support_num, ...]
    query_set = inputs[:, args.support_num:, ...]
    prototypes = support_set.mean(dim=1, keepdim=True)

    # Baseline check: squared_euclidean must still match the original formula.
    _, baseline_scores = compute_prototypical_loss(
        inputs,
        labels,
        args.support_num,
        distance_type="squared_euclidean",
    )
    original_scores = -((rearrange(query_set, "b q n l -> b (q n) 1 l") - prototypes) ** 2).sum(dim=-1)
    original_scores = rearrange(original_scores, "b n c -> (b n) c")
    baseline_diff = (baseline_scores - original_scores).abs().max().item()

    loop_scores, _ = compute_graph_prototypical_scores_loop(
        support_set=support_set,
        query_set=query_set,
        prototypes=prototypes,
        labels=labels,
        graph_k=3,
        graph_query_k_global=3,
        graph_query_min_per_class=1,
        distance_norm="mean",
    )
    batched_scores, _ = compute_graph_prototypical_scores(
        support_set=support_set,
        query_set=query_set,
        prototypes=prototypes,
        labels=labels,
        graph_k=3,
        graph_query_k_global=3,
        graph_query_min_per_class=1,
        distance_norm="mean",
    )
    graph_diff = (loop_scores - batched_scores).abs().max().item()

    loss, _ = compute_prototypical_loss(
        inputs,
        labels,
        args.support_num,
        distance_type="graph",
        graph_alpha=1.0,
        graph_edge_weight="squared_euclidean",
        distance_norm="mean",
        graph_mode="label_aware_global",
        graph_k=3,
        graph_query_k_global=3,
        graph_query_min_per_class=1,
        graph_fallback="squared_euclidean",
        transductive=False,
    )
    loss.backward()
    grad_ok = inputs.grad is not None and torch.isfinite(inputs.grad).all().item()

    print(f"baseline max diff: {baseline_diff:.8g}")
    print(f"loop vs batched graph max diff: {graph_diff:.8g}")
    print(f"graph backward finite: {grad_ok}")

    if baseline_diff > args.tolerance:
        raise SystemExit(f"Baseline mismatch: {baseline_diff} > {args.tolerance}")
    if graph_diff > args.tolerance:
        raise SystemExit(f"Graph implementation mismatch: {graph_diff} > {args.tolerance}")
    if not grad_ok:
        raise SystemExit("Graph backward failed or produced non-finite gradients.")


if __name__ == "__main__":
    main()

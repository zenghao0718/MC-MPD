""" 
    Prototypical loss from https://arxiv.org/abs/1703.05175. 
"""

from einops import rearrange
import torch
import torch.nn.functional as F


def compute_squared_euclidean_scores(query_set, prototypes):
    """Compute original FSD scores with squared Euclidean distance."""

    distances = ((rearrange(query_set, 'b q n l -> b (q n) 1 l') - prototypes) ** 2).sum(dim=-1)
    scores = -distances
    return rearrange(scores, 'b n c -> (b n) c')


def _validate_graph_options(
    graph_alpha,
    graph_edge_weight,
    distance_norm,
    graph_mode,
    graph_fallback,
    transductive,
):
    """第一版只实现主方法，其他选项直接报错，避免悄悄跑成未验证方案。"""

    if graph_mode != "label_aware_global":
        raise NotImplementedError("第一版暂时只支持 graph_mode='label_aware_global'。")
    if transductive:
        raise NotImplementedError("第一版暂时只支持 transductive=False，即每张 query 单独建图。")
    if graph_edge_weight != "squared_euclidean":
        raise NotImplementedError("第一版暂时只支持 graph_edge_weight='squared_euclidean'。")
    if graph_fallback != "squared_euclidean":
        raise NotImplementedError("第一版暂时只支持 graph_fallback='squared_euclidean'。")
    if distance_norm not in ("mean", "none"):
        raise NotImplementedError("第一版暂时只支持 distance_norm='mean' 或 'none'。")
    if abs(float(graph_alpha) - 1.0) > 1e-12:
        raise NotImplementedError("第一版暂时只支持 graph_alpha=1.0，不做欧氏距离和图距离混合。")


def _pairwise_squared_euclidean(nodes):
    """Pairwise squared Euclidean distance for graph edge weights."""

    diff = nodes[:, None, :] - nodes[None, :, :]
    return (diff ** 2).sum(dim=-1)


def _connect_undirected(adj, dist_matrix, src_idx, dst_idx):
    """Connect two nodes with differentiable edge weight from dist_matrix."""

    adj[src_idx, dst_idx] = dist_matrix[src_idx, dst_idx]
    adj[dst_idx, src_idx] = dist_matrix[src_idx, dst_idx]


def build_label_aware_graph(
    support_by_class,
    prototypes,
    query,
    graph_k,
    graph_query_k_global,
    graph_query_min_per_class,
):
    """构建类别感知全局图。

    support-support 只连同类；prototype 连接本类全部 support；
    query 连接全局最近 support，并保证每个类别至少可达。
    """

    if graph_k < 0 or graph_query_k_global < 0 or graph_query_min_per_class < 0:
        raise ValueError("graph_k、graph_query_k_global、graph_query_min_per_class 不能为负数。")

    class_num, support_num, feature_dim = support_by_class.shape
    support_nodes = support_by_class.reshape(class_num * support_num, feature_dim)
    nodes = torch.cat([support_nodes, prototypes, query.unsqueeze(0)], dim=0).float()

    support_total = class_num * support_num
    proto_start = support_total
    query_idx = support_total + class_num
    node_num = query_idx + 1

    dist_matrix = _pairwise_squared_euclidean(nodes)
    adj = torch.full(
        (node_num, node_num),
        float("inf"),
        device=nodes.device,
        dtype=nodes.dtype,
    )
    adj.fill_diagonal_(0)

    # 1. support-support：只在同类别 support 内部建 kNN 边。
    support_k = min(graph_k, max(support_num - 1, 0))
    if support_k > 0:
        for class_idx in range(class_num):
            start = class_idx * support_num
            end = start + support_num
            class_indices = torch.arange(start, end, device=nodes.device)
            class_dist = dist_matrix[class_indices][:, class_indices].clone()
            class_dist.fill_diagonal_(float("inf"))
            nearest = torch.topk(class_dist, k=support_k, largest=False, dim=-1).indices
            for local_src in range(support_num):
                src_idx = start + local_src
                for local_dst in nearest[local_src]:
                    dst_idx = start + int(local_dst.item())
                    _connect_undirected(adj, dist_matrix, src_idx, dst_idx)

    # 2. prototype-support：第一版连接本类全部 support，避免 prototype 孤立。
    for class_idx in range(class_num):
        proto_idx = proto_start + class_idx
        for support_idx in range(support_num):
            node_idx = class_idx * support_num + support_idx
            _connect_undirected(adj, dist_matrix, proto_idx, node_idx)

    # 3. query-support：先连全局最近邻，再保证每个类别至少连接若干 support。
    support_indices = torch.arange(0, support_total, device=nodes.device)
    query_to_support = dist_matrix[query_idx, support_indices]
    global_k = min(graph_query_k_global, support_total)
    if global_k > 0:
        global_nearest = torch.topk(query_to_support, k=global_k, largest=False).indices
        for dst in support_indices[global_nearest]:
            _connect_undirected(adj, dist_matrix, query_idx, int(dst.item()))

    per_class_k = min(graph_query_min_per_class, support_num)
    if per_class_k > 0:
        for class_idx in range(class_num):
            start = class_idx * support_num
            end = start + support_num
            class_indices = torch.arange(start, end, device=nodes.device)
            class_dist = dist_matrix[query_idx, class_indices]
            class_nearest = torch.topk(class_dist, k=per_class_k, largest=False).indices
            for dst in class_indices[class_nearest]:
                _connect_undirected(adj, dist_matrix, query_idx, int(dst.item()))

    proto_indices = torch.arange(proto_start, proto_start + class_num, device=nodes.device)
    direct_proto_distances = dist_matrix[query_idx, proto_indices]
    return adj, query_idx, proto_indices, direct_proto_distances


def floyd_warshall_torch(adj):
    """Use PyTorch Floyd-Warshall so graph edge weights keep gradients."""

    dist = adj.clone()
    node_num = dist.shape[0]
    for k in range(node_num):
        dist = torch.minimum(dist, dist[:, k:k + 1] + dist[k:k + 1, :])
    return dist


def mean_normalize_distance(distances, eps=1e-12):
    """均值归一化：只缩放距离，不改变距离正负和大小顺序。"""

    return distances / (distances.detach().mean().clamp_min(eps))


def _compute_margin(distances, labels):
    """Margin = 最近错误类距离 - 正确类距离；正数表示正确类更近。"""

    targets = labels.argmax(dim=-1) if labels.dim() > 1 else labels
    targets = targets.to(device=distances.device, dtype=torch.long)
    true_distances = distances.gather(1, targets.view(-1, 1)).squeeze(1)
    wrong_mask = torch.ones_like(distances, dtype=torch.bool)
    wrong_mask.scatter_(1, targets.view(-1, 1), False)
    nearest_wrong = distances.masked_fill(~wrong_mask, float("inf")).min(dim=1).values
    return nearest_wrong - true_distances


def compute_graph_prototypical_scores(
    support_set,
    query_set,
    prototypes,
    labels,
    graph_k=3,
    graph_query_k_global=3,
    graph_query_min_per_class=1,
    distance_norm="mean",
):
    """Compute first-version label-aware graph distance scores."""

    batch_size, _, class_num, _ = support_set.shape
    query_num = query_set.shape[1]

    batch_scores = []
    raw_distance_values = []
    final_distance_values = []
    unreachable_count = 0
    fallback_count = 0

    for batch_idx in range(batch_size):
        support_by_class = support_set[batch_idx].permute(1, 0, 2).float()
        proto_by_class = prototypes[batch_idx, 0].float()
        episode_distances = []
        episode_raw_distances = []

        for query_idx in range(query_num):
            for query_class_idx in range(class_num):
                query = query_set[batch_idx, query_idx, query_class_idx].float()
                adj, graph_query_idx, proto_indices, direct_proto_distances = build_label_aware_graph(
                    support_by_class=support_by_class,
                    prototypes=proto_by_class,
                    query=query,
                    graph_k=graph_k,
                    graph_query_k_global=graph_query_k_global,
                    graph_query_min_per_class=graph_query_min_per_class,
                )
                shortest_paths = floyd_warshall_torch(adj)
                graph_distances = shortest_paths[graph_query_idx, proto_indices]

                invalid_mask = ~torch.isfinite(graph_distances)
                invalid_num = int(invalid_mask.detach().sum().item())
                unreachable_count += invalid_num
                fallback_count += invalid_num
                graph_distances = torch.where(invalid_mask, direct_proto_distances, graph_distances)

                episode_raw_distances.append(graph_distances)
                episode_distances.append(graph_distances)

        episode_raw_distances = torch.stack(episode_raw_distances, dim=0)
        episode_distances = torch.stack(episode_distances, dim=0)
        raw_distance_values.append(episode_raw_distances.detach().reshape(-1))

        if distance_norm == "mean":
            episode_distances = mean_normalize_distance(episode_distances)
        elif distance_norm != "none":
            raise NotImplementedError("第一版暂时只支持 distance_norm='mean' 或 'none'。")

        final_distance_values.append(episode_distances.detach().reshape(-1))
        batch_scores.append(-episode_distances)

    distances_for_scores = torch.cat([(-scores) for scores in batch_scores], dim=0)
    scores = torch.cat(batch_scores, dim=0)
    raw_distances = torch.cat(raw_distance_values, dim=0)
    final_distances = torch.cat(final_distance_values, dim=0)
    margins = _compute_margin(distances_for_scores, labels).detach()

    stats = {
        "mean_graph_distance": raw_distances.mean(),
        "min_graph_distance": raw_distances.min(),
        "max_graph_distance": raw_distances.max(),
        "fallback_count": float(fallback_count),
        "unreachable_count": float(unreachable_count),
        "mean_margin": margins.mean(),
        "mean_final_distance": final_distances.mean(),
    }
    return scores, stats


def compute_prototypical_loss(
    inputs,
    labels,
    support_num,
    distance_type="squared_euclidean",
    graph_alpha=1.0,
    graph_edge_weight="squared_euclidean",
    distance_norm="mean",
    graph_mode="label_aware_global",
    graph_k=3,
    graph_query_k_global=3,
    graph_query_min_per_class=1,
    graph_fallback="squared_euclidean",
    transductive=False,
    return_stats=False,
):
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

    if distance_type == "squared_euclidean":
        # 原始 FSD baseline：平方欧氏距离，不做图距离和归一化。
        scores = compute_squared_euclidean_scores(query_set, prototypes)
        stats = {}
    elif distance_type == "graph":
        _validate_graph_options(
            graph_alpha=graph_alpha,
            graph_edge_weight=graph_edge_weight,
            distance_norm=distance_norm,
            graph_mode=graph_mode,
            graph_fallback=graph_fallback,
            transductive=transductive,
        )
        scores, stats = compute_graph_prototypical_scores(
            support_set=support_set,
            query_set=query_set,
            prototypes=prototypes,
            labels=labels,
            graph_k=graph_k,
            graph_query_k_global=graph_query_k_global,
            graph_query_min_per_class=graph_query_min_per_class,
            distance_norm=distance_norm,
        )
        stats["graph_alpha"] = float(graph_alpha)
        stats["graph_k"] = float(graph_k)
    else:
        raise NotImplementedError("distance_type 只能是 'squared_euclidean' 或 'graph'。")

    loss = F.cross_entropy(scores, labels)

    if return_stats:
        return loss, scores, stats
    return loss, scores


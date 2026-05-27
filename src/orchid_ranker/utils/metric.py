import numpy as np
import torch


def compute_accuracy(labels, scores, threshold=0.5):
    """Return binary accuracy for labels and probability-like scores."""
    y_true = _binary_labels(labels)
    y_score = _finite_vector(scores, "scores")
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("labels and scores must have the same length")
    if y_true.size == 0:
        return 0.0
    y_pred = (y_score >= float(threshold)).astype(int)
    return float(np.mean(y_pred == y_true))


def compute_auc(labels, scores):
    """Return ROC AUC, falling back to 0.5 when only one class is present."""
    y_true = _binary_labels(labels)
    y_score = _finite_vector(scores, "scores")
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("labels and scores must have the same length")
    if y_true.size == 0:
        return 0.0
    if np.unique(y_true).size < 2:
        return 0.5

    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    sorted_scores = y_score[order]
    start = 0
    while start < sorted_scores.size:
        end = start + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + end + 1) / 2.0
        start = end

    positives = y_true == 1
    n_pos = int(np.sum(positives))
    n_neg = int(y_true.size - n_pos)
    rank_sum_pos = float(np.sum(ranks[positives]))
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def precision_recall_ndcg_at_k(model, val_df, k, num_users, num_items, device="cpu", implicit=True):
    """
    Compute Precision@K, Recall@K, and NDCG@K for recommendation.

    Args:
        model: trained recommendation model
        val_df: validation DataFrame with columns ['u', 'i', 'label']
        k: cutoff for metrics
        num_users: total number of users
        num_items: total number of items
        device: torch device
        implicit: whether labels are implicit (0/1) or explicit (rating values)
    """
    # Build ground truth dictionary
    if implicit:
        # Only store positive items for each user
        gt = {u: set() for u in range(num_users)}
        for row in val_df.itertuples(index=False):
            if row.label > 0:
                gt[row.u].add(row.i)
    else:
        # Store item-rating pairs for explicit feedback
        gt = {u: {} for u in range(num_users)}
        for row in val_df.itertuples(index=False):
            gt[row.u][row.i] = row.label

    precisions, recalls, ndcgs = [], [], []

    model.eval()
    with torch.no_grad():
        for u in range(num_users):
            # Predict scores for all items for user u
            user_tensor = torch.full((num_items,), u, dtype=torch.long, device=device)
            item_tensor = torch.arange(num_items, dtype=torch.long, device=device)
            scores = model(user_tensor, item_tensor).cpu().numpy()

            # Remove items already seen in training+val
            if implicit:
                # For implicit, ground truth is a set
                relevant_items = gt[u]
            else:
                # For explicit, ground truth is a dict
                relevant_items = set(gt[u].keys())

            if not relevant_items:
                continue  # skip users with no positives

            # Rank items by predicted score
            ranked_items = np.argsort(-scores)[:k]

            # Precision & Recall (binary, even for explicit here)
            hits = sum(1 for i in ranked_items if i in relevant_items)
            precisions.append(hits / k)
            recalls.append(hits / len(relevant_items))

            # NDCG computation
            if implicit:
                dcg = sum(1.0 / np.log2(idx + 2) for idx, i in enumerate(ranked_items) if i in relevant_items)
                idcg = sum(1.0 / np.log2(idx + 2) for idx in range(min(len(relevant_items), k)))
            else:
                # Use actual ratings as relevance scores
                dcg = sum(gt[u].get(i, 0) / np.log2(idx + 2) for idx, i in enumerate(ranked_items))
                ideal_ratings = sorted(gt[u].values(), reverse=True)[:k]
                idcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_ratings))

            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return np.mean(precisions), np.mean(recalls), np.mean(ndcgs)


def _finite_vector(values, name):
    array = np.asarray(list(values), dtype=float).reshape(-1)
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _binary_labels(values):
    labels = _finite_vector(values, "labels")
    return (labels >= 0.5).astype(int)


__all__ = ["compute_accuracy", "compute_auc", "precision_recall_ndcg_at_k"]

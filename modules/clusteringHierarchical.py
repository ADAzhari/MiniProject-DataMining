# modules/clusteringHierarchical_manual.py
# Full manual hierarchical agglomerative clustering (no SciPy/sklearn)
# Supports linkage methods: "single", "complete", "average"
# Returns linkage matrix and labels; can print step-by-step merges.

import numpy as np
from math import sqrt
from typing import List, Tuple
import matplotlib.pyplot as plt


def pairwise_euclidean(X: np.ndarray) -> np.ndarray:
    """
    Compute full pairwise distance matrix (n x n) using Euclidean distance.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(X[i] - X[j])
            D[i, j] = d
            D[j, i] = d
    return D


def _cluster_distance(cluster_a: List[int], cluster_b: List[int],
                      point_dist_matrix: np.ndarray,
                      method: str) -> float:
    """
    Compute distance between two clusters (lists of indices) using linkage method.
    method: "single" (min), "complete" (max), "average" (mean)
    """
    # gather pairwise distances between members
    dists = []
    for i in cluster_a:
        for j in cluster_b:
            dists.append(point_dist_matrix[i, j])
    if not dists:
        return np.inf
    if method == "single":
        return float(np.min(dists))
    elif method == "complete":
        return float(np.max(dists))
    elif method == "average":
        return float(np.mean(dists))
    else:
        raise ValueError(f"Unsupported method '{method}'. Choose single/complete/average.")


def hierarchical_agglomerative(X: np.ndarray,
                               method: str = "average",
                               display_steps: bool = False) -> Tuple[np.ndarray, List[set]]:
    """
    Perform agglomerative hierarchical clustering manually.
    Args:
        X: (n_samples, n_features) numeric array
        method: "single", "complete", or "average"
        display_steps: if True, prints each merge with distances and sizes
    Returns:
        linkage: ndarray shape (n-1, 4) with rows [idx1, idx2, dist, new_size]
                 idx1/idx2 follow original-index convention:
                 - initial observations: 0..n-1
                 - newly formed clusters get ids n, n+1, ...
        clusters_final: list of sets of original indices after full merging (one set)
    Note: Complexity O(n^3) in worst case; suitable for small n (n <= ~300 recommended).
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if n == 0:
        return np.empty((0, 4)), []

    # initial clusters: each point is its own cluster
    clusters = [[i] for i in range(n)]
    # track cluster ids (the ids used in linkage rows)
    cluster_ids = [i for i in range(n)]
    next_cluster_id = n

    # compute pointwise pairwise distances once
    point_dist = pairwise_euclidean(X)

    linkage_rows = []

    step = 0
    while len(clusters) > 1:
        step += 1
        # find closest pair of clusters
        best_pair = (None, None)
        best_dist = np.inf

        m = len(clusters)
        # brute-force check all cluster pairs
        for i in range(m):
            for j in range(i + 1, m):
                d = _cluster_distance(clusters[i], clusters[j], point_dist, method)
                if d < best_dist:
                    best_dist = d
                    best_pair = (i, j)

        i, j = best_pair
        # cluster ids for reporting
        id_i = cluster_ids[i]
        id_j = cluster_ids[j]

        # new cluster = union of two clusters
        new_cluster = clusters[i] + clusters[j]
        new_size = len(new_cluster)

        # record linkage row: [id_i, id_j, dist, new_size]
        linkage_rows.append([id_i, id_j, best_dist, new_size])

        if display_steps:
            print(f"Step {step}: merge clusters {id_i} & {id_j} | dist={best_dist:.6f} | size={new_size}")

        # replace clusters: remove j then i (j>i) to keep indices valid, then append new cluster and id
        # ensure we remove higher index first
        if j > i:
            del clusters[j]
            del clusters[i]
            del cluster_ids[j]
            del cluster_ids[i]
        else:
            del clusters[i]
            del clusters[j]
            del cluster_ids[i]
            del cluster_ids[j]

        clusters.append(new_cluster)
        cluster_ids.append(next_cluster_id)
        next_cluster_id += 1

    linkage = np.array(linkage_rows, dtype=float)
    return linkage, clusters  # clusters will be a list with single set representing all merged indices


def cut_tree(linkage: np.ndarray, n_samples: int, k: int) -> np.ndarray:
    """
    Given linkage matrix (as returned by hierarchical_agglomerative) and original n_samples,
    cut to produce k clusters. Returns labels array of length n_samples with integers 0..k-1.
    Approach (manual):
      - Start with each original sample as its own set
      - Apply merges sequentially from linkage rows until number of sets == k
      - Assign cluster labels by index order of remaining sets
    """
    if k < 1 or k > n_samples:
        raise ValueError("k must satisfy 1 <= k <= n_samples")

    # start sets: list of sets of original indices
    sets = [{i} for i in range(n_samples)]

    # mapping from cluster id to set index is not needed here; linkage rows give ids in terms of original/new ids
    # We will process rows in order, but we need to find which current sets contain the ids mentioned in linkage.
    # Simpler approach: maintain a dict mapping cluster_id -> set_of_indices
    cluster_map = {i: {i} for i in range(n_samples)}
    next_id = n_samples

    for row in linkage:
        id1, id2, dist, size = row
        id1 = int(id1); id2 = int(id2)
        # get sets
        set1 = cluster_map.pop(id1)
        set2 = cluster_map.pop(id2)
        merged = set1.union(set2)
        cluster_map[next_id] = merged
        next_id += 1

        # check current number of top-level clusters = number of entries in cluster_map
        if len(cluster_map) == k:
            break

    # at this point cluster_map contains k clusters (as sets of original indices)
    clusters = list(cluster_map.values())

    # create labels array
    labels = np.empty(n_samples, dtype=int)
    for label_idx, s in enumerate(clusters):
        for idx in s:
            labels[idx] = label_idx

    return labels


# Helper: convenience wrapper to get labels + linkage and optionally print steps
def hierarchical_manual(X: np.ndarray, method: str = "average", k: int = 2, display_steps: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full wrapper doing:
      - compute linkage manually
      - optionally show steps
      - cut linkage to k clusters and return labels
    Returns:
      labels (n_samples,) and linkage (n-1,4)
    """
    linkage, _ = hierarchical_agglomerative(X, method=method, display_steps=display_steps)
    labels = cut_tree(linkage, n_samples=X.shape[0], k=k)
    return labels, linkage

def plot_dendrogram_manual(linkage: np.ndarray, labels=None, title="Dendrogram Manual", figsize=(8, 5)):
    """
    Membuat visualisasi dendrogram sederhana dari linkage matrix manual.
    Tidak menggunakan scipy.hierarchy.
    """
    if linkage is None or len(linkage) == 0:
        print("Linkage kosong, tidak bisa plot dendrogram.")
        return

    n_samples = int(np.max(linkage[:, :2])) + 1
    cluster_positions = {i: i for i in range(n_samples)}  # posisi x setiap titik awal
    cluster_heights = {i: 0 for i in range(n_samples)}    # tinggi awal semua nol
    next_cluster_id = n_samples

    fig, ax = plt.subplots(figsize=figsize)
    for row in linkage:
        id1, id2, dist, size = row
        id1, id2 = int(id1), int(id2)

        x1, x2 = cluster_positions[id1], cluster_positions[id2]
        h1, h2 = cluster_heights[id1], cluster_heights[id2]

        # posisi x cluster baru = rata-rata dua x lama
        new_x = (x1 + x2) / 2
        cluster_positions[next_cluster_id] = new_x
        cluster_heights[next_cluster_id] = dist

        # Gambar garis vertikal dan horizontal
        ax.plot([x1, x1], [h1, dist], c='gray')
        ax.plot([x2, x2], [h2, dist], c='gray')
        ax.plot([x1, x2], [dist, dist], c='gray')

        next_cluster_id += 1

    ax.set_xlabel("Sampel")
    ax.set_ylabel("Jarak")
    ax.set_title(title)
    ax.invert_yaxis()  # biar akar di bawah seperti dendrogram normal
    if labels is not None:
        ax.set_xticks(range(n_samples))
        ax.set_xticklabels(labels, rotation=90)
    plt.tight_layout()
    return fig

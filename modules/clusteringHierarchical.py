# modules/clusteringHierarchical.py
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import streamlit as st


def get_linkage_recommendations():
    """
    Memberikan rekomendasi metode linkage berdasarkan karakteristik data
    """
    recommendations = {
        "single": {
            "description": "Single Linkage (Minimum)",
            "use_case": "Data dengan cluster yang memanjang atau tidak berbentuk bulat",
            "pros": "Dapat mendeteksi cluster dengan bentuk tidak teratur",
            "cons": "Sensitif terhadap outlier dan noise, cenderung membuat chain effect",
        },
        "complete": {
            "description": "Complete Linkage (Maximum)",
            "use_case": "Data dengan cluster yang kompak dan terpisah jelas",
            "pros": "Menghasilkan cluster yang kompak dan seimbang",
            "cons": "Kurang baik untuk cluster dengan bentuk tidak teratur",
        },
        "average": {
            "description": "Average Linkage (UPGMA)",
            "use_case": "Pilihan umum yang seimbang untuk berbagai jenis data",
            "pros": "Seimbang antara single dan complete, lebih stabil",
            "cons": "Mungkin tidak optimal untuk data dengan karakteristik khusus",
        },
    }
    return recommendations


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


def _cluster_distance(
    cluster_a: List[int],
    cluster_b: List[int],
    point_dist_matrix: np.ndarray,
    method: str,
) -> float:
    """
    Compute distance between two clusters (lists of indices) using linkage method.
    """
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
        raise ValueError(
            f"Unsupported method '{method}'. Choose single/complete/average."
        )


def hierarchical_agglomerative(
    X: np.ndarray, method: str = "average", display_steps: bool = False
) -> Tuple[np.ndarray, List[set]]:
    """
    Perform agglomerative hierarchical clustering manually.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if n == 0:
        return np.empty((0, 4)), []

    clusters = [[i] for i in range(n)]
    cluster_ids = [i for i in range(n)]
    next_cluster_id = n
    point_dist = pairwise_euclidean(X)
    linkage_rows = []

    while len(clusters) > 1:
        best_pair = (None, None)
        best_dist = np.inf
        m = len(clusters)

        for i in range(m):
            for j in range(i + 1, m):
                d = _cluster_distance(clusters[i], clusters[j], point_dist, method)
                if d < best_dist:
                    best_dist = d
                    best_pair = (i, j)

        # ==== PERBAIKAN: Cek jika tidak ada pasangan yang ditemukan ====
        if best_pair == (None, None):
            raise RuntimeError(
                "Tidak dapat menemukan pasangan klaster untuk digabungkan. "
                "Ini bisa terjadi jika ada nilai NaN pada data setelah normalisasi "
                "(misal: ada kolom data dengan nilai yang sama semua)."
            )
        # ============================================================

        i, j = best_pair
        id_i = cluster_ids[i]
        id_j = cluster_ids[j]

        new_cluster = clusters[i] + clusters[j]
        new_size = len(new_cluster)
        linkage_rows.append([id_i, id_j, best_dist, new_size])

        if display_steps:
            print(
                f"Step {len(linkage_rows)}: merge clusters {id_i} & {id_j} | dist={best_dist:.6f} | size={new_size}"
            )

        if j > i:
            del clusters[j], clusters[i]
            del cluster_ids[j], cluster_ids[i]
        else:
            del clusters[i], clusters[j]
            del cluster_ids[i], cluster_ids[j]

        clusters.append(new_cluster)
        cluster_ids.append(next_cluster_id)
        next_cluster_id += 1

    linkage = np.array(linkage_rows, dtype=float)
    return linkage, clusters


def cut_tree(linkage: np.ndarray, n_samples: int, k: int) -> np.ndarray:
    """
    Cut linkage matrix to produce k clusters.
    """
    if k < 1 or k > n_samples:
        raise ValueError("k must satisfy 1 <= k <= n_samples")

    # ==== PERBAIKAN: Logika yang lebih efisien dan jelas ====
    # Untuk mendapatkan k cluster, kita perlu melakukan (n_samples - k) penggabungan.
    num_merges = n_samples - k

    cluster_map = {i: {i} for i in range(n_samples)}
    next_id = n_samples

    # Lakukan merge sebanyak num_merges
    for i in range(num_merges):
        row = linkage[i]
        id1, id2, _, _ = row
        id1, id2 = int(id1), int(id2)

        set1 = cluster_map.pop(id1)
        set2 = cluster_map.pop(id2)
        merged = set1.union(set2)
        cluster_map[next_id] = merged
        next_id += 1
    # ========================================================

    clusters = list(cluster_map.values())
    labels = np.empty(n_samples, dtype=int)
    for label_idx, s in enumerate(clusters):
        for idx in s:
            labels[idx] = label_idx

    return labels


def hierarchical_manual(
    X: np.ndarray, method: str = "average", k: int = 2, display_steps: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full wrapper for manual hierarchical clustering.
    """
    linkage, _ = hierarchical_agglomerative(
        X, method=method, display_steps=display_steps
    )
    if linkage.shape[0] == 0:
        return np.zeros(X.shape[0], dtype=int), np.empty((0, 4))
    labels = cut_tree(linkage, n_samples=X.shape[0], k=k)
    return labels, linkage


def plot_dendrogram_manual(
    linkage: np.ndarray, labels=None, title="Dendrogram", figsize=(10, 6)
):
    """
    Membuat visualisasi dendrogram dari linkage matrix manual.
    """
    if linkage is None or len(linkage) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "Linkage kosong, tidak bisa plot dendrogram.",
            ha="center",
            va="center",
        )
        return fig

    n_samples = int(linkage[:, :2].max()) + 1
    cluster_positions = {i: (i, 0) for i in range(n_samples)}  # (posisi_x, tinggi)
    next_cluster_id = n_samples

    fig, ax = plt.subplots(figsize=figsize)

    for row in linkage:
        id1, id2, dist, _ = row
        id1, id2 = int(id1), int(id2)

        x1, h1 = cluster_positions[id1]
        x2, h2 = cluster_positions[id2]

        new_x = (x1 + x2) / 2

        ax.plot([x1, x1], [h1, dist], c="C0")
        ax.plot([x2, x2], [h2, dist], c="C0")
        ax.plot([x1, x2], [dist, dist], c="C0")

        cluster_positions[next_cluster_id] = (new_x, dist)
        next_cluster_id += 1

    ax.set_xlabel("Indeks Sampel")
    ax.set_ylabel("Jarak")
    ax.set_title(title)

    if labels is not None:
        ax.set_xticks(range(n_samples))
        ax.set_xticklabels(labels, rotation=90)

    plt.tight_layout()
    return fig

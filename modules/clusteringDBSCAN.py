# modules/clusteringDBSCAN.py
# Implementasi manual algoritma DBSCAN

import numpy as np
from typing import List, Tuple
import streamlit as st


def get_dbscan_parameter_suggestions(X: np.ndarray) -> dict:
    """
    Memberikan saran parameter untuk DBSCAN berdasarkan karakteristik data
    """
    n_samples, n_features = X.shape

    # Hitung jarak rata-rata ke k nearest neighbors
    distances = []
    k = min(4, n_samples - 1) 

    for i in range(n_samples):
        dists_from_i = []
        for j in range(n_samples):
            if i != j:
                dist = np.linalg.norm(X[i] - X[j])
                dists_from_i.append(dist)
        dists_from_i.sort()
        if len(dists_from_i) >= k:
            distances.append(dists_from_i[k - 1]) 

    distances = np.array(distances)
    distances.sort()

    # Estimasi eps berdasarkan knee point (75th percentile sebagai pendekatan)
    suggested_eps = np.percentile(distances, 75)

    # Estimasi min_samples berdasarkan dimensi dan ukuran data
    if n_features <= 2:
        suggested_min_samples = 4
    elif n_features <= 5:
        suggested_min_samples = 2 * n_features
    else:
        suggested_min_samples = 2 * n_features

    # Batasi min_samples berdasarkan ukuran dataset
    suggested_min_samples = min(suggested_min_samples, max(3, n_samples // 10))

    suggestions = {
        "eps_range": {
            "min": round(suggested_eps * 0.5, 2),
            "suggested": round(suggested_eps, 2),
            "max": round(suggested_eps * 1.5, 2),
        },
        "min_samples_range": {
            "min": max(2, suggested_min_samples // 2),
            "suggested": suggested_min_samples,
            "max": suggested_min_samples * 2,
        },
        "explanation": {
            "eps": f"Berdasarkan analisis {k}-nearest neighbors, eps optimal sekitar {suggested_eps:.2f}",
            "min_samples": f"Untuk {n_features} dimensi, min_samples disarankan {suggested_min_samples}",
        },
    }

    return suggestions


def _get_neighbors(X: np.ndarray, point_index: int, eps: float) -> List[int]:
    """
    Mencari semua tetangga dari sebuah titik dalam radius epsilon (eps).
    Menggunakan jarak Euclidean.
    """
    neighbors = []
    p1 = X[point_index]
    for i in range(len(X)):
        p2 = X[i]
        if np.linalg.norm(p1 - p2) < eps:
            neighbors.append(i)
    return neighbors


def dbscan_manual(X: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """
    Melakukan clustering DBSCAN secara manual.

    Args:
        X (np.ndarray): Array data (n_samples, n_features).
        eps (float): Jarak maksimum untuk dianggap sebagai tetangga.
        min_samples (int): Jumlah minimum sampel untuk membentuk core point.

    Returns:
        np.ndarray: Array label untuk setiap titik data. Noise diberi label -1.
    """
    n_samples = X.shape[0]
    # Inisialisasi label: 0 = belum dikunjungi, -1 = noise
    labels = np.zeros(n_samples, dtype=int)
    cluster_id = 0

    for i in range(n_samples):
        # Lewati titik yang sudah menjadi bagian dari cluster lain
        if labels[i] != 0:
            continue

        # Temukan tetangga
        neighbors = _get_neighbors(X, i, eps)

        # Jika jumlah tetangga kurang dari min_samples, tandai sebagai noise (sementara)
        if len(neighbors) < min_samples:
            labels[i] = -1
            continue

        # Jika cukup tetangga, ini adalah core point. Mulai cluster baru.
        cluster_id += 1
        labels[i] = cluster_id

        # Gunakan 'seeds' sebagai antrian untuk memperluas cluster
        seeds = list(neighbors)

        head = 0
        while head < len(seeds):
            current_point_idx = seeds[head]
            head += 1

            # Jika titik ini sebelumnya ditandai noise, sekarang ia adalah border point
            if labels[current_point_idx] == -1:
                labels[current_point_idx] = cluster_id

            # Jika titik ini belum dikunjungi, kunjungi dan proses
            if labels[current_point_idx] == 0:
                labels[current_point_idx] = cluster_id

                # Temukan tetangganya
                current_neighbors = _get_neighbors(X, current_point_idx, eps)

                # Jika titik ini juga core point, tambahkan tetangganya ke antrian
                if len(current_neighbors) >= min_samples:
                    seeds.extend(current_neighbors)

    return labels

# modules/clusteringDBSCAN.py
# Implementasi manual algoritma DBSCAN

import numpy as np
from typing import List

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
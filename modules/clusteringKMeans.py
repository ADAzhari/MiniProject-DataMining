import numpy as np


def kmeans(X, k=3, max_iters=100):
    np.random.seed(42)
    if len(X) < k:
        raise ValueError("Jumlah data lebih sedikit dari jumlah cluster.")

    centroids = X[np.random.choice(len(X), size=k, replace=False)]

    for _ in range(max_iters):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array(
            [
                (
                    X[labels == j].mean(axis=0)
                    if np.any(labels == j)
                    else X[np.random.choice(len(X))]
                )
                for j in range(k)
            ]
        )

        if np.allclose(centroids, new_centroids, atol=1e-4):
            break
        centroids = new_centroids

    return labels, centroids


def silhouette_score(X, labels):
    n = len(X)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0

    sil_scores = []
    for i in range(n):
        same_cluster = X[labels == labels[i]]
        other_clusters = [X[labels == l] for l in unique_labels if l != labels[i]]

        # rata-rata jarak ke titik lain di cluster yang sama
        a = (
            np.mean(
                [
                    np.linalg.norm(X[i] - p)
                    for p in same_cluster
                    if not np.array_equal(p, X[i])
                ]
            )
            if len(same_cluster) > 1
            else 0
        )

        # jarak minimum ke cluster lain
        b_vals = [
            np.mean([np.linalg.norm(X[i] - p) for p in c])
            for c in other_clusters
            if len(c) > 0
        ]
        b = min(b_vals) if b_vals else 0

        s = (b - a) / max(a, b) if max(a, b) > 0 else 0
        sil_scores.append(s)

    return float(np.nanmean(sil_scores))


def find_best_k(X, min_k=2, max_k=10):
    best_k, best_score, scores = min_k, -1, {}
    for k in range(min_k, max_k + 1):
        try:
            labels, _ = kmeans(X, k)
            score = silhouette_score(X, labels)
            scores[k] = score
            print(f"k={k}: Silhouette Score = {score:.4f}")
            if score > best_score:
                best_score, best_k = score, k
        except Exception as e:
            print(f"⚠️ k={k} gagal: {e}")

    if best_score < 0:
        best_k, best_score = 2, 0
        print("⚠️ Tidak ada k valid, default ke k=2.")

    return best_k, best_score, scores


def calculate_wcss(X, k):
    """
    Menghitung Within-Cluster Sum of Squares (WCSS) untuk elbow method
    """
    labels, centroids = kmeans(X, k)
    wcss = 0
    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            wcss += np.sum((cluster_points - centroids[i]) ** 2)
    return wcss


def elbow_method(X, min_k=1, max_k=10):
    """
    Implementasi Elbow Method untuk menentukan K optimal
    """
    wcss_values = []
    k_range = range(min_k, max_k + 1)

    for k in k_range:
        try:
            if k == 1:
                # Untuk k=1, WCSS adalah total varians dari mean
                mean_point = np.mean(X, axis=0)
                wcss = np.sum((X - mean_point) ** 2)
            else:
                wcss = calculate_wcss(X, k)
            wcss_values.append(wcss)
            print(f"k={k}: WCSS = {wcss:.4f}")
        except Exception as e:
            print(f"⚠️ k={k} gagal: {e}")
            wcss_values.append(0)

    return list(k_range), wcss_values


def get_cluster_summary(X, labels, feature_names=None):
    """
    Membuat ringkasan rata-rata nilai per fitur per cluster
    """
    import pandas as pd

    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]

    # Buat DataFrame dari data dan labels
    df = pd.DataFrame(X, columns=feature_names)
    df["Cluster"] = labels

    # Hitung rata-rata per cluster
    cluster_summary = df.groupby("Cluster").mean()

    # Tambahkan jumlah anggota per cluster
    cluster_counts = df["Cluster"].value_counts().sort_index()
    cluster_summary["Member_Count"] = cluster_counts

    return cluster_summary

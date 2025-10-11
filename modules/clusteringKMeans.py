import numpy as np

def kmeans(X, k=3, max_iters=100):
    np.random.seed(42)
    if len(X) < k:
        raise ValueError("Jumlah data lebih sedikit dari jumlah cluster.")

    centroids = X[np.random.choice(len(X), size=k, replace=False)]

    for _ in range(max_iters):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([
            X[labels == j].mean(axis=0) if np.any(labels == j)
            else X[np.random.choice(len(X))]
            for j in range(k)
        ])

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
        a = np.mean([np.linalg.norm(X[i] - p)
                     for p in same_cluster if not np.array_equal(p, X[i])]) if len(same_cluster) > 1 else 0

        # jarak minimum ke cluster lain
        b_vals = [np.mean([np.linalg.norm(X[i] - p) for p in c]) for c in other_clusters if len(c) > 0]
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

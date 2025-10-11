import numpy as np

def pca_manual(X, n_components=2):
    X = np.array(X, dtype=float)

    if np.isnan(X).any():
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])

    X_centered = X - np.mean(X, axis=0)

    variances = np.var(X_centered, axis=0)
    non_constant_cols = variances > 1e-10
    X_centered = X_centered[:, non_constant_cols]

    cov_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_idx]
    components = eigenvectors[:, :n_components]
    X_reduced = np.dot(X_centered, components)

    # 🔹 Return juga kolom mana yang dipakai
    return X_reduced, eigenvectors, non_constant_cols

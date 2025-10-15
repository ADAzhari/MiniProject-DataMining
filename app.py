import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from io import BytesIO

# ===== IMPORT MODULES =====
from modules.utils import read_csv
from modules.preprocessing import clean_data, transform_data, normalize_data
from modules.clusteringKMeans import kmeans, find_best_k
from modules.pca import pca_manual
from modules.clusteringHierarchical import hierarchical_manual, plot_dendrogram_manual
from modules.clusteringDBSCAN import dbscan_manual # <-- IMPORT BARU

# ===== STREAMLIT UI =====
st.title("🧭 Analisis Motivasi Belajar Mahasiswa")

uploaded_file = st.file_uploader("📤 Upload file CSV hasil kuesioner:", type="csv")

if uploaded_file:
    try:
        rows = read_csv(uploaded_file)
        drop_cols = ["timestamp"]
        rows = clean_data(rows, drop_cols)

        # === Load keyword dari file JSON ===
        try:
            with open("config/keyword.json", "r", encoding="utf-8") as f:
                keywords_dict = json.load(f)
            st.success("✅ Keyword custom dari JSON digunakan.")
        except FileNotFoundError:
            st.warning("⚠️ Keyword JSON tidak ditemukan, pakai default.")
            keywords_dict = {
                "karir": ["kerja", "karier", "pekerjaan", "masa depan"],
                "wawasan": ["ilmu", "pengetahuan", "wawasan", "belajar hal baru"],
                "prestasi": ["nilai", "prestasi", "target akademik"],
                "motivasi_diri": ["semangat", "diri sendiri", "pengembangan diri"],
                "tugas": ["kewajiban", "tugas/ujian", "tugas", "kuliah"]
            }

        df_rows = transform_data(rows, keywords_dict)

# === Tambahkan tahap normalisasi ===
        df_rows = normalize_data(df_rows, method="minmax")

        df = pd.DataFrame(df_rows)
        df_original = df.copy()

        # === BUAT TAB MENU === # <-- TAMBAH TAB 4
        tab1, tab2, tab3, tab4 = st.tabs([
            "🧹 Preprocessing", 
            "📊 Clustering K-Means", 
            "🌳 Hierarchical", 
            "🔬 Clustering DBSCAN"
        ])

        # ---------------- TAB 1: PREPROCESSING ----------------
        with tab1:
            st.subheader("🧹 Hasil Preprocessing Data")
            st.dataframe(df_original.head())
            buffer_pre = BytesIO()
            df_original.to_csv(buffer_pre, index=False, encoding="utf-8-sig", float_format="%.6f")
            buffer_pre.seek(0)
            st.download_button("⬇️ Download Hasil Preprocessing (CSV)", buffer_pre, "hasil_preprocessing.csv", "text/csv")

        # (Kode untuk Tab 2 K-Means dan Tab 3 Hierarchical tetap sama seperti sebelumnya)
        # ---------------- TAB 2: K-MEANS ----------------
        with tab2:
            st.subheader("📊 Hasil Clustering (K-Means)")
            df_kmeans = df_original.copy()
            exclude_cols = ["apakah kamu siap meluangkan waktu sekitar 5 menit untuk mengisi kuesioner ini?", "nama (opsional)", "nim (opsional)", "apa alasan utama kamu belajar?"]
            df_filtered = df_kmeans.drop(columns=[c for c in exclude_cols if c in df_kmeans.columns], errors="ignore")
            numeric_df = df_filtered.select_dtypes(include=[float, int])
            if numeric_df.empty:
                st.error("❌ Tidak ada kolom numerik untuk clustering.")
            else:
                X = numeric_df.values.astype(float)
                with st.spinner("⏳ Sedang mencari K optimal..."):
                    best_k, best_score, scores = find_best_k(X)
                st.subheader("📈 Evaluasi Silhouette Score")
                st.line_chart(pd.DataFrame({"Silhouette Score": scores.values()}, index=scores.keys()))
                st.write(f"**K optimal:** {best_k}  (Silhouette Score = {best_score:.4f})")
                labels, centroids = kmeans(X, k=best_k)
                df_kmeans["Cluster"] = labels
                st.bar_chart(df_kmeans["Cluster"].value_counts().sort_index())
                X_reduced, eigenvectors, mask = pca_manual(X, n_components=2)
                centroids_filtered = centroids[:, mask]
                mean_filtered = np.mean(X[:, mask], axis=0)
                centroids_reduced = np.dot(centroids_filtered - mean_filtered, eigenvectors[:, :2])
                fig, ax = plt.subplots(figsize=(6, 6))
                scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap="viridis", alpha=0.7)
                ax.scatter(centroids_reduced[:, 0], centroids_reduced[:, 1], c="red", marker="X", s=200, label="Centroid")
                ax.set_title(f"Visualisasi Klaster (k={best_k})")
                ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2"); ax.legend()
                st.pyplot(fig)
                buffer = BytesIO()
                df_kmeans.to_csv(buffer, index=False, encoding="utf-8-sig", float_format="%.6f")
                buffer.seek(0)
                st.download_button("⬇️ Download Hasil Clustering (CSV)", buffer, "hasil_clustering_kmeans.csv", "text/csv")

        # ---------------- TAB 3: HIERARCHICAL ----------------
        with tab3:
            st.subheader("🌳 Hierarchical Clustering")
            df_hier = df_original.copy()
            exclude_cols = ["apakah kamu siap meluangkan waktu sekitar 5 menit untuk mengisi kuesioner ini?", "nama (opsional)", "nim (opsional)", "apa alasan utama kamu belajar?"]
            df_filtered = df_hier.drop(columns=[c for c in exclude_cols if c in df_hier.columns], errors="ignore")
            numeric_df = df_filtered.select_dtypes(include=[float, int])
            if numeric_df.empty:
                st.error("❌ Tidak ada kolom numerik untuk clustering.")
            else:
                X = numeric_df.values.astype(float)
                data_range = X.max(axis=0) - X.min(axis=0)
                data_range[data_range == 0] = 1
                X_norm = (X - X.min(axis=0)) / data_range
                method = st.selectbox("Pilih metode linkage:", ["single", "complete", "average"])
                k_hier = st.slider("Jumlah cluster (k)", 2, 10, 3)
                with st.spinner("⏳ Menghitung Hierarchical Clustering..."):
                    labels_hier, linkage = hierarchical_manual(X_norm, method=method, k=k_hier)
                    df_hier["Cluster_Hierarchical"] = labels_hier
                st.success("✅ Proses Hierarchical Clustering selesai!")
                st.bar_chart(df_hier["Cluster_Hierarchical"].value_counts().sort_index())
                st.subheader("📈 Dendrogram")
                if X_norm.shape[0] > 50:
                    st.warning("⚠️ Jumlah sampel > 50. Plot dendrogram mungkin lambat dan sulit dibaca.")
                fig = plot_dendrogram_manual(linkage, title=f"Dendrogram (Linkage: {method})")
                st.pyplot(fig)
                st.subheader("💾 Simpan Hasil")
                buffer_hier = BytesIO()
                df_hier.to_csv(buffer_hier, index=False, encoding="utf-8-sig", float_format="%.6f")
                buffer_hier.seek(0)
                st.download_button("⬇️ Download hasil clustering (CSV)", buffer_hier, f"hasil_clustering_hierarchical_{method}.csv", "text/csv")

        # =========================================================
        # TAB 4 — DBSCAN (BARU)
        # =========================================================
        with tab4:
            st.subheader("🔬 Clustering dengan DBSCAN")
            
            df_dbscan = df_original.copy()
            exclude_cols = ["apakah kamu siap meluangkan waktu sekitar 5 menit untuk mengisi kuesioner ini?", "nama (opsional)", "nim (opsional)", "apa alasan utama kamu belajar?"]
            df_filtered = df_dbscan.drop(columns=[c for c in exclude_cols if c in df_dbscan.columns], errors="ignore")
            numeric_df = df_filtered.select_dtypes(include=[float, int])

            if numeric_df.empty:
                st.error("❌ Tidak ada kolom numerik untuk clustering.")
            else:
                X = numeric_df.values.astype(float)
                
                # Normalisasi data sangat penting untuk DBSCAN
                data_range = X.max(axis=0) - X.min(axis=0)
                data_range[data_range == 0] = 1 
                X_norm = (X - X.min(axis=0)) / data_range

                st.info("DBSCAN sensitif terhadap parameter. Silakan sesuaikan **Epsilon** dan **Min Samples** untuk mendapatkan hasil terbaik.")
                
                # UI untuk parameter DBSCAN
                eps = st.slider("Epsilon (eps)", 0.1, 2.0, 0.8, 0.05, help="Jarak maksimum antara dua sampel untuk dianggap sebagai tetangga.")
                min_samples = st.slider("Min Samples", 2, 20, 5, 1, help="Jumlah minimum sampel dalam lingkungan eps agar sebuah titik dianggap titik inti (core point).")

                with st.spinner("⏳ Menjalankan DBSCAN..."):
                    labels_dbscan = dbscan_manual(X_norm, eps=eps, min_samples=min_samples)
                    df_dbscan["Cluster_DBSCAN"] = labels_dbscan
                
                st.success("✅ Proses DBSCAN selesai!")
                
                # Hitung hasil
                n_clusters = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
                n_noise = list(labels_dbscan).count(-1)
                
                st.write(f"**Jumlah klaster ditemukan:** **{n_clusters}**")
                st.write(f"**Jumlah titik noise (pencilan):** **{n_noise}**")

                st.write("Distribusi cluster (Label -1 adalah Noise):")
                st.bar_chart(df_dbscan["Cluster_DBSCAN"].value_counts().sort_index())

                # Visualisasi hasil dengan PCA
                st.subheader("📊 Visualisasi Klaster (PCA)")
                X_reduced, _, _ = pca_manual(X, n_components=2)
                
                fig, ax = plt.subplots(figsize=(6, 6))
                scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels_dbscan, cmap="viridis", alpha=0.7)
                ax.set_title(f"Visualisasi Klaster DBSCAN (eps={eps}, min_samples={min_samples})")
                ax.set_xlabel("PC 1")
                ax.set_ylabel("PC 2")
                
                # Membuat legenda manual
                unique_labels = set(labels_dbscan)
                handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {l}',
                                      markerfacecolor=plt.cm.viridis(i/len(unique_labels)), markersize=8)
                           for i, l in enumerate(sorted(unique_labels))]
                ax.legend(handles=handles, title="Clusters")

                st.pyplot(fig)

                # Tombol download
                st.subheader("💾 Simpan Hasil")
                buffer_dbscan = BytesIO()
                df_dbscan.to_csv(buffer_dbscan, index=False, encoding="utf-8-sig", float_format="%.6f")
                buffer_dbscan.seek(0)
                st.download_button(
                    "⬇️ Download hasil clustering DBSCAN (CSV)", 
                    buffer_dbscan, 
                    f"hasil_clustering_dbscan_eps{eps}_ms{min_samples}.csv", 
                    "text/csv"
                )

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan: {e}")
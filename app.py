import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from io import BytesIO

# ===== IMPORT MODULES =====
from modules.utils import read_csv
from modules.preprocessing import clean_data, transform_data
from modules.clusteringKMeans import kmeans, find_best_k
from modules.pca import pca_manual
from modules.clusteringHierarchical import hierarchical_manual, plot_dendrogram_manual

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
                "tugas": ["kewajiban", "tugas", "kuliah"]
            }

        # === Transformasi data ===
        df_rows = transform_data(rows, keywords_dict)
        df = pd.DataFrame(df_rows)

        # === BUAT TAB MENU ===
        tab1, tab2, tab3 = st.tabs(["🧹 Preprocessing", "📊 Clustering K-Means", "huerarchiak"])

        # ---------------- TAB 1: PREPROCESSING ----------------
        with tab1:
            st.subheader("🧹 Hasil Preprocessing Data")
            st.dataframe(df.head())

            buffer_pre = BytesIO()
            df.to_csv(buffer_pre, index=False, encoding="utf-8-sig")
            buffer_pre.seek(0)
            st.download_button(
                label="⬇️ Download Hasil Preprocessing (CSV)",
                data=buffer_pre,
                file_name="hasil_preprocessing.csv",
                mime="text/csv"
            )

        # ---------------- TAB 2: K-MEANS ----------------
        with tab2:
            st.subheader("📊 Hasil Clustering (K-Means)")

            exclude_cols = [
                "apakah kamu siap meluangkan waktu sekitar 5 menit untuk mengisi kuesioner ini?",
                "nama (opsional)",
                "nim (opsional)",
                "apa alasan utama kamu belajar?"
            ]

            df_filtered = df.drop(columns=[c for c in exclude_cols if c in df.columns], errors="ignore")
            numeric_df = df_filtered.select_dtypes(include=[float, int])

            st.write("Kolom yang digunakan untuk clustering:", list(numeric_df.columns))

            if numeric_df.empty:
                st.error("❌ Tidak ada kolom numerik untuk clustering.")
            else:
                X = numeric_df.values.astype(float)

                with st.spinner("⏳ Sedang mencari K optimal..."):
                    best_k, best_score, scores = find_best_k(X)

                st.subheader("📈 Evaluasi Silhouette Score")
                st.line_chart(pd.DataFrame({"Silhouette Score": scores.values()}, index=scores.keys()))
                st.write(f"**K optimal:** {best_k}  (Silhouette Score = {best_score:.4f})")

                # Jalankan clustering
                labels, centroids = kmeans(X, k=best_k)
                df["Cluster"] = labels
                st.bar_chart(df["Cluster"].value_counts().sort_index())

                # ====== PCA VISUALIZATION ======
                X_reduced, eigenvectors, mask = pca_manual(X, n_components=2)
                centroids_filtered = centroids[:, mask]
                mean_filtered = np.mean(X[:, mask], axis=0)
                centroids_reduced = np.dot(centroids_filtered - mean_filtered, eigenvectors[:, :2])

                fig, ax = plt.subplots(figsize=(6, 6))
                scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap="viridis", alpha=0.7)
                ax.scatter(centroids_reduced[:, 0], centroids_reduced[:, 1],
                           c="red", marker="X", s=200, label="Centroid")
                ax.set_title(f"Visualisasi Klaster (k={best_k})")
                ax.set_xlabel("PC 1")
                ax.set_ylabel("PC 2")
                ax.legend()
                st.pyplot(fig)

                # Simpan hasil clustering
                buffer = BytesIO()
                df.to_csv(buffer, index=False, encoding="utf-8-sig")
                buffer.seek(0)
                st.download_button(
                    label="⬇️ Download Hasil Clustering (CSV)",
                    data=buffer,
                    file_name="hasil_clustering_kmeans.csv",
                    mime="text/csv"
                )

# =========================================================
        # TAB 3 — HIERARCHICAL (baru, tapi pakai data & exclude yang sama)
        # =========================================================
        with tab3:
            st.subheader("🌳 Hierarchical Clustering (Manual)")

            exclude_cols = [
                "nama", "nim", "nama (opsional)", "nim (opsional)",
                "apa alasan utama kamu belajar?"
            ]
            df_filtered = df.drop(columns=exclude_cols, errors="ignore")
            numeric_df = df_filtered.select_dtypes(include=[float, int])

            if numeric_df.empty:
                st.error("❌ Tidak ada kolom numerik untuk clustering.")
            else:
                # Gunakan data & normalisasi yang sama
                X = numeric_df.values.astype(float)
                X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

                method = st.selectbox("Pilih metode linkage:", ["single", "complete", "average"])
                k_hier = st.slider("Jumlah cluster (k)", 2, 10, 2)

                with st.spinner("⏳ Menghitung Hierarchical Clustering..."):
                    labels_hier, linkage = hierarchical_manual(X, method=method, k=k_hier)
                    df["Cluster_Hier"] = labels_hier

                st.success("✅ Proses Hierarchical Clustering selesai!")

                st.write("Distribusi cluster:")
                st.bar_chart(df["Cluster_Hier"].value_counts().sort_index())

                # ====== Dendrogram Manual ======
                st.subheader("📈 Dendrogram Manual")
                fig = plot_dendrogram_manual(linkage, title=f"Dendrogram (Linkage: {method})")
                st.pyplot(fig)

                # ====== Simpan hasil ke CSV ======
                st.subheader("💾 Simpan Hasil Clustering Hierarchical")

                buffer_hier = BytesIO()
                df.to_csv(buffer_hier, index=False, encoding="utf-8-sig")
                buffer_hier.seek(0)

                st.download_button(
                    label="⬇️ Download hasil clustering (CSV)",
                    data=buffer_hier,
                    file_name="hasil_clustering_hierarchical.csv",
                    mime="text/csv"
                )
                st.info("Klik tombol di atas untuk mengunduh hasil clustering Hierarchical.")


    except Exception as e:
        st.error(f"❌ Terjadi kesalahan: {e}")

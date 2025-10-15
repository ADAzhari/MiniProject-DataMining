import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from io import BytesIO

# ===== IMPORT MODULES =====
from modules.utils import read_csv
from modules.preprocessing import clean_data, transform_data, normalize_data
from modules.clusteringKMeans import kmeans, find_best_k, elbow_method, get_cluster_summary
from modules.pca import pca_manual
from modules.clusteringHierarchical import hierarchical_manual, plot_dendrogram_manual, get_linkage_recommendations
from modules.clusteringDBSCAN import dbscan_manual, get_dbscan_parameter_suggestions

# ===== STREAMLIT CONFIG =====
st.set_page_config(
    page_title="Analisis Motivasi Belajar Mahasiswa",
    page_icon="🧭",
    layout="wide"
)

# ===== STREAMLIT UI =====
st.title("🧭 Analisis Motivasi Belajar Mahasiswa")
st.markdown("### Dashboard Analisis Clustering untuk Data Kuesioner Motivasi Belajar")
st.markdown("IDENTIFIKASI MAHASISWA BERDASARKAN MOTIVASI BELAJAR, DISTRAKSI DIGITAL, DAN PRODUKTIVITAS AKADEMIK")

st.markdown(
    """
    <div style='font-size:14px;'>
        <i>Kelompok 6 Data Mining 2025</i>
        <ul style='margin-top:4px;'>
            <li>Luthfi Hamam Arsyada - 140810230007</li>
            <li>Muhammad Ainur Rafiq N. - 140810230009</li>
            <li>Muhammad Ghazi Ichsan - 140810230029</li>
            <li>Achmad Dzaki Azhari - 140810230034</li>
            <li>Dzikri Bassyril Mu’minin - 140810230071</li>
            <br>
            <br>
        </ul>
    </div>
    """, unsafe_allow_html=True
)

uploaded_file = st.file_uploader("📤 Upload file CSV atau XLSX hasil kuesioner:", type=["csv", "xlsx"])




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
        df_rows = normalize_data(df_rows, method="minmax")

        df = pd.DataFrame(df_rows)
        df_original = df.copy()

        # === BUAT TAB MENU ===
        tab1, tab2, tab3, tab4 = st.tabs([
            "🧹 Preprocessing", 
            "📊 Clustering K-Means", 
            "🌳 Hierarchical Clustering", 
            "🔬 DBSCAN Clustering"
        ])

        # ---------------- TAB 1: PREPROCESSING ----------------
        with tab1:
            st.subheader("🧹 Hasil Preprocessing Data")
            
            st.write("### 📋 Sample Data Hasil Preprocessing")
            st.dataframe(df_original.head())
            
            buffer_pre = BytesIO()
            df_original.to_csv(buffer_pre, index=False, encoding="utf-8-sig", float_format="%.6f")
            buffer_pre.seek(0)
            st.download_button("⬇️ Download Hasil Preprocessing (CSV)", buffer_pre, "hasil_preprocessing.csv", "text/csv")

        # ---------------- TAB 2: K-MEANS ----------------
        with tab2:
            st.subheader("📊 Hasil Clustering K-Means")
            df_kmeans = df_original.copy()
            exclude_cols = ["apakah kamu siap meluangkan waktu sekitar 5 menit untuk mengisi kuesioner ini?", "nama (opsional)", "nim (opsional)", "apa alasan utama kamu belajar?"]
            df_filtered = df_kmeans.drop(columns=[c for c in exclude_cols if c in df_kmeans.columns], errors="ignore")
            numeric_df = df_filtered.select_dtypes(include=[float, int])
            
            if numeric_df.empty:
                st.error("❌ Tidak ada kolom numerik untuk clustering.")
            else:
                X = numeric_df.values.astype(float)
                
                # Elbow Method
                st.subheader("📈 Elbow Method untuk Menentukan K Optimal")
                with st.spinner("⏳ Menghitung Elbow Method..."):
                    k_range, wcss_values = elbow_method(X, min_k=1, max_k=10)
                
                # Plot Elbow Method
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(k_range, wcss_values, 'bo-', linewidth=2, markersize=8)
                    ax.set_xlabel('Jumlah Cluster (K)')
                    ax.set_ylabel('Within-Cluster Sum of Squares (WCSS)')
                    ax.set_title('Elbow Method untuk Optimal K')
                    ax.grid(True, alpha=0.3)
                    
                    # Tambahkan annotasi untuk setiap titik
                    for i, (k, wcss) in enumerate(zip(k_range, wcss_values)):
                        ax.annotate(f'K={k}\nWCSS={wcss:.1f}', 
                                   (k, wcss), 
                                   textcoords="offset points", 
                                   xytext=(0,10), 
                                   ha='center',
                                   fontsize=8,
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
                    
                    st.pyplot(fig)
                
                # Silhouette Analysis
                with col2:
                    st.subheader("📈 Silhouette Analysis")
                    with st.spinner("⏳ Sedang mencari K optimal..."):
                        best_k, best_score, scores = find_best_k(X)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    k_values = list(scores.keys())
                    score_values = list(scores.values())
                    ax.plot(k_values, score_values, 'ro-', linewidth=2, markersize=8)
                    ax.set_xlabel('Jumlah Cluster (K)')
                    ax.set_ylabel('Silhouette Score')
                    ax.set_title('Silhouette Analysis')
                    ax.grid(True, alpha=0.3)
                    ax.axvline(x=best_k, color='green', linestyle='--', alpha=0.7, label=f'Optimal K = {best_k}')
                    ax.legend()
                    st.pyplot(fig)
                
                st.write(f"**🎯 K optimal berdasarkan Silhouette Score:** {best_k}  (Score = {best_score:.4f})")
                
                # Clustering dengan K optimal
                labels, centroids = kmeans(X, k=best_k)
                df_kmeans["Cluster"] = labels
                
                # Distribusi cluster
                st.subheader("📊 Distribusi Cluster")
                col1, col2 = st.columns(2)
                
                with col1:
                    cluster_counts = df_kmeans["Cluster"].value_counts().sort_index()
                    fig, ax = plt.subplots(figsize=(8, 6))
                    cluster_counts.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
                    ax.set_title('Distribusi Jumlah Anggota per Cluster')
                    ax.set_xlabel('Cluster')
                    ax.set_ylabel('Jumlah Anggota')
                    plt.xticks(rotation=0)
                    
                    # Tambahkan nilai di atas bar
                    for i, v in enumerate(cluster_counts.values):
                        ax.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
                    
                    st.pyplot(fig)
                
                # Cluster Summary
                with col2:
                    st.write("**📋 Ringkasan Cluster:**")
                    feature_names = numeric_df.columns.tolist()
                    cluster_summary = get_cluster_summary(X, labels, feature_names)
                    st.dataframe(cluster_summary, use_container_width=True)
                
                # Visualisasi PCA
                st.subheader("🎯 Visualisasi Cluster (PCA)")
                X_reduced, eigenvectors, mask = pca_manual(X, n_components=2)
                centroids_filtered = centroids[:, mask]
                mean_filtered = np.mean(X[:, mask], axis=0)
                centroids_reduced = np.dot(centroids_filtered - mean_filtered, eigenvectors[:, :2])
                
                fig, ax = plt.subplots(figsize=(12, 8))
                scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap="viridis", alpha=0.7, s=50)
                ax.scatter(centroids_reduced[:, 0], centroids_reduced[:, 1], c="red", marker="X", s=200, label="Centroid", edgecolors='black', linewidth=2)
                ax.set_title(f"Visualisasi Klaster K-Means (k={best_k})")
                ax.set_xlabel("Principal Component 1")
                ax.set_ylabel("Principal Component 2")
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Tambahkan colorbar
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Cluster')
                
                st.pyplot(fig)
                
                # Interpretasi cluster
                st.subheader("🔍 Interpretasi Cluster")
                for cluster_id in range(best_k):
                    cluster_data = cluster_summary.loc[cluster_id]
                    st.write(f"**Cluster {cluster_id}** ({int(cluster_data['Member_Count'])} anggota):")
                    
                    # Analisis fitur tertinggi dan terendah
                    feature_values = cluster_data.drop('Member_Count')
                    top_features = feature_values.nlargest(3)
                    bottom_features = feature_values.nsmallest(3)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("🔥 **Fitur Tertinggi:**")
                        for feature, value in top_features.items():
                            st.write(f"- {feature}: {value:.3f}")
                    
                    with col2:
                        st.write("❄️ **Fitur Terendah:**")
                        for feature, value in bottom_features.items():
                            st.write(f"- {feature}: {value:.3f}")
                    
                    st.markdown("---")
                
                buffer = BytesIO()
                df_kmeans.to_csv(buffer, index=False, encoding="utf-8-sig", float_format="%.6f")
                buffer.seek(0)
                st.download_button("⬇️ Download Hasil Clustering K-Means (CSV)", buffer, "hasil_clustering_kmeans.csv", "text/csv")

        # ---------------- TAB 3: HIERARCHICAL ----------------
        with tab3:
            st.subheader("🌳 Hierarchical Clustering")
            
            # Tampilkan rekomendasi linkage
            st.subheader("📋 Rekomendasi Metode Linkage")
            recommendations = get_linkage_recommendations()
            
            col1, col2, col3 = st.columns(3)
            for i, (method, info) in enumerate(recommendations.items()):
                with [col1, col2, col3][i]:
                    st.info(f"**{info['description']}**")
                    st.write(f"**Kapan digunakan:** {info['use_case']}")
                    st.write(f"**Kelebihan:** {info['pros']}")
                    st.write(f"**Kekurangan:** {info['cons']}")
            
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
                
                col1, col2 = st.columns(2)
                with col1:
                    method = st.selectbox("Pilih metode linkage:", ["single", "complete", "average"], 
                                        help="Pilih berdasarkan rekomendasi di atas")
                with col2:
                    k_hier = st.slider("Jumlah cluster (k)", 2, 10, 3)
                
                with st.spinner("⏳ Menghitung Hierarchical Clustering..."):
                    labels_hier, linkage = hierarchical_manual(X_norm, method=method, k=k_hier)
                    df_hier["Cluster_Hierarchical"] = labels_hier
                
                st.success("✅ Proses Hierarchical Clustering selesai!")
                
                # Distribusi cluster
                cluster_counts = df_hier["Cluster_Hierarchical"].value_counts().sort_index()
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Distribusi Cluster")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    cluster_counts.plot(kind='bar', ax=ax, color='lightcoral', edgecolor='black')
                    ax.set_title(f'Distribusi Cluster ({method} linkage)')
                    ax.set_xlabel('Cluster')
                    ax.set_ylabel('Jumlah Anggota')
                    plt.xticks(rotation=0)
                    
                    # Tambahkan nilai di atas bar
                    for i, v in enumerate(cluster_counts.values):
                        ax.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
                    
                    st.pyplot(fig)
                
                with col2:
                    st.subheader("📋 Ringkasan Cluster")
                    feature_names = numeric_df.columns.tolist()
                    cluster_summary_hier = get_cluster_summary(X, labels_hier, feature_names)
                    st.dataframe(cluster_summary_hier, use_container_width=True)
                
                st.subheader("📈 Dendrogram")
                if X_norm.shape[0] > 50:
                    st.warning("⚠️ Jumlah sampel > 50. Plot dendrogram mungkin lambat dan sulit dibaca.")
                fig = plot_dendrogram_manual(linkage, title=f"Dendrogram (Linkage: {method})")
                st.pyplot(fig)
                
                st.subheader("💾 Simpan Hasil")
                buffer_hier = BytesIO()
                df_hier.to_csv(buffer_hier, index=False, encoding="utf-8-sig", float_format="%.6f")
                buffer_hier.seek(0)
                st.download_button("⬇️ Download hasil clustering Hierarchical (CSV)", buffer_hier, f"hasil_clustering_hierarchical_{method}.csv", "text/csv")

        # ---------------- TAB 4: DBSCAN ----------------
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

                # Parameter suggestions
                st.subheader("💡 Saran Parameter DBSCAN")
                suggestions = get_dbscan_parameter_suggestions(X_norm)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info("**🎯 Epsilon (eps)**")
                    st.write(f"**Range yang disarankan:** {suggestions['eps_range']['min']} - {suggestions['eps_range']['max']}")
                    st.write(f"**Nilai optimal:** {suggestions['eps_range']['suggested']}")
                    st.write(f"**Penjelasan:** {suggestions['explanation']['eps']}")
                
                with col2:
                    st.info("**👥 Min Samples**")
                    st.write(f"**Range yang disarankan:** {suggestions['min_samples_range']['min']} - {suggestions['min_samples_range']['max']}")
                    st.write(f"**Nilai optimal:** {suggestions['min_samples_range']['suggested']}")
                    st.write(f"**Penjelasan:** {suggestions['explanation']['min_samples']}")

                st.info("DBSCAN sensitif terhadap parameter. Gunakan saran di atas sebagai panduan awal, lalu sesuaikan berdasarkan hasil.")
                
                # UI untuk parameter DBSCAN dengan nilai default dari suggestions
                col1, col2 = st.columns(2)
                with col1:
                    eps = st.slider("Epsilon (eps)", 
                                  float(suggestions['eps_range']['min']), 
                                  float(suggestions['eps_range']['max']), 
                                  float(suggestions['eps_range']['suggested']), 
                                  0.01, 
                                  help="Jarak maksimum antara dua sampel untuk dianggap sebagai tetangga.")
                with col2:
                    min_samples = st.slider("Min Samples", 
                                           suggestions['min_samples_range']['min'], 
                                           suggestions['min_samples_range']['max'], 
                                           suggestions['min_samples_range']['suggested'], 
                                           1, 
                                           help="Jumlah minimum sampel dalam lingkungan eps agar sebuah titik dianggap titik inti (core point).")

                with st.spinner("⏳ Menjalankan DBSCAN..."):
                    labels_dbscan = dbscan_manual(X_norm, eps=eps, min_samples=min_samples)
                    df_dbscan["Cluster_DBSCAN"] = labels_dbscan
                
                st.success("✅ Proses DBSCAN selesai!")
                
                # Hitung hasil
                n_clusters = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
                n_noise = list(labels_dbscan).count(-1)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Jumlah Cluster", n_clusters)
                with col2:
                    st.metric("Titik Noise", n_noise)
                with col3:
                    st.metric("% Noise", f"{(n_noise/len(labels_dbscan)*100):.1f}%")

                # Distribusi cluster dan ringkasan
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Distribusi Cluster")
                    cluster_counts = df_dbscan["Cluster_DBSCAN"].value_counts().sort_index()
                    fig, ax = plt.subplots(figsize=(8, 6))
                    colors = ['red' if x == -1 else 'lightgreen' for x in cluster_counts.index]
                    cluster_counts.plot(kind='bar', ax=ax, color=colors, edgecolor='black')
                    ax.set_title('Distribusi Cluster DBSCAN\n(Merah = Noise)')
                    ax.set_xlabel('Cluster (-1 = Noise)')
                    ax.set_ylabel('Jumlah Anggota')
                    plt.xticks(rotation=0)
                    
                    # Tambahkan nilai di atas bar
                    for i, v in enumerate(cluster_counts.values):
                        ax.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
                    
                    st.pyplot(fig)
                
                with col2:
                    if n_clusters > 0:
                        st.subheader("📋 Ringkasan Cluster")
                        # Filter out noise points for summary
                        mask_no_noise = labels_dbscan != -1
                        if np.any(mask_no_noise):
                            X_no_noise = X[mask_no_noise]
                            labels_no_noise = labels_dbscan[mask_no_noise]
                            feature_names = numeric_df.columns.tolist()
                            cluster_summary_dbscan = get_cluster_summary(X_no_noise, labels_no_noise, feature_names)
                            st.dataframe(cluster_summary_dbscan, use_container_width=True)
                        else:
                            st.warning("Semua titik diklasifikasikan sebagai noise!")
                    else:
                        st.warning("Tidak ada cluster yang terbentuk. Coba ubah parameter eps dan min_samples.")

                # Visualisasi hasil dengan PCA
                st.subheader("📊 Visualisasi Klaster (PCA)")
                X_reduced, _, _ = pca_manual(X, n_components=2)
                
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Buat colormap khusus untuk DBSCAN (noise = hitam)
                unique_labels = set(labels_dbscan)
                colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
                
                for k, col in zip(unique_labels, colors):
                    if k == -1:
                        # Noise points
                        col = 'black'
                        marker = 'x'
                        size = 30
                        alpha = 0.5
                        label = 'Noise'
                    else:
                        marker = 'o'
                        size = 50
                        alpha = 0.7
                        label = f'Cluster {k}'
                    
                    class_member_mask = (labels_dbscan == k)
                    xy = X_reduced[class_member_mask]
                    ax.scatter(xy[:, 0], xy[:, 1], c=[col], marker=marker, s=size, alpha=alpha, label=label)
                
                ax.set_title(f"Visualisasi Klaster DBSCAN (eps={eps}, min_samples={min_samples})")
                ax.set_xlabel("Principal Component 1")
                ax.set_ylabel("Principal Component 2")
                ax.legend()
                ax.grid(True, alpha=0.3)

                st.pyplot(fig)
                
                # Analisis kualitas clustering
                st.subheader("📈 Analisis Kualitas Clustering")
                if n_clusters > 1:
                    from modules.clusteringKMeans import silhouette_score
                    mask_no_noise = labels_dbscan != -1
                    if np.any(mask_no_noise):
                        X_no_noise = X[mask_no_noise]
                        labels_no_noise = labels_dbscan[mask_no_noise]
                        sil_score = silhouette_score(X_no_noise, labels_no_noise)
                        st.metric("Silhouette Score (tanpa noise)", f"{sil_score:.4f}")
                        
                        if sil_score > 0.5:
                            st.success("🎉 Kualitas clustering sangat baik!")
                        elif sil_score > 0.25:
                            st.info("👍 Kualitas clustering cukup baik.")
                        else:
                            st.warning("⚠️ Kualitas clustering kurang baik. Pertimbangkan untuk menyesuaikan parameter.")
                    else:
                        st.warning("Tidak dapat menghitung Silhouette Score - semua titik adalah noise!")
                else:
                    st.warning("Silhouette Score tidak dapat dihitung - cluster terlalu sedikit.")

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
        st.error("Pastikan file yang diupload memiliki format yang benar dan tidak corrupt.")
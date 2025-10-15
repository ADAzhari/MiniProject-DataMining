import numpy as np

# ==========================================================
# 🌐 GLOBAL CONFIG
# ==========================================================
IDENTITY_COLS = [
    "nama", 
    "nim",
    "nama (opsional)", 
    "nim (opsional)",
    "umur", 
    "semester",
    "apa alasan utama kamu belajar?",
    "apakah kamu siap meluangkan waktu sekitar 5 menit untuk mengisi kuesioner ini?"
]


# ==========================================================
# 🧹 CLEANING
# ==========================================================
def clean_data(rows, drop_cols):
    """
    Membersihkan data mentah:
    - Hilangkan kolom yang tidak diperlukan
    - Hilangkan spasi ekstra
    - Konversi ke lowercase untuk kolom teks open-ended
    """
    cleaned_rows = []
    for r in rows:
        new_r = {}
        for k, v in r.items():
            key = k.strip().lower()
            if key not in [c.lower() for c in drop_cols]:
                if isinstance(v, str):
                    v = " ".join(v.strip().lower().split())
                new_r[key] = v
        cleaned_rows.append(new_r)
    print("✅ Data Cleaning selesai.")
    return cleaned_rows


# ==========================================================
# 🔄 TRANSFORMATION
# ==========================================================
def transform_data(rows, keywords_dict):
    """
    Transformasi data:
    - Konversi ordinal & likert ke angka
    - One-Hot Encoding untuk alasan belajar
    - Hapus kolom identitas dari hasil akhir
    """
    if not rows:
        print("❌ Data kosong!")
        return []

    # --- MAPPING ORDINAL ---
    ordinal_maps = {
        "dalam sehari, berapa jam rata-rata kamu gunakan untuk media sosial (instagram, tiktok, x, dll) di luar kebutuhan kuliah?":
            {'<1 jam': 1, '1–2 jam': 2, '3–4 jam': 3, '5–6 jam': 4, '>6 jam': 5},
        "berapa rata-rata jam belajar mandiri kamu per minggu (di luar kelas)?":
            {'<2 jam': 1, '2–4 jam': 2, '5–7 jam': 3, '8–10 jam': 4, '>10 jam': 5}
    }

    # --- MAPPING SKALA LIKERT ---
    likert_text_to_num = {
        "tidak pernah": 1.25, "jarang": 2.5, "lumayan sering": 3.75, "sangat sering": 5,
        "sangat tidak bisa": 1, "tidak bisa": 2, "cukup bisa": 3, "bisa": 4, "sangat bisa": 5,
        "sangat tidak puas": 1, "tidak puas": 2, "cukup puas": 3, "puas": 4, "sangat puas": 5,
        "ya": 5, "tidak": 1
    }

    ohe_col = "apa alasan utama kamu belajar?"

    # --- KONVERSI ORDINAL & LIKERT ---
    for r in rows:
        # Ordinal
        for col_name, mapping in ordinal_maps.items():
            for actual_col in r.keys():
                if col_name.lower() in actual_col.lower():
                    r[actual_col] = mapping.get(r[actual_col], 3)

        # Likert
        for k, v in r.items():
            if k.lower() in [c.lower() for c in IDENTITY_COLS]:
                continue
            if isinstance(v, str):
                val = likert_text_to_num.get(v, None)
                if val is not None:
                    r[k] = val
                else:
                    try:
                        r[k] = float(v)
                    except (ValueError, TypeError):
                        r[k] = 3.0
            elif v is None:
                r[k] = 3.0

    # --- ONE-HOT ENCODING UNTUK "ALASAN BELAJAR" ---
    if ohe_col.lower() not in [k.lower() for k in rows[0].keys()]:
        print(f"⚠️ Kolom '{ohe_col}' tidak ditemukan di data.")
        print("Kolom yang ada:", list(rows[0].keys()))
        return rows

    transformed_rows = []
    for r in rows:
        new_row = {}

        # Ambil jawaban alasan belajar
        jawaban = str(r.get(ohe_col, "")).lower()
        jawaban = jawaban.replace("/", " ").replace(",", " ").replace(".", " ").strip()

        # One-Hot Encoding berdasarkan keywords_dict
        for category, keyword_list in keywords_dict.items():
            colname = "alasan_" + category.lower().replace(" ", "_")
            new_row[colname] = 0
            for kw in keyword_list:
                kw_clean = kw.lower().replace("/", " ").replace(",", " ").replace(".", " ").strip()
                if kw_clean in jawaban:
                    new_row[colname] = 1
                    break

        # Tambahkan kolom non-identitas lainnya
        for k, v in r.items():
            if k.lower() not in [c.lower() for c in IDENTITY_COLS]:
                new_row[k] = v

        transformed_rows.append(new_row)

    print("✅ Transformasi data selesai (kolom identitas dihapus).")
    return transformed_rows


def normalize_data(rows, method="minmax"):
    """
    Normalisasi data numerik (kecuali OHE, kolom konstan, dan kolom identitas).
    method:
      - 'minmax' : skala 0–1
      - 'zscore' : standarisasi (mean 0, std 1)
    """
    import numpy as np

    if not rows:
        print("❌ Data kosong!")
        return []

    # Tentukan kolom yang layak dinormalisasi
    numeric_cols = []
    for k in rows[0].keys():
        values = [r.get(k) for r in rows]
        if all(isinstance(v, (int, float)) for v in values):
            unique_vals = set(values)
            if (
                len(unique_vals) > 1 and
                not unique_vals.issubset({0, 1}) and
                k.lower() not in [c.lower() for c in IDENTITY_COLS]
            ):
                numeric_cols.append(k)

    if not numeric_cols:
        print("⚠️ Tidak ada kolom numerik yang perlu dinormalisasi.")
        return rows

    data_matrix = np.array([[r[col] for col in numeric_cols] for r in rows], dtype=float)

    # Ganti NaN dengan rata-rata kolom
    if np.isnan(data_matrix).any():
        col_means = np.nanmean(data_matrix, axis=0)
        inds = np.where(np.isnan(data_matrix))
        data_matrix[inds] = np.take(col_means, inds[1])

    if method == "minmax":
        min_vals = data_matrix.min(axis=0)
        max_vals = data_matrix.max(axis=0)
        denom = np.where(max_vals - min_vals == 0, 1, max_vals - min_vals)
        normalized = (data_matrix - min_vals) / denom
        
    elif method == "zscore":
        means = data_matrix.mean(axis=0)
        stds = np.where(data_matrix.std(axis=0) == 0, 1, data_matrix.std(axis=0))
        normalized = (data_matrix - means) / stds
    else:
        raise ValueError("method harus 'minmax' atau 'zscore'")

    # Simpan hasil ke rows dengan pembulatan dan clamping
    for i, r in enumerate(rows):
        for j, col in enumerate(numeric_cols):
            val = float(normalized[i][j])
            val = max(0, min(1, round(val, 6))) if method == "minmax" else round(val, 6)
            r[col] = val

    print(f"✅ Normalisasi selesai ({method}) untuk {len(numeric_cols)} kolom (tanpa OHE/konstan/identitas).")
    return rows

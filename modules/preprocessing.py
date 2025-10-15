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
                    v = v.strip().lower()
                new_r[key] = v
        cleaned_rows.append(new_r)
    print("✅ Data Cleaning selesai.")
    return cleaned_rows

def transform_data(rows, keywords_dict):
    """
    Transformasi data:
    - Konversi ordinal dan likert ke angka
    - Scan jawaban alasan belajar jadi OHE multi-label
    - Pertahankan kolom identitas tanpa diubah
    """
    if not rows:
        print("❌ Data kosong!")
        return []

    identity_cols = ["nama", "nim", "umur", "semester", "apa alasan utama kamu belajar?"]

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
            if k.lower() in identity_cols:
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
        new_row = {k: v for k, v in r.items()}
    
        # Ambil jawaban dan normalisasi teks
        jawaban = str(r.get(ohe_col, "")).lower()
        jawaban = jawaban.replace("/", " ").replace(",", " ").replace(".", " ").strip()
    
        for category, keyword_list in keywords_dict.items():
            colname = "alasan_" + category.lower().replace(" ", "_")
            new_row[colname] = 0
        
            for kw in keyword_list:
             # hapus juga tanda baca di keyword biar match adil
                kw_clean = kw.lower().replace("/", " ").replace(",", " ").replace(".", " ").strip()
                if kw_clean in jawaban:
                    new_row[colname] = 1
                    break
    
        transformed_rows.append(new_row)

    for r in rows:
        new_row = {k: v for k, v in r.items()}
        jawaban = str(r.get(ohe_col, "")).lower()
        print(f"\nJawaban asli: '{jawaban}'")
        for category, keyword_list in keywords_dict.items():
            colname = "alasan_" + category.lower().replace(" ", "_")
            new_row[colname] = 0
            for kw in keyword_list:
                if kw.lower() in jawaban:
                    print(f"  ✅ Cocok: '{kw}' → {colname}")
                    new_row[colname] = 1
                    break
    transformed_rows.append(new_row)

    print("✅ Transformasi data selesai.")
    return transformed_rows



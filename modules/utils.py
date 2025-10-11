import csv
import io

def read_csv(file):
    """
    Membaca CSV dari:
    - File path lokal (string)
    - Streamlit UploadedFile

    Hasil: list of dict (tiap baris = dictionary)
    """
    try:
        # Jika file dari Streamlit
        if hasattr(file, "getvalue"):
            content = file.getvalue().decode("utf-8-sig", errors="ignore")
            f = io.StringIO(content)
        elif hasattr(file, "read"):  # fallback jika UploadedFile belum punya getvalue
            content = file.read().decode("utf-8-sig", errors="ignore")
            f = io.StringIO(content)
        else:
            f = open(file, "r", encoding="utf-8-sig", errors="ignore")

        reader = csv.DictReader(f)
        rows = [row for row in reader if any(row.values())]  # skip baris kosong

        if not rows:
            print("⚠️ File CSV kosong atau tidak memiliki data.")
        elif None in reader.fieldnames or "" in reader.fieldnames:
            print("⚠️ Header CSV mengandung kolom kosong, harap periksa file.")

        f.close()
        return rows

    except Exception as e:
        print(f"❌ Gagal membaca CSV: {e}")
        return []

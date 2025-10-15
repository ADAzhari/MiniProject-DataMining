import csv
import io
import pandas as pd


def read_csv(file):
    """
    Membaca CSV atau XLSX dari:
    - File path lokal (string)
    - Streamlit UploadedFile

    Hasil: list of dict (tiap baris = dictionary)
    """
    try:
        # Deteksi jenis file
        file_name = getattr(file, "name", str(file))
        is_xlsx = file_name.lower().endswith(".xlsx")

        if is_xlsx:
            # Handle XLSX files
            if hasattr(file, "getvalue"):
                content = file.getvalue()
                df = pd.read_excel(io.BytesIO(content))
            elif hasattr(file, "read"):
                content = file.read()
                df = pd.read_excel(io.BytesIO(content))
            else:
                df = pd.read_excel(file)

            # Convert DataFrame to list of dictionaries
            df = df.dropna(how="all")  # Remove empty rows
            rows = df.to_dict("records")

            # Convert column names to string and clean
            clean_rows = []
            for row in rows:
                clean_row = {}
                for k, v in row.items():
                    key = str(k).strip()
                    # Handle NaN values
                    if pd.isna(v):
                        clean_row[key] = ""
                    else:
                        clean_row[key] = (
                            str(v) if not isinstance(v, (int, float)) else v
                        )
                clean_rows.append(clean_row)

            return clean_rows

        else:
            # Handle CSV files (original logic)
            if hasattr(file, "getvalue"):
                content = file.getvalue().decode("utf-8-sig", errors="ignore")
                f = io.StringIO(content)
            elif hasattr(
                file, "read"
            ):  # fallback jika UploadedFile belum punya getvalue
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
        print(f"❌ Gagal membaca file: {e}")
        return []

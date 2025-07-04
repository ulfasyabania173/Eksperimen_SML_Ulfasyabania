import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_ionosphere(input_csv: str = "ionosphere_raw.csv", output_csv: str = "ionosphere_preprocessing.csv") -> pd.DataFrame:
    """
    Melakukan preprocessing otomatis pada dataset Ionosphere:
    1. Membaca file CSV mentah
    2. Menghapus duplikat
    3. Meng-encode target 'Class' menjadi numerik
    4. Standarisasi fitur numerik (Attribute1-Attribute34)
    5. Menyimpan hasil ke file CSV baru
    6. Mengembalikan DataFrame hasil preprocessing
    """
    # 1. Baca data
    import os
    if not os.path.isfile(input_csv):
        # Cek di parent folder jika file tidak ditemukan di folder saat ini
        parent_path = os.path.join(os.path.dirname(__file__), '..', 'ionosphere_raw.csv')
        if os.path.isfile(parent_path):
            input_csv = parent_path
    df = pd.read_csv(input_csv)

    # 2. Hapus duplikat
    df = df.drop_duplicates()

    # 3. Encode target
    df['ClassNum'] = df['Class'].map({'g': 1, 'b': 0})

    # 4. Standarisasi fitur numerik
    feature_cols = [f"Attribute{i}" for i in range(1, 35)]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # 5. Simpan hasil ke CSV
    df.to_csv(output_csv, index=False)

    return df

if __name__ == "__main__":
    df = preprocess_ionosphere()
    print("✅ Preprocessing selesai. File hasil: ionosphere_preprocessing.csv")

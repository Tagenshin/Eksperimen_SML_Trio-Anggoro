import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

def load_data(file_path):
    """
    Memuat dataset dari file CSV.
    """
    print(f"Memuat data dari: {file_path}")
    return pd.read_csv(file_path)

def clean_data(df):
    """
    Membersihkan data:
    1. Mengubah TotalCharges menjadi numerik.
    2. Menghapus baris dengan nilai null.
    """
    print("Membersihkan data...")
    # Mengubah TotalCharges ke numerik, error jadi NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip(), errors='coerce')
    
    # Menghapus baris dengan nilai NaN (terutama dari TotalCharges)
    df.dropna(inplace=True)
    
    return df

def select_features(df):
    """
    Memilih fitur berdasarkan hasil analisis korelasi (Cramer's V > 0.3)
    dan fitur numerik penting.
    """
    print("Melakukan seleksi fitur...")
    # Kolom terpilih berdasarkan insight notebook
    selected_columns = [
        'InternetService', 
        'OnlineSecurity', 
        'TechSupport', 
        'Contract', 
        'PaymentMethod', 
        'SeniorCitizen', 
        'MonthlyCharges', 
        'Churn'
    ]
    return df[selected_columns].copy()

def feature_engineering(df):
    """
    Melakukan binning pada MonthlyCharges.
    """
    print("Melakukan Feature Engineering (Binning)...")
    # Binning MonthlyCharges menjadi 3 kategori
    # Menggunakan pd.cut langsung pada kolom baru, lalu drop kolom lama jika perlu
    # Di sini kita ikuti logika notebook: replace nilai numerik dengan kategori
    df['MonthlyCharges_Category'] = pd.cut(
        df['MonthlyCharges'], 
        bins=3, 
        labels=['Rendah', 'Sedang', 'Tinggi']
    )
    
    # Hapus kolom MonthlyCharges original yang numerik, 
    # dan rename kolom kategori menjadi MonthlyCharges agar sesuai output notebook
    df = df.drop('MonthlyCharges', axis=1)
    df = df.rename(columns={'MonthlyCharges_Category': 'MonthlyCharges'})
    
    # Mengatur ulang urutan kolom agar Churn tetap di akhir (opsional tapi rapi)
    cols = [col for col in df.columns if col != 'Churn'] + ['Churn']
    df = df[cols]
    
    return df

def encode_data(df):
    """
    Mengubah data kategorikal menjadi angka menggunakan Manual Mapping
    yang disesuaikan dengan file referensi.
    """
    print("Melakukan Encoding data menggunakan Mapping...")
    
    # Definisi Mapping berdasarkan file preproses.csv
    mappings = {
        'InternetService': {
            'DSL': 0, 
            'Fiber optic': 1, 
            'No': 2
        },
        'OnlineSecurity': {
            'No': 0, 
            'No internet service': 1, 
            'Yes': 2
        },
        'TechSupport': {
            'No': 0, 
            'No internet service': 1, 
            'Yes': 2
        },
        'Contract': {
            'Month-to-month': 0, 
            'One year': 1, 
            'Two year': 2
        },
        'PaymentMethod': {
            'Bank transfer (automatic)': 0, 
            'Credit card (automatic)': 1, 
            'Electronic check': 2, 
            'Mailed check': 3
        },
        'MonthlyCharges': {
            'Rendah': 0, 
            'Sedang': 1, 
            'Tinggi': 2
        },
        'Churn': {
            'No': 0, 
            'Yes': 1
        }
    }
    
    # Terapkan mapping
    for col, mapping in mappings.items():
        if col in df.columns:
            print(f" - Mapping kolom: {col}")
            # Menggunakan replace agar aman terhadap tipe data category
            df[col] = df[col].replace(mapping)
            
            # Pastikan tipe data menjadi integer setelah mapping
            try:
                df[col] = df[col].astype(int)
            except ValueError as e:
                print(f"   Warning: Gagal convert ke int untuk kolom {col}. Error: {e}")

    return df

def save_data(df, output_path):
    """
    Menyimpan data hasil preprocessing ke file CSV.
    """
    # Pastikan folder tujuan ada
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Menyimpan data hasil preprocessing ke: {output_path}")
    df.to_csv(output_path, index=False)
    print("Proses selesai!")

def run_preprocessing():
    # --- Konfigurasi Path ---
    # Sesuaikan path ini dengan struktur folder repository Anda
    # Asumsi script ini ada di dalam folder 'preprocessing/'
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(base_dir) # Naik satu level ke root repo
    
    input_file = os.path.join(root_dir, 'Telco-Customer-Churn.csv') # Sesuaikan nama file raw
    # Jika file raw ada di folder khusus, misal: os.path.join(root_dir, 'data_raw', 'Telco-Customer-Churn.csv')
    
    output_file = os.path.join(base_dir, 'Telco-Customer-Churn_preprocessing.csv')

    # --- Eksekusi Pipeline ---
    try:
        # 1. Load
        df = load_data(input_file)
        
        # 2. Clean
        df = clean_data(df)
        
        # 3. Select Features
        df = select_features(df)
        
        # 4. Feature Engineering (Binning)
        df = feature_engineering(df)
        
        # 5. Encoding
        df_processed = encode_data(df)
        
        # 6. Save
        save_data(df_processed, output_file)
        
    except FileNotFoundError:
        print(f"Error: File dataset tidak ditemukan di {input_file}")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

if __name__ == "__main__":
    run_preprocessing()
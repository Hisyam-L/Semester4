import numpy as np
import pandas as pd

def load_dataset():
    df_dataset_asli = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
    # Preprocessing (Sama seperti sebelumnya)
    df_dataset_asli[['Sistolik', 'Diastolik']] = df_dataset_asli['Blood Pressure'].str.split('/', expand=True).astype(float)
    bmi_mapping = {'Normal': 1, 'Normal Weight': 1, 'Overweight': 2, 'Obese': 3} # Kadang di dataset ada 'Normal Weight'
    df_dataset_asli['BMI Angka'] = df_dataset_asli['BMI Category'].map(bmi_mapping)
    return df_dataset_asli

def bobot_masing_masing():
    bobot = [
    0.10, # Sleep Duration
    0.20, # Quality of Sleep
    0.05, # Physical Activity Level
    0.20, # Stress Level
    0.20, # BMI Category
    0.05, # Sistolik
    0.05, # Diastolik
    0.10, # Heart Rate
    0.05  # Daily Steps
    ]
    return bobot

def Normalisasi_WP(df_aseli):
    df_norm = pd.DataFrame()
    # COST: Min / X
    for col in ['Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Daily Steps']:
        df_norm[col] = df_aseli[col].min() / df_aseli[col]
    # BENEFIT: X / Max
    for col in ['Stress Level', 'BMI Angka', 'Sistolik', 'Diastolik', 'Heart Rate']:
        df_norm[col] = df_aseli[col] / df_aseli[col].max()
    return df_norm

def cari_skor_tertinggi(df_norm, df_aseli, bobot):
    cols = ['Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level',
            'BMI Angka', 'Sistolik', 'Diastolik', 'Heart Rate', 'Daily Steps']
    df_aseli['Skor_Akhir'] = np.dot(df_norm[cols].values, bobot)

def Hitung_Akurasi(df_hasil, threshold):
    aktual = (df_hasil['Sleep Disorder'].notna()) & (df_hasil['Sleep Disorder'] != 'None')
    prediksi = df_hasil['Skor_Akhir'] >= threshold
    return (prediksi == aktual).mean() * 100

def Evaluasi_Detail(df_hasil, threshold):
    """ Menampilkan persentase akurasi tiap gangguan """
    df_hasil['Sleep Disorder'] = df_hasil['Sleep Disorder'].fillna('None')
    df_hasil['Prediksi_Ada_Gangguan'] = df_hasil['Skor_Akhir'] >= threshold

    # 1. Akurasi Insomnia
    insomnia_data = df_hasil[df_hasil['Sleep Disorder'] == 'Insomnia']
    acc_insomnia = (insomnia_data['Prediksi_Ada_Gangguan']).mean() * 100 if len(insomnia_data) > 0 else 0

    # 2. Akurasi Sleep Apnea
    apnea_data = df_hasil[df_hasil['Sleep Disorder'] == 'Sleep Apnea']
    acc_apnea = (apnea_data['Prediksi_Ada_Gangguan']).mean() * 100 if len(apnea_data) > 0 else 0

    # 3. Akurasi Orang Sehat (Harus Prediksi False)
    normal_data = df_hasil[df_hasil['Sleep Disorder'] == 'None']
    acc_normal = (~normal_data['Prediksi_Ada_Gangguan']).mean() * 100 if len(normal_data) > 0 else 0

    print(f"\n--- ANALISIS DETAIL AKURASI (Threshold: {threshold:.2f}) ---")
    print(f"Akurasi Deteksi Insomnia    : {acc_insomnia:.2f}%")
    print(f"Akurasi Deteksi Sleep Apnea : {acc_apnea:.2f}%")
    print(f"Akurasi Deteksi Orang Sehat : {acc_normal:.2f}%")

if __name__ == "__main__":
    df_aseli = load_dataset()
    bobot = bobot_masing_masing()
    df_normalisasi = Normalisasi_WP(df_aseli)
    cari_skor_tertinggi(df_normalisasi, df_aseli, bobot)
    print(df_aseli.head(10))

     # Optimasi mencari threshold terbaik
    best_t, max_acc = 0, 0
    for t in np.arange(0.3, 0.8, 0.01):
        acc = Hitung_Akurasi(df_aseli, t)
        if acc > max_acc:
            max_acc, best_t = acc, t

    print(f"Akurasi Total Tertinggi: {max_acc:.2f}% pada Threshold {best_t:.2f}")

    Evaluasi_Detail(df_aseli, best_t)
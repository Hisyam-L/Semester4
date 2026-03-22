import numpy as np
import pandas as pd

# 1. Load Data
df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')

def Scoring_Data():
    # --- FUNGSI ASLI LU (TIDAK DIUBAH) ---
    df_skor = pd.DataFrame()
    df_systolic = df["Blood Pressure"].str.split("/").str[0].astype(int)
    df_diastolic = df["Blood Pressure"].str.split("/").str[1].astype(int)

    df_skor["Sleep Duration"] = pd.cut(df["Sleep Duration"], bins=[0, 6.0, 6.5, 7.0, 8.0, float("inf")], labels=[1, 2, 3, 4, 5], right=False).astype(int)
    df_skor["Quality of Sleep"] = pd.cut(df["Quality of Sleep"], bins=[0, 5, 6, 7, 9, float("inf")], labels=[1, 2, 3, 4, 5], right=False).astype(int)
    df_skor["Physical Activity Level"] = pd.cut(df["Physical Activity Level"], bins=[0, 31, 46, 61, 81, float("inf")], labels=[1, 2, 3, 4, 5], right=False).astype(int)
    df_skor["Stress Level"] = pd.cut(df["Stress Level"], bins=[0, 5, 6, 7, 9, float("inf")], labels=[1, 2, 3, 4, 5], right=False).astype(int)

    bmi_map = {"Normal": 1, "Normal Weight": 1, "Overweight": 3, "Obese": 5}
    # Tambah strip() biar gak AttributeError di string
    df_skor["BMI Category"] = df["BMI Category"].str.strip().map(bmi_map).fillna(1).astype(int)

    df_skor["Sistolik"] = pd.cut(df_systolic, bins=[0, 120, 130, 140, 160, float("inf")], labels=[1, 2, 3, 4, 5], right=False).astype(int)
    df_skor["Diastolik"] = pd.cut(df_diastolic, bins=[0, 80, 85, 90, 100, float("inf")], labels=[1, 2, 3, 4, 5], right=False).astype(int)
    df_skor["Heart Rate"] = pd.cut(df["Heart Rate"], bins=[0, 66, 71, 78, 84, float("inf")], labels=[1, 2, 3, 4, 5], right=False).astype(int)
    df_skor["Daily Steps"] = pd.cut(df["Daily Steps"], bins=[0, 3001, 5001, 7001, 9001, float("inf")], labels=[1, 2, 3, 4, 5], right=False).astype(int)
    return df_skor

def Normalisasi_SAW(df_scored):
    df_norm = pd.DataFrame()
    # COST: Min / X
    for col in ['Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Daily Steps']:
        df_norm[col] = df_scored[col].min() / df_scored[col]
    # BENEFIT: X / Max
    for col in ['Stress Level', 'BMI Category', 'Sistolik', 'Diastolik', 'Heart Rate']:
        df_norm[col] = df_scored[col] / df_scored[col].max()
    return df_norm

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
    df_skor = Scoring_Data()
    df_normalized = Normalisasi_SAW(df_skor)

    # Urutan kolom disamakan dengan bobot
    cols = ['Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level',
            'BMI Category', 'Sistolik', 'Diastolik', 'Heart Rate', 'Daily Steps']

    bobot = [0.10, 0.15, 0.05, 0.20, 0.15, 0.10, 0.10, 0.10, 0.05]
    df['Skor_Akhir'] = np.dot(df_normalized[cols].values, bobot)

    # Optimasi mencari threshold terbaik
    best_t, max_acc = 0, 0
    for t in np.arange(0.3, 0.8, 0.01):
        acc = Hitung_Akurasi(df, t)
        if acc > max_acc:
            max_acc, best_t = acc, t

    print(f"Akurasi Total Tertinggi: {max_acc:.2f}% pada Threshold {best_t:.2f}")

    # OUTPUT YANG LU MINTA:
    Evaluasi_Detail(df, best_t)
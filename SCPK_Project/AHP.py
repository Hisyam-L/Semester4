import numpy as np
import pandas as pd

def pair_Wise_Comparison(df_ahp, baris, kolom, nilai):
    df_ahp.at[baris, kolom] = float(nilai)
    df_ahp.at[kolom, baris] = 1 / float(nilai)

def cari_perbedaan(val1, val2):
    diff = round(abs(val1 - val2), 2)
    if diff == 0:
        return 1
    elif diff == 0.05:
        return 3
    elif diff == 0.10:
        return 5
    elif diff == 0.15:
        return 7
    else:
        return 1

def isi_data_pairwise():
    kriteria = [
    "Sleep Duration", "Quality of Sleep", "Physical Activity Level",
    "Stress Level", "BMI Category", "Sistolik", "Diastolik",
    "Heart Rate", "Daily Steps"
    ]
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

    df_ahp = pd.DataFrame(1.0, index=kriteria, columns=kriteria)

    for i in range(len(kriteria)):
        for j in range(i + 1, len(kriteria)):
            k1 = kriteria[i]
            k2 = kriteria[j]

            w1 = bobot[i]
            w2 = bobot[j]

            skala = cari_perbedaan(w1, w2)

            if w1 > w2:
                pair_Wise_Comparison(df_ahp, k1, k2, skala)
            elif w2 > w1:
                pair_Wise_Comparison(df_ahp, k1, k2, 1/skala)

    return df_ahp

def normalisasi(df_ahp):
    df_normalisasi = df_ahp.copy()
    for i in range(9):
        pembagi = df_normalisasi.iloc[:, i].sum()
        for j in range(9):
            df_normalisasi.iloc[j,i] = df_normalisasi.iloc[j,i]/pembagi
    return df_normalisasi

def cek_normalisasi(df_normalisasi):
    total_per_kolom = df_normalisasi.sum(axis=0)
    for i, val in enumerate(total_per_kolom):
        print(f"Kolom ke-{i} totalnya adalah: {val}")

def cari_weight(df_normalisasi):
    df_normalisasi['Weight'] = df_normalisasi.mean(axis=1)

def cek_consistency_ratio(df_ahp):
    n = 9
    matriks = df_ahp.iloc[:n, :n].to_numpy()
    eigenValues, eigenVector = np.linalg.eig(matriks)
    lambda_max = max(eigenValues).real
    n = len(df_ahp)
    CI = (lambda_max-n)/(n-1)
    RI = 1.45
    print(f"Consistency Ratio : {CI/RI}")

def cari_skor_alternatif(df_asli):
    df_kriteria = pd.DataFrame()
    df_kriteria['Person ID'] = df_asli['Person ID']

    # LOGIKA DIBALIK: Agar sakit = skor tinggi, maka Cost ditukar jadi Benefit, dan sebaliknya
    df_kriteria['Sleep Duration'] = df_asli['Sleep Duration'].min() / df_asli['Sleep Duration']
    df_kriteria['Quality of Sleep'] = df_asli['Quality of Sleep'].min() / df_asli['Quality of Sleep']
    df_kriteria['Physical Activity Level'] = df_asli['Physical Activity Level'].min() / df_asli['Physical Activity Level']
    df_kriteria['Daily Steps'] = df_asli['Daily Steps'].min() / df_asli['Daily Steps']

    df_kriteria['Stress Level'] = df_asli['Stress Level'] / df_asli['Stress Level'].max()
    df_kriteria['BMI Category'] = df_asli['BMI Angka'] / df_asli['BMI Angka'].max()
    df_kriteria['Sistolik'] = df_asli['Sistolik'] / df_asli['Sistolik'].max()
    df_kriteria['Diastolik'] = df_asli['Diastolik'] / df_asli['Diastolik'].max()
    df_kriteria['Heart Rate'] = df_asli['Heart Rate'] / df_asli['Heart Rate'].max()

    return df_kriteria

def load_dataset():
    df_dataset_asli = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
    df_dataset_asli[['Sistolik', 'Diastolik']] = df_dataset_asli['Blood Pressure'].str.split('/', expand=True).astype(float)
    bmi_mapping = {'Normal': 1, 'Normal Weight': 1, 'Overweight': 2, 'Obese': 3}
    df_dataset_asli['BMI Angka'] = df_dataset_asli['BMI Category'].map(bmi_mapping)
    return df_dataset_asli

def menghitung_skor_akhir(df_normalisasi_alternatif, df_normalisasi_ahp, df_asli):
    df_hasil = pd.DataFrame()
    df_hasil['Person ID'] = df_normalisasi_alternatif['Person ID']

    df_hasil['Total Skor'] = 0.0

    kriteria = [
        "Sleep Duration", "Quality of Sleep", "Physical Activity Level",
        "Stress Level", "BMI Category", "Sistolik", "Diastolik",
        "Heart Rate", "Daily Steps"
    ]

    for k in kriteria:
        bobot = df_normalisasi_ahp.loc[k, 'Weight']
        df_hasil['Total Skor'] += df_normalisasi_alternatif[k] * bobot

    df_hasil = pd.merge(df_hasil, df_asli[['Person ID', 'Sleep Disorder']], on='Person ID')

    df_hasil = df_hasil.sort_values(by='Total Skor', ascending=False).reset_index(drop=True)

    df_hasil.index += 1
    df_hasil['Ranking'] = df_hasil.index

    return df_hasil

def cek_akurasi(df_hasil):
    df_eval = df_hasil.copy()

    df_eval['Sleep Disorder'] = df_eval['Sleep Disorder'].fillna('None')

    df_eval['Aktual'] = df_eval['Sleep Disorder'].apply(
        lambda x: 'Sehat' if str(x).strip().lower() == 'none' else 'Sakit'
    )

    jumlah_sehat = (df_eval['Aktual'] == 'Sehat').sum()

    # LOGIKA DIBALIK: Skor tertinggi sekarang = Sakit. Maka orang Sehat ada di ranking paling Bawah.
    df_eval['Prediksi'] = 'Sakit'
    df_eval.iloc[-jumlah_sehat:, df_eval.columns.get_loc('Prediksi')] = 'Sehat'

    benar = (df_eval['Aktual'] == df_eval['Prediksi']).sum()
    total = len(df_eval)
    akurasi = (benar / total) * 100

    print(f"\n=== HASIL EVALUASI AKURASI MODEL ===")
    print(f"Total Data         : {total} Pasien")
    print(f"Aktual Pasien Sehat: {jumlah_sehat} Orang")
    print(f"Tebakan Benar      : {benar} Orang")
    print(f"Akurasi Sistem     : {akurasi:.2f}%")

    return akurasi, df_eval


if __name__ == "__main__":
    df_ahp = isi_data_pairwise()
    df_normalisasi = normalisasi(df_ahp)
    cari_weight(df_normalisasi)
    cek_consistency_ratio(df_ahp)

    df_asli = load_dataset()
    df_kriteria = cari_skor_alternatif(df_asli)
    df_hasil = menghitung_skor_akhir(df_kriteria, df_normalisasi, df_asli)

    # Karena skor tinggi = sakit, Top 5 Teratas sekarang = Paling Sakit
    print("\n=== TOP 5 POLA HIDUP TERBURUK (Paling Berisiko Sakit) ===")
    print(df_hasil[['Ranking', 'Person ID', 'Total Skor', 'Sleep Disorder']].head(5))

    # Dan Top 5 Terbawah = Paling Sehat
    print("\n=== TOP 5 POLA HIDUP TERBAIK (Paling Sehat) ===")
    print(df_hasil[['Ranking', 'Person ID', 'Total Skor', 'Sleep Disorder']].tail(5))

    akurasi, detail_evaluasi = cek_akurasi(df_hasil)

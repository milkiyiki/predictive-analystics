# Laporan Proyek Machine Learning - [Risqie Nur Salsabila]

## Domain Proyek

Pasar saham merupakan salah satu pilar penting dalam perekonomian modern yang mencerminkan kondisi keuangan perusahaan dan sentimen investor. Prediksi harga saham menjadi tantangan menarik karena dipengaruhi oleh banyak faktor dan memiliki pola yang kompleks secara temporal. Dalam proyek ini, dilakukan prediksi tren harga saham harian PT Bank Rakyat Indonesia Tbk (BBRI.JK) dengan memanfaatkan pendekatan deep learning berbasis data historis selama lima tahun.

BBRI dipilih karena merupakan salah satu saham blue-chip dengan likuiditas tinggi di Bursa Efek Indonesia, serta memiliki pergerakan harga yang dinamis. Informasi prediksi tren saham dapat digunakan untuk membantu investor dalam pengambilan keputusan beli/jual yang lebih terinformasi, sekaligus sebagai dasar untuk pengembangan sistem rekomendasi berbasis data.

Penelitian dari Zhang et al. (2020) mengemukakan pendekatan berbasis LSTM mampu menangkap pola temporal dalam data keuangan secara lebih efektif dibanding model lainnya untuk data time-series. [1]

## Business Understanding

### Problem Statement

Bagaimana memanfaatkan data historis harga saham harian BBRI untuk memprediksi harga penutupan di masa depan secara akurat menggunakan pendekatan machine learning?

### Goals

- Mengembangkan model prediksi harga penutupan saham BBRI berbasis time series.
- Mengevaluasi performa model dalam memprediksi harga penutupan dengan metrik regresi.

### Solution Statements

- Menggunakan model **Long Short-Term Memory (LSTM)** karena kemampuannya memahami dependensi jangka panjang dalam data sekuensial.
- Melakukan feature engineering dengan teknik moving average untuk menyaring noise.
- Melakukan normalisasi data menggunakan **MinMaxScaler** agar training stabil.
- Melakukan evaluasi model dengan **MAE**, **RMSE**, dan **R²**.

## Data Understanding

Dataset yang digunakan merupakan data historis saham harian BBRI dari Yahoo Finance selama tahun 2018–2025. Dataset ini mencakup kolom-kolom berikut:

- `Date`: Tanggal perdagangan
- `Open`: Harga pembukaan
- `High`: Harga tertinggi
- `Low`: Harga terendah
- `Close`: Harga penutupan (target prediksi)
- `Volume`: Volume transaksi harian

Jumlah data: 1.730 baris, tanpa missing value.

Link dataset: [https://finance.yahoo.com/quote/BBRI.JK/](https://finance.yahoo.com/quote/BBRI.JK/)

### Exploratory Data Analysis (EDA)

- Distribusi harga menunjukkan volatilitas tinggi.
- Harga cenderung naik dengan pola musiman tertentu.
- Moving average dan visualisasi tren digunakan untuk membantu identifikasi pola.

## Data Preparation

Langkah-langkah yang dilakukan:

1. **Konversi tanggal**: Mengubah kolom `Date` menjadi format datetime agar dapat diurutkan dan divisualisasikan.
2. **Pemilihan fitur**: Menggunakan hanya fitur `Close` untuk memprediksi harga penutupan berikutnya.
3. **Normalisasi**: Menggunakan `MinMaxScaler` untuk menskalakan harga penutupan ke rentang 0–1 agar lebih stabil dalam pelatihan model LSTM.
4. **Windowing**: Menggunakan window **30 hari** untuk memprediksi harga pada hari ke-31. Artinya, tiap input memiliki dimensi `(30,)`.
5. **Train-test split**: Proporsi 80:20, tanpa shuffle, untuk menjaga urutan waktu.

Seluruh langkah disusun secara konsisten dan berurutan antara notebook dan laporan ini.

## Modeling

### Arsitektur Model

Model LSTM yang digunakan memiliki struktur sebagai berikut:

- LSTM layer: 64 unit
- Dense output layer: 1 neuron
- Optimizer: Adam
- Loss function: Mean Squared Error (MSE)

```python
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
```

### Cara Kerja LSTM

LSTM (Long Short-Term Memory) adalah tipe jaringan saraf berulang (RNN) yang dirancang untuk belajar dari data sekuensial. Berbeda dengan RNN biasa, LSTM memiliki **cell state** dan tiga jenis **gate**:

- **Forget gate**: Memutuskan informasi apa yang akan dibuang dari cell state.
- **Input gate**: Memutuskan informasi baru apa yang akan ditambahkan.
- **Output gate**: Menentukan output berdasarkan cell state.

Dengan struktur ini, LSTM mampu mengingat informasi jangka panjang dan menghindari masalah vanishing gradient. Ini sangat sesuai untuk memodelkan urutan seperti harga saham harian.

### Parameter training:

- Epoch: 100
- Batch size: 32
- Callbacks: EarlyStopping dan ModelCheckpoint

## Evaluation

### Metrik yang digunakan:

- **MAE (Mean Absolute Error)**: Rata-rata nilai absolut dari selisih prediksi dan nilai aktual.
- **RMSE (Root Mean Squared Error)**: Akar dari rata-rata selisih kuadrat.
- **R² (R-squared)**: Proporsi variasi data yang bisa dijelaskan oleh model.

### Rumus:

- \(\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|\)
- \(\text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 }\)
- \(R^2 = 1 - \frac{SS_{res}}{SS_{tot}}\)

### Hasil Evaluasi Model:

- **MAE**: 65.29
- **RMSE**: 84.20
- **R²**: 0.9635

Model menunjukkan performa yang sangat baik, terbukti dari nilai R² yang mendekati 1 dan visualisasi prediksi yang mengikuti tren aktual dengan akurat.

---

> Catatan:
>
> - Model akhir disimpan dalam format `.keras` untuk keperluan deployment.
> - Scaler disimpan menggunakan `joblib` agar dapat digunakan ulang untuk data baru.

## Referensi

[1] Zhang, Y., Aggarwal, C. C., & Qi, G. J. (2020). *Stock Price Prediction via Discovering Multi-Frequency Trading Patterns*. Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining.
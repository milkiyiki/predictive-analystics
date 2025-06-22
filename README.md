# Predictive Analysis of BBRI Stock Prices Using LSTM

Proyek ini merupakan implementasi machine learning untuk memprediksi harga saham harian PT Bank Rakyat Indonesia Tbk (BBRI.JK) menggunakan pendekatan time series berbasis deep learning (LSTM). Model ini dibangun untuk membantu investor memahami pola tren harga dan membuat keputusan lebih terinformasi.

> Domain: Finance
> Pendekatan: Regresi Time Series (LSTM)
> Data: Yahoo Finance (BBRI.JK) 2018–2025
> Tools: Python, TensorFlow, Scikit-learn, YFinance, Matplotlib

---

## File Repositori

| File                    | Deskripsi                                                                                    |
| ----------------------- | -------------------------------------------------------------------------------------------- |
| `predictive.ipynb`      | Notebook utama yang memuat EDA, preprocessing, dan modeling LSTM                             |
| `predictive.py`         | Versi Python script dari notebook untuk kebutuhan production atau pipeline                   |
| `laporan.md`            | Laporan lengkap proyek sesuai format submission yang mencakup domain, modeling, dan evaluasi |
| `stock_price.csv`       | Dataset harga saham harian BBRI dari Yahoo Finance                                           |

---

## Cara Menjalankan Proyek

### Prasyarat

Pastikan Anda telah menginstal library berikut:

```bash
pip install yfinance pandas numpy scikit-learn matplotlib seaborn tensorflow joblib
```

### Langkah 1: Unduh Dataset

Dataset otomatis diunduh dari Yahoo Finance menggunakan `yfinance`. Anda juga dapat langsung menggunakan file `stock_price.csv` yang sudah disimpan.

### Langkah 2: Jalankan Notebook

Buka file `predictive.ipynb` di Google Colab atau Jupyter Notebook untuk menjalankan semua tahapan:

* Exploratory Data Analysis (EDA)
* Preprocessing (Normalisasi, Windowing)
* Training Model LSTM
* Evaluasi dan visualisasi hasil

Atau jalankan `predictive.py` dari terminal/IDE:

```bash
python predictive.py
```

---

## Model Arsitektur

Model LSTM terdiri dari:

* 1 LSTM Layer dengan 64 unit
* 1 Dense Output Layer
* Loss: MSE
* Optimizer: Adam

Pelatihan dilakukan selama 100 epoch dengan early stopping dan validasi time-series.

---

## Hasil Evaluasi

| Metrik | Nilai  |
| ------ | ------ |
| MAE    | 65.29  |
| RMSE   | 84.20  |
| R²     | 0.9635 |

Model menunjukkan performa yang sangat baik dalam memprediksi harga penutupan saham dengan nilai R² mendekati 1.

---

## Referensi

\[1] Zhang, Y., Aggarwal, C. C., & Qi, G. J. (2020). *Stock Price Prediction via Discovering Multi-Frequency Trading Patterns*. KDD 2020.

---

## Penulis
**Risqie Nur Salsabila**

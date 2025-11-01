# 🎓 UTBK Streamlit Portfolio

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nilaiutbk.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Made with Streamlit](https://img.shields.io/badge/Made%20with-Streamlit-red?logo=streamlit)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Aplikasi Streamlit interaktif untuk analisis data UTBK dan prediksi kelulusan berbasis Machine Learning.**

---

## 🧭 Deskripsi Proyek

Proyek ini dikembangkan oleh **Rusdi Ahmad** sebagai bagian dari tugas *Portfolio Building with Streamlit (MLOps)*.  
Aplikasi ini menampilkan portofolio interaktif yang menggabungkan:
- Analisis Data UTBK (Eksplorasi & Visualisasi)
- Pelatihan Model Logistic Regression untuk memprediksi kelulusan
- Upload dataset baru untuk prediksi
- Tampilan portofolio pribadi yang responsif

Aplikasi dapat diakses secara publik di:
👉 **[https://nilaiutbk.streamlit.app/](https://nilaiutbk.streamlit.app/)**

---

## 🧩 Fitur Utama

| Halaman | Deskripsi |
|----------|------------|
| **About** | Informasi pribadi dan latar belakang pembuat aplikasi |
| **Projects** | Daftar proyek yang pernah dikerjakan |
| **Data Viz** | Fitur upload dataset dan eksplorasi data (EDA) interaktif |
| **Train Model** | Melatih model Logistic Regression langsung dari dataset yang diunggah |
| **Predict** | Upload dataset baru untuk prediksi `Lulus/Tidak Lulus` menggunakan model tersimpan |

---

## 🧮 Teknologi yang Digunakan

- **Python 3.10+**
- **Streamlit** untuk antarmuka web interaktif  
- **Pandas & NumPy** untuk pengolahan data  
- **Scikit-learn** untuk machine learning pipeline  
- **Matplotlib & Seaborn** untuk visualisasi  

---

## 🧠 Model Machine Learning

Model yang digunakan: **Logistic Regression**  
Langkah kerja:
1. Data dibersihkan dan di-*impute* menggunakan `SimpleImputer(strategy='median')`
2. Fitur diskalakan menggunakan `StandardScaler`
3. Model Logistic Regression dilatih untuk memprediksi kolom target `lulus`
4. Model tersimpan dalam file `model/utbk_model.pkl`

---

## 📊 Visualisasi

- **Histogram** distribusi skor dan rata-rata nilai
- **Korelasi antar fitur**
- **Bar chart** untuk rata-rata tiap kolom numerik
- **Distribusi Target (0 = Tidak Lulus, 1 = Lulus)** setelah training

---

## ⚙️ Cara Menjalankan Secara Lokal

1. Clone repository:
   ```bash
   git clone https://github.com/rusdiahmad/utbk.git
   cd utbk

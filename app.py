# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------
# Page config / Title
# ---------------------------
st.set_page_config(page_title="My AI & ML Portfolio: UTBK Score Prediction Dashboard",
                   page_icon="🤖",
                   layout="wide")

st.title("🤖 My AI & ML Portfolio: UTBK Score Prediction Dashboard")
st.markdown("""
Aplikasi portofolio untuk Bootcamp **AI & ML** — fokus pada analisis dan prediksi nilai UTBK per subtes (TO1..TO7) berdasarkan jurusan dan fitur terkait.
""")

# ---------------------------
# Sidebar navigation
# ---------------------------
menu = st.sidebar.radio("Navigasi", ["Home", "Tentang Saya", "Proyek Saya", "Analisis & Prediksi UTBK"])

# ---------------------------
# Home
# ---------------------------
if menu == "Home":
    st.header("Welcome")
    st.write("Gunakan sidebar untuk navigasi. Aplikasi ini membaca file `NILAI UTBK ANGK 4.xlsx` dari repository.")

# ---------------------------
# Tentang Saya
# ---------------------------
elif menu == "Tentang Saya":
    st.header("👋 Tentang Saya")
    # show profile photo if exists
    try:
        st.image("Pas Photo.jpg", width=180, caption="Rusdi Ahmad")
    except Exception:
        st.info("Letakkan `Pas Photo.jpg` di folder repo untuk menampilkan foto profil.")
    st.markdown("""
**Nama:** Rusdi Ahmad  
**Posisi:** Guru Matematika & Peserta Bootcamp AI & ML  
**Keahlian:** Machine Learning, Visualisasi Data, Streamlit, Pendidikan  
    """)
    st.write("---")
    st.write("Kontak: rusdiahmad979@gmail.com")

# ---------------------------
# Proyek Saya
# ---------------------------
elif menu == "Proyek Saya":
    st.header("💼 Proyek Saya")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("🏫 UTBK Score Analysis & Prediction")
        st.image("https://cdn-icons-png.flaticon.com/512/2232/2232688.png", width=120)
        st.write("Analisis nilai UTBK serta model prediksi nilai tiap subtes (TO1..TO7) berdasarkan jurusan dan fitur lain.")
    with c2:
        st.subheader("🏠 House Price Prediction")
        st.image("https://cdn-icons-png.flaticon.com/512/619/619153.png", width=120)
        st.write("Contoh proyek ML untuk regresi (Random Forest) pada dataset House Prices (sebagai referensi).")
    with c3:
        st.subheader("🧮 Mathematics Question Generator (AI)")
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135706.png", width=120)
        st.write("Eksperimen AI untuk menghasilkan soal matematika yang relevan dengan kurikulum.")




# ---------------------------
# Analisis & Prediksi UTBK
# ---------------------------
elif menu == "Visualisasi Data":
    st.header("📊 Visualisasi Dataset UTBK")

    pd.read_excel("NILAI_UTBK_ANGK_4.xlsx")
    st.write("### Cuplikan Data")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    st.write("### Distribusi Fitur")
    selected_col = st.selectbox("Pilih kolom untuk dilihat distribusinya:", numeric_cols)
    fig, ax = plt.subplots()
    sns.histplot(df[selected_col], kde=True, color="teal", ax=ax)
    st.pyplot(fig)

    st.write("### Korelasi antar Fitur Numerik")
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    st.pyplot(fig)

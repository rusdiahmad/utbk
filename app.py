import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.exceptions import NotFittedError

# ================================
# Konfigurasi Aplikasi
# ================================
st.set_page_config(page_title="My Portfolio with Streamlit", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["About", "Projects", "Data Viz", "Train Model", "Predict"])

# ================================
# Halaman ABOUT
# ================================
if page == "About":
    st.title("My Portfolio with Streamlit")
    st.markdown("""
    ### 👋 Halo! Saya **Rusdi Ahmad**
    - 🎓 **Latar Belakang:** S2 Matematika UNAND  
    - 🧮 **Profesi:** Guru & Analis Data  
    - 💡 **Keahlian:** Data Science, Machine Learning, Streamlit, dan Pendidikan  
    - 📊 **Proyek ini** menampilkan portofolio interaktif berbasis data UTBK.
    """)

# ================================
# Halaman PROJECTS
# ================================
elif page == "Projects":
    st.header("My Projects")
    st.write("Berikut beberapa proyek yang pernah saya kerjakan:")
    st.markdown("""
    1. **Analisis Nilai UTBK dan Prediksi Kelulusan**  
       Analisis performa peserta UTBK menggunakan model Logistic Regression.
    2. **Dashboard Streamlit Portofolio**  
       Aplikasi ini sendiri — menampilkan data, grafik, dan prediksi.
    3. **Pengajaran Data Science untuk Siswa SMA**  
       Modul interaktif dengan Python & Streamlit.
    """)
    st.info("Anda dapat menambahkan gambar, tautan GitHub, atau tombol interaktif di sini.")

# ================================
# Halaman DATA VIZ
# ================================
elif page == "Data Viz":
    st.header("Data Visualization")
    data_path = "data/NILAI UTBK ANGK 4.xlsx"

if os.path.exists(data_path):
    df = pd.read_excel(data_path)
    st.success("✅ Dataset UTBK berhasil dimuat otomatis dari folder data/")
    st.dataframe(df.head())
else:
    st.error("❌ Dataset belum ditemukan. Harap tambahkan file ke folder data/")


    if uploaded is not None:
        try:
            if uploaded.name.endswith('.csv'):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)

            st.subheader("Preview Data")
            st.dataframe(df.head())

            st.subheader("Deskripsi Statistik")
            st.write(df.describe())

            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(num_cols) >= 2:
                st.subheader("Korelasi antar fitur")
                st.dataframe(df[num_cols].corr())

                st.subheader("Visualisasi Rata-rata Kolom Numerik")
                st.bar_chart(df[num_cols].mean())

        except Exception as e:
            st.error(f"Error membaca file: {e}")

# ================================
# Halaman TRAIN MODEL
# ================================
elif page == "Train Model":
    st.header("Train Model on Uploaded UTBK Dataset")
    st.write("Upload dataset untuk melatih model prediksi kelulusan otomatis.")

    uploaded = st.file_uploader("Upload file Excel atau CSV", type=['csv', 'xlsx'])

    if uploaded is not None:
        # Baca file
        if uploaded.name.endswith('.csv'):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)

        st.subheader("Preview Data")
        st.dataframe(df.head())

        # Pilih kolom numeric
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        st.write("Kolom numerik terdeteksi:", num_cols)

        # Pilih metode pembuatan target
        target_choice = st.selectbox("Pilih metode target:",
                                     ["Berdasarkan median skor", "Gunakan kolom tertentu"])

        if target_choice == "Gunakan kolom tertentu":
            target_col = st.selectbox("Pilih kolom target:", df.columns)
            df['lulus'] = df[target_col].astype(str).str.contains('lulus|ya|1', case=False).astype(int)
        else:
            main_col = st.selectbox("Gunakan kolom utama untuk threshold:", num_cols)
            median_val = df[main_col].median()
            df['lulus'] = (df[main_col] >= median_val).astype(int)
            st.info(f"Target dibuat otomatis berdasarkan median {main_col}: {median_val:.2f}")

        # Training model
        if st.button("Train Model"):
            from sklearn.model_selection import train_test_split
            from sklearn.pipeline import Pipeline
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LogisticRegression
            import seaborn as sns
            import matplotlib.pyplot as plt

            features = [c for c in num_cols if c != 'lulus']
            X = df[features]
            y = df['lulus']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(max_iter=1000))
            ])

            pipe.fit(X_train, y_train)
            acc = pipe.score(X_test, y_test)
            st.success(f"✅ Model trained successfully! Akurasi test: {acc:.2f}")

            # Simpan model
            os.makedirs("model", exist_ok=True)
            with open("model/utbk_model.pkl", "wb") as f:
                pickle.dump(pipe, f)
            st.info("Model disimpan ke folder model/utbk_model.pkl")

            # Visualisasi target
            st.subheader("Distribusi Target (0 = Tidak Lulus, 1 = Lulus)")
            fig, ax = plt.subplots()
            sns.countplot(x='lulus', data=df, ax=ax)
            st.pyplot(fig)

# ================================
# Halaman PREDICT
# ================================
elif page == "Predict":
    st.header("Predict Lulus/Tidak (menggunakan model tersimpan)")
    st.write("Upload dataset baru untuk melakukan prediksi dengan model yang telah dilatih.")

    uploaded = st.file_uploader("Upload data untuk prediksi", type=['csv', 'xlsx'])
    model_path = "model/utbk_model.pkl"

    if uploaded is not None:
        try:
            if uploaded.name.endswith('.csv'):
                data = pd.read_csv(uploaded)
            else:
                data = pd.read_excel(uploaded)
            st.subheader("Preview Data")
            st.dataframe(data.head())

            # Load model
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            # Ambil kolom numerik yang digunakan model
            num_cols = model.named_steps['preprocessor'].transformers[0][2] \
                if 'preprocessor' in model.named_steps else model.named_steps['imputer'].feature_names_in_
            X = data[num_cols]

            preds = model.predict(X)
            data['pred_lulus'] = preds
            st.subheader("Hasil Prediksi")
            st.dataframe(data.head())

            st.success("Prediksi selesai ✅ (kolom baru: pred_lulus)")

        except FileNotFoundError:
            st.error("Model file tidak ditemukan. Latih dulu model di menu Train Model.")
        except NotFittedError:
            st.error("Model belum terlatih.")
        except Exception as e:
            st.error(f"Prediction error: {e}")

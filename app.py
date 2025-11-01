# app.py
"""
My AI & ML Portfolio:
UTBK Subtest Score Analysis & Prediction Dashboard (Bahasa Indonesia, Professional)

Pastikan:
- File excel bernama 'NILAI UTBK ANGK 4.xlsx' ada di root repository.
- Foto profil 'Pas Photo.jpg' ada di root repository (opsional).
- requirements.txt berisi openpyxl agar Streamlit Cloud dapat membaca .xlsx
"""

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
import io

# ---------------------------
# Konfigurasi halaman
# ---------------------------
st.set_page_config(
    page_title="UTBK Subtest Score Analysis & Prediction Dashboard",
    page_icon="🎓",
    layout="wide",
)

# Gaya singkat
st.title("🎓 My AI & ML Portfolio: UTBK Subtest Score Analysis & Prediction Dashboard")
st.markdown(
    "Aplikasi portofolio (Bootcamp AI & ML) untuk **analisis** dan **prediksi** skor UTBK per subtes "
    "(PU, PK, PPU, PBM, LIND, LING, PM, Rata-rata) berdasarkan jurusan/prodi dan fitur terkait."
)
st.write("---")

# ---------------------------
# Sidebar navigasi
# ---------------------------
st.sidebar.header("Navigasi")
page = st.sidebar.radio("Pilih halaman", ["Beranda", "Tentang Saya", "Proyek Saya", "Analisis & Prediksi UTBK"])

# ---------------------------
# Utility functions
# ---------------------------
@st.cache_data
def load_excel(path: str):
    return pd.read_excel(path)

def detect_subtest_columns(df: pd.DataFrame):
    # Target subtest (expected): PU, PK, PPU, PBM, LIND, LING, PM, Rata-rata
    expected = ["PU", "PK", "PPU", "PBM", "LIND", "LING", "PM", "Rata-rata"]
    detected = [c for c in expected if c in df.columns]
    # If 'Rata-rata' named differently, attempt to find close names
    if "Rata-rata" not in detected:
        for c in df.columns:
            if c.strip().lower() in ["rata-rata", "rata rata", "rata2", "rata_rata", "rata"]:
                detected.append(c)
                break
    return detected

def encode_features(X: pd.DataFrame):
    encoders = {}
    X_enc = pd.DataFrame(index=X.index)
    for col in X.columns:
        if X[col].dtype == object or X[col].dtype.name == 'category':
            le = LabelEncoder()
            vals = X[col].astype(str).fillna("___NA___")
            X_enc[col] = le.fit_transform(vals)
            encoders[col] = le
        else:
            X_enc[col] = X[col].fillna(X[col].median())
    return X_enc, encoders

def transform_new(X_new: pd.DataFrame, encoders: dict, model_features: list):
    X_t = pd.DataFrame(index=X_new.index)
    for col in model_features:
        if col in X_new.columns:
            series = X_new[col]
        else:
            series = pd.Series([np.nan]*len(X_new), index=X_new.index)

        if col in encoders:
            le = encoders[col]
            mapped = []
            classes = list(map(str, le.classes_))
            for v in series.astype(str).fillna("___NA___"):
                if v in classes:
                    mapped.append(int(np.where(le.classes_ == v)[0][0]))
                else:
                    # unseen label -> map to -1
                    mapped.append(-1)
            X_t[col] = mapped
        else:
            X_t[col] = pd.to_numeric(series, errors='coerce').fillna(np.nanmedian(np.array(series.dropna(), dtype=float)) if series.dropna().size>0 else 0)
    return X_t

# ---------------------------
# BERANDA
# ---------------------------
if page == "Beranda":
    st.header("Beranda")
    st.write(
        "Selamat datang — gunakan sidebar untuk masuk ke halaman 'Analisis & Prediksi UTBK' "
        "atau melihat profil & proyek."
    )
    st.info("Pastikan `NILAI UTBK ANGK 4.xlsx` dan `Pas Photo.jpg` berada di root repository sebelum deploy.")

# ---------------------------
# TENTANG SAYA
# ---------------------------
elif page == "Tentang Saya":
    st.header("👋 Tentang Saya")
    col1, col2 = st.columns([1, 3])
    with col1:
        try:
            st.image("Pas Photo.jpg", width=180, caption="Rusdi Ahmad")
        except Exception:
            st.info("Letakkan `Pas Photo.jpg` pada folder repo untuk menampilkan foto profil.")
    with col2:
        st.markdown(
            "**Nama:** Rusdi Ahmad  \n"
            "**Peran:** Guru Matematika & Peserta Bootcamp AI & ML  \n"
            "**Keahlian:** Machine Learning, Visualisasi Data, Streamlit, Pendidikan  \n\n"
            "> \"Mengaplikasikan AI untuk mendukung pembelajaran dan keputusan pendidikan.\""
        )
    st.write("---")
    st.markdown("**Kontak**: rusdiahmad979@gmail.com")

# ---------------------------
# PROYEK SAYA
# ---------------------------
elif page == "Proyek Saya":
    st.header("💼 Proyek Saya")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("🏫 UTBK Score Analysis & Prediction")
        st.image("https://cdn-icons-png.flaticon.com/512/2232/2232688.png", width=110)
        st.write("Analisis mendalam nilai UTBK dan model multi-output untuk memprediksi nilai per subtes.")
    with c2:
        st.subheader("🏠 House Price Prediction (Referensi)")
        st.image("https://cdn-icons-png.flaticon.com/512/619/619153.png", width=110)
        st.write("Contoh proyek regresi menggunakan dataset House Prices (Kaggle).")
    with c3:
        st.subheader("🧮 Mathematics Question Generator (AI)")
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135706.png", width=110)
        st.write("Eksperimen AI untuk menghasilkan soal matematika otomatis — relevan dengan latar pendidikan.")

# ---------------------------
# ANALISIS & PREDIKSI UTBK
# ---------------------------
elif page == "Analisis & Prediksi UTBK":
    st.header("📊 Analisis & Prediksi Nilai Subtes UTBK (Semua Subtes Sekaligus)")

    # Load data
    try:
        df = load_excel("NILAI UTBK ANGK 4.xlsx")
    except FileNotFoundError:
        st.error("File `NILAI UTBK ANGK 4.xlsx` tidak ditemukan di root repository. Upload terlebih dahulu.")
        st.stop()
    except Exception as e:
        st.error(f"Gagal membaca file Excel: {e}")
        st.stop()

    # Normalisasi nama kolom
    df.columns = [c.strip() for c in df.columns]

    # Tampilkan preview dan kolom
    with st.expander("Pratinjau data & kolom"):
        st.dataframe(df.head(10))
        st.write("Kolom terdeteksi:")
        st.write(list(df.columns))

    # Deteksi kolom subtes
    subtests = detect_subtest_columns(df)
    st.markdown(f"**Subtest terdeteksi yang akan dianalisis / diprediksi:** `{subtests}`")

    if len(subtests) < 1:
        st.warning("Kolom subtes (PU, PK, PPU, PBM, LIND, LING, PM, Rata-rata) tidak ditemukan. Pastikan kolom sesuai.")
    else:
        # Statistik deskriptif
        st.subheader("Statistik Deskriptif Subtes")
        st.dataframe(df[subtests].describe().T)

        # Pilih subtes untuk visual
        st.subheader("Visualisasi Distribusi & Boxplot")
        sel = st.selectbox("Pilih subtes untuk lihat distribusi", subtests)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(df[sel].dropna(), kde=True, ax=axes[0])
        axes[0].set_title(f"Distribusi - {sel}")
        sns.boxplot(x=df[sel].dropna(), ax=axes[1])
        axes[1].set_title(f"Boxplot - {sel}")
        st.pyplot(fig)

        # Korelasi heatmap
        st.subheader("Korelasi Antar Subtes")
        corr_fig, corr_ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[subtests].corr(), annot=True, fmt=".2f", cmap="vlag", center=0, ax=corr_ax)
        st.pyplot(corr_fig)

        # Perbandingan per jurusan/prodi
        st.subheader("Perbandingan Rata-rata Per Jurusan / Prodi")
        group_col = None
        if "JURUSAN/PRODI" in df.columns:
            group_col = "JURUSAN/PRODI"
        else:
            # fallback
            for cand in ["RUMPUN", "SUB RUMPUN", "KAMPUS"]:
                if cand in df.columns:
                    group_col = cand
                    break

        if group_col:
            st.markdown(f"Grouping berdasarkan: **{group_col}**")
            top_n = st.slider("Tampilkan top N jurusan (berdasarkan jumlah siswa)", 3, 20, 7)
            counts = df[group_col].value_counts().head(top_n)
            st.bar_chart(counts)
            # rata-rata per group
            mean_by_group = df.groupby(group_col)[subtests].mean().reset_index()
            st.dataframe(mean_by_group.head(top_n).set_index(group_col))
        else:
            st.info("Kolom jurusan/prodi tidak ditemukan (cari kolom bernama 'JURUSAN/PRODI' atau 'RUMPUN').")

    # Persiapan fitur & target untuk model
    st.write("---")
    st.subheader("🔧 Persiapan Model & Training (Multi-output Regression)")

    # Fitur kandidat otomatis
    candidate_features = [
        "JURUSAN/PRODI", "RUMPUN", "PILIHAN 1 PTN-PRODI", "PILIHAN 2 PTN-PRODI",
        "PILIHAN 3 PTN-PRODI", "PILIHAN 4 PTN-PRODI",
        "RATA- RATA TO 4 S.D 7", "ESTIMASI RATA-RATA", "ESTIMASI NILAI MINIMUM",
        "ESTIMASI NILAI MAKSIMUM", "Rata-rata"
    ]
    features = [f for f in candidate_features if f in df.columns]

    # Tambahkan beberapa kolom numerik pendukung jika ada (maks 3)
    numeric_candidates = [c for c in df.columns if df[c].dtype in [np.float64, np.int64] and c not in subtests]
    features += numeric_candidates[:3]

    st.markdown(f"**Fitur yang terdeteksi otomatis:** `{features}` (bisa disesuaikan di dataset)")

    # Tombol training
    if st.button("▶️ Latih Model Multi-output (Random Forest)"):
        # Persiapan data training
        df_model = df.copy()
        # drop rows tanpa target lengkap
        df_model = df_model.dropna(subset=subtests, how='any').reset_index(drop=True)
        if df_model.shape[0] < 10:
            st.warning("Data pelatihan kurang (<10 baris) setelah menghapus baris missing target. Model mungkin tidak stabil.")

        if not features:
            # fallback ke JURUSAN/PRODI jika ada
            if "JURUSAN/PRODI" in df_model.columns:
                features = ["JURUSAN/PRODI"]
                st.info("Fallback: menggunakan 'JURUSAN/PRODI' sebagai fitur tunggal.")
            else:
                st.error("Tidak ada fitur yang cocok untuk training. Tambahkan kolom fitur ke dataset.")
                st.stop()

        X = df_model[features].copy()
        y = df_model[subtests].astype(float).copy()

        # Encode categorical & handle numeric
        X_proc, encoders = encode_features(X)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X_proc, y, test_size=0.2, random_state=42)

        # Train model
        with st.spinner("Melatih model..."):
            base = RandomForestRegressor(n_estimators=200, random_state=42)
            model = MultiOutputRegressor(base, n_jobs=-1)
            model.fit(X_train, y_train)

        # Evaluasi
        y_pred = model.predict(X_test)
        metrics = []
        for i, col in enumerate(subtests):
            mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
            rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
            r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
            metrics.append({"Subtest": col, "MAE": mae, "RMSE": rmse, "R2": r2})
        metrics_df = pd.DataFrame(metrics).set_index("Subtest")

        st.success("✅ Pelatihan selesai")
        st.subheader("Hasil evaluasi pada test set")
        st.dataframe(metrics_df)

        # Simpan ke session state
        st.session_state["model"] = model
        st.session_state["encoders"] = encoders
        st.session_state["model_features"] = X_proc.columns.tolist()
        st.session_state["targets"] = subtests

    # Bagian prediksi
    st.write("---")
    st.subheader("📥 Prediksi pada Data Baru (Upload CSV atau Input Manual)")

    if "model" in st.session_state:
        mode = st.radio("Pilih mode input untuk prediksi:", ["Upload CSV", "Input Manual"])
        model = st.session_state["model"]
        encoders = st.session_state["encoders"]
        model_features = st.session_state["model_features"]
        targets = st.session_state["targets"]

        if mode == "Upload CSV":
            uploaded = st.file_uploader("Upload file CSV berisi baris siswa (header harus memuat fitur yang sama)", type=["csv"])
            if uploaded is not None:
                newdf = pd.read_csv(uploaded)
                st.markdown("Preview data upload (5 baris pertama)")
                st.dataframe(newdf.head())

                # Build X_new
                X_new = pd.DataFrame()
                for f in model_features:
                    X_new[f] = newdf[f] if f in newdf.columns else np.nan

                X_new_proc = transform_new(X_new, encoders, model_features)
                preds = model.predict(X_new_proc)
                preds_df = pd.DataFrame(preds, columns=targets)

                # If jurusan column exists in uploaded, include it; else try to include from newdf
                out_df = preds_df.copy()
                if "JURUSAN/PRODI" in newdf.columns:
                    out_df["JURUSAN/PRODI"] = newdf["JURUSAN/PRODI"].values
                else:
                    out_df["JURUSAN/PRODI"] = newdf[model_features[0]].values if model_features and model_features[0] in newdf.columns else "Unknown"

                # compute rata-rata prediksi dan ranking
                out_df["Pred_Rata-rata"] = out_df[targets].mean(axis=1)
                out_df["Ranking (prediksi)"] = out_df["Pred_Rata-rata"].rank(ascending=False, method="min").astype(int)

                st.markdown("Hasil prediksi (5 baris pertama)")
                st.dataframe(out_df.head())

                # Download button
                csv = out_df.to_csv(index=False).encode("utf-8")
                st.download_button("💾 Download Hasil Prediksi (CSV)", data=csv, file_name="prediksi_utbk.csv", mime="text/csv")

        else:
            # Manual input single row
            st.markdown("Isi fitur untuk 1 siswa (input manual):")
            input_vals = {}
            for f in model_features:
                if f in df.columns and df[f].dtype == object:
                    default = str(df[f].dropna().iloc[0]) if not df[f].dropna().empty else ""
                    input_vals[f] = st.text_input(f, value=default)
                else:
                    default_num = float(df[f].dropna().median()) if f in df.columns and not df[f].dropna().empty else 0.0
                    input_vals[f] = st.number_input(f"{f}", value=default_num)

            if st.button("Run Prediksi (Manual)"):
                X_manual = pd.DataFrame([input_vals])
                X_manual_proc = transform_new(X_manual, encoders, model_features)
                pred = model.predict(X_manual_proc)
                pred_df = pd.DataFrame(pred, columns=targets)
                pred_df["Pred_Rata-rata"] = pred_df.mean(axis=1)
                pred_df["Ranking (prediksi)"] = 1
                st.markdown("Hasil prediksi (manual)")
                st.dataframe(pred_df.T)

                # Download
                csv = pred_df.to_csv(index=False).encode("utf-8")
                st.download_button("💾 Download Hasil Prediksi (Manual)", data=csv, file_name="prediksi_manual_utbk.csv", mime="text/csv")

    else:
        st.info("Silakan latih model terlebih dahulu dengan menekan tombol 'Latih Model Multi-output (Random Forest)' di atas.")

    st.write("---")
    st.markdown(
        "Catatan: Model ini dibuat sebagai contoh portofolio Bootcamp AI & ML. "
        "Untuk keperluan produksi/performance lebih baik lakukan feature engineering lanjutan, cross-validation, "
        "hyperparameter tuning, dan validasi eksternal."
    )

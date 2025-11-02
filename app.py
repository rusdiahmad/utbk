# app.py
"""
UTBK Subtest Score Analysis & Prediction Dashboard
Versi profesional siap untuk GitHub + Streamlit Cloud.

Pastikan file berikut ada di root repo:
- NILAI UTBK ANGK 4.xlsx
- Pas Photo.jpg
- buku.jpg (gambar proyek UTBK)
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

# ---------------------------
# Page config & theme hints
# ---------------------------
st.set_page_config(
    page_title="UTBK Subtest Score Analysis & Prediction Dashboard",
    page_icon="🎓",
    layout="wide",
)

# custom seaborn style for consistent visuals
sns.set_style("whitegrid")

# ---------------------------
# Header (professional)
# ---------------------------
st.title("📈 UTBK Subtest Score Analysis & Prediction Dashboard")
st.markdown(
    """
Dashboard ini dirancang untuk menyajikan **analisis mendalam** atas nilai per-subtes UTBK
dan menyediakan **prediksi nilai per-subtes** (PU, PK, PPU, PBM, LIND, LING, PM, Rata-rata)
menggunakan pendekatan Machine Learning.  
Aplikasi ini dapat membantu evaluasi akademik, perencanaan pembelajaran, dan pengambilan keputusan berbasis data.
"""
)
st.write("---")

# ---------------------------
# Sidebar navigation (3 pages, default ke "Tentang Saya")
# ---------------------------
st.sidebar.header("Navigasi")
page = st.sidebar.radio(
    "Pilih halaman:",
    ["Tentang Saya", "Proyek Saya", "Analisis & Prediksi UTBK"],
    index=0,  # default ke Tentang Saya
)

# ---------------------------
# Helper functions
# ---------------------------
@st.cache_data
def load_excel(path: str):
    return pd.read_excel(path)

def detect_subtests(df: pd.DataFrame):
    expected = ["PU", "PK", "PPU", "PBM", "LIND", "LING", "PM", "Rata-rata"]
    found = [c for c in expected if c in df.columns]
    # try to find variants for Rata-rata
    if "Rata-rata" not in found:
        for c in df.columns:
            if c.strip().lower() in ["rata-rata","rata rata","rata2","rata_rata","rata"]:
                found.append(c)
                break
    return found

def encode_features(X: pd.DataFrame):
    encoders = {}
    X_enc = pd.DataFrame(index=X.index)
    for col in X.columns:
        if X[col].dtype == object or X[col].dtype.name == "category":
            le = LabelEncoder()
            vals = X[col].astype(str).fillna("___NA___")
            X_enc[col] = le.fit_transform(vals)
            encoders[col] = {cls: i for i, cls in enumerate(le.classes_)}
        else:
            X_enc[col] = X[col].fillna(X[col].median())
    return X_enc, encoders

def transform_new(X_new: pd.DataFrame, encoders: dict, model_features: list):
    X_t = pd.DataFrame(index=X_new.index)
    for col in model_features:
        if col in X_new.columns:
            series = X_new[col].astype(str).fillna("___NA___")
        else:
            series = pd.Series(["___NA___"]*len(X_new), index=X_new.index)
        if col in encoders:
            mapping = encoders[col]
            X_t[col] = series.map(lambda v: mapping.get(v, -1)).astype(int)
        else:
            # numeric fallback
            X_t[col] = pd.to_numeric(series, errors='coerce').fillna(0)
    return X_t

# ---------------------------
# Page: Tentang Saya (interactive CV)
# ---------------------------
if page == "Tentang Saya":
    st.header("👨‍🏫 Tentang Saya")
    col1, col2 = st.columns([1, 2.4])
    with col1:
        try:
            st.image("Pas Photo.jpg", width=220, caption="Rusdi Ahmad")
        except Exception:
            st.warning("Letakkan file 'Pas Photo.jpg' di root repository untuk menampilkan foto profil.")
        st.markdown("### 📬 Kontak")
        st.markdown("- 📧 rusdiahmad979@gmail.com")
        st.markdown("- 🔗 [LinkedIn](https://www.linkedin.com/in/rusdi-ahmad-a2948a1a4)")
        st.markdown("- 📍 Kota Bogor, Indonesia")
    with col2:
        st.markdown("## Rusdi Ahmad")
        st.markdown("**Guru Matematika | AI & Machine Learning Enthusiast | Data Educator**")
        st.markdown(
            "> Mengaplikasikan AI dan analisis data untuk meningkatkan mutu pembelajaran dan membantu pengambilan keputusan pendidikan."
        )
        st.write("---")
        st.markdown("### 🎓 Pendidikan")
        st.markdown(
            "- **S2 Matematika — Universitas Andalas (UNAND)**\n"
            "  _Fokus: Analisis & Pendidikan Matematika_\n"
            "- **S1 Pendidikan Matematika — UIN Imam Bonjol Padang_"
        )
        st.write("---")
        st.markdown("### 💼 Pengalaman Kerja (Ringkasan)")
        st.markdown(
            "- **Guru Matematika — Bimbingan Belajar Bintang Pelajar (2024–Sekarang)**\n"
            "- **Guru Matematika — SMAN Agam Cendekia**\n"
            "- **Instruktur — Bimbel CPNS DINNDA**\n"
            "- **Pembina Ekstrakurikuler Robotik & Coding (SMA)**"
        )
        st.write("---")
        st.markdown("### 🏆 Prestasi & Keterlibatan")
        st.markdown(
            "- Finalis Olimpiade Nasional Matematika (ON MIPA) UNAND\n"
            "- Juara Kompetisi Sains Madrasah (KSM) Nasional\n"
            "- Pembina & panitia kegiatan pendidikan dan lomba matematika"
        )
        st.write("---")
        st.markdown("### 🧠 Keahlian Teknis")
        sk1, sk2 = st.columns(2)
        with sk1:
            st.write("Python (pandas, scikit-learn, Streamlit)")
            st.progress(90)
            st.write("Data Visualization (Matplotlib, Seaborn)")
            st.progress(85)
            st.write("Machine Learning")
            st.progress(80)
        with sk2:
            st.write("Google Colab & Jupyter")
            st.progress(90)
            st.write("Excel Analytics & Dashboarding")
            st.progress(95)
            st.write("Data Cleaning & Preprocessing")
            st.progress(75)
        st.write("---")
        st.markdown("### 🤝 Soft Skills")
        st.write("- Leadership • Public Speaking • Curriculum Design • Mentoring")
        st.write("---")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("[💬 Kirim Email](mailto:rusdiahmad979@gmail.com)")
        with c2:
            st.markdown("[🔗 Kunjungi LinkedIn](https://www.linkedin.com/in/rusdi-ahmad-a2948a1a4)")

# ---------------------------
# Page: Proyek Saya
# ---------------------------
elif page == "Proyek Saya":
    st.header("💼 Proyek Saya")
    st.markdown("Kumpulan proyek unggulan yang merepresentasikan kemampuan dalam AI, Machine Learning, dan Analisis Pendidikan.")
    st.write("---")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📘 UTBK Score Analysis & Prediction")
        try:
            st.image("buku.jpg", use_column_width=True)
        except Exception:
            st.image("https://cdn-icons-png.flaticon.com/512/2232/2232688.png", width=120)
        st.markdown(
            "Analisis komprehensif nilai UTBK per subtes serta model multi-output untuk memprediksi skor berdasarkan jurusan/prodi dan fitur terkait."
        )
        st.markdown("**Teknologi:** Python, pandas, scikit-learn, Streamlit, RandomForest (MultiOutput).")
    with c2:
        st.subheader("🏠 House Price Prediction (Referensi)")
        st.image("https://cdn-icons-png.flaticon.com/512/619/619153.png", width=140)
        st.markdown("Model regresi Random Forest pada dataset House Prices (Kaggle) — referensi metode dan visualisasi.")
        st.markdown("**Teknologi:** Python, scikit-learn, EDA, Feature Engineering.")

# ---------------------------
# Page: Analisis & Prediksi UTBK
# ---------------------------
elif page == "Analisis & Prediksi UTBK":
    st.header("📊 Analisis & Prediksi Nilai Subtes UTBK")
    st.markdown(
        "Unggah file Excel (NILAI UTBK ANGK 4.xlsx) di bawah untuk memulai analisis dan melatih model prediksi."
    )

    # Upload or read from repo file
    upload = st.file_uploader("Upload file Excel (.xlsx) — atau biarkan kosong untuk memakai file di repo", type=["xlsx"])
    if upload is None:
        # try reading from repo root
        try:
            df = load_excel("NILAI UTBK ANGK 4.xlsx")
            st.success("Dataset dimuat dari repository.")
        except FileNotFoundError:
            st.info("File tidak ditemukan di repository. Silakan upload file Excel via tombol di atas.")
            st.stop()
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            st.stop()
    else:
        try:
            df = pd.read_excel(upload)
            st.success("Dataset berhasil di-upload dan dimuat.")
        except Exception as e:
            st.error(f"Gagal membaca file upload: {e}")
            st.stop()

    # clean column names
    df.columns = [c.strip() for c in df.columns]

    # detect subtests
    subtests = detect_subtests(df)
    st.markdown(f"**Subtest detected:** {subtests}")

    if not subtests:
        st.warning("Subtest columns tidak ditemukan. Pastikan kolom: PU, PK, PPU, PBM, LIND, LING, PM, Rata-rata.")
        st.stop()

    # EDA: descriptive
    st.subheader("Statistik Deskriptif Subtes")
    st.dataframe(df[subtests].describe().T)

    # Distribution & boxplot
    st.subheader("Distribusi Nilai Per Subtest (Pilih subtest)")
    sel = st.selectbox("Pilih subtest", subtests)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df[sel].dropna(), kde=True, ax=axs[0])
    axs[0].set_title(f"Distribusi {sel}")
    sns.boxplot(x=df[sel].dropna(), ax=axs[1])
    axs[1].set_title(f"Boxplot {sel}")
    st.pyplot(fig)

    # correlation heatmap
    st.subheader("Korelasi Antar Subtest")
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[subtests].corr(), annot=True, cmap="vlag", center=0, ax=ax_corr)
    st.pyplot(fig_corr)

    # group by jurusan/prodi
    st.subheader("Perbandingan Rata-rata per Jurusan / Prodi")
    group_col = "JURUSAN/PRODI" if "JURUSAN/PRODI" in df.columns else None
    if group_col:
        top_n = st.slider("Tampilkan top N jurusan (berdasarkan jumlah siswa)", 3, 20, 7)
        counts = df[group_col].value_counts().head(top_n)
        st.bar_chart(counts)
        mean_by_group = df.groupby(group_col)[subtests].mean().reset_index()
        st.dataframe(mean_by_group.head(top_n).set_index(group_col))
    else:
        st.info("Kolom jurusan/prodi tidak ditemukan. Jika ada, beri nama 'JURUSAN/PRODI' pada dataset.")

    # Model training (multi-output)
    st.write("---")
    st.subheader("🔧 Latih Model Multi-output (Prediksi Semua Subtest Sekaligus)")

    # candidate features
    candidate_features = [
        "JURUSAN/PRODI", "RUMPUN", "PILIHAN 1 PTN-PRODI", "PILIHAN 2 PTN-PRODI",
        "PILIHAN 3 PTN-PRODI", "PILIHAN 4 PTN-PRODI", "RATA- RATA TO 4 S.D 7",
        "ESTIMASI RATA-RATA", "ESTIMASI NILAI MINIMUM", "ESTIMASI NILAI MAKSIMUM", "Rata-rata"
    ]
    features = [f for f in candidate_features if f in df.columns]
    # add up to 3 numeric extras
    numeric_candidates = [c for c in df.columns if df[c].dtype in [np.float64, np.int64] and c not in subtests]
    features += numeric_candidates[:3]

    st.markdown(f"**Fitur otomatis terdeteksi:** `{features}`")

    if st.button("▶️ Latih Model (Random Forest MultiOutput)"):
        df_model = df.dropna(subset=subtests, how="any").reset_index(drop=True)
        if df_model.shape[0] < 10:
            st.warning("Data sangat sedikit setelah drop missing target; hasil model mungkin tidak stabil.")
        if not features:
            if "JURUSAN/PRODI" in df_model.columns:
                features = ["JURUSAN/PRODI"]
                st.info("Fallback: menggunakan 'JURUSAN/PRODI' sebagai fitur.")
            else:
                st.error("Tidak ada fitur yang cocok untuk training. Tambahkan kolom fitur pada dataset.")
                st.stop()

        X = df_model[features].copy()
        y = df_model[subtests].astype(float).copy()

        X_enc, encoders = encode_features(X)

        X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.2, random_state=42)

        with st.spinner("Melatih model... (harap tunggu beberapa saat)"):
            base = RandomForestRegressor(n_estimators=200, random_state=42)
            model = MultiOutputRegressor(base, n_jobs=-1)
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        metrics = []
        for i, col in enumerate(subtests):
            mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
            rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
            r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
            metrics.append({"Subtest": col, "MAE": mae, "RMSE": rmse, "R2": r2})
        metrics_df = pd.DataFrame(metrics).set_index("Subtest")
        st.success("✅ Pelatihan selesai")
        st.subheader("Hasil Evaluasi (Test set)")
        st.dataframe(metrics_df)

        # save model to session
        st.session_state["model"] = model
        st.session_state["encoders"] = encoders
        st.session_state["model_features"] = X_enc.columns.tolist()
        st.session_state["targets"] = subtests

    # Prediction section
    st.write("---")
    st.subheader("📥 Prediksi untuk Data Baru (Upload CSV / Input Manual)")
    if "model" in st.session_state:
        mode = st.radio("Mode input:", ["Upload CSV", "Input Manual"])
        model = st.session_state["model"]
        encoders = st.session_state["encoders"]
        model_features = st.session_state["model_features"]
        targets = st.session_state["targets"]

        if mode == "Upload CSV":
            uploaded = st.file_uploader("Upload CSV (header perlu mengandung fitur yang sama)", type=["csv"])
            if uploaded is not None:
                newdf = pd.read_csv(uploaded)
                st.markdown("Preview data upload")
                st.dataframe(newdf.head())

                X_new = pd.DataFrame()
                for f in model_features:
                    X_new[f] = newdf[f] if f in newdf.columns else np.nan

                X_new_proc = transform_new(X_new, encoders, model_features)
                preds = model.predict(X_new_proc)
                preds_df = pd.DataFrame(preds, columns=targets)

                # attach jurusan if present
                out_df = preds_df.copy()
                if "JURUSAN/PRODI" in newdf.columns:
                    out_df["JURUSAN/PRODI"] = newdf["JURUSAN/PRODI"].values
                else:
                    out_df["JURUSAN/PRODI"] = newdf[model_features[0]].values if model_features and model_features[0] in newdf.columns else "Unknown"

                out_df["Pred_Rata-rata"] = out_df[targets].mean(axis=1)
                out_df["Ranking (prediksi)"] = out_df["Pred_Rata-rata"].rank(ascending=False, method="min").astype(int)

                st.markdown("Hasil prediksi (5 baris pertama)")
                st.dataframe(out_df.head())

                csv = out_df.to_csv(index=False).encode("utf-8")
                st.download_button("💾 Download Hasil Prediksi", data=csv, file_name="prediksi_utbk.csv", mime="text/csv")

        else:
            st.markdown("Isi fitur untuk satu siswa (manual):")
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
                csv = pred_df.to_csv(index=False).encode("utf-8")
                st.download_button("💾 Download Hasil Prediksi (Manual)", data=csv, file_name="prediksi_manual_utbk.csv", mime="text/csv")
    else:
        st.info("Latih model terlebih dahulu dengan tombol 'Latih Model (Random Forest MultiOutput)' di atas.")

    st.write("---")
    st.markdown(
        "Catatan: Aplikasi ini dibuat sebagai portofolio dan proof-of-concept. "
        "Untuk produksi lakukan feature engineering, cross-validation, hyperparameter tuning, dan validasi eksternal."
    )

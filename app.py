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
elif menu == "Analisis & Prediksi UTBK":
    st.header("📊 Analisis Data & Prediksi Nilai Per Subtes (TO1..TO7)")

    # Try read excel
    try:
        df = pd.read_excel("NILAI UTBK ANGK 4.xlsx")
    except Exception as e:
        st.error(f"Gagal membaca file `NILAI UTBK ANGK 4.xlsx`. Pastikan file ada di root repo. Error: {e}")
        st.stop()

    st.subheader("Cuplikan Data")
    st.dataframe(df.head())

    # Standardize column names (strip spaces)
    df.columns = [c.strip() for c in df.columns]

    # Important column names (based on yang kamu berikan)
    to_cols = [c for c in df.columns if c.upper().startswith("TO " ) or c.upper().startswith("TO") and ("TO" in c)]
    # fallback: look for TO1..TO7 explicitly
    fallback = []
    for i in range(1, 8):
        name = f"TO {i}"
        if name in df.columns:
            fallback.append(name)
    if fallback and not to_cols:
        to_cols = fallback

    # also accept columns like 'TO1' without space
    for i in range(1,8):
        if f"TO{i}" in df.columns and f"TO {i}" not in to_cols:
            to_cols.append(f"TO{i}")

    st.write("Deteksi kolom subtest (TO):", to_cols)

    # Basic EDA
    st.subheader("Statistik Deskriptif Nilai Subtes")
    if to_cols:
        st.write(df[to_cols].describe().T)
        # Distribution selector
        st.write("### Distribusi Nilai Per Subtes")
        sel = st.selectbox("Pilih subtes untuk melihat distribusi", to_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[sel].dropna(), kde=True, ax=ax)
        ax.set_xlabel(sel)
        st.pyplot(fig)

        # Correlation heatmap for TO columns + Rata-rata if exists
        corr_cols = to_cols.copy()
        if "Rata-rata" in df.columns:
            corr_cols.append("Rata-rata")
        st.write("### Korelasi antar subtes")
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        st.pyplot(fig)
    else:
        st.info("Kolom subtest (TO1..TO7) tidak ditemukan otomatis. Pastikan kolom bernama 'TO 1', 'TO 2', dst.")

    # Show top ranking or pass/fail summary if exists
    if "LULUS JALUR UTBK-SNBT" in df.columns:
        st.write("Distribusi Lulus Jalur UTBK-SNBT:")
        st.write(df["LULUS JALUR UTBK-SNBT"].value_counts())

    # Prepare features & targets for modeling
    st.write("---")
    st.subheader("🔧 Persiapan Model")

    # Choose features: use JURUSAN/PRODI, PILIHAN 1..4, Rata-rata, RATA- RATA TO 4 S.D 7, ESTIMASI RATA-RATA if available
    feature_candidates = ["JURUSAN/PRODI", "PILIHAN 1 PTN-PRODI", "PILIHAN 2 PTN-PRODI", "PILIHAN 3 PTN-PRODI", "PILIHAN 4 PTN-PRODI",
                          "RATA- RATA TO 4 S.D 7", "ESTIMASI RATA-RATA", "Rata-rata", "RATA-RATA", "ESTIMASI NILAI MINIMUM"]
    features = [c for c in feature_candidates if c in df.columns]

    # Always include any numeric columns that look useful (Rata-rata, ESTIMASI etc)
    numeric_extra = [c for c in df.columns if c not in features and df[c].dtype in [np.float64, np.int64] and c not in to_cols]
    # but keep numeric_extra short
    if numeric_extra:
        numeric_extra = numeric_extra[:3]
        features += numeric_extra

    st.write("Fitur yang dipakai (detected):", features if features else "Tidak ada fitur kategori/estimasi ditemukan; model akan menggunakan jurusan jika ada.")

    # Build dataset for modeling
    # targets = to_cols (TO 1..TO 7)
    targets = to_cols.copy()
    if not targets:
        st.warning("Tidak ada kolom target (TO1..TO7) untuk dilatih. Tidak bisa membangun model.")
        st.stop()

    model_button = st.button("Latih Model (Multi-output Random Forest)")

    # Encode categorical features if present
    encoders = {}
    if model_button:
        st.info("Memulai pelatihan model. Mohon tunggu...")

        df_model = df.copy()

        # keep only rows where all target values are present (or at least non-null)
        df_model = df_model.dropna(subset=targets, how="any").reset_index(drop=True)
        if df_model.shape[0] < 10:
            st.warning("Data pelatihan kurang (<10 baris) setelah drop missing target — pelatihan mungkin tidak optimal.")

        X = pd.DataFrame()
        # If features exist, use them; otherwise fallback to 'JURUSAN/PRODI' only if exists
        if features:
            for f in features:
                if f in df_model.columns:
                    X[f] = df_model[f]
        else:
            if "JURUSAN/PRODI" in df_model.columns:
                X["JURUSAN/PRODI"] = df_model["JURUSAN/PRODI"]
            else:
                st.error("Tidak ada fitur kategorikal/estimasi untuk dipakai. Silakan tambahkan fitur ke dataset.")
                st.stop()

        # Preprocess X: encode categoricals, fill numeric NaN
        X_proc = pd.DataFrame()
        for col in X.columns:
            if X[col].dtype == object or X[col].dtype.name == 'category':
                le = LabelEncoder()
                X_proc[col] = le.fit_transform(X[col].astype(str))
                encoders[col] = le
            else:
                X_proc[col] = X[col].fillna(X[col].median())

        y = df_model[targets].astype(float)

        # train/test split
        X_train, X_test, y_train, y_test = train_test_split(X_proc, y, test_size=0.2, random_state=42)

        # model
        base = RandomForestRegressor(n_estimators=200, random_state=42)
        multi = MultiOutputRegressor(base, n_jobs=-1)
        multi.fit(X_train, y_train)

        # preds
        y_pred = multi.predict(X_test)

        # metrics per target
        metrics = []
        st.write("### Hasil Evaluasi (Test set)")
        for i, col in enumerate(targets):
            mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
            rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
            r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
            metrics.append((col, mae, rmse, r2))
        metrics_df = pd.DataFrame(metrics, columns=["Subtest", "MAE", "RMSE", "R2"])
        st.dataframe(metrics_df.set_index("Subtest"))

        st.success("Pelatihan selesai. Model siap digunakan untuk prediksi pada data baru.")

        # Save model objects in session state for later prediction
        st.session_state['model'] = multi
        st.session_state['encoders'] = encoders
        st.session_state['model_features'] = X_proc.columns.tolist()
        st.session_state['targets'] = targets

    # Prediction UI
    st.write("---")
    st.subheader("📥 Prediksi Nilai (Upload CSV atau Input Manual)")

    if 'model' in st.session_state:
        predict_mode = st.radio("Mode Prediksi", ["Upload CSV", "Input Manual"])
        model = st.session_state['model']
        encoders = st.session_state['encoders']
        model_features = st.session_state['model_features']
        targets = st.session_state['targets']

        if predict_mode == "Upload CSV":
            up = st.file_uploader("Upload CSV (baris: tiap siswa) untuk prediksi", type=["csv"])
            if up is not None:
                newdf = pd.read_csv(up)
                st.write("Cuplikan data yang diupload:")
                st.dataframe(newdf.head())

                # Build X_new with same features
                X_new = pd.DataFrame()
                for f in model_features:
                    if f in newdf.columns:
                        X_new[f] = newdf[f]
                    else:
                        # if feature missing, create default
                        X_new[f] = np.nan

                # encode categorical with saved encoders
                for col in X_new.columns:
                    if col in encoders:
                        le = encoders[col]
                        X_new[col] = X_new[col].astype(str).map(lambda x: x if x in le.classes_ else None)
                        # For unseen labels, fit_transform won't accept -> fallback: map unseen to -1 or add
                        # Safe approach: transform with unseen mapped to -1
                        X_new[col] = X_new[col].fillna("___UNK___")
                        # extend classes_ temporarily
                        classes = list(le.classes_)
                        if "___UNK___" not in classes:
                            classes.append("___UNK___")
                        le2 = LabelEncoder()
                        le2.classes_ = np.array(classes)
                        # map
                        X_new[col] = X_new[col].apply(lambda v: v if v in le2.classes_ else "___UNK___")
                        # numeric labels via mapping
                        mapping = {c: i for i, c in enumerate(le2.classes_)}
                        X_new[col] = X_new[col].map(mapping)
                    else:
                        X_new[col] = pd.to_numeric(X_new[col], errors='coerce').fillna(newdf[col].median() if col in newdf.columns else 0)

                preds = model.predict(X_new)
                preds_df = pd.DataFrame(preds, columns=targets)
                st.write("Hasil Prediksi:")
                st.dataframe(preds_df.head())

                csv = preds_df.to_csv(index=False).encode('utf-8')
                st.download_button("💾 Download Prediksi", data=csv, file_name="prediksi_utbk.csv", mime="text/csv")

        else:  # Manual input
            st.write("Masukkan nilai fitur (jurusan / estimasi) untuk memprediksi nilai TO:")
            input_dict = {}
            for f in model_features:
                if f in df.columns and df[f].dtype == object:
                    val = st.text_input(f"Masukkan {f}", value=str(df[f].dropna().iloc[0]) if not df[f].dropna().empty else "")
                    input_dict[f] = [val]
                else:
                    val = st.number_input(f"Masukkan {f}", value=float(df[f].dropna().median()) if f in df.columns else 0.0)
                    input_dict[f] = [val]
            X_manual = pd.DataFrame(input_dict)

            # encode manual
            for col in X_manual.columns:
                if col in encoders:
                    le = encoders[col]
                    v = X_manual.loc[0, col]
                    if v in list(le.classes_):
                        X_manual[col] = le.transform([v])
                    else:
                        # unseen -> map to a code
                        X_manual[col] = -1
                else:
                    X_manual[col] = pd.to_numeric(X_manual[col], errors='coerce').fillna(0)

            if st.button("Run Prediksi (Manual)"):
                pred = model.predict(X_manual)
                pred_df = pd.DataFrame(pred, columns=targets)
                st.write("Hasil Prediksi:")
                st.dataframe(pred_df.T)

    else:
        st.info("Latih model terlebih dahulu dengan menekan tombol 'Latih Model' di atas.")

    st.write("---")
    st.write("Catatan: Model ini bersifat contoh untuk tugas bootcamp. Untuk produksi, lakukan feature engineering, hyperparameter tuning, cross-validation, dan validasi lebih lanjut.")

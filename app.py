# app.py
"""
Professional Streamlit app for:
"My AI & ML Portfolio: UTBK Score Prediction Dashboard"

Requirements (put into requirements.txt):
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
openpyxl
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
# Page config
# ---------------------------
st.set_page_config(
    page_title="My AI & ML Portfolio: UTBK Score Prediction Dashboard",
    page_icon="🤖",
    layout="wide",
)

# ---------------------------
# Styles (small)
# ---------------------------
MAIN_TITLE = "🤖 My AI & ML Portfolio: UTBK Score Prediction Dashboard"
st.title(MAIN_TITLE)
st.markdown(
    "A professional portfolio app for the **AI & ML Bootcamp** — "
    "data exploration and multi-output prediction of UTBK subtest scores (TO1..TO7)."
)

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.header("Navigation")
menu = st.sidebar.radio(
    "Go to",
    ["Home", "Tentang Saya", "Proyek Saya", "Analisis & Prediksi UTBK"]
)

# ---------------------------
# Helper utilities
# ---------------------------
@st.cache_data
def load_excel(path: str):
    return pd.read_excel(path)

def detect_to_columns(df: pd.DataFrame):
    # try variants: "TO 1", "TO1", "TO 2", etc.
    to_cols = []
    for i in range(1, 8):
        for variant in (f"TO {i}", f"TO{i}", f"TO{i}".upper(), f"TO {i}".upper()):
            if variant in df.columns and variant not in to_cols:
                to_cols.append(variant)
    # also include columns that start with "TO " or "TO"
    auto = [c for c in df.columns if c.strip().upper().startswith("TO")]
    for c in auto:
        if c not in to_cols:
            to_cols.append(c)
    return to_cols

def encode_features(X: pd.DataFrame, encoders: dict = None):
    encoders = {} if encoders is None else encoders
    X_proc = pd.DataFrame(index=X.index)
    for col in X.columns:
        if X[col].dtype == object or X[col].dtype.name == "category":
            le = LabelEncoder()
            X_proc[col] = le.fit_transform(X[col].astype(str).fillna("___NA___"))
            encoders[col] = le
        else:
            X_proc[col] = X[col].fillna(X[col].median())
    return X_proc, encoders

def transform_new_features(X_new: pd.DataFrame, encoders: dict, model_features: list):
    X_tmp = pd.DataFrame(index=X_new.index)
    for col in model_features:
        if col in X_new.columns:
            val = X_new[col]
        else:
            val = [np.nan] * len(X_new)
        if col in encoders:
            le = encoders[col]
            # map unseen labels to a special code (encoded as -1)
            mapped = []
            classes = set(le.classes_.astype(str))
            for v in val.astype(str).fillna("___NA___"):
                if v in classes:
                    mapped.append(int(np.where(le.classes_ == v)[0][0]))
                else:
                    mapped.append(-1)
            X_tmp[col] = mapped
        else:
            # numeric
            X_tmp[col] = pd.to_numeric(val, errors="coerce").fillna(np.nanmedian(np.array(val, dtype=float)))
    return X_tmp

# ---------------------------
# HOME
# ---------------------------
if menu == "Home":
    st.header("Welcome")
    st.markdown(
        "Use the left sidebar to navigate. \n\n"
        "- **Analisis & Prediksi UTBK**: main page to explore and train models. \n"
        "- **Tentang Saya / Proyek Saya**: portfolio sections for presentation."
    )
    st.info("Make sure `NILAI UTBK ANGK 4.xlsx` and `Pas Photo.jpg` are placed in the repository root before deploying.")

# ---------------------------
# TENTANG SAYA
# ---------------------------
elif menu == "Tentang Saya":
    st.header("👋 Tentang Saya")
    col1, col2 = st.columns([1, 3])
    with col1:
        try:
            st.image("Pas Photo.jpg", width=200, caption="Rusdi Ahmad")
        except Exception:
            st.empty()
            st.markdown("> (Letakkan `Pas Photo.jpg` di root repo untuk menampilkan foto.)")
    with col2:
        st.markdown(
            "**Nama:** Rusdi Ahmad  \n"
            "**Peran:** Guru Matematika & Peserta Bootcamp AI & ML  \n"
            "**Keahlian:** Machine Learning, Visualisasi Data, Streamlit, Pendidikan  \n\n"
            "> \"Mengintegrasikan AI dalam pembelajaran untuk hasil yang berdampak.\""
        )
    st.write("---")
    st.markdown("**Kontak**: rusdiahmad979@gmail.com")

# ---------------------------
# PROYEK SAYA
# ---------------------------
elif menu == "Proyek Saya":
    st.header("💼 Proyek Saya")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("🏫 UTBK Score Analysis & Prediction")
        st.image("https://cdn-icons-png.flaticon.com/512/2232/2232688.png", width=120)
        st.markdown(
            "Analisis mendalam terhadap nilai UTBK dan model prediksi nilai per subtes (TO1..TO7) "
            "berbasis fitur jurusan/estimasi."
        )
    with c2:
        st.subheader("🏠 House Price Prediction (Reference)")
        st.image("https://cdn-icons-png.flaticon.com/512/619/619153.png", width=120)
        st.markdown("Proyek referensi regresi: model Random Forest untuk prediksi harga rumah.")
    with c3:
        st.subheader("🧮 Mathematics Question Generator (AI)")
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135706.png", width=120)
        st.markdown("Eksperimen AI untuk menghasilkan soal matematika otomatis — relevan dengan latar pendidikanku.")

# ---------------------------
# ANALISIS & PREDIKSI UTBK
# ---------------------------
elif menu == "Analisis & Prediksi UTBK":
    st.header("📊 Analisis Data & Prediksi Nilai Per Subtes (TO1..TO7)")

    # --- Load file
    try:
        df = load_excel("NILAI UTBK ANGK 4.xlsx")
    except FileNotFoundError:
        st.error("File `NILAI UTBK ANGK 4.xlsx` tidak ditemukan di root repository.")
        st.stop()
    except Exception as e:
        st.error(f"Gagal membaca file Excel: {e}")
        st.stop()

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Show basic info
    with st.expander("Preview data & info"):
        st.markdown("**Data preview (first rows)**")
        st.dataframe(df.head())
        st.markdown("**Columns detected**")
        st.write(list(df.columns))

    # Detect TO columns
    to_cols = detect_to_columns(df)
    st.markdown(f"**Detected subtest columns (TO):** `{to_cols}`")

    if not to_cols:
        st.warning("Tidak ditemukan kolom TO1..TO7. Pastikan kolom bernama 'TO 1', 'TO1', dll.")
    else:
        # EDA area
        st.subheader("Exploratory Data Analysis")
        # Summary stats for TO columns
        st.markdown("**Statistik deskriptif per subtes**")
        st.dataframe(df[to_cols].describe().T)

        # Distribution and boxplot
        st.markdown("**Distribusi & Boxplot per subtes**")
        sel_to = st.selectbox("Pilih subtest", to_cols)
        fig_distr, ax = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(df[sel_to].dropna(), kde=True, ax=ax[0])
        ax[0].set_title(f"Distribusi - {sel_to}")
        sns.boxplot(x=df[sel_to].dropna(), ax=ax[1])
        ax[1].set_title(f"Boxplot - {sel_to}")
        st.pyplot(fig_distr)

        # Correlation heatmap among TO columns
        st.markdown("**Korelasi antar subtes**")
        corr_fig, corr_ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[to_cols].corr(), annot=True, fmt=".2f", cmap="vlag", center=0, ax=corr_ax)
        st.pyplot(corr_fig)

    # Additional summaries
    st.subheader("Summary per Jurusan / Prodi")
    if "JURUSAN/PRODI" in df.columns:
        group_col = "JURUSAN/PRODI"
    else:
        # fallback to 'RUMPUN' or others
        group_col = None
        for candidate in ["RUMPUN", "SUB RUMPUN", "KAMPUS", "STATUS"]:
            if candidate in df.columns:
                group_col = candidate
                break

    if group_col:
        st.markdown(f"Grouping by **{group_col}**")
        top = st.slider("Show top N groups by count", 3, 10, 5)
        counts = df[group_col].value_counts().head(top)
        st.bar_chart(counts)
        # show mean per group for average TO if exists
        if to_cols:
            mean_by_group = df.groupby(group_col)[to_cols].mean().reset_index()
            st.dataframe(mean_by_group.head(top).set_index(group_col))
    else:
        st.info("Tidak ditemukan kolom jurusan/prodi untuk summary per jurusan. Jika ada, beri nama kolom 'JURUSAN/PRODI' atau 'RUMPUN'.")

    # Modeling preparation
    st.write("---")
    st.subheader("🔧 Model: Multi-output Regression (Predict TO1..TO7)")

    # Pick features automatically (prefer certain columns)
    # Candidate features: JURUSAN/PRODI, PILIHAN 1..4, RATA- RATA TO 4 S.D 7, ESTIMASI RATA-RATA, Rata-rata, ESTIMASI NILAI MINIMUM, ESTIMASI NILAI MAKSIMUM
    candidate_features = [
        "JURUSAN/PRODI", "PILIHAN 1 PTN-PRODI", "PILIHAN 2 PTN-PRODI",
        "PILIHAN 3 PTN-PRODI", "PILIHAN 4 PTN-PRODI",
        "RATA- RATA TO 4 S.D 7", "ESTIMASI RATA-RATA", "Rata-rata",
        "ESTIMASI NILAI MINIMUM", "ESTIMASI NILAI MAKSIMUM"
    ]
    features = [c for c in candidate_features if c in df.columns]

    # also include a few numeric extras if exist (limit 3)
    numeric_candidates = [c for c in df.columns if df[c].dtype in [np.float64, np.int64] and c not in to_cols]
    numeric_extra = numeric_candidates[:3]
    features += numeric_extra

    st.markdown(f"**Features detected (will be used for training):** {features if features else 'No features detected automatically.'}")

    # Train button
    train_btn = st.button("▶️ Train Multi-output Model")
    if train_btn:
        # Prepare data: drop rows with missing targets
        df_model = df.copy()
        df_model = df_model.dropna(subset=to_cols, how="any").reset_index(drop=True)
        if df_model.shape[0] < 10:
            st.warning("Data pelatihan setelah drop missing target kurang dari 10 baris. Model mungkin tidak stabil.")

        # Build X
        if not features and "JURUSAN/PRODI" in df_model.columns:
            features = ["JURUSAN/PRODI"]
            st.info("Fallback: using JURUSAN/PRODI as single feature.")
        elif not features:
            st.error("Tidak ada fitur yang ditemukan untuk melatih model. Tambahkan kolom fitur ke dataset.")
            st.stop()

        X = df_model[features].copy()
        y = df_model[to_cols].astype(float).copy()

        # Encode categorical and clean numeric
        X_proc, encoders = encode_features(X)

        # train/test split
        X_train, X_test, y_train, y_test = train_test_split(X_proc, y, test_size=0.2, random_state=42)

        # model training
        with st.spinner("Training model (Random Forest, multi-output)..."):
            base = RandomForestRegressor(n_estimators=150, random_state=42)
            model = MultiOutputRegressor(base, n_jobs=-1)
            model.fit(X_train, y_train)

        # predictions & metrics
        y_pred = model.predict(X_test)
        metrics = []
        for i, col in enumerate(to_cols):
            mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
            rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
            r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
            metrics.append({"Subtest": col, "MAE": mae, "RMSE": rmse, "R2": r2})
        metrics_df = pd.DataFrame(metrics).set_index("Subtest")

        st.success("✅ Training completed")
        st.subheader("Model evaluation (on test set)")
        st.dataframe(metrics_df)

        # Save model to session state
        st.session_state.model = model
        st.session_state.encoders = encoders
        st.session_state.model_features = X_proc.columns.tolist()
        st.session_state.targets = to_cols

    # Prediction block
    st.write("---")
    st.subheader("📥 Prediksi untuk Data Baru")

    if "model" in st.session_state:
        mode = st.radio("Pilih mode input untuk prediksi:", ["Upload CSV", "Single-row Manual Input"])

        model = st.session_state.model
        encoders = st.session_state.encoders
        model_features = st.session_state.model_features
        targets = st.session_state.targets

        if mode == "Upload CSV":
            uploaded = st.file_uploader("Upload CSV berisi baris siswa untuk diprediksi (header harus mengandung fitur yang sama)", type=["csv"])
            if uploaded is not None:
                newdf = pd.read_csv(uploaded)
                st.markdown("Preview uploaded data")
                st.dataframe(newdf.head())

                X_new = pd.DataFrame()
                for f in model_features:
                    if f in newdf.columns:
                        X_new[f] = newdf[f]
                    else:
                        X_new[f] = np.nan

                X_new_proc = transform_new_features(X_new, encoders, model_features)
                preds = model.predict(X_new_proc)
                preds_df = pd.DataFrame(preds, columns=targets)
                st.markdown("Hasil prediksi (first rows)")
                st.dataframe(preds_df.head())

                csv = preds_df.to_csv(index=False).encode("utf-8")
                st.download_button("💾 Download hasil prediksi", data=csv, file_name="prediksi_utbk.csv", mime="text/csv")

        else:
            st.markdown("Isi fitur manual (untuk satu siswa)")
            input_vals = {}
            for f in model_features:
                if f in df.columns and df[f].dtype == object:
                    default = str(df[f].dropna().iloc[0]) if not df[f].dropna().empty else ""
                    input_vals[f] = st.text_input(f, value=default)
                else:
                    default_num = float(df[f].dropna().median()) if f in df.columns and not df[f].dropna().empty else 0.0
                    input_vals[f] = st.number_input(f, value=default_num)

            if st.button("Run prediction (manual)"):
                X_manual = pd.DataFrame([input_vals])
                X_manual_proc = transform_new_features(X_manual, encoders, model_features)
                pred = model.predict(X_manual_proc)
                pred_df = pd.DataFrame(pred, columns=targets)
                st.markdown("Hasil prediksi (manual):")
                st.dataframe(pred_df.T)

    else:
        st.info("Tekan tombol **Train Multi-output Model** untuk melatih model terlebih dahulu.")

    st.write("---")
    st.markdown(
        "Catatan: Aplikasi ini dibuat sebagai portofolio dan contoh implementasi ML. "
        "Untuk penggunaan produksi lakukan feature engineering lebih matang, cross-validation, "
        "hyperparameter tuning, dan evaluasi lebih lengkap."
    )

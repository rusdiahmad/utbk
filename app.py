# app.py
"""
UTBK Subtest Score Analysis & Jurusan Prediction Dashboard (Profesional)
- Pastikan file: NILAI UTBK ANGK 4.xlsx, Pas Photo.jpg, buku.jpg ada di root repo.
- requirements.txt minimal:
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
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import io

# ---------------------------
# Page & style
# ---------------------------
st.set_page_config(page_title="UTBK Subtest Analysis & Jurusan Predictor", page_icon="🎓", layout="wide")
sns.set_style("whitegrid")

# Header (professional)
st.title("📈 UTBK Subtest Analysis & Jurusan Prediction Dashboard")
st.markdown(
    "Dashboard profesional untuk menganalisis nilai per-subtes UTBK dan merekomendasikan *rumpun* serta "
    "jurusan/prodi yang paling cocok berdasarkan pola nilai sub-test (PU, PK, PPU, PBM, LIND, LING, PM)."
)
st.write("---")

# Sidebar navigation
st.sidebar.header("Navigasi")
page = st.sidebar.radio("Pilih halaman:", ["Tentang Saya", "Proyek Saya", "Analisis UTBK", "Prediksi Jurusan"], index=0)

# ---------------------------
# Utilities
# ---------------------------
@st.cache_data
def load_excel(path: str):
    return pd.read_excel(path)

def detect_subtests(df: pd.DataFrame):
    expected = ["PU", "PK", "PPU", "PBM", "LIND", "LING", "PM", "Rata-rata"]
    found = [c for c in expected if c in df.columns]
    # try variants for rata-rata
    if "Rata-rata" not in found:
        for c in df.columns:
            if c.strip().lower() in ["rata-rata", "rata rata", "rata2", "rata_rata", "rata"]:
                found.append(c)
                break
    return found

def prepare_features(df: pd.DataFrame, feature_cols):
    X = df[feature_cols].copy()
    # fill numeric with median and categorical as string
    for c in X.columns:
        if X[c].dtype in [np.float64, np.int64]:
            X[c] = X[c].fillna(X[c].median())
        else:
            X[c] = X[c].astype(str).fillna("___NA___")
    return X

def encode_X(X: pd.DataFrame):
    encoders = {}
    X_enc = pd.DataFrame(index=X.index)
    for col in X.columns:
        if X[col].dtype == object or X[col].dtype.name == "category":
            le = LabelEncoder()
            X_enc[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
        else:
            X_enc[col] = X[col].astype(float)
    return X_enc, encoders

def transform_X_new(X_new: pd.DataFrame, encoders: dict, model_features: list):
    X_t = pd.DataFrame(index=X_new.index)
    for col in model_features:
        if col in X_new.columns:
            series = X_new[col].astype(str).fillna("___NA___")
        else:
            series = pd.Series(["___NA___"] * len(X_new), index=X_new.index)
        if col in encoders:
            le = encoders[col]
            # map unseen labels to -1
            mapping = {c: i for i, c in enumerate(le.classes_)}
            X_t[col] = series.map(lambda v: mapping.get(v, -1))
        else:
            # numeric
            X_t[col] = pd.to_numeric(series, errors='coerce').fillna(0)
    return X_t

def plot_radar(values, labels, title="Radar Chart"):
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    # repeat first value to close the circle
    values = values.tolist()
    values += values[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title(title)
    ax.grid(True)
    return fig

# ---------------------------
# PAGE: TENTANG SAYA (interactive CV)
# ---------------------------
if page == "Tentang Saya":
    st.header("👨‍🏫 Tentang Saya")
    col1, col2 = st.columns([1, 2.5])
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
            "> Mengaplikasikan AI dan analisis data untuk meningkatkan mutu pembelajaran dan pengambilan keputusan pendidikan."
        )
        st.write("---")
        st.markdown("### 🎓 Pendidikan")
        st.markdown(
            "- **S2 Matematika — Universitas Andalas (UNAND)**\n"
            "- **S1 Pendidikan Matematika — Universitas Negeri Padang (UNP)**"
        )
        st.write("---")
        st.markdown("### 💼 Pengalaman Kerja")
        st.markdown(
            "- **Guru Matematika — Bimbingan Belajar Bintang Pelajar (2024–Sekarang)**\n"
            "- **Guru Matematika — SMAN Agam Cendekia**\n"
            "- **Instruktur — Bimbel CPNS DINNDA**\n"
            "- **Pembina Ekstrakurikuler Robotik & Coding (SMA)**"
        )
        st.write("---")
        st.markdown("### 🧠 Keahlian Teknis")
        left, right = st.columns(2)
        with left:
            st.write("Python (pandas, scikit-learn, Streamlit)")
            st.progress(90)
            st.write("Data Visualization (Matplotlib, Seaborn)")
            st.progress(85)
            st.write("Machine Learning")
            st.progress(80)
        with right:
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
# PAGE: PROYEK SAYA
# ---------------------------
elif page == "Proyek Saya":
    st.header("💼 Proyek Saya")
    st.markdown("Koleksi proyek yang merepresentasikan kemampuan saya dalam AI/ML untuk pendidikan dan bisnis.")
    st.write("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("📘 Analisis Data UTBK")
        try:
            st.image("buku.jpg", use_column_width=True)
        except Exception:
            st.image("https://cdn-icons-png.flaticon.com/512/2232/2232688.png", width=120)
        st.write("Analisis nilai per-subtes UTBK serta rekomendasi rumpun/jurusan berdasarkan profil nilai siswa.")
        st.markdown("**Teknologi:** Python, pandas, scikit-learn, Streamlit")
    with c2:
        st.subheader("📶 Telco Customer Churn")
        st.image("https://cdn-icons-png.flaticon.com/512/2910/2910768.png", width=120)
        st.write("Proyek klasifikasi churn pelanggan telco: EDA, feature engineering, dan model Random Forest.")
        st.markdown("**Teknologi:** Python, scikit-learn, SHAP (explainability)")
    with c3:
        st.subheader("🏪 Sales Supermarket Analytics")
        st.image("https://cdn-icons-png.flaticon.com/512/2620/2620608.png", width=120)
        st.write("Analisis penjualan supermarket: trend, RFM, dan forecasting sederhana.")
        st.markdown("**Teknologi:** pandas, Prophet/ARIMA (opsional), Streamlit")

# ---------------------------
# PAGE: ANALISIS UTBK
# ---------------------------
elif page == "Analisis UTBK":
    st.header("📊 Analisis Nilai Per-Subtest UTBK")
    st.markdown(
        "Halaman ini menampilkan analisis hubungan antar nilai sub-test (PU, PK, PPU, PBM, LIND, LING, PM) "
        "dan bagaimana pola nilai tersebut berkorelasi dengan rumpun/jurusan."
    )
    # load data
    upload = st.file_uploader("Upload file Excel UTBK (.xlsx) — atau kosongkan untuk baca dari repo", type=["xlsx"])
    if upload is None:
        try:
            df = load_excel("NILAI UTBK ANGK 4.xlsx")
            st.success("Dataset dimuat dari repository.")
        except FileNotFoundError:
            st.info("File 'NILAI UTBK ANGK 4.xlsx' tidak ditemukan di repo. Silakan upload.")
            st.stop()
        except Exception as e:
            st.error(f"Gagal membaca file Excel: {e}")
            st.stop()
    else:
        try:
            df = pd.read_excel(upload)
            st.success("Dataset berhasil di-upload.")
        except Exception as e:
            st.error(f"Gagal membaca file upload: {e}")
            st.stop()

    df.columns = [c.strip() for c in df.columns]

    subtests = detect_subtests(df)
    if not subtests:
        st.warning("Kolom subtests (PU, PK, PPU, PBM, LIND, LING, PM, Rata-rata) tidak ditemukan.")
        st.stop()

    st.subheader("Statistik Deskriptif Subtests")
    st.dataframe(df[subtests].describe().T)

    st.subheader("Distribusi Nilai")
    sel = st.selectbox("Pilih subtest untuk visualisasi", subtests)
    fig, axs = plt.subplots(1,2, figsize=(12,4))
    sns.histplot(df[sel].dropna(), kde=True, ax=axs[0])
    axs[0].set_title(f"Distribusi {sel}")
    sns.boxplot(x=df[sel].dropna(), ax=axs[1])
    axs[1].set_title(f"Boxplot {sel}")
    st.pyplot(fig)

    st.subheader("Korelasi Antar Subtest")
    fig_corr, ax_corr = plt.subplots(figsize=(8,6))
    sns.heatmap(df[subtests].corr(), annot=True, cmap="vlag", center=0, ax=ax_corr)
    st.pyplot(fig_corr)

    # relationship to rumpun
    st.subheader("Analisis Rata-rata Nilai per Rumpun / Jurusan")
    group_col = None
    if "RUMPUN" in df.columns:
        group_col = "RUMPUN"
    elif "JURUSAN/PRODI" in df.columns:
        group_col = "JURUSAN/PRODI"
    else:
        st.info("Kolom 'RUMPUN' atau 'JURUSAN/PRODI' tidak ditemukan dalam dataset.")
    if group_col:
        top_n = st.slider("Top N rumpun/jurusan tampilkan (berdasarkan jumlah siswa)", 2, 10, 5)
        counts = df[group_col].value_counts().head(top_n)
        st.markdown("Jumlah siswa per group (top)")
        st.bar_chart(counts)

        mean_by_group = df.groupby(group_col)[subtests].mean().reset_index()
        # show top groups
        st.markdown("Rata-rata nilai per subtest per group (top groups)")
        display = mean_by_group.set_index(group_col).loc[counts.index]
        st.dataframe(display)

        # radar chart for selected group
        st.subheader("Radar Chart: Profil Subtest per Group")
        sel_group = st.selectbox("Pilih group untuk radar chart", options=display.index.tolist())
        vals = display.loc[sel_group].values
        fig_radar = plot_radar(vals, subtests, title=f"Profil Subtest: {sel_group}")
        st.pyplot(fig_radar)

        # Provide simple rule-based insight example
        st.markdown("**Insight contoh (rule-based)**")
        st.write(
            "- Jika **PM & PK** rata-rata tinggi → kecenderungan **MIPA / Teknik**.\n"
            "- Jika **PPU & PBM** tinggi → kecenderungan **Soshum / Humaniora**.\n"
            "- Jika **LIND / LING** kuat → nilai bahasa / communicative skills lebih baik (cocok prodi berbahasa).\n"
        )

# ---------------------------
# PAGE: PREDIKSI JURUSAN (REVISI ROBUST)
# ---------------------------
elif page == "Prediksi Jurusan":
    st.header("🤖 Prediksi Rumpun & Rekomendasi Jurusan dari Nilai Subtests")
    st.markdown(
        "Masukkan nilai subtests untuk memprediksi **rumpun** dan melihat rekomendasi jurusan/prodi. "
        "Model dilatih pada data yang ada di file `NILAI UTBK ANGK 4.xlsx`."
    )

    # --- Load dataset (required)
    try:
        df_all = load_excel("NILAI UTBK ANGK 4.xlsx")
    except FileNotFoundError:
        st.error("File 'NILAI UTBK ANGK 4.xlsx' tidak ditemukan. Upload file lalu kembali ke halaman ini.")
        st.stop()
    except Exception as e:
        st.error(f"Gagal membaca file Excel: {e}")
        st.stop()

    # Normalize
    df_all.columns = [c.strip() for c in df_all.columns]

    # Detect subtests
    subtests = detect_subtests(df_all)
    if not subtests:
        st.warning("Kolom subtests tidak ditemukan. Pastikan ada kolom: PU, PK, PPU, PBM, LIND, LING, PM, Rata-rata")
        st.stop()

    # Choose target: prefer RUMPUN, else use JURUSAN/PRODI as proxy
    if "RUMPUN" in df_all.columns:
        target_col = "RUMPUN"
    elif "JURUSAN/PRODI" in df_all.columns:
        target_col = "JURUSAN/PRODI"
        st.info("Kolom 'RUMPUN' tidak ditemukan; menggunakan 'JURUSAN/PRODI' sebagai target proxy.")
    else:
        st.error("Tidak ditemukan kolom target 'RUMPUN' atau 'JURUSAN/PRODI' dalam dataset.")
        st.stop()

    # --- Prepare training data
    df_train = df_all.dropna(subset=subtests + [target_col]).reset_index(drop=True)
    n_rows = df_train.shape[0]
    if n_rows == 0:
        st.error("Tidak ada baris lengkap (subtests + target). Periksa dataset.")
        st.stop()
    if n_rows < 30:
        st.warning(f"Data latih kecil ({n_rows} baris). Model mungkin kurang stabil, namun tetap akan dilatih.")

    # X: numeric subtests only
    X = df_train[subtests].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors='coerce').fillna(X[c].median())

    # y: target (rumpun / jurusan)
    y_raw = df_train[target_col].astype(str).fillna("Unknown")
    le_rumpun = LabelEncoder()
    y = le_rumpun.fit_transform(y_raw)

    # if only one class present, classifier won't work well — guard
    if len(np.unique(y)) < 2:
        st.error("Hanya satu kelas rumpun ditemukan di data. Perlu setidaknya 2 rumpun berbeda untuk melakukan klasifikasi.")
        st.stop()

    # train/test split (try stratify if possible)
    try:
        stratify = y if len(np.unique(y)) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
    except Exception:
        # fallback no stratify
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # train classifier
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    with st.spinner("Melatih model classifier..."):
        clf.fit(X_train, y_train)

    # evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.subheader("Evaluasi Model (Rumpun)")
    st.metric("Akurasi (test set)", f"{acc:.3f}")
    st.text("Classification report:")
    st.text(classification_report(y_test, y_pred, target_names=le_rumpun.classes_, zero_division=0))

    # confusion matrix
    try:
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=le_rumpun.classes_, yticklabels=le_rumpun.classes_, cmap="Blues", ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        st.pyplot(fig_cm)
    except Exception:
        st.info("Confusion matrix skipped (masalah plotting).")

    # Prepare top jurusan per rumpun mapping (if available)
    top_jurusan_by_rumpun = {}
    if "RUMPUN" in df_train.columns and "JURUSAN/PRODI" in df_train.columns:
        # list jurusan ordered by frequency per rumpun
        tmp = df_train.groupby("RUMPUN")["JURUSAN/PRODI"].apply(lambda s: s.value_counts().index.tolist())
        top_jurusan_by_rumpun = tmp.to_dict()

    # --- Prediction UI (manual)
    st.write("---")
    st.subheader("Input nilai subtests untuk prediksi rumpun & rekomendasi jurusan")
    with st.form("pred_form"):
        cols = st.columns(4)
        inputs = {}
        for i, s in enumerate(subtests):
            c = cols[i % 4]
            default_val = float(df_all[s].dropna().median()) if s in df_all.columns and not df_all[s].dropna().empty else 50.0
            inputs[s] = c.number_input(f"{s}", value=default_val, min_value=0.0, max_value=1000.0, step=1.0)
        submit = st.form_submit_button("🔮 Prediksi")

    if submit:
        try:
            X_new = pd.DataFrame([inputs])
            # ensure numeric and fill medians from training X
            for c in X_new.columns:
                X_new[c] = pd.to_numeric(X_new[c], errors='coerce').fillna(X[c].median() if c in X.columns else 0)

            # predict rumpun
            pred_code = clf.predict(X_new)[0]
            pred_rumpun = le_rumpun.inverse_transform([pred_code])[0]
            st.success(f"Prediksi Rumpun: **{pred_rumpun}**")

            # probabilities (if available)
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(X_new)[0]
                prob_df = pd.DataFrame({"Rumpun": le_rumpun.classes_, "Prob": probs}).sort_values("Prob", ascending=False)
                st.subheader("Probabilitas Rumpun (desc)")
                st.dataframe(prob_df)
            else:
                st.info("Model tidak menyediakan probabilitas (predict_proba).")

            # Recommend top-3 jurusan:
            recommendations = []
            if pred_rumpun in top_jurusan_by_rumpun and top_jurusan_by_rumpun[pred_rumpun]:
                recommendations = top_jurusan_by_rumpun[pred_rumpun][:3]
            else:
                # fallback: nearest neighbors in training set
                try:
                    X_vals = X.values
                    new_vals = X_new.values
                    dists = np.linalg.norm(X_vals - new_vals, axis=1)
                    nearest_idx = np.argsort(dists)[:20]
                    nearest = df_train.iloc[nearest_idx]
                    if "JURUSAN/PRODI" in nearest.columns:
                        recommendations = nearest["JURUSAN/PRODI"].value_counts().index.tolist()[:3]
                except Exception:
                    recommendations = ["(Rekomendasi tidak tersedia)"]

            st.subheader("Rekomendasi Jurusan (Top 3)")
            if recommendations:
                for i, r in enumerate(recommendations, start=1):
                    st.markdown(f"{i}. **{r}**")
            else:
                st.write("Tidak ada rekomendasi jurusan yang tersedia.")

            # Radar chart: show average profile of predicted rumpun if available
            if "RUMPUN" in df_all.columns and pred_rumpun in df_all["RUMPUN"].unique():
                avg_profile = df_all[df_all["RUMPUN"] == pred_rumpun][subtests].mean().fillna(0)
                st.subheader(f"Profil rata-rata subtest untuk rumpun: {pred_rumpun}")
                fig_radar = plot_radar(avg_profile.values, subtests, title=f"Profil {pred_rumpun}")
                st.pyplot(fig_radar)

            # package output for download
            out = {"Predicted_Rumpun": pred_rumpun}
            out.update(inputs)
            out_df = pd.DataFrame([out])
            csv = out_df.to_csv(index=False).encode("utf-8")
            st.download_button("💾 Download Hasil Prediksi (CSV)", data=csv, file_name="prediksi_rumpun.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Gagal melakukan prediksi: {e}")
            st.exception(e)


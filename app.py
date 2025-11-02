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
# PAGE: PREDIKSI JURUSAN (VERSI SIMPLIFIKASI PROFESIONAL)
# ---------------------------
elif page == "Prediksi Jurusan":
    st.header("🎯 Rekomendasi Jurusan Berdasarkan Nilai Subtes UTBK")
    st.markdown(
        "Masukkan nilai per subtes (PU, PK, PPU, PBM, LIND, LING, PM, dan Rata-rata) untuk mengetahui "
        "jurusan atau rumpun yang paling sesuai dengan profil kemampuanmu."
    )

    # --- Input nilai subtes
    st.write("### 🧮 Input Nilai Subtes")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        PU = st.number_input("PU (Penalaran Umum)", min_value=0.0, max_value=1000.0, value=600.0)
        PK = st.number_input("PK (Pengetahuan Kuantitatif)", min_value=0.0, max_value=1000.0, value=600.0)
    with col2:
        PPU = st.number_input("PPU (Pemahaman Bacaan & Menulis)", min_value=0.0, max_value=1000.0, value=600.0)
        PBM = st.number_input("PBM (Pengetahuan & Pemahaman Umum)", min_value=0.0, max_value=1000.0, value=600.0)
    with col3:
        LIND = st.number_input("LIND (Literasi Bahasa Indonesia)", min_value=0.0, max_value=1000.0, value=600.0)
        LING = st.number_input("LING (Literasi Bahasa Inggris)", min_value=0.0, max_value=1000.0, value=600.0)
    with col4:
        PM = st.number_input("PM (Penalaran Matematika)", min_value=0.0, max_value=1000.0, value=600.0)
        rata = st.number_input("Rata-rata", min_value=0.0, max_value=1000.0, value=600.0)

    # --- Tombol prediksi
    if st.button("🔍 Lihat Rekomendasi Jurusan"):
        # Analisis berbasis aturan sederhana
        hasil = []

        # Logika dasar berbasis pola nilai subtes
        if PM > 650 and PK > 650:
            hasil.append("💻 Teknik Informatika / Ilmu Komputer")
            hasil.append("⚙️ Teknik Industri / Elektro / Mesin")
            hasil.append("🧮 Matematika / Statistika / Fisika")
        elif PPU > 650 and PBM > 650:
            hasil.append("📚 Hukum / Ilmu Sosial / Komunikasi")
            hasil.append("🏛️ Ilmu Pemerintahan / Hubungan Internasional")
            hasil.append("🧠 Psikologi / Pendidikan")
        elif LIND > 650 and LING > 650:
            hasil.append("🗣️ Sastra Inggris / Pendidikan Bahasa")
            hasil.append("📖 Jurnalistik / Ilmu Komunikasi")
            hasil.append("🌏 Bahasa dan Kebudayaan / Linguistik")
        elif PK > 650 and PBM > 650:
            hasil.append("📊 Ekonomi / Manajemen / Akuntansi")
            hasil.append("💼 Administrasi Bisnis / Perbankan")
            hasil.append("🏦 Keuangan / Aktuaria")
        elif PM < 550 and PPU > 650:
            hasil.append("🧠 Psikologi / Ilmu Pendidikan / Bimbingan Konseling")
            hasil.append("🎨 Desain Komunikasi Visual / Seni")
            hasil.append("📚 Sastra dan Bahasa")
        else:
            hasil.append("📘 Multidisiplin / Jurusan Umum")
            hasil.append("📈 Bisnis dan Manajemen")
            hasil.append("💡 Pendidikan atau Sosial Humaniora")

        # --- Output hasil rekomendasi
        st.success("Berdasarkan profil nilai kamu, jurusan yang paling sesuai adalah:")
        for i, j in enumerate(hasil, start=1):
            st.markdown(f"**{i}. {j}**")

        # Tambahan insight
        st.write("---")
        st.info(
            "💡 *Keterangan:* Rekomendasi ini bersifat orientatif berdasarkan pola nilai subtes. "
            "Untuk hasil lebih akurat, sebaiknya dipadukan dengan minat dan rencana karier pribadi."
        )

        # Radar Chart visualisasi
        subtests = ["PU", "PK", "PPU", "PBM", "LIND", "LING", "PM", "Rata-rata"]
        values = [PU, PK, PPU, PBM, LIND, LING, PM, rata]
        fig = plot_radar(values, subtests, title="Profil Nilai Subtes Kamu")
        st.pyplot(fig)

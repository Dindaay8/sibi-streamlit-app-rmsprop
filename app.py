import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# ======================
# KONFIGURASI MODEL
# ======================
IMG_SIZE = (96, 96)
MODEL_PATH = "best_RMSprop.h5"
CLASS_NAMES_PATH = "class_names.npy"

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Sistem Pengenalan SIBI",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================
# SESSION STATE
# ======================
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# ======================
# LOAD MODEL
# ======================
@st.cache_resource
def load_cnn_model():
    try:
        model = load_model(MODEL_PATH, compile=False)
        class_names = np.load(CLASS_NAMES_PATH, allow_pickle=True)
        return model, class_names
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()


model, class_names = load_cnn_model()

# ======================
# SIDEBAR TOGGLE
# ======================
st.sidebar.title("⚙️ Pengaturan Tampilan")
st.session_state.dark_mode = st.sidebar.toggle(
    "🌙 Dark Mode",
    value=st.session_state.dark_mode
)

# ======================
# WARNA TEMA
# ======================
if st.session_state.dark_mode:
    BG = "#0e1117"
    SIDEBAR_BG = "#0b0f14"
    CARD = "#161b22"
    TEXT = "#e6edf3"
    MUTED = "#9ba3b4"
    ACCENT = "#58a6ff"
    BORDER = "#30363d"
else:
    BG = "#f5f7fa"
    SIDEBAR_BG = "#ffffff"
    CARD = "#ffffff"
    TEXT = "#000000"
    MUTED = "#6c757d"
    ACCENT = "#1f77b4"
    BORDER = "#d0d7de"

# ======================
# CSS GLOBAL (FULL DARK MODE FIX)
# ======================
st.markdown(f"""
<style>

/* ===== APP BACKGROUND ===== */
html, body, [data-testid="stApp"] {{
    background-color: {BG};
    color: {TEXT};
}}

/* ===== STREAMLIT HEADER (HILANGKAN PUTIH) ===== */
[data-testid="stHeader"] {{
    background-color: {BG};
}}

[data-testid="stHeader"]::after {{
    background: none;
}}

[data-testid="stToolbar"] {{
    background-color: {BG};
}}

/* ===== SIDEBAR ===== */
[data-testid="stSidebar"] {{
    background-color: {SIDEBAR_BG};
    border-right: 1px solid {BORDER};
}}

[data-testid="stSidebar"] * {{
    color: {TEXT} !important;
}}

/* ===== TEXT ===== */
h1, h2, h3, h4, h5, h6, p, span, label, div {{
    color: {TEXT};
}}

/* ===== FILE UPLOADER ===== */
[data-testid="stFileUploader"] {{
    background-color: {CARD};
    border-radius: 12px;
    padding: 15px;
    border: 1px solid {BORDER};
}}

/* ===== CARD ===== */
.card {{
    background-color: {CARD};
    padding: 30px;
    border-radius: 16px;
    border: 1px solid {BORDER};
    box-shadow: 0 6px 20px rgba(0,0,0,0.25);
}}

/* ===== RESULT LETTER ===== */
.result-letter {{
    font-size: 80px;
    font-weight: bold;
    color: {ACCENT};
    text-align: center;
}}

/* ===== SUBTITLE ===== */
.subtitle {{
    color: {MUTED};
    font-size: 16px;
}}

</style>
""", unsafe_allow_html=True)

# ======================
# SIDEBAR INFO
# ======================
st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Tentang Aplikasi")
st.sidebar.write("""
Sistem ini menggunakan **Convolutional Neural Network (CNN)**  
untuk mengenali **huruf Bahasa Isyarat Indonesia (SIBI)**  
berdasarkan citra tangan.
""")

st.sidebar.markdown("---")
st.sidebar.write("**Input:** Citra tangan")
st.sidebar.write("**Output:** Huruf A–Z")
st.sidebar.write("**Model:** CNN + RMSprop")
st.sidebar.markdown("---")
st.sidebar.caption("© Sistem Pengenalan SIBI")

# ======================
# HEADER UTAMA
# ======================
st.title("Sistem Pengenalan Bahasa Isyarat Indonesia (SIBI)")
st.markdown(
    "<p class='subtitle'>Unggah citra isyarat tangan untuk memprediksi huruf SIBI menggunakan model CNN.</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# ======================
# MAIN LAYOUT
# ======================
col1, col2 = st.columns([1, 1.2])

# ======================
# LEFT COLUMN (UPLOAD)
# ======================
with col1:
    st.markdown("### 📤 Upload Gambar")
    uploaded_file = st.file_uploader(
        "Pilih gambar isyarat tangan",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Citra Input", use_column_width=True)

# ======================
# RIGHT COLUMN (RESULT)
# ======================
with col2:
    st.markdown("### 📊 Hasil Prediksi")

    if uploaded_file:
        img = image.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array, verbose=0)
        idx = np.argmax(preds)
        pred_class = class_names[idx]
        confidence = float(np.max(preds))

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='result-letter'>{pred_class}</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<p style='text-align:center;'>Confidence: <b>{confidence*100:.2f}%</b></p>",
            unsafe_allow_html=True
        )
        st.progress(confidence)
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("📈 Detail Probabilitas Semua Kelas"):
            prob_dict = {
                class_names[i]: float(preds[0][i])
                for i in range(len(class_names))
            }
            st.bar_chart(prob_dict)
    else:
    st.info("Silakan upload gambar untuk melihat hasil prediksi.")

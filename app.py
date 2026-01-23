import streamlit as st
import numpy as np
import tensorflow as tf
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
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )
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
# HEADER
# ======================
st.title("Sistem Pengenalan Bahasa Isyarat Indonesia (SIBI)")
st.markdown("Unggah citra isyarat tangan untuk memprediksi huruf SIBI menggunakan CNN.")
st.markdown("---")

col1, col2 = st.columns([1, 1.2])

with col1:
    uploaded_file = st.file_uploader(
        "📤 Upload gambar isyarat tangan",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Citra Input", use_column_width=True)

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

        st.markdown(f"### 🔤 Huruf: **{pred_class}**")
        st.progress(confidence)
        st.write(f"Confidence: **{confidence*100:.2f}%**")

    else:
        st.info("Silakan upload gambar untuk melihat hasil prediksi.")

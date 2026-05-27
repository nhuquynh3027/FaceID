import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import urllib.request
import os
import base64
from io import BytesIO

# =========================================================================
# CẤU HÌNH TRANG
# =========================================================================
st.set_page_config(
    page_title="Face Recognition AI",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =========================================================================
# CSS TOÀN CỤC — DARK BIOMETRIC THEME
# =========================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
@import url('https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@latest/tabler-icons.min.css');

/* ---- Reset & root ---- */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #0a0c0f !important;
    color: #e8e6e1 !important;
    font-family: 'Syne', sans-serif !important;
}
[data-testid="stHeader"], [data-testid="stToolbar"] { display: none !important; }
[data-testid="stMainBlockContainer"] { padding: 2rem 1.5rem 3rem !important; max-width: 700px !important; }
section[data-testid="stFileUploaderDropzone"] { background: rgba(29,158,117,0.04) !important; border: 0.5px dashed rgba(29,158,117,0.35) !important; border-radius: 12px !important; }
section[data-testid="stFileUploaderDropzone"]:hover { background: rgba(29,158,117,0.08) !important; border-color: rgba(29,158,117,0.6) !important; }
[data-testid="stFileUploaderDropzoneInstructions"] p { color: #888780 !important; font-family: 'JetBrains Mono', monospace !important; font-size: 13px !important; }
[data-testid="stCameraInputButton"] button, .stButton > button {
    background: linear-gradient(135deg, #0f6e56, #1d9e75) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #fff !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    letter-spacing: 0.02em !important;
    padding: 0.65rem 1.5rem !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
.stButton > button:hover { background: linear-gradient(135deg, #1d9e75, #5dcaa5) !important; transform: translateY(-1px) !important; }
[data-testid="stRadio"] > div { gap: 10px !important; }
[data-testid="stRadio"] label { background: rgba(255,255,255,0.03) !important; border: 0.5px solid rgba(255,255,255,0.1) !important; border-radius: 10px !important; padding: 10px 14px !important; cursor: pointer !important; color: #888780 !important; font-family: 'Syne', sans-serif !important; font-size: 14px !important; }
[data-testid="stRadio"] label:has(input:checked) { background: rgba(29,158,117,0.1) !important; border-color: rgba(29,158,117,0.4) !important; color: #e8e6e1 !important; }
[data-testid="stImage"] img { border-radius: 10px !important; border: 0.5px solid rgba(255,255,255,0.08) !important; }
[data-testid="stSpinner"] { color: #5dcaa5 !important; }
[data-testid="stAlert"] { border-radius: 10px !important; font-family: 'JetBrains Mono', monospace !important; }
hr { border-color: rgba(255,255,255,0.06) !important; margin: 1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# =========================================================================
# HELPER: BADGE STATUS
# =========================================================================
def badge(text, color="#1d9e75", bg="rgba(29,158,117,0.12)", border="rgba(29,158,117,0.3)"):
    return f"""
    <span style="display:inline-flex;align-items:center;gap:6px;background:{bg};
        border:0.5px solid {border};border-radius:20px;padding:4px 12px;
        font-size:11px;font-family:'JetBrains Mono',monospace;color:{color};letter-spacing:0.05em;">
        <span style="width:6px;height:6px;border-radius:50%;background:{color};
            box-shadow:0 0 6px {color};display:inline-block;animation:none;"></span>
        {text}
    </span>"""

def confidence_bar(conf: float) -> str:
    is_ok  = conf >= 65
    fill   = "linear-gradient(90deg,#0f6e56,#5dcaa5)" if is_ok else "linear-gradient(90deg,#854f0b,#ef9f27)"
    c_text = "#5dcaa5" if is_ok else "#ef9f27"
    return f"""
    <div style="margin:12px 0 8px;">
        <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
            <span style="font-size:11px;color:#888780;font-family:'JetBrains Mono',monospace;letter-spacing:0.1em;">ĐỘ TIN CẬY</span>
            <span style="font-size:12px;color:{c_text};font-family:'JetBrains Mono',monospace;font-weight:500;">{conf:.1f}%</span>
        </div>
        <div style="height:4px;background:rgba(255,255,255,0.08);border-radius:2px;overflow:hidden;">
            <div style="height:100%;width:{conf}%;background:{fill};border-radius:2px;
                transition:width 0.8s cubic-bezier(0.4,0,0.2,1);"></div>
        </div>
    </div>"""

def result_card(name: str, conf: float) -> str:
    is_ok = conf >= 65
    border_col = "rgba(29,158,117,0.35)" if is_ok else "rgba(239,159,39,0.35)"
    bg_col     = "rgba(29,158,117,0.06)"  if is_ok else "rgba(239,159,39,0.06)"
    status_badge = (
        badge("Nhận diện thành công") if is_ok
        else badge("Cần xác minh lại","#ef9f27","rgba(239,159,39,0.1)","rgba(239,159,39,0.3)")
    )
    warn_html = ""
    if not is_ok:
        warn_html = """
        <div style="background:rgba(239,159,39,0.06);border:0.5px solid rgba(239,159,39,0.25);
            border-radius:8px;padding:12px 14px;font-size:12px;color:#ef9f27;
            font-family:'JetBrains Mono',monospace;margin-top:12px;">
            ⚠ Độ tin cậy thấp — hãy thử căn chỉnh lại góc mặt hoặc tăng độ sáng.
        </div>"""
    return f"""
    <div style="background:{bg_col};border:0.5px solid {border_col};border-radius:12px;
        padding:20px 22px;margin-top:20px;animation:fadeIn 0.4s ease;">
        <div style="font-size:11px;color:#5dcaa5;font-family:'JetBrains Mono',monospace;
            letter-spacing:0.1em;margin-bottom:4px;">DANH TÍNH PHÁT HIỆN</div>
        <div style="font-size:22px;font-weight:800;letter-spacing:-0.01em;
            margin-bottom:14px;color:#e8e6e1;">{name}</div>
        {confidence_bar(conf)}
        <div style="margin-top:10px;">{status_badge}</div>
        {warn_html}
    </div>
    <style>@keyframes fadeIn{{from{{opacity:0;transform:translateY(8px)}}to{{opacity:1;transform:translateY(0)}}}}</style>"""

# =========================================================================
# LOAD MODEL
# =========================================================================
@st.cache_resource(show_spinner=False)
def load_model_ai():
    model_path = "60anh_model.h5"
    if not os.path.exists(model_path):
        url = (
            "https://www.dropbox.com/scl/fi/0naoddly80pf587648upu/"
            "60anh_model.h5?rlkey=ksxohalnnkvrek3kysezmmzlm&st=1y87dwyi&dl=1"
        )
        with st.spinner("Đang tải mô hình AI lần đầu — vui lòng đợi..."):
            urllib.request.urlretrieve(url, model_path)
    return tf.keras.models.load_model(model_path, compile=False)

LABELS = {
    0: "BUI DANG KHOI",           1: "DANG NGUYEN PHUONG NGHI",
    2: "HA PHUONG THAO",          3: "HOANG BAO TRAN",
    4: "HOANG BUI TRA MY",        5: "LE HUYNH DUC HUY",
    6: "LE MINH TRIET",           7: "LE THAI BAO",
    8: "LE THI NHU QUYNH",        9: "LE TRAN QUY ANH",
    10: "LE TRONG DAI",           11: "MAI HO QUOC TUY",
    12: "NGUYEN BAO HAN",         13: "NGUYEN DONG HAI",
    14: "NGUYEN HOANG BAO",       15: "NGUYEN HUU TOAN",
    16: "NGUYEN KHAC LUU VU",     17: "NGUYEN NGOC KHANH UYEN",
    18: "NGUYEN NGOC KIM TUYET",  19: "NGUYEN THI THANH HA",
    20: "NGUYEN TRONG MINH",      21: "NHAN MANH TUAN",
    22: "PHAM DUC THANH CONG",    23: "PHAM LY BAO LAM",
    24: "PHAM MAI PHUONG",        25: "THAI TUAN PHAT",
    26: "TRAN GIA HAN",           27: "TRAN MINH HOANG",
    28: "TRAN NGOC THAO ANH",     29: "TRAN THE DANG KHOA",
    30: "TRINH THUY HANG",
}

def predict(model, pil_image: Image.Image):
    img = pil_image.convert("RGB").resize((200, 200))
    arr = np.expand_dims(np.array(img) / 255.0, axis=0)
    res = model.predict(arr, verbose=0)
    idx = int(np.argmax(res))
    conf = float(res[0][idx]) * 100
    return LABELS.get(idx, f"ID chưa đặt tên: {idx}"), conf

# =========================================================================
# UI — HEADER
# =========================================================================
st.markdown(badge("HỆ THỐNG ĐANG HOẠT ĐỘNG"), unsafe_allow_html=True)
st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

st.markdown("""
<h1 style='font-family:Syne,sans-serif;font-size:30px;font-weight:800;
    letter-spacing:-0.02em;line-height:1.1;margin-bottom:6px;color:#e8e6e1;'>
    Nhận Diện<br><span style='color:#5dcaa5;'>Khuôn Mặt AI</span>
</h1>
<p style='font-family:"JetBrains Mono",monospace;font-size:12px;color:#888780;margin-bottom:28px;'>
    // deep_vision_v2.0 · 31 danh tính · accuracy_threshold: 65%
</p>
""", unsafe_allow_html=True)

# =========================================================================
# UI — CHẾ ĐỘ NẠP ẢNH
# =========================================================================
mode = st.radio(
    "Chọn cách nạp ảnh khuôn mặt:",
    ["📁  Tải file ảnh lên", "📷  Chụp bằng Camera"],
    horizontal=True,
    label_visibility="collapsed",
)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

image_source: Image.Image | None = None

if "Tải file" in mode:
    uploaded = st.file_uploader(
        "Chọn ảnh khuôn mặt",
        type=["jpg", "png", "jpeg"],
        label_visibility="collapsed",
    )
    if uploaded:
        image_source = Image.open(uploaded)
else:
    cam = st.camera_input("Đưa khuôn mặt vào giữa khung hình", label_visibility="collapsed")
    if cam:
        image_source = Image.open(cam)

# =========================================================================
# UI — XEM TRƯỚC & DỰ ĐOÁN
# =========================================================================
if image_source:
    if "Tải file" in mode:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # Đóng khung biometric quanh ảnh preview
        buf = BytesIO()
        image_source.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        st.markdown(f"""
        <div style="position:relative;border-radius:10px;overflow:hidden;
            border:0.5px solid rgba(255,255,255,0.08);margin-bottom:16px;">
            <img src="data:image/png;base64,{b64}"
                style="width:100%;display:block;max-height:320px;object-fit:cover;border-radius:10px;">
            <div style="position:absolute;top:0;left:0;right:0;height:2px;
                background:linear-gradient(90deg,transparent,#1d9e75,transparent);opacity:0.7;"></div>
            <!-- corners -->
            <div style="position:absolute;top:8px;left:8px;width:16px;height:16px;
                border-top:2px solid #1d9e75;border-left:2px solid #1d9e75;opacity:0.8;"></div>
            <div style="position:absolute;top:8px;right:8px;width:16px;height:16px;
                border-top:2px solid #1d9e75;border-right:2px solid #1d9e75;opacity:0.8;"></div>
            <div style="position:absolute;bottom:8px;left:8px;width:16px;height:16px;
                border-bottom:2px solid #1d9e75;border-left:2px solid #1d9e75;opacity:0.8;"></div>
            <div style="position:absolute;bottom:8px;right:8px;width:16px;height:16px;
                border-bottom:2px solid #1d9e75;border-right:2px solid #1d9e75;opacity:0.8;"></div>
        </div>
        """, unsafe_allow_html=True)

    analyze = st.button(" Phân tích khuôn mặt")

    if analyze:
        try:
            model = load_model_ai()
        except Exception as e:
            st.error(f"Lỗi tải mô hình: {e}")
            st.stop()

        with st.spinner("AI đang quét và phân tích khuôn mặt..."):
            name, conf = predict(model, image_source)

        st.markdown(result_card(name, conf), unsafe_allow_html=True)

# =========================================================================
# FOOTER
# =========================================================================
st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
st.markdown("""<hr>
<p style='font-family:"JetBrains Mono",monospace;font-size:11px;color:#444441;text-align:center;'>
    deep_vision · face_id_system · model: 60anh_v1
</p>""", unsafe_allow_html=True)
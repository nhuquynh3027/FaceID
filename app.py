import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import urllib.request
import os

st.set_page_config(page_title="Face Recognition AI", layout="centered")
st.title("👤 Hệ Thống Nhận Diện Khuôn Mặt AI")
st.write("Chọn phương thức nạp dữ liệu dưới đây để AI tiến hành nhận diện danh tính.")

# =========================================================================
# 1. TẢI MÔ HÌNH TỪ DROPBOX (Giữ nguyên link của bạn)
# =========================================================================
@st.cache_resource
def load_model_ai():
    model_path = '60anh_model.h5'
    if not os.path.exists(model_path):
        with st.spinner('Đang tải mô hình AI từ server (Chỉ tải lần đầu, vui lòng đợi)...'):
            url = "https://www.dropbox.com/scl/fi/0naoddly80pf587648upu/60anh_model.h5?rlkey=ksxohalnnkvrek3kysezmmzlm&st=1y87dwyi&dl=1"
            urllib.request.urlretrieve(url, model_path)
    return tf.keras.models.load_model(model_path, compile=False)

try:
    model = load_model_ai()
    
    labels = {
        0: "BUI DANG KHOI",
        1: "DANG NGUYEN PHUONG NGHI",
        2: "HA PHUONG THAO",
        3: "HOANG BAO TRAN",
        4: "HOANG BUI TRA MY",
        5: "LE HUYNH DUC HUY",
        6: "LE MINH TRIET",
        7: "LE THAI BAO",
        8: "LE THI NHU QUYNH",
        9: "LE TRAN QUY ANH",
        10: "LE TRONG DAI",
        11: "MAI HO QUOC TUY",
        12: "NGUYEN BAO HAN",
        13: "NGUYEN DONG HAI",
        14: "NGUYEN HOANG BAO",
        15: "NGUYEN HUU TOAN",
        16: "NGUYEN KHAC LUU VU",
        17: "NGUYEN NGOC KHANH UYEN",
        18: "NGUYEN NGOC KIM TUYET",
        19: "NGUYEN THI THANH HA",
        20: "NGUYEN TRONG MINH",
        21: "NHAN MANH TUAN",
        22: "PHAM DUC THANH CONG",
        23: "PHAM LY BAO LAM",
        24: "PHAM MAI PHUONG",
        25: "THAI TUAN PHAT",
        26: "TRAN GIA HAN",
        27: "TRAN MINH HOANG",
        28: "TRAN NGOC THAO ANH",
        29: "TRAN THE DANG KHOA",
        30: "TRINH THUY HANG"
    }
    
except Exception as e:
    st.error(f"Lỗi khởi tạo hệ thống: {e}")

# =========================================================================
# 2. GIAO DIỆN CHỌN PHƯƠNG THỨC: TẢI ẢNH HOẶC CAMERA
# =========================================================================
# Tạo một thanh chọn phương thức ở menu bên trái hoặc ngay màn hình chính
hinh_thuc = st.radio("Chọn cách nạp ảnh khuôn mặt:", ("Tải file ảnh lên", "Chụp bằng Camera"))

image_source = None

if hinh_thuc == "Tải file ảnh lên":
    uploaded_file = st.file_uploader("Chọn ảnh khuôn mặt từ thiết bị...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image_source = Image.open(uploaded_file)
else:
    camera_file = st.camera_input("Đưa khuôn mặt vào giữa khung hình camera")
    if camera_file is not None:
        image_source = Image.open(camera_file)

# =========================================================================
# 3. TIỀN XỬ LÝ VÀ TIẾN HÀNH DỰ ĐOÁN
# =========================================================================
if image_source is not None:
    # Hiển thị ảnh đang được xử lý (Nếu dùng camera_input thì nó đã tự hiện ảnh chụp rồi)
    if hinh_thuc == "Tải file ảnh lên":
        st.image(image_source, caption='Ảnh bạn đã tải lên', use_container_width=True)
    
    with st.spinner('AI đang quét và phân tích khuôn mặt...'):
        # Đảm bảo ảnh luôn ở hệ màu RGB và ép kích thước chuẩn 200x200 như lúc train
        img_resized = image_source.convert('RGB').resize((200, 200))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Mô hình dự đoán
        res = model.predict(img_array)
        idx = np.argmax(res)
        confidence = res[0][idx] * 100
        
        # Lấy tên người từ danh sách nhãn
        person_name = labels.get(idx, f"ID chưa đặt tên: {idx}")
        
    # Hiển thị kết quả một cách đẹp mắt
    st.success(f"### Kết quả nhận diện: **{person_name.upper()}**")
    
    # Nếu độ chính xác thấp (ví dụ dưới 65%), hiển thị thanh cảnh báo màu cam
    if confidence > 65:
        st.info(f"Độ tin cậy của mô hình: **{confidence:.2f}%**")
    else:
        st.warning(f"Độ tin cậy khá thấp: **{confidence:.2f}%** (Hãy thử căn chỉnh lại góc mặt hoặc tăng độ sáng)")
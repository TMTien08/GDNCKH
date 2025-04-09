import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from PIL import Image

# Cấu hình giao diện Streamlit
st.set_page_config(page_title="Dự Đoán Rủi Ro Tín Dụng", page_icon="💰", layout="wide")

# Tùy chỉnh CSS
st.markdown("""
    <style>
    .main {background-color: #F7F9FC;}
    .stButton>button {
        background-color: #2E86C1;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1B4F72;
    }
    .stSlider .st-dn {background-color: #2E86C1;}
    .stRadio>label {font-size: 16px;}
    .stSelectbox>label {font-size: 16px;}
    .stNumberInput>label {font-size: 16px;}
    .tooltip {
        position: relative;
        display: inline-block;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 220px;
        background-color: #566573;
        color: white;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -110px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    </style>
""", unsafe_allow_html=True)

# Load dữ liệu gốc để tính tỷ lệ
file_path = "german_credit_data.csv"  # Thay bằng đường dẫn thực tế
df = pd.read_csv(file_path)
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Tính tỷ lệ rủi ro xấu cho từng đặc trưng
def calculate_risk_rates(df, feature):
    risk_rates = df.groupby(feature)["Risk"].value_counts(normalize=True).unstack().fillna(0)
    risk_rates["Bad_Rate"] = risk_rates["bad"] * 100
    return risk_rates["Bad_Rate"].to_dict()

age_risk_dict = calculate_risk_rates(df, "Age")
job_risk_dict = calculate_risk_rates(df, "Job")
credit_amount_risk_dict = calculate_risk_rates(df, "Credit amount")
duration_risk_dict = calculate_risk_rates(df, "Duration")
sex_risk_dict = calculate_risk_rates(df, "Sex")
housing_risk_dict = calculate_risk_rates(df, "Housing")
saving_risk_dict = calculate_risk_rates(df, "Saving accounts")
checking_risk_dict = calculate_risk_rates(df, "Checking account")
purpose_risk_dict = calculate_risk_rates(df, "Purpose")

# Load mô hình và bộ xử lý dữ liệu
@st.cache_resource
def load_model():
    return joblib.load("credit_risk_model.pkl")

@st.cache_resource
def load_preprocessor():
    return joblib.load("preprocessor.pkl")

mo_hinh = load_model()
preprocessor = load_preprocessor()

# Header
st.markdown("<h1 style='text-align: center; color: #2E86C1; font-family: Arial;'>🔍 Dự Đoán Rủi Ro Tín Dụng</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #566573; font-family: Arial;'>Phân tích khả năng hoàn trả khoản vay một cách nhanh chóng và chính xác</h4>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #7F8C8D; font-family: Arial;'>NCKH: P.Nam, H.Nam, P.Huy, T.Tiến, V.Vinh</p>", unsafe_allow_html=True)

# Nhập dữ liệu khách hàng
st.markdown("---")
st.markdown("<h3 style='color: #2E86C1; font-family: Arial;'>📋 Nhập thông tin khách hàng</h3>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")
with col1:
    with st.expander("Thông tin cá nhân", expanded=True):
        age = st.slider("📆 Tuổi", 18, 100, 30, help="Chọn tuổi của khách hàng")
        sex = st.radio("🚻 Giới tính", ["Nam", "Nữ"], horizontal=True)
        sex = "male" if sex == "Nam" else "female"
        job = st.selectbox("👔 Loại công việc", ["Không có kỹ năng & không cư trú", "Không có kỹ năng & cư trú", "Có kỹ năng", "Rất có kỹ năng"])
        job_mapping = {"Không có kỹ năng & không cư trú": 0, "Không có kỹ năng & cư trú": 1, "Có kỹ năng": 2, "Rất có kỹ năng": 3}
        job = job_mapping[job]

with col2:
    with st.expander("Thông tin tài chính & mục đích vay", expanded=True):
        credit_amount = st.number_input("💵 Khoản vay (DM)", min_value=500, max_value=50000, value=10000, step=100)
        duration = st.slider("🕒 Thời hạn vay (tháng)", 6, 72, 24)
        purpose = st.selectbox("🎯 Mục đích vay", ["Mua ô tô", "Mua nội thất/trang thiết bị", "Mua radio/TV", "Mua thiết bị gia dụng", "Sửa chữa", "Giáo dục", "Kinh doanh", "Du lịch/Khác"])
        purpose_mapping = {"Mua ô tô": "car", "Mua nội thất/trang thiết bị": "furniture/equipment", "Mua radio/TV": "radio/TV", "Mua thiết bị gia dụng": "domestic appliances",
                           "Sửa chữa": "repairs", "Giáo dục": "education", "Kinh doanh": "business", "Du lịch/Khác": "vacation/others"}
        purpose = purpose_mapping[purpose]

col3, col4 = st.columns([1, 1], gap="large")
with col3:
    with st.expander("Tình trạng nhà ở", expanded=True):
        housing = st.selectbox("🏠 Hình thức nhà ở", ["Sở hữu", "Thuê", "Miễn phí"])
        housing_mapping = {"Sở hữu": "own", "Thuê": "rent", "Miễn phí": "free"}
        housing = housing_mapping[housing]

with col4:
    with st.expander("Tài khoản ngân hàng", expanded=True):
        st.markdown("""
            <div class="tooltip">
                💰 Tài khoản tiết kiệm
                <span class="tooltiptext">Không có: 0 DM<br>Ít: 1-500 DM<br>Trung bình: 501-1000 DM<br>Khá nhiều: 1001-5000 DM<br>Nhiều: >5000 DM</span>
            </div>
        """, unsafe_allow_html=True)
        saving_accounts = st.selectbox("", ["Không có", "Ít", "Trung bình", "Khá nhiều", "Nhiều"], key="savings")

import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

# Load model và pipeline
with open('credit_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('credit_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# App UI
st.set_page_config(page_title="Dự đoán rủi ro tín dụng", layout="centered")

# CSS nâng cấp giao diện
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, #f0f4f8 0%, #ffffff 100%);
    }

    .stExpander {
        background-color: #ffffff;
        border: 1px solid #e1e1e1;
        border-radius: 16px !important;
        padding: 20px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }

    .footer {
        text-align: center;
        margin-top: 30px;
        color: gray;
        font-size: 14px;
    }

    .card {
        padding: 20px;
        border-radius: 15px;
        margin-top: 20px;
        text-align: center;
    }

    .low-risk {
        background-color: #e8f8f5;
        color: #1abc9c;
    }

    .high-risk {
        background-color: #fdecea;
        color: #e74c3c;
    }

    h1 {
        background: -webkit-linear-gradient(45deg, #2E86C1, #1ABC9C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 36px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Dự đoán rủi ro tín dụng")

with st.expander("📋 Nhập thông tin khách hàng", expanded=True):
    credit_amount = st.number_input("💰 Số tiền vay (DM)", min_value=0, value=1000)
    duration = st.slider("⏳ Thời hạn vay (tháng)", 4, 72, 24)
    
    checking_status = st.selectbox("🏦 Trạng thái tài khoản ngân hàng", 
                                   ['<0', '0<=X<200', '>=200', 'no checking'])
    credit_history = st.selectbox("📜 Lịch sử tín dụng", 
                                  ['no credits/all paid', 'all paid', 'existing paid', 'delayed previously', 'critical/other existing credit'])
    purpose = st.selectbox("🎯 Mục đích vay", 
                           ['radio/tv', 'education', 'furniture/equipment', 'new car', 'used car', 'business', 'repairs', 'other'])

    age = st.slider("🎂 Tuổi", 18, 75, 30)
    housing = st.radio("🏘️ Loại nhà ở", ['own', 'for free', 'rent'])
    job = st.radio("💼 Nghề nghiệp", ['unskilled resident', 'skilled', 'highly skilled', 'unemployed/non-resident'])

    sex = st.radio("⚧️ Giới tính", ['male', 'female'])

# Gộp dữ liệu
input_data = {
    "checking_status": checking_status,
    "duration": duration,
    "credit_history": credit_history,
    "purpose": purpose,
    "credit_amount": credit_amount,
    "age": age,
    "housing": housing,
    "job": job,
    "sex": sex
}

input_df = pd.DataFrame([input_data])

# Xử lý dữ liệu
X_transformed = pipeline.transform(input_df)

# Dự đoán
risk_score = model.predict_proba(X_transformed)[0][1]

# Hiển thị kết quả
risk_level = "⚠️ Rủi ro cao" if risk_score > 0.5 else "✅ Rủi ro thấp"
risk_class = "high-risk" if risk_score > 0.5 else "low-risk"

st.markdown(f"""
    <div class="card {risk_class}">
        <h3>{risk_level}</h3>
        <p><b>Xác suất rủi ro: {risk_score:.2%}</b></p>
    </div>
""", unsafe_allow_html=True)

# Hiển thị bảng dữ liệu đầu vào (tùy chọn)
with st.expander("🔎 Xem lại dữ liệu đã nhập"):
    st.dataframe(input_df.style.set_properties(**{
        'background-color': '#FAFAFA',
        'color': '#111',
        'border-color': 'lightgray'
    }))

# Footer
st.markdown("""
<div class="footer">
    📘 Nghiên cứu khoa học 2025 • Đại học XYZ<br>
    Liên hệ: <a href="mailto:info@university.edu">info@university.edu</a>
</div>
""", unsafe_allow_html=True)

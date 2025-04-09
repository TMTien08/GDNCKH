import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from streamlit_lottie import st_lottie
import json

# Load model and encoders
model = joblib.load("best_model_xgb.pkl")
ohe = joblib.load("encoder_ohe.pkl")
scaler = joblib.load("scaler.pkl")

# Set page config
st.set_page_config(page_title="Dự đoán rủi ro tín dụng", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f4f6f7;
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h3, h4 {
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        background-color: #2980b9;
        color: white;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: bold;
        border: none;
        transition: 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1c5980;
        transform: scale(1.03);
    }
    .stSelectbox label, .stRadio label, .stSlider label, .stNumberInput label {
        font-weight: 600;
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)

# Load Lottie animation
@st.cache_data
def load_lottie(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

success_anim = load_lottie("success_animation.json")
warning_anim = load_lottie("warning_animation.json")

# Title
st.title("📊 Ứng dụng dự đoán rủi ro tín dụng")

# Input columns
col1, col2 = st.columns([1.2, 1], gap="medium")
col3, col4 = st.columns([1.2, 1], gap="medium")

with col1:
    checking_account = st.selectbox("Tài khoản séc", ["none", "little", "moderate", "rich"])
    credit_history = st.selectbox("Lịch sử tín dụng", ["critical", "good", "perfect", "poor"])
    purpose = st.selectbox("Mục đích vay", ["car", "radio/tv", "education", "furniture", "business", "repairs", "vacation"])
    housing = st.radio("Loại nhà ở", ["own", "rent", "free"])

with col2:
    amount = st.number_input("Số tiền vay (Euro)", 100, 20000, 1000, step=100)
    duration = st.slider("Thời hạn vay (tháng)", 6, 72, 12)
    age = st.slider("Tuổi", 18, 75, 30)
    job = st.selectbox("Nghề nghiệp", ["unskilled", "skilled", "highly skilled"])

with col3:
    savings_account = st.selectbox("Tài khoản tiết kiệm", ["unknown", "little", "moderate", "rich"])
    employment = st.selectbox("Thời gian làm việc (năm)", ["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"])

with col4:
    other_debtors = st.radio("Người bảo lãnh", ["none", "guarantor", "co-applicant"])
    property = st.selectbox("Tài sản thế chấp", ["real estate", "car", "savings", "other"])

# Predict
if st.button("Dự đoán rủi ro"):
    input_data = pd.DataFrame({
        "checking_account": [checking_account],
        "credit_history": [credit_history],
        "purpose": [purpose],
        "housing": [housing],
        "job": [job],
        "savings_account": [savings_account],
        "employment": [employment],
        "other_debtors": [other_debtors],
        "property": [property],
        "amount": [amount],
        "duration": [duration],
        "age": [age]
    })

    cat_cols = [
        "checking_account", "credit_history", "purpose", "housing",
        "job", "savings_account", "employment", "other_debtors", "property"
    ]
    num_cols = ["amount", "duration", "age"]

    input_cat = ohe.transform(input_data[cat_cols])
    input_num = scaler.transform(input_data[num_cols])

    import numpy as np
    input_final = np.hstack((input_cat, input_num))

    prediction = model.predict_proba(input_final)
    risk_score = prediction[0][1]

    if risk_score > 0.5:
        st.subheader("🚨 Khách hàng có **NGUY CƠ CAO** không trả nợ!")
        st_lottie(warning_anim, height=200)
    else:
        st.subheader("✅ Khách hàng có **rủi ro thấp**, có thể cho vay!")
        st_lottie(success_anim, height=200)

    # Gauge chart
    fig1 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score * 100,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Xác suất rủi ro (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "red" if risk_score > 0.5 else "green"},
            "steps": [
                {"range": [0, 50], "color": "lightgreen"},
                {"range": [50, 75], "color": "orange"},
                {"range": [75, 100], "color": "red"},
            ]
        }
    ))
    fig1.update_layout(paper_bgcolor="#f4f6f7", font={"color": "#2c3e50", "family": "Segoe UI"})

    # Pie chart
    fig3 = px.pie(
        names=["Không rủi ro", "Rủi ro cao"],
        values=[1 - risk_score, risk_score],
        title="Tỷ lệ dự đoán",
        color_discrete_sequence=["green", "red"]
    )
    fig3.update_traces(textinfo="percent+label", textfont_size=14)
    fig3.update_layout(paper_bgcolor="#f4f6f7", legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))

    # Show charts
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig3, use_container_width=True)

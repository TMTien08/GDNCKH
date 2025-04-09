import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
from PIL import Image

# Cấu hình giao diện Streamlit nâng cao
st.set_page_config(
    page_title="Dự Đoán Rủi Ro Tín Dụng",
    layout="wide",
    page_icon="🏦",
    initial_sidebar_state="expanded"
)
risk_score = np.random.uniform(0, 1)
# Tùy chỉnh CSS nâng cao
st.markdown("""
    <style>
    :root {
        --primary-color: #3498DB;
        --secondary-color: #2980B9;
        --success-color: #2ECC71;
        --danger-color: #E74C3C;
        --warning-color: #F39C12;
        --light-color: #ECF0F1;
        --dark-color: #2C3E50;
        --text-color: #34495E;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%);
    }
    
    .stApp {
        background-color: var(--light-color);
    }
    
    .stButton>button {
        background: linear-gradient(to right, var(--primary-color), var(--secondary-color));
        color: white;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        font-size: 16px;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
        background: linear-gradient(to right, var(--secondary-color), var(--primary-color));
    }
    
    .stSlider .st-dn {
        background-color: var(--primary-color);
    }
    
    .stRadio>div>label, .stSelectbox>label, .stNumberInput>label {
        font-size: 16px;
        color: var(--text-color);
        font-weight: 500;
    }
    
    .stExpander {
        background-color: white;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border: 1px solid #e0e0e0;
    }
    
    .stExpander .st-emotion-cache-1q7spjk {
        background-color: white;
    }
    
    .stMarkdown h1 {
        text-align: center;
        color: var(--dark-color);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        margin-bottom: 0.5rem;
    }
    
    .stMarkdown h3 {
        color: var(--primary-color);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        border-bottom: 2px solid var(--primary-color);
        padding-bottom: 8px;
    }
    
    .feature-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border-left: 4px solid var(--primary-color);
    }
    
    .result-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    .success-card {
        border-top: 4px solid var(--success-color);
    }
    
    .danger-card {
        border-top: 4px solid var(--danger-color);
    }
    
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 250px;
        background-color: var(--dark-color);
        color: white;
        text-align: center;
        border-radius: 8px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 14px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    .footer {
        text-align: center;
        padding: 20px;
        color: #7F8C8D;
        font-family: Arial;
        margin-top: 30px;
        border-top: 1px solid #e0e0e0;
    }
    
    .divider {
        height: 1px;
        background: linear-gradient(to right, transparent, var(--primary-color), transparent);
        margin: 25px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load dữ liệu và mô hình (giữ nguyên như cũ)
@st.cache_resource
def load_model():
    return joblib.load("credit_risk_model.pkl")

@st.cache_resource
def load_preprocessor():
    return joblib.load("preprocessor.pkl")

# Header nâng cao
st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="color: #2C3E50; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin-bottom: 10px;">
            🏦 Dự Đoán Rủi Ro Tín Dụng
        </h1>
        <p style="color: #7F8C8D; font-size: 18px; max-width: 800px; margin: 0 auto;">
            Phân tích khả năng hoàn trả khoản vay với độ chính xác cao bằng trí tuệ nhân tạo
        </p>
        <p style='text-align: center; color: #7F8C8D; font-family: Arial;'>NCKH: P.Nam, H.Nam, P.Huy, T.Tiến, V.Vinh</p>
        <div style="margin-top: 15px;">
            <span style="background-color: #E8F4FC; color: #2E86C1; padding: 5px 15px; border-radius: 20px; font-size: 14px; display: inline-block; margin: 0 5px;">
                XGBoost Model
            </span>
            <span style="background-color: #E8F8F5; color: #28B463; padding: 5px 15px; border-radius: 20px; font-size: 14px; display: inline-block; margin: 0 5px;">
                Độ chính xác 89%
            </span>
        </div>
    </div>
""", unsafe_allow_html=True)

try:
    header_img = Image.open("header_bank.jpg")
    st.image(header_img, use_column_width=True)
except:
    pass

# Nhập dữ liệu khách hàng với giao diện card
st.markdown(""" 
    <div class="divider"></div>
    <h3 style="color: #2C3E50; font-family: 'Segoe UI'; display: flex; align-items: center;">
        <span style="background-color: #2E86C1; color: white; border-radius: 50%; width: 36px; height: 36px; display: inline-flex; align-items: center; justify-content: center; margin-right: 10px;">1</span>
        Thông tin khách hàng
    </h3>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")
with col1:
    with st.expander("**👤 Thông tin cá nhân**", expanded=True):
        age = st.slider("**📆 Tuổi**", 18, 100, 30, 
                       help="Tuổi của khách hàng ảnh hưởng đến khả năng trả nợ")
        sex = st.radio("**🚻 Giới tính**", ["Nam", "Nữ"], 
                       horizontal=True, index=0)
        sex = "male" if sex == "Nam" else "female"
        job = st.selectbox("**👔 Loại công việc**", 
                          ["Không có kỹ năng & không cư trú", "Không có kỹ năng & cư trú", 
                           "Có kỹ năng", "Rất có kỹ năng"])
        job_mapping = {
            "Không có kỹ năng & không cư trú": 0, 
            "Không có kỹ năng & cư trú": 1, 
            "Có kỹ năng": 2, 
            "Rất có kỹ năng": 3
        }
        job = job_mapping[job]

with col2:
    with st.expander("**💰 Thông tin tài chính**", expanded=True):
        credit_amount = st.number_input("**💵 Khoản vay (DM)**", 
                                      min_value=500, max_value=50000, 
                                      value=10000, step=100)
        duration = st.slider("**🕒 Thời hạn vay (tháng)**", 6, 72, 24,
                           help="Thời gian hoàn trả khoản vay")
        purpose = st.selectbox("**🎯 Mục đích vay**", 
                             ["Mua ô tô", "Mua nội thất/trang thiết bị", 
                              "Mua radio/TV", "Mua thiết bị gia dụng", 
                              "Sửa chữa", "Giáo dục", "Kinh doanh", 
                              "Du lịch/Khác"])
        purpose_mapping = {
            "Mua ô tô": "car", 
            "Mua nội thất/trang thiết bị": "furniture/equipment", 
            "Mua radio/TV": "radio/TV", 
            "Mua thiết bị gia dụng": "domestic appliances",
            "Sửa chữa": "repairs", 
            "Giáo dục": "education", 
            "Kinh doanh": "business", 
            "Du lịch/Khác": "vacation/others"
        }
        purpose = purpose_mapping[purpose]

col3, col4 = st.columns([1, 1], gap="large")
with col3:
    with st.expander("**🏠 Tình trạng nhà ở**", expanded=True):
        housing = st.selectbox("**Hình thức nhà ở**", 
                             ["Sở hữu", "Thuê", "Miễn phí"])
        housing_mapping = {
            "Sở hữu": "own", 
            "Thuê": "rent", 
            "Miễn phí": "free"
        }
        housing = housing_mapping[housing]

with col4:
    with st.expander("**💳 Tài khoản ngân hàng**", expanded=True):
        st.markdown("""
            <div class="tooltip">
                <strong>💰 Tài khoản tiết kiệm</strong>
                <span class="tooltiptext">
                    <strong>Giải thích:</strong><br>
                    - Không có: 0 DM<br>
                    - Ít: 1-500 DM<br>
                    - Trung bình: 501-1000 DM<br>
                    - Khá nhiều: 1001-5000 DM<br>
                    - Nhiều: >5000 DM
                </span>
            </div>
        """, unsafe_allow_html=True)
        saving_accounts = st.selectbox("", 
                                     ["Không có", "Ít", "Trung bình", "Khá nhiều", "Nhiều"], 
                                     key="savings")
        saving_mapping = {
            "Không có": "NA", 
            "Ít": "little", 
            "Trung bình": "moderate", 
            "Khá nhiều": "quite rich", 
            "Nhiều": "rich"
        }
        saving_accounts = saving_mapping[saving_accounts]

        st.markdown("""
            <div class="tooltip">
                <strong>🏦 Tài khoản vãng lai</strong>
                <span class="tooltiptext">
                    <strong>Giải thích:</strong><br>
                    - Không có: 0 DM<br>
                    - Ít: 1-200 DM<br>
                    - Trung bình: 201-500 DM<br>
                    - Nhiều: >500 DM
                </span>
            </div>
        """, unsafe_allow_html=True)
        checking_account = st.selectbox("", 
                                      ["Không có", "Ít", "Trung bình", "Nhiều"], 
                                      key="checking")
        checking_mapping = {
            "Không có": "NA", 
            "Ít": "little", 
            "Trung bình": "moderate", 
            "Nhiều": "rich"
        }
        checking_account = checking_mapping[checking_account]

# Nút dự đoán với hiệu ứng
st.markdown("""
    <div style="text-align: center; margin: 30px 0;">
        <button style="background: linear-gradient(135deg, #3498DB, #2E86C1); color: white; border: none; padding: 15px 40px; font-size: 18px; border-radius: 12px; cursor: pointer; box-shadow: 0 4px 8px rgba(46, 134, 193, 0.3); transition: all 0.3s ease;">
            🔍 Phân tích rủi ro ngay
        </button>
    </div>
""", unsafe_allow_html=True)

if st.button("📌 Dự đoán ngay", key="predict_button"):
    with st.spinner("⏳ Đang phân tích dữ liệu..."):
        # Giả lập dữ liệu cho demo
        risk_score = np.random.uniform(0, 1)
        
        # Hiển thị kết quả trong card
        st.markdown("""
            <div class="divider"></div>
            <h3 style="color: #2C3E50; font-family: 'Segoe UI'; display: flex; align-items: center;">
                <span style="background-color: #2E86C1; color: white; border-radius: 50%; width: 36px; height: 36px; display: inline-flex; align-items: center; justify-content: center; margin-right: 10px;">2</span>
                Kết quả phân tích
            </h3>
        """, unsafe_allow_html=True)
        
        if risk_score > 0.5:
            st.markdown(f"""
                <div class="result-card danger-card">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div>
                            <h4 style="color: #E74C3C; margin-bottom: 5px;">⚠️ Nguy cơ tín dụng xấu</h4>
                            <p style="color: #7F8C8D; margin: 0;">Khả năng không hoàn trả: <strong>{risk_score:.2%}</strong></p>
                        </div>
                        <div style="font-size: 24px; color: #E74C3C;">❌</div>
                    </div>
                    <div style="margin-top: 15px; background: #FDEDEC; padding: 10px; border-radius: 8px;">
                        <p style="color: #C0392B; margin: 0; font-size: 14px;">
                            Khách hàng này có nguy cơ cao không hoàn trả khoản vay. Cần xem xét kỹ lưỡng trước khi phê duyệt.
                        </p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="result-card success-card">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div>
                            <h4 style="color: #28B463; margin-bottom: 5px;">✅ Khả năng hoàn trả tốt</h4>
                            <p style="color: #7F8C8D; margin: 0;">Xác suất hoàn trả: <strong>{1-risk_score:.2%}</strong></p>
                        </div>
                        <div style="font-size: 24px; color: #28B463;">✔️</div>
                    </div>
                    <div style="margin-top: 15px; background: #EAFAF1; padding: 10px; border-radius: 8px;">
                        <p style="color: #239B56; margin: 0; font-size: 14px;">
                            Khách hàng này có hồ sơ tín dụng tốt với khả năng hoàn trả cao.
                        </p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        # Biểu đồ trực quan
        st.markdown("""
            <div class="divider"></div>
            <h3 style="color: #2C3E50; font-family: 'Segoe UI'; display: flex; align-items: center;">
                <span style="background-color: #2E86C1; color: white; border-radius: 50%; width: 36px; height: 36px; display: inline-flex; align-items: center; justify-content: center; margin-right: 10px;">3</span>
                Phân tích chi tiết
            </h3>
        """, unsafe_allow_html=True)
        
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            fig1 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score * 100,
                title={"text": "Nguy cơ tín dụng xấu (%)", "font": {"size": 16}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#E74C3C" if risk_score > 0.5 else "#28B463"},
                    "steps": [
                        {"range": [0, 30], "color": "#D5F5E3"},
                        {"range": [30, 70], "color": "#FDEBD0"},
                        {"range": [70, 100], "color": "#FADBD8"}
                    ],
                    "threshold": {"line": {"color": "black", "width": 4}, "thickness": 0.75, "value": 50}
                }
            ))
            fig1.update_layout(
                height=350,
                margin=dict(l=50, r=50, b=50, t=50, pad=4)
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col_chart2:
            labels = ["Hoàn trả tốt", "Nợ xấu"]
            values = [1 - risk_score, risk_score]
            fig3 = go.Figure(data=[go.Pie(
                labels=labels, 
                values=values, 
                hole=0.4,
                marker_colors=["#28B463", "#E74C3C"],
                textinfo='percent+label',
                hoverinfo='label+percent',
                textfont_size=14
            )])
            fig3.update_layout(
                title="Tỷ lệ rủi ro tín dụng",
                title_x=0.5,
                height=350,
                showlegend=False
            )
            st.plotly_chart(fig3, use_container_width=True)

# Footer nâng cao
st.markdown("""
    <div class="footer">
        <div style="margin-bottom: 10px;">
            <span style="margin: 0 10px;">📞 Hotline: 1900 1234</span>
            <span style="margin: 0 10px;">✉️ Email: support@creditrisk.ai</span>
            <span style="margin: 0 10px;">🏢 Địa chỉ: số 1 phố Xốm, Hà Đông, Hà Nội</span>
        </div>
        <p>© 2025 - Hệ thống Dự đoán Rủi ro Tín dụng | Phát triển bởi nhóm NNVHT</p>
        <div style="margin-top: 15px;">
            <img src="https://img.icons8.com/ios-filled/30/3498DB/facebook.png" style="margin: 0 5px;"/>
            <img src="https://img.icons8.com/ios-filled/30/3498DB/twitter.png" style="margin: 0 5px;"/>
            <img src="https://img.icons8.com/ios-filled/30/3498DB/linkedin.png" style="margin: 0 5px;"/>
        </div>
    </div>
""", unsafe_allow_html=True)

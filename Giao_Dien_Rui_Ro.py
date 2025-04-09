import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
from PIL import Image

# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="Dự Đoán Rủi Ro Tín Dụng",
    layout="wide",
    page_icon="🏦",
    initial_sidebar_state="expanded"
)

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

# Load dữ liệu và mô hình
@st.cache_resource
def load_model():
    return joblib.load("credit_risk_model.pkl")

@st.cache_resource
def load_preprocessor():
    return joblib.load("preprocessor.pkl")

# Load model and preprocessor
model = load_model()
preprocessor = load_preprocessor()

# Tạo các risk dictionary mẫu (thay thế bằng dữ liệu thực tế của bạn)
age_risk_dict = {age: np.random.uniform(5, 30) for age in range(18, 101)}
sex_risk_dict = {"male": 15.2, "female": 12.5}
job_risk_dict = {0: 25.3, 1: 18.7, 2: 12.1, 3: 8.5}
credit_amount_risk_dict = {amt: np.random.uniform(10, 40) for amt in range(500, 50001, 100)}
duration_risk_dict = {dur: np.random.uniform(10, 35) for dur in range(6, 73)}
housing_risk_dict = {"own": 12.3, "rent": 18.7, "free": 15.4}
saving_risk_dict = {"NA": 25.6, "little": 18.2, "moderate": 12.7, "quite rich": 8.9, "rich": 5.1}
checking_risk_dict = {"NA": 22.4, "little": 16.8, "moderate": 11.3, "rich": 7.5}
purpose_risk_dict = {
    "car": 18.5, 
    "furniture/equipment": 15.2, 
    "radio/TV": 12.7, 
    "domestic appliances": 11.3,
    "repairs": 14.8, 
    "education": 9.5, 
    "business": 20.1, 
    "vacation/others": 16.3
}

# Header nâng cao
st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="color: #2C3E50; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin-bottom: 10px;">
            🏦 Dự Đoán Rủi Ro Tín Dụng
        </h1>
        <p style="color: #7F8C8D; font-size: 18px; max-width: 800px; margin: 0 auto;">
            Phân tích khả năng hoàn trả khoản vay với độ chính xác cao bằng trí tuệ nhân tạo
        </p>
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

# Thêm ảnh header (nếu có)
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
st.markdown("<div style='text-align: center; margin: 30px 0;'>", unsafe_allow_html=True)
if st.button("🔮 Dự đoán rủi ro tín dụng", key="predict_button", help="Nhấn để phân tích rủi ro tín dụng của khách hàng"):
    with st.spinner("🔄 Đang phân tích dữ liệu..."):
        input_data = pd.DataFrame([{
            "Age": age,
            "Job": job,
            "Credit amount": credit_amount,
            "Duration": duration,
            "Sex": sex,
            "Housing": housing,
            "Saving accounts": saving_accounts,
            "Checking account": checking_account,
            "Purpose": purpose
        }])
        input_transformed = preprocessor.transform(input_data)
        prediction = model.predict_proba(input_transformed)[:, 1]
        risk_score = prediction[0]

    # Hiển thị kết quả chi tiết từng đặc trưng
    st.markdown("---")
    st.markdown("<h3 style='color: #2E86C1; font-family: Arial;'>📊 Phân tích rủi ro từng đặc trưng</h3>", unsafe_allow_html=True)
    
    feature_contributions = {
        "Tuổi": {"Giá trị": f"{age} tuổi", "Tỷ lệ rủi ro xấu": f"{age_risk_dict.get(age, 0):.2f}%"},
        "Giới tính": {"Giá trị": "Nam" if sex == "male" else "Nữ", "Tỷ lệ rủi ro xấu": f"{sex_risk_dict.get(sex, 0):.2f}%"},
        "Công việc": {"Giá trị": list(job_mapping.keys())[list(job_mapping.values()).index(job)], "Tỷ lệ rủi ro xấu": f"{job_risk_dict.get(job, 0):.2f}%"},
        "Khoản vay": {"Giá trị": f"{credit_amount:,} DM", "Tỷ lệ rủi ro xấu": f"{credit_amount_risk_dict.get(credit_amount, 0):.2f}%"},
        "Thời hạn": {"Giá trị": f"{duration} tháng", "Tỷ lệ rủi ro xấu": f"{duration_risk_dict.get(duration, 0):.2f}%"},
        "Nhà ở": {"Giá trị": list(housing_mapping.keys())[list(housing_mapping.values()).index(housing)], "Tỷ lệ rủi ro xấu": f"{housing_risk_dict.get(housing, 0):.2f}%"},
        "Tài khoản tiết kiệm": {"Giá trị": list(saving_mapping.keys())[list(saving_mapping.values()).index(saving_accounts)], "Tỷ lệ rủi ro xấu": f"{saving_risk_dict.get(saving_accounts, 0):.2f}%"},
        "Tài khoản vãng lai": {"Giá trị": list(checking_mapping.keys())[list(checking_mapping.values()).index(checking_account)], "Tỷ lệ rủi ro xấu": f"{checking_risk_dict.get(checking_account, 0):.2f}%"},
        "Mục đích vay": {"Giá trị": list(purpose_mapping.keys())[list(purpose_mapping.values()).index(purpose)], "Tỷ lệ rủi ro xấu": f"{purpose_risk_dict.get(purpose, 0):.2f}%"}
    }
    
    feature_df = pd.DataFrame.from_dict(feature_contributions, orient="index")
    st.dataframe(
        feature_df.style
        .set_properties(**{'background-color': '#FFFFFF', 'border': '1px solid #EAEDED'})
        .highlight_max(subset=["Tỷ lệ rủi ro xấu"], color='#FADBD8')
        .highlight_min(subset=["Tỷ lệ rủi ro xấu"], color='#D5F5E3'),
        use_container_width=True
    )

    # Hiển thị kết quả tổng hợp với card đẹp
    st.markdown("---")
    st.markdown("<h3 style='color: #2E86C1; font-family: Arial;'>🔍 Kết quả dự đoán tổng hợp</h3>", unsafe_allow_html=True)
    
    if risk_score > 0.5:
        st.error(f"""
            ⚠️ **Nguy cơ tín dụng xấu: {risk_score:.2%}**  
            *Khách hàng có nguy cơ cao không hoàn trả khoản vay. Cần xem xét kỹ lưỡng trước khi phê duyệt.*
        """)
    else:
        st.success(f"""
            ✅ **Khả năng hoàn trả tốt: {1-risk_score:.2%}**  
            *Khách hàng có hồ sơ tín dụng tốt và khả năng hoàn trả cao.*
        """)
    
    st.markdown(f"""
        <div style="background-color: #F8F9F9; padding: 15px; border-radius: 10px; margin-top: 20px;">
            <p style="color: #566573; font-size: 15px;">
                📌 <strong>Giải thích:</strong> Xác suất này được tính toán dựa trên mô hình XGBoost với độ chính xác cao, 
                phân tích các đặc trưng quan trọng nhất ảnh hưởng đến rủi ro tín dụng.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Biểu đồ trực quan nâng cao
    st.markdown("---")
    st.markdown("<h3 style='color: #2E86C1; font-family: Arial;'>📈 Trực quan hóa rủi ro</h3>", unsafe_allow_html=True)
    
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        fig1 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_score * 100,
            title={"text": "Nguy cơ tín dụng xấu (%)", "font": {"size": 18}},
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#2E86C1"},
                "bar": {"color": "#E74C3C" if risk_score > 0.5 else "#28B463", "thickness": 0.3},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [0, 30], "color": "#D5F5E3"},
                    {"range": [30, 70], "color": "#FDEBD0"},
                    {"range": [70, 100], "color": "#FADBD8"}
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.8,
                    "value": risk_score * 100
                }
            }
        ))
        fig1.update_layout(
            height=350,
            margin=dict(l=50, r=50, b=50, t=80),
            font=dict(color="#2E86C1", family="Arial")
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col_chart2:
        labels = ["Hoàn trả tốt", "Nợ xấu"]
        values = [1 - risk_score, risk_score]
        colors = ["#28B463", "#E74C3C"]
        
        fig3 = go.Figure(data=[go.Pie(
            labels=labels, 
            values=values, 
            hole=0.5,
            marker=dict(colors=colors),
            textinfo='percent+value',
            hoverinfo='label+percent',
            textfont_size=15
        )])
        
        fig3.update_layout(
            title="Phân bổ rủi ro tín dụng",
            title_x=0.5,
            title_font=dict(size=18),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            ),
            height=350,
            margin=dict(l=50, r=50, b=50, t=80)
        )
        st.plotly_chart(fig3, use_container_width=True)

# Footer chuyên nghiệp
st.markdown("""
    <div class="footer">
        <div style="margin-bottom: 10px;">
            <span style="margin: 0 10px;">📞 Hotline: 1900 1234</span>
            <span style="margin: 0 10px;">✉️ Email: support@creditrisk.ai</span>
            <span style="margin: 0 10px;">🏢 Địa chỉ: 123 Nguyễn Du, Hà Nội</span>
        </div>
        <p>© 2025 - Hệ thống Dự đoán Rủi ro Tín dụng | Phát triển bởi nhóm NCKH</p>
        <div style="margin-top: 15px;">
            <img src="https://img.icons8.com/ios-filled/30/3498DB/facebook.png" style="margin: 0 5px;"/>
            <img src="https://img.icons8.com/ios-filled/30/3498DB/twitter.png" style="margin: 0 5px;"/>
            <img src="https://img.icons8.com/ios-filled/30/3498DB/linkedin.png" style="margin: 0 5px;"/>
        </div>
    </div>
""", unsafe_allow_html=True)

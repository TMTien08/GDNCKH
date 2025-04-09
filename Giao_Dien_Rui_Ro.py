import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import base64

# Hàm để thêm background image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/jpeg;base64,{encoded_string.decode()});
        background-size: cover;
        background-attachment: fixed;
        background-opacity: 0.1;
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

# Thêm background (đảm bảo bạn có file bank_background.jpg trong cùng thư mục)
add_bg_from_local('bank_background.jpg')  # Thay bằng tên file hình ảnh của bạn

# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="Dự Đoán Rủi Ro Tín Dụng", 
    page_icon="💰", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tùy chỉnh CSS nâng cao với background trong suốt
st.markdown("""
    <style>
    :root {
        --primary: #2E86C1;
        --secondary: rgba(247, 249, 252, 0.9);
        --success: #28B463;
        --danger: #E74C3C;
        --text: #34495E;
        --light-text: #7F8C8D;
        --card-bg: rgba(255, 255, 255, 0.95);
        --shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .main {background-color: transparent;}
    
    /* Làm trong suốt các container chính */
    .block-container {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: var(--shadow);
    }
    
    /* Nút bấm */
    .stButton>button {
        background-color: var(--primary);
        color: white;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: bold;
        border: none;
        box-shadow: var(--shadow);
        transition: all 0.3s ease;
        width: 100%;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #1B4F72;
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    /* Thanh trượt */
    .stSlider .st-dn {background-color: var(--primary);}
    
    /* Ô nhập liệu */
    .stTextInput>div>div>input, 
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>select {
        border-radius: 8px;
        border: 1px solid #D5DBDB;
        padding: 10px;
        background-color: rgba(255, 255, 255, 0.8);
    }
    
    /* Thẻ mở rộng */
    .stExpander {
        background-color: var(--card-bg);
        border-radius: 12px;
        box-shadow: var(--shadow);
        padding: 16px;
        margin-bottom: 16px;
    }
    .stExpander .streamlit-expanderHeader {
        font-weight: bold;
        color: var(--primary);
        font-size: 18px;
    }
    
    /* Bảng */
    .stDataFrame {
        border-radius: 10px;
        box-shadow: var(--shadow);
        background-color: var(--card-bg);
    }
    
    /* Tiêu đề */
    h1, h2, h3, h4 {
        color: var(--text) !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Tooltip */
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted var(--primary);
        cursor: help;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 220px;
        background-color: var(--text);
        color: white;
        text-align: center;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 14px;
        box-shadow: var(--shadow);
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Kết quả */
    .stAlert {
        border-radius: 10px;
        background-color: rgba(255, 255, 255, 0.9);
    }
    .stAlert.success {
        background-color: rgba(40, 180, 99, 0.2);
        border-left: 5px solid var(--success);
    }
    .stAlert.error {
        background-color: rgba(231, 76, 60, 0.2);
        border-left: 5px solid var(--danger);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        color: var(--light-text);
        font-size: 14px;
        margin-top: 40px;
        border-top: 1px solid #EAEDED;
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
    }
    
    /* Card highlight */
    .highlight-card {
        background: linear-gradient(135deg, rgba(46, 134, 193, 0.9) 0%, rgba(27, 79, 114, 0.9) 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: var(--shadow);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ... (phần còn lại của mã giữ nguyên như bạn đã có)


# Load dữ liệu gốc để tính tỷ lệ
@st.cache_data
def load_data():
    file_path = "german_credit_data.csv"
    df = pd.read_csv(file_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df

df = load_data()

# Tính tỷ lệ rủi ro xấu cho từng đặc trưng
@st.cache_data
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
# Header nâng cao
st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="color: #2C3E50; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin-bottom: 10px;">
            🏦 Dự Đoán Rủi Ro Tín Dụng
        </h1>
        <p style="color: #7F8C8D; font-size: 18px; max-width: 800px; margin: 0 auto;">
            Phân tích khả năng hoàn trả khoản vay với độ chính xác cao bằng trí tuệ nhân tạo
        </p>
        <p style="color: #7F8C8D; font-size: 18px; max-width: 800px; margin: 0 auto;'>
            NCKH: P.Nam, H.Nam, P.Huy, T.Tiến, V.Vinh
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
# Nhập dữ liệu khách hàng
st.markdown("---")
st.markdown("<h3 style='color: #2E86C1; font-family: Arial;'>📋 Thông tin khách hàng</h3>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")
with col1:
    with st.expander("🔍 Thông tin cá nhân", expanded=True):
        age = st.slider("📆 Tuổi", 18, 100, 30, help="Chọn tuổi của khách hàng")
        sex = st.radio("🚻 Giới tính", ["Nam", "Nữ"], horizontal=True)
        sex = "male" if sex == "Nam" else "female"
        job = st.selectbox("👔 Loại công việc", ["Không có kỹ năng & không cư trú", "Không có kỹ năng & cư trú", "Có kỹ năng", "Rất có kỹ năng"])
        job_mapping = {"Không có kỹ năng & không cư trú": 0, "Không có kỹ năng & cư trú": 1, "Có kỹ năng": 2, "Rất có kỹ năng": 3}
        job = job_mapping[job]

with col2:
    with st.expander("💰 Thông tin tài chính", expanded=True):
        credit_amount = st.number_input("💵 Khoản vay (DM)", min_value=500, max_value=50000, value=10000, step=100)
        duration = st.slider("🕒 Thời hạn vay (tháng)", 6, 72, 24)
        purpose = st.selectbox("🎯 Mục đích vay", ["Mua ô tô", "Mua nội thất/trang thiết bị", "Mua radio/TV", "Mua thiết bị gia dụng", "Sửa chữa", "Giáo dục", "Kinh doanh", "Du lịch/Khác"])
        purpose_mapping = {"Mua ô tô": "car", "Mua nội thất/trang thiết bị": "furniture/equipment", "Mua radio/TV": "radio/TV", "Mua thiết bị gia dụng": "domestic appliances",
                           "Sửa chữa": "repairs", "Giáo dục": "education", "Kinh doanh": "business", "Du lịch/Khác": "vacation/others"}
        purpose = purpose_mapping[purpose]

col3, col4 = st.columns([1, 1], gap="large")
with col3:
    with st.expander("🏠 Tình trạng nhà ở", expanded=True):
        housing = st.selectbox("Hình thức nhà ở", ["Sở hữu", "Thuê", "Miễn phí"])
        housing_mapping = {"Sở hữu": "own", "Thuê": "rent", "Miễn phí": "free"}
        housing = housing_mapping[housing]

with col4:
    with st.expander("🏦 Tài khoản ngân hàng", expanded=True):
        st.markdown("""
            <div class="tooltip">
                💰 Tài khoản tiết kiệm
                <span class="tooltiptext">Không có: 0 DM<br>Ít: 1-500 DM<br>Trung bình: 501-1000 DM<br>Khá nhiều: 1001-5000 DM<br>Nhiều: >5000 DM</span>
            </div>
        """, unsafe_allow_html=True)
        saving_accounts = st.selectbox("savings", ["Không có", "Ít", "Trung bình", "Khá nhiều", "Nhiều"], key="savings", label_visibility="collapsed")
        saving_mapping = {"Không có": "NA", "Ít": "little", "Trung bình": "moderate", "Khá nhiều": "quite rich", "Nhiều": "rich"}
        saving_accounts = saving_mapping[saving_accounts]

        st.markdown("""
            <div class="tooltip">
                🏦 Tài khoản vãng lai
                <span class="tooltiptext">Không có: 0 DM<br>Ít: 1-200 DM<br>Trung bình: 201-500 DM<br>Nhiều: >500 DM</span>
            </div>
        """, unsafe_allow_html=True)
        checking_account = st.selectbox("checking", ["Không có", "Ít", "Trung bình", "Nhiều"], key="checking", label_visibility="collapsed")
        checking_mapping = {"Không có": "NA", "Ít": "little", "Trung bình": "moderate", "Nhiều": "rich"}
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
        prediction = mo_hinh.predict_proba(input_transformed)[:, 1]
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

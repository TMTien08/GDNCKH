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
add_bg_from_local('data:image/jpeg;base64,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')  # Thay bằng tên file hình ảnh của bạn

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

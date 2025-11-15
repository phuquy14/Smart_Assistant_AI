import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# ----------------------------------------------------
# 1. Cấu hình và Tiêu đề
# ----------------------------------------------------
st.set_page_config(layout="wide") # Thiết lập ứng dụng web hiển thị rộng hơn
st.title("💡 Trợ Lý Điện Thông Minh (AI Power Assistant)")
st.write("Ứng dụng dự đoán tiêu thụ điện năng 7 ngày tới dựa trên dữ liệu lịch sử do người dùng cung cấp.")

# Tạo ô để người dùng điều chỉnh ngưỡng cảnh báo (Sidebar)
st.sidebar.subheader("Cài đặt Cảnh báo")
ALERT_THRESHOLD = st.sidebar.number_input(
    "Đặt Ngưỡng Cảnh báo (kWh):", 
    min_value=10000, 
    max_value=50000, 
    value=35000, 
    step=100,
    help="AI sẽ cảnh báo những thời điểm dự đoán tiêu thụ vượt qua ngưỡng này."
)

# ----------------------------------------------------
# 2. HÀM HUẤN LUYỆN VÀ DỰ BÁO (Sử dụng Cache để tối ưu tốc độ)
# ----------------------------------------------------
# @st.cache_resource: Lệnh này bảo Streamlit chỉ chạy hàm này 1 lần 
# và lưu kết quả vào bộ nhớ. Nếu file CSV không đổi, nó không chạy lại, giúp ứng dụng siêu nhanh.
@st.cache_resource 
def train_and_predict(df_input):
    # Chuẩn bị Dữ liệu cho Prophet (đảm bảo cột 'ds' và 'y')
    
    # Kiểm tra các cột bắt buộc
    required_cols = ['Datetime', 'PowerConsumption_Zone1']
    if not all(col in df_input.columns for col in required_cols):
        st.error("LỖI DỮ LIỆU: File CSV của bạn phải có các cột 'Datetime' và 'PowerConsumption_Zone1'.")
        return None, None

    df_prophet = df_input[required_cols].copy()
    df_prophet.rename(columns={'Datetime': 'ds', 'PowerConsumption_Zone1': 'y'}, inplace=True)
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

    # Khởi tạo và Huấn luyện Mô hình
    model = Prophet(interval_width=0.95, daily_seasonality=True)
    model.fit(df_prophet)

    # Tạo Khung Thời Gian và Dự báo (7 ngày = 1008 điểm)
    future = model.make_future_dataframe(periods=1008, freq='10min')
    forecast = model.predict(future)
    
    return model, forecast

# ----------------------------------------------------
# 3. HÀM CHÍNH ĐỂ CHẠY ỨNG DỤNG
# ----------------------------------------------------
def run_app():
    
    st.sidebar.subheader("Tải lên Dữ liệu của Bạn")
    
    # Kích hoạt tính năng tải file lên cho NGƯỜI DÙNG KHÁC
    uploaded_file = st.sidebar.file_uploader(
        "Vui lòng tải lên file CSV chứa dữ liệu tiêu thụ điện của bạn (phải có cột 'Datetime' và 'PowerConsumption_Zone1')", 
        type=['csv']
    )
    
    # Kiểm tra xem người dùng đã tải file lên chưa
    if uploaded_file is None:
        st.info("Vui lòng tải lên file dữ liệu tiêu thụ điện của bạn để bắt đầu phân tích.")
        return # Thoát khỏi hàm nếu chưa có file

    # Nếu có file, đọc file đó
    df = pd.read_csv(uploaded_file)
    
    st.subheader("1. Xử lý và Huấn luyện Mô hình")
    
    # Bắt đầu quá trình huấn luyện
    with st.spinner('Đang huấn luyện mô hình Prophet... (Vui lòng chờ) '):
        model, forecast = train_and_predict(df)

    if model is None:
        return # Thoát nếu có lỗi dữ liệu

    st.success("Huấn luyện mô hình AI hoàn tất!")

    # ----------------------------------------------------
    # 4. HIỂN THỊ KẾT QUẢ TRỰC QUAN
    # ----------------------------------------------------
    st.subheader("2. Biểu đồ Dự báo 7 Ngày tới")
    
    # Vẽ biểu đồ 1 và hiển thị bằng st.pyplot
    fig1 = model.plot(forecast)
    plt.title("Dự báo Tiêu Thụ Điện (Zone 1) - Hiện tại & Tương lai")
    plt.xlabel("Thời gian")
    plt.ylabel("Tiêu thụ Điện (kWh)")
    st.pyplot(fig1)

    # ----------------------------------------------------
    # 5. CHẠY CẢNH BÁO THÔNG MINH
    # ----------------------------------------------------
    st.subheader("3. Cảnh báo và Lời khuyên")
    
    # Lọc ra chỉ 7 ngày dự báo cuối cùng
    final_forecast = forecast.tail(1008)
    
    # Lọc ra các thời điểm vượt ngưỡng do người dùng đặt
    alerts = final_forecast[final_forecast['yhat'] > ALERT_THRESHOLD]

    if alerts.empty:
        st.success(f"🎉 Chúc mừng! 7 ngày tới dự kiến không có mức tiêu thụ điện nào vượt quá ngưỡng **{ALERT_THRESHOLD} kWh**.")
    else:
        st.warning(f"⚠️ CẢNH BÁO: Phát hiện **{len(alerts)}** thời điểm tiêu thụ điện dự kiến rất cao.")
        
        # Tìm thời điểm tiêu thụ đỉnh điểm
        peak_consumption = alerts['yhat'].max()
        peak_time_row = alerts[alerts['yhat'] == peak_consumption].iloc[0]
        
        date_str = peak_time_row['ds'].strftime('%Y-%m-%d')
        time_str = peak_time_row['ds'].strftime('%H:%M')
        
        # Hiển thị kết quả cảnh báo nổi bật bằng HTML
        st.markdown(f"""
        <div style="background-color:#ffe6e6; padding:15px; border-radius:10px;">
            <h4>💡 THỜI ĐIỂM TIÊU THỤ ĐỈNH ĐIỂM DỰ KIẾN:</h4>
            <p><strong>Ngày:</strong> {date_str} lúc <strong>{time_str}</strong></p>
            <p><strong>Tiêu thụ dự kiến:</strong> {peak_consumption:.2f} kWh</p>
            <p>🔥 <strong>LỜI KHUYÊN:</strong> Hãy cân nhắc điều chỉnh việc sử dụng thiết bị công suất lớn vào thời điểm này để tiết kiệm chi phí!</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**10 Cảnh báo chi tiết đầu tiên:**")
        # Hiển thị 10 cảnh báo đầu tiên dưới dạng bảng
        st.dataframe(alerts[['ds', 'yhat']].head(10).rename(columns={'ds': 'Thời gian', 'yhat': 'Dự kiến (kWh)'}))

# Chạy ứng dụng web
run_app()

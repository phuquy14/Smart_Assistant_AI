import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# ----------------------------------------------------
# 1. CẤU HÌNH VÀ TIÊU ĐỀ
# ----------------------------------------------------
st.set_page_config(layout="wide") # Thiết lập ứng dụng web hiển thị rộng hơn
st.title("💡 Trợ Lý Điện Thông Minh Cá Nhân Hóa (AI Power Assistant)")
st.write("Ứng dụng dự đoán tiêu thụ điện năng 7 ngày tới dựa trên dữ liệu lịch sử do người dùng cung cấp.")

# ----------------------------------------------------
# 2. HÀM HUẤN LUYỆN VÀ DỰ BÁO (Sử dụng Cache để tối ưu tốc độ)
# ----------------------------------------------------
@st.cache_resource 
def train_and_predict(df_input, value_col):
    # Kiểm tra các cột bắt buộc
    required_cols = ['Datetime', value_col]
    if not all(col in df_input.columns for col in required_cols):
        st.error(f"LỖI DỮ LIỆU: File CSV của bạn phải có cột 'Datetime' và cột giá trị '{value_col}'.")
        return None, None

    # Chuẩn bị Dữ liệu cho Prophet (đảm bảo cột 'ds' và 'y')
    df_prophet = df_input[required_cols].copy()
    df_prophet.rename(columns={'Datetime': 'ds', value_col: 'y'}, inplace=True)
    
    try:
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    except Exception:
        st.error("LỖI ĐỊNH DẠNG: Không thể chuyển đổi cột 'Datetime' sang định dạng thời gian. Vui lòng kiểm tra lại dữ liệu.")
        return None, None

    # Khởi tạo và Huấn luyện Mô hình
    model = Prophet(interval_width=0.95, daily_seasonality=True)
    model.fit(df_prophet)

    # Tạo Khung Thời Gian và Dự báo (7 ngày = 1008 điểm với tần suất 10 phút)
    future = model.make_future_dataframe(periods=1008, freq='10min')
    forecast = model.predict(future)
    
    return model, forecast

# ----------------------------------------------------
# 3. HÀM CHÍNH ĐỂ CHẠY ỨNG DỤNG
# ----------------------------------------------------
def run_app():
    
    # 3.1. Cấu hình Tải file và Cảnh báo
    st.sidebar.subheader("Tải lên Dữ liệu của Bạn")
    uploaded_file = st.sidebar.file_uploader(
        "Vui lòng tải lên file CSV chứa dữ liệu tiêu thụ điện.", 
        type=['csv']
    )
    
    # Thoát nếu chưa có file
    if uploaded_file is None:
        st.info("Vui lòng tải lên file dữ liệu tiêu thụ điện của bạn để bắt đầu phân tích. File phải có cột 'Datetime' và ít nhất một cột giá trị (ví dụ: 'PowerConsumption_Zone1').")
        return 
    
    # Đọc file và hiển thị lựa chọn cột
    df = pd.read_csv(uploaded_file)
    
    # Tìm tất cả các cột kiểu số (có khả năng là cột tiêu thụ điện)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Bắt buộc phải có cột Datetime
    if 'Datetime' not in df.columns:
        st.error("LỖI DỮ LIỆU: File CSV của bạn phải có cột tên là 'Datetime'.")
        return
        
    st.sidebar.markdown("---")
    st.sidebar.subheader("Cấu hình Dự đoán")
    
    # Cho phép người dùng chọn cột giá trị
    value_col = st.sidebar.selectbox(
        "Chọn Cột Giá Trị Cần Dự Đoán:",
        options=numeric_cols,
        index=numeric_cols.index('PowerConsumption_Zone1') if 'PowerConsumption_Zone1' in numeric_cols else 0,
        help="Chọn cột chứa giá trị tiêu thụ điện mà bạn muốn AI dự đoán."
    )
    
    # Tạo ô để người dùng điều chỉnh ngưỡng cảnh báo
    ALERT_THRESHOLD = st.sidebar.number_input(
        f"Đặt Ngưỡng Cảnh báo cho {value_col} (kWh):", 
        min_value=df[value_col].min(), 
        max_value=df[value_col].max() * 1.2, # Ngưỡng tối đa cao hơn mức lớn nhất một chút
        value=df[value_col].mean() * 1.5, # Giá trị mặc định là 1.5 lần trung bình
        step=100
    )


    # 3.2. Huấn luyện Mô hình
    st.subheader("1. Xử lý và Huấn luyện Mô hình")
    
    with st.spinner(f'Đang huấn luyện mô hình Prophet dựa trên cột {value_col}...'):
        model, forecast = train_and_predict(df, value_col)

    if model is None:
        return # Thoát nếu có lỗi

    st.success("Huấn luyện mô hình AI hoàn tất!")

    # ----------------------------------------------------
    # 4. HIỂN THỊ KẾT QUẢ TRỰC QUAN
    # ----------------------------------------------------
    st.subheader(f"2. Biểu đồ Dự báo 7 Ngày tới cho {value_col}")
    
    # Vẽ biểu đồ 1 và hiển thị bằng st.pyplot
    fig1 = model.plot(forecast)
    plt.title(f"Dự báo Tiêu Thụ Điện ({value_col}) - Hiện tại & Tương lai")
    plt.xlabel("Thời gian")
    plt.ylabel("Tiêu thụ Điện (kWh)")
    st.pyplot(fig1)

    # ----------------------------------------------------
    # 5. CHẠY CẢNH BÁO THÔNG MINH
    # ----------------------------------------------------
    st.subheader(f"3. Cảnh báo và Lời khuyên cho {value_col}")
    
    final_forecast = forecast.tail(1008)
    alerts = final_forecast[final_forecast['yhat'] > ALERT_THRESHOLD]

    if alerts.empty:
        st.success(f"🎉 Chúc mừng! 7 ngày tới dự kiến không có mức tiêu thụ điện nào vượt quá ngưỡng **{ALERT_THRESHOLD:.2f} kWh**.")
    else:
        st.warning(f"⚠️ CẢNH BÁO: Phát hiện **{len(alerts)}** thời điểm tiêu thụ điện dự kiến rất cao.")
        
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
        
        st.markdown(f"**10 Cảnh báo chi tiết đầu tiên vượt ngưỡng {ALERT_THRESHOLD:.2f} kWh:**")
        st.dataframe(alerts[['ds', 'yhat']].head(10).rename(columns={'ds': 'Thời gian', 'yhat': 'Dự kiến (kWh)'}))

# Chạy ứng dụng web
run_app()

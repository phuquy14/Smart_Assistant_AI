import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np

# Thiết lập ngôn ngữ và font cho biểu đồ
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False 

# ----------------------------------------------------
# 1. CẤU HÌNH CHUNG VÀ GIAO DIỆN
# ----------------------------------------------------
st.set_page_config(layout="wide") 
st.title("💡 Trợ Lý Điện Thông Minh Cá Nhân Hóa")
st.markdown("Ứng dụng dự đoán mức tiêu thụ điện năng 7 ngày tới dựa trên dữ liệu lịch sử do người dùng cung cấp. **Vui lòng tải file CSV của bạn lên ở thanh bên.**")

# ----------------------------------------------------
# 2. HÀM HUẤN LUYỆN VÀ DỰ BÁO (Tối ưu tốc độ với Cache)
# ----------------------------------------------------
@st.cache_resource 
def train_and_predict(df_input, date_col, value_col):
    # Select and rename columns for Prophet: 'ds' (Date/Time) and 'y' (Value)
    
    # Check if the required columns exist
    if date_col not in df_input.columns or value_col not in df_input.columns:
        st.error(f"LỖI CỘT: Không tìm thấy cột '{date_col}' hoặc '{value_col}' trong file của bạn.")
        return None, None

    # Select and rename columns
    df_prophet = df_input[[date_col, value_col]].copy()
    df_prophet.rename(columns={date_col: 'ds', value_col: 'y'}, inplace=True)
    
    # Attempt to convert the date column
    try:
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    except Exception:
        st.error(f"LỖỖI ĐỊNH DẠNG: Không thể chuyển đổi cột '{date_col}' sang định dạng thời gian. Vui lòng kiểm tra dữ liệu trong cột này.")
        return None, None
        
    # Ensure 'y' column is numeric
    if not pd.api.types.is_numeric_dtype(df_prophet['y']):
        st.error(f"LỖI DỮ LIỆU: Cột giá trị '{value_col}' phải là dạng số (ví dụ: kWh, Ampe...).")
        return None, None
    
    # Filter out any non-finite values that Prophet can't handle
    df_prophet.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_prophet.dropna(inplace=True)

    # Check for sufficient data
    if len(df_prophet) < 50:
         st.error("LỖI DỮ LIỆU: Cần tối thiểu 50 điểm dữ liệu (50 hàng) để mô hình huấn luyện hiệu quả.")
         return None, None

    # Initialize and Train the Prophet Model
    # Assumes data is at 10-minute intervals for prediction frequency
    model = Prophet(interval_width=0.95, daily_seasonality=True)
    model.fit(df_prophet)

    # Create Future Dataframe for prediction (7 days = 1008 points at 10-minute frequency)
    future = model.make_future_dataframe(periods=1008, freq='10min')
    forecast = model.predict(future)
    
    return model, forecast

# ----------------------------------------------------
# 3. HÀM CHÍNH ĐỂ CHẠY ỨNG DỤNG
# ----------------------------------------------------
def run_app():
    
    # 3.1. Cấu hình Tải file
    st.sidebar.subheader("1. Tải lên Dữ liệu")
    uploaded_file = st.sidebar.file_uploader(
        "Vui lòng tải lên file CSV chứa dữ liệu của bạn.", 
        type=['csv']
    )
    
    if uploaded_file is None:
        st.info("Vui lòng tải lên file CSV để bắt đầu phân tích.")
        return 
    
    # Read the uploaded file
    df = pd.read_csv(uploaded_file)
    
    # 3.2. Cấu hình Cột
    st.sidebar.markdown("---")
    st.sidebar.subheader("2. Cấu hình Cột Dữ liệu")
    
    all_cols = df.columns.tolist()
    
    # Cột Ngày/Giờ
    date_col = st.sidebar.selectbox(
        "Chọn Cột Ngày/Giờ (Time Series):",
        options=all_cols,
        index=all_cols.index('Datetime') if 'Datetime' in all_cols else 0, # Default to 'Datetime' or the first column
        help="Chọn cột chứa thông tin thời gian (Ngày, Giờ, Tháng...). Tên cột không cần là 'Datetime'."
    )

    # Cột Giá trị
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if not numeric_cols:
        st.error("LỖI: File của bạn không có cột nào ở dạng số để dự đoán. Vui lòng kiểm tra lại dữ liệu.")
        return

    # Set default index for value column based on existing column names
    default_value_index = 0
    if 'PowerConsumption_Zone1' in numeric_cols:
        default_value_index = numeric_cols.index('PowerConsumption_Zone1')
    elif 'PowerConsumption_Zone2' in numeric_cols:
        default_value_index = numeric_cols.index('PowerConsumption_Zone2')
    elif 'PowerConsumption_Zone3' in numeric_cols:
        default_value_index = numeric_cols.index('PowerConsumption_Zone3')
        
    value_col = st.sidebar.selectbox(
        "Chọn Cột Giá Trị Cần Dự Đoán:",
        options=numeric_cols,
        index=default_value_index,
        help="Chọn cột chứa giá trị (ví dụ: kWh) mà bạn muốn AI dự đoán."
    )
    
    # 3.3. Cấu hình Ngưỡng Cảnh báo
    st.sidebar.markdown("---")
    st.sidebar.subheader("3. Cấu hình Cảnh báo")

    try:
        min_val = df[value_col].min()
        max_val = df[value_col].max()
        default_val = df[value_col].mean() * 1.5 if df[value_col].mean() > min_val else min_val * 1.5
    except:
        min_val = 10000.0
        max_val = 50000.0
        default_val = 35000.0

    ALERT_THRESHOLD = st.sidebar.number_input(
        f"Đặt Ngưỡng Cảnh báo ({value_col}):", 
        min_value=min_val, 
        max_value=max_val * 1.2,
        value=default_val, 
        step=100.0, 
        format="%.2f",
        help="Đặt mức tiêu thụ cao nhất mà bạn muốn AI cảnh báo."
    )

    # 3.4. Huấn luyện Mô hình
    st.subheader(f"1. Huấn luyện Mô hình ({date_col} vs. {value_col})")
    
    with st.spinner(f'Đang huấn luyện mô hình Prophet...'):
        model, forecast = train_and_predict(df, date_col, value_col)

    if model is None:
        return # Exit if there was a data error

    st.success("Huấn luyện mô hình AI hoàn tất!")

    # ----------------------------------------------------
    # 4. HIỂN THỊ KẾT QUẢ TRỰC QUAN
    # ----------------------------------------------------
    st.subheader(f"2. Biểu đồ Dự báo 7 Ngày tới")
    
    # Draw and display the first chart (Forecast)
    fig1 = model.plot(forecast)
    plt.title(f"Dự báo Tiêu Thụ ({value_col}) - Hiện tại & Tương lai")
    plt.xlabel("Thời gian")
    plt.ylabel(f"Giá trị ({value_col})")
    st.pyplot(fig1)

    # ----------------------------------------------------
    # 5. CHẠY CẢNH BÁO THÔNG MINH
    # ----------------------------------------------------
    st.subheader(f"3. Cảnh báo và Lời khuyên")
    
    final_forecast = forecast.tail(1008)
    alerts = final_forecast[final_forecast['yhat'] > ALERT_THRESHOLD]

    if alerts.empty:
        st.success(f"🎉 Chúc mừng! 7 ngày tới dự kiến không có mức tiêu thụ nào vượt quá ngưỡng **{ALERT_THRESHOLD:.2f}**.")
    else:
        st.warning(f"⚠️ CẢNH BÁO: Phát hiện **{len(alerts)}** thời điểm tiêu thụ dự kiến rất cao.")
        
        peak_consumption = alerts['yhat'].max()
        peak_time_row = alerts[alerts['yhat'] == peak_consumption].iloc[0]
        
        date_str = peak_time_row['ds'].strftime('%Y-%m-%d')
        time_str = peak_time_row['ds'].strftime('%H:%M')
        
        # Display the highlighted alert result
        st.markdown(f"""
        <div style="background-color:#ffe6e6; padding:15px; border-radius:10px;">
            <h4>💡 THỜI ĐIỂM TIÊU THỤ ĐỈNH ĐIỂM DỰ KIẾN:</h4>
            <p><strong>Ngày:</strong> {date_str} lúc <strong>{time_str}</strong></p>
            <p><strong>Tiêu thụ dự kiến:</strong> {peak_consumption:.2f}</p>
            <p>🔥 <strong>LỜI KHUYÊN:</strong> Hãy cân nhắc điều chỉnh việc sử dụng thiết bị công suất lớn vào thời điểm này để tiết kiệm chi phí!</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"**10 Cảnh báo chi tiết đầu tiên vượt ngưỡng {ALERT_THRESHOLD:.2f}:**")
        st.dataframe(alerts[['ds', 'yhat']].head(10).rename(columns={'ds': 'Thời gian', 'yhat': f'Dự kiến ({value_col})'}))

# Run the web application
run_app()

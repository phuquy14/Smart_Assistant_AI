import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# ----------------------------------------------------
# 1. Cấu hình và Tiêu đề
# ----------------------------------------------------
st.set_page_config(layout="wide") # Thiết lập ứng dụng web hiển thị rộng hơn
st.title("💡 Trợ Lý Điện Thông Minh (AI Power Assistant)")
st.write("Ứng dụng dự đoán tiêu thụ điện năng 7 ngày tới dựa trên file powerconsumption.csv.")

# Tạo ô để người dùng điều chỉnh ngưỡng cảnh báo (Sidebar)
st.sidebar.subheader("Cài đặt Cảnh báo")
ALERT_THRESHOLD = st.sidebar.number_input(
    "Đặt Ngưỡng Cảnh báo (kWh):", 
    min_value=10000, 
    max_value=50000, 
    value=35000, 
    step=100
)

# ----------------------------------------------------
# 2. HÀM CHÍNH ĐỂ CHẠY VÀ HIỂN THỊ MÔ HÌNH
# ----------------------------------------------------
# Dùng cache (lưu trữ tạm thời) để huấn luyện mô hình 
# chỉ một lần duy nhất, giúp ứng dụng chạy nhanh hơn.
@st.cache_resource 
def train_and_predict(df_input):
    # Chuẩn bị Dữ liệu cho Prophet
    df_prophet = df_input[['Datetime', 'PowerConsumption_Zone1']].copy()
    df_prophet.rename(columns={'Datetime': 'ds', 'PowerConsumption_Zone1': 'y'}, inplace=True)
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

    # Khởi tạo và Huấn luyện Mô hình
    model = Prophet(interval_width=0.95, daily_seasonality=True)
    model.fit(df_prophet)

    # Tạo Khung Thời Gian và Dự báo (7 ngày = 1008 điểm)
    future = model.make_future_dataframe(periods=1008, freq='10min')
    forecast = model.predict(future)
    
    return model, forecast

def run_app():
    st.subheader("1. Xử lý và Huấn luyện Mô hình")
    
    # Tải Dữ liệu
    try:
        df = pd.read_csv('powerconsumption.csv')
    except FileNotFoundError:
        st.error("LỖI: Không tìm thấy file 'powerconsumption.csv'. Hãy đảm bảo file này nằm cùng thư mục.")
        return

    # Huấn luyện mô hình và dự báo
    with st.spinner('Đang huấn luyện mô hình Prophet... (Quá trình này có thể mất 30-60 giây)'):
        model, forecast = train_and_predict(df)
    st.success("Huấn luyện mô hình AI hoàn tất!")

    # ----------------------------------------------------
    # 3. HIỂN THỊ KẾT QUẢ TRỰC QUAN
    # ----------------------------------------------------
    st.subheader("2. Biểu đồ Dự báo 7 Ngày tới")
    
    # Vẽ biểu đồ 1 và hiển thị bằng st.pyplot
    fig1 = model.plot(forecast)
    plt.title("Dự báo Tiêu Thụ Điện (Zone 1) - Hiện tại & Tương lai")
    plt.xlabel("Thời gian")
    plt.ylabel("Tiêu thụ Điện (kWh)")
    st.pyplot(fig1)

    # ----------------------------------------------------
    # 4. CHẠY CẢNH BÁO THÔNG MINH
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
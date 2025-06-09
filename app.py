import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency, pearsonr
from utils.ChartStorage import ChartStorage
from utils.Data import Data
from utils.DataAnalyzer import DataAnalyzer
from utils.PDF import PDF
from scipy.stats import pearsonr 
from fpdf import FPDF
import base64
import tempfile
from datetime import datetime
import re 
import os
import unicodedata # Thêm thư viện này để xử lý tiếng Việt không dấu
MODEL_DIR = 'model'
MODEL_FILENAME = 'lgbm_revenue_prediction_model.joblib'
LABEL_ENCODERS_FILENAME = 'label_encoders.joblib'

MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
LABEL_ENCODERS_PATH = os.path.join(MODEL_DIR, LABEL_ENCODERS_FILENAME)
# Các imports khác đã có sẵn
from utils.Data import Data
from utils.DataAnalyzer import DataAnalyzer


def remove_vietnamese_accents(text):
    """
    Chuyển đổi chuỗi tiếng Việt có dấu thành không dấu.
    """
    if not isinstance(text, str):
        return text # Giữ nguyên nếu không phải chuỗi (ví dụ: số, None)

    # NFD (Normalization Form Canonical Decomposition) tách ký tự có dấu thành ký tự cơ bản và dấu
    # Mn (Mark, Nonspacing) là loại unicode cho các dấu
    text = unicodedata.normalize('NFD', text)
    text = ''.join([c for c in text if not unicodedata.category(c) == 'Mn'])
    return text.encode('ascii', 'ignore').decode('utf-8') # Loại bỏ các ký tự không phải ASCII


# Tiêu đề ứng dụng.
st.title('Phân tích dữ liệu doanh số bán lẻ')


# Tạo các tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 Trang chính",
                                      "📈 Phân tích dữ liệu", "🔮 Dự đoán doanh thu", "📥 Tải tài liệu","Giới thiệu"])


if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'chart_storage' not in st.session_state:
    st.session_state.chart_storage = ChartStorage()

# Thêm biến cờ để kiểm soát việc lưu file
if 'processed_data_saved' not in st.session_state:
    st.session_state.processed_data_saved = False

@st.cache_resource # Sử dụng st.cache_resource để chỉ tải một lần
def load_model_and_encoders(model_p, encoders_p):
    try:
        model = joblib.load(model_p)
        encoders = joblib.load(encoders_p)
        st.success(f"Mô hình và LabelEncoders đã được tải thành công từ `{MODEL_DIR}/`.")
        return model, encoders
    except FileNotFoundError:
        st.error(f"Lỗi: Không tìm thấy file mô hình tại `{model_p}` hoặc LabelEncoders tại `{encoders_p}`.")
        st.info("Vui lòng đảm bảo bạn đã chạy script huấn luyện mô hình để tạo các file này.")
        return None, None
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình hoặc LabelEncoders: {e}")
        return None, None

if 'revenue_prediction_model' not in st.session_state:
    st.session_state.revenue_prediction_model = None

# Tải mô hình khi ứng dụng bắt đầu (hoặc khi cần)
if st.session_state.revenue_prediction_model is None:
    st.session_state.revenue_prediction_model, st.session_state.loaded_label_encoders = \
        load_model_and_encoders(MODEL_PATH, LABEL_ENCODERS_PATH)
        
        
# --- TRANG CHÍNH (Tab 1) ---
with tab1:
    st.subheader('Tải dữ liệu lên')
    # Sử dụng st.file_uploader để người dùng tải lên tệp CSV
    uploaded_file = st.file_uploader(
        "Vui lòng tải lên tệp dữ liệu CSV của bạn",
        type=["csv"],
        help="Chọn một tệp .csv từ máy tính của bạn."
    )


    
    # Khởi tạo df_original và df_processed ở cấp độ session state
    # để chúng được giữ lại khi các tab hoặc widget khác được tương tác.
    

    if uploaded_file is not None:
        st.info(f"Tệp đã chọn: **{uploaded_file.name}**")

        try:
            encodings_to_try = ['utf-8', 'windows-1252', 'latin-1', 'ISO-8859-1']
            temp_df = None
            for encoding in encodings_to_try:
                try:
                    uploaded_file.seek(0)
                    temp_df = pd.read_csv(uploaded_file, encoding=encoding)
                    st.success(f"Đọc tệp thành công với encoding: `{encoding}`")
                    break
                except UnicodeDecodeError:
                    pass
                except Exception as e:
                    st.error(f"Đã xảy ra lỗi không mong muốn khi đọc tệp với encoding `{encoding}`: {e}")
            
            if temp_df is not None:
                # Chỉ xử lý và lưu khi file mới được tải lên hoặc chưa được xử lý
                # (Sử dụng hash của file hoặc kiểm tra df_original đã có chưa)
                # Đơn giản nhất là kiểm tra một biến cờ
                if st.session_state.df_original is None or \
                   not st.session_state.df_original.equals(temp_df): # Kiểm tra nếu file mới là khác
                    
                    st.session_state.df_original = temp_df.copy()
                    
                    data_loader = Data(file_csv=None)
                    data_loader.original_shape = st.session_state.df_original.shape

                    df_after_missing = data_loader.process_missing_values(st.session_state.df_original.copy())
                    st.session_state.df_processed = data_loader.process_outliers(df_after_missing.copy())

                    required_cols = ['Sales', 'Quantity', 'Discount', 'Profit']
                    all_cols_exist = True
                    for col in required_cols:
                        if col in st.session_state.df_processed.columns:
                            st.session_state.df_processed[col] = pd.to_numeric(st.session_state.df_processed[col], errors='coerce').fillna(0)
                        else:
                            st.warning(f"Cột '{col}' không tìm thấy trong dữ liệu đã xử lý. Không thể tính toán Revenue.")
                            st.session_state.df_processed['Revenue'] = 0
                            all_cols_exist = False
                            break

                    if all_cols_exist:
                        st.session_state.df_processed['Revenue'] = (
                            st.session_state.df_processed['Sales'] * st.session_state.df_processed['Quantity']
                            - st.session_state.df_processed['Discount'] * st.session_state.df_processed['Sales']
                            + st.session_state.df_processed['Profit']
                        )

                    # --- Logic lưu file analyzed_data.csv ---
                    output_dir = 'processed'
                    os.makedirs(output_dir, exist_ok=True)

                    max_file_number = 0
                    for filename in os.listdir(output_dir):
                        match = re.match(r'analyzed_data(\d*)\.csv', filename)
                        if match:
                            current_number = int(match.group(1)) if match.group(1) else 0
                            if current_number > max_file_number:
                                max_file_number = current_number

                    new_file_number = max_file_number + 1
                    output_filename = f'analyzed_data{new_file_number}.csv'
                    output_path = os.path.join(output_dir, output_filename)

                    st.session_state.df_processed.to_csv(output_path, index=False)
                    st.success(f"File '{output_filename}' đã được lưu thành công!")
                    
                    # Đặt biến cờ thành True sau khi lưu
                    st.session_state.processed_data_saved = True
                else:
                    st.info("Tệp đã được tải lên và xử lý trước đó. Không cần xử lý lại.")



                st.dataframe(st.session_state.df_processed.head(10))

                st.markdown("---") 

                data_analyzer = DataAnalyzer(st.session_state.df_processed)

                st.subheader('Thống kê tổng quan')
                col1, col2 = st.columns(2)
                col3, col4 = st.columns(2)

                # Ensure 'Revenue' column is available for get_total_revenue()
                if 'Revenue' in st.session_state.df_processed.columns:
                    revenue_total = st.session_state.df_processed['Revenue'].sum() # Calculate total revenue from the new column
                else:
                    revenue_total = 0 # Or handle the error appropriately
                    st.warning("Cột 'Revenue' không có sẵn để tính tổng doanh thu.")
                
                
                with col1:
                    st.metric("Tổng doanh thu ($)", f"{revenue_total:,.2f}")
                with col2:
                    mean_revenue = data_analyzer.get_total_sales()/ data_analyzer.get_number_of_transactions()
                    # st.metric('Doanh thu trung bình', f"{data_analyzer.get_total_quantity_sold():,}")
                    st.metric('Doanh thu trung bình', f"{mean_revenue:,.2f}")
                with col3:
                    st.metric('Tổng số giao dịch', f"{data_analyzer.get_number_of_transactions():,}")
                with col4:
                    st.metric('Tổng lợi nhuận', f"{data_analyzer.get_total_profit():,.2f}")
            else:
                st.error("Không thể đọc tệp CSV với bất kỳ encoding nào đã thử.")
        except Exception as e:
            st.error(f"Đã xảy ra lỗi khi xử lý tệp: {e}. Vui lòng đảm bảo tệp đúng định dạng CSV.")
    else:
        st.info("Vui lòng tải lên tệp CSV để bắt đầu.")

# --- PHÂN TÍCH (Tab 2) ---
df_analysis = None
with tab2:
    st.header('Phân tích Doanh thu theo thời gian')

    # Lấy df_processed từ session state
    df_processed_tab2 = st.session_state.df_processed

    # Đảm bảo df_processed tồn tại và có các cột cần thiết từ Tab 1
    if df_processed_tab2 is not None and 'Order Date' in df_processed_tab2.columns and 'Sales' in df_processed_tab2.columns:
        df_analysis = df_processed_tab2.copy()

        
        df_analysis['Order Date'] = pd.to_datetime(df_analysis['Order Date'])
        df_analysis['Month'] = df_analysis['Order Date'].dt.month
        df_analysis['Year'] = df_analysis['Order Date'].dt.year
        
        
        

        # Lấy danh sách các năm có trong dữ liệu
        available_years = sorted(df_analysis['Year'].unique())
        if not available_years:
            st.warning("Không có dữ liệu năm nào được tìm thấy. Vui lòng kiểm tra cột 'Order Date'.")
            st.stop()

        # DateTimePicker (chọn năm)
        selected_year = st.selectbox(
            "Chọn Năm:",
            options=available_years,
            index=len(available_years) - 1 # Mặc định chọn năm mới nhất
        )

        # ComboBox chọn tháng từ...
        col_month_start, col_month_end = st.columns(2)
        with col_month_start:
            month_start = st.selectbox(
                "Tháng Bắt Đầu:",
                options=list(range(1, 13)),
                index=0 # Mặc định là tháng 1
            )
        with col_month_end:
            month_end = st.selectbox(
                "Tháng Kết Thúc:",
                options=list(range(1, 13)),
                index=11 # Mặc định là tháng 12
            )

        # Điều kiện: tháng bắt đầu phải <= tháng kết thúc
        if month_start > month_end:
            st.warning("Tháng Bắt Đầu không được lớn hơn Tháng Kết Thúc. Vui lòng chọn lại.")
        else:
            st.markdown("---")
            st.subheader(f"Doanh thu từ tháng {month_start} đến tháng {month_end} năm {selected_year}")

            # Lọc dữ liệu theo năm và tháng đã chọn
            filtered_df = df_analysis[
                (df_analysis['Year'] == selected_year) &
                (df_analysis['Month'] >= month_start) &
                (df_analysis['Month'] <= month_end)
            ]

            if not filtered_df.empty:
                # Nhóm dữ liệu để tính tổng doanh thu theo tháng
                monthly_sales = filtered_df.groupby('Month')['Revenue'].sum().reset_index()
                
                # Sắp xếp theo tháng để biểu đồ trông hợp lý
                monthly_sales = monthly_sales.sort_values('Month')

                # Vẽ biểu đồ doanh thu theo tháng
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Month', y='Revenue', data=monthly_sales, ax=ax, palette='viridis')
                ax.set_title(f'Tổng Doanh thu theo Tháng (Năm {selected_year})')
                ax.set_xlabel('Tháng')
                ax.set_ylabel('Doanh thu')
                ax.ticklabel_format(style='plain', axis='y')
                
                for container in ax.containers:
                    ax.bar_label(container, fmt='%.2f', label_type='edge')

                st.pyplot(fig)
                st.session_state.chart_storage.monthly_sales_chart = plt.gcf()

                # --- 3 biến gán giá trị từ metric ---
                total_sales_selected_period = monthly_sales['Revenue'].sum()
                st.metric("Tổng doanh thu trong kỳ ($)", f"{total_sales_selected_period:,.2f}")
                
                month_highest_sales = monthly_sales.loc[monthly_sales['Revenue'].idxmax()]
                st.metric("Tháng có doanh thu cao nhất", f"Tháng {int(month_highest_sales['Month'])}: {month_highest_sales['Revenue']:,.2f} $")

                month_lowest_sales = monthly_sales.loc[monthly_sales['Revenue'].idxmin()]
                st.metric("Tháng có doanh thu thấp nhất", f"Tháng {int(month_lowest_sales['Month'])}: {month_lowest_sales['Revenue']:,.2f} $")
                
                _total_sales_period_value = total_sales_selected_period
                _month_highest_sales_value = month_highest_sales['Revenue']
                _month_lowest_sales_value = month_lowest_sales['Revenue']

                # st.write(f"Giá trị gán vào biến (chỉ để minh họa):")
                # st.code(f"_total_sales_period_value = {_total_sales_period_value:,.2f}")
                # st.code(f"_month_highest_sales_value = {_month_highest_sales_value:,.2f}")
                # st.code(f"_month_lowest_sales_value = {_month_lowest_sales_value:,.2f}")

            else:
                st.warning(f"Không có dữ liệu doanh thu cho năm {selected_year} từ tháng {month_start} đến tháng {month_end}.")
    

        # --- Biểu đồ Doanh thu theo Thành phố ---
        st.markdown("---")
        st.header('Phân tích Doanh thu theo Thành phố')
        
        # SỬ DỤNG filtered_df Ở ĐÂY
        if 'City' in filtered_df.columns:
            top_n_cities = 15
            city_sales = filtered_df.groupby('City')['Revenue'].sum().nlargest(top_n_cities).reset_index()

            if not city_sales.empty:
                fig_city, ax_city = plt.subplots(figsize=(12, 7))
                sns.barplot(x='Revenue', y='City', data=city_sales, ax=ax_city, palette='coolwarm')
                ax_city.set_title(f'Top {top_n_cities} Thành phố có Doanh thu cao nhất (Tháng {month_start}-{month_end}, Năm {selected_year})')
                ax_city.set_xlabel('Doanh thu ($)')
                ax_city.set_ylabel('Thành phố')
                ax_city.ticklabel_format(style='plain', axis='x')

                for container in ax_city.containers:
                    ax_city.bar_label(container, fmt='%.2f', label_type='edge', padding=3)

                st.pyplot(fig_city)
                st.session_state.chart_storage.city_sales_chart = plt.gcf()
                
            else:
                st.info(f"Không có dữ liệu doanh thu theo thành phố để hiển thị trong khoảng thời gian đã chọn (Tháng {month_start}-{month_end}, Năm {selected_year}).")
        else:
            st.warning("Không tìm thấy cột 'City' trong dữ liệu. Vui lòng kiểm tra lại tên cột.")

        # --- Biểu đồ Doanh thu theo Sản phẩm ---
        st.markdown("---")
        st.header('Phân tích Doanh thu theo Sản phẩm')

        # SỬ DỤNG filtered_df Ở ĐÂY
        if 'Product Name' in filtered_df.columns:
            top_n_products = 15
            product_sales = filtered_df.groupby('Product Name')['Revenue'].sum().nlargest(top_n_products).reset_index()

            if not product_sales.empty:
                fig_product, ax_product = plt.subplots(figsize=(12, 8))
                sns.barplot(x='Revenue', y='Product Name', data=product_sales, ax=ax_product, palette='magma')
                ax_product.set_title(f'Top {top_n_products} Sản phẩm có Doanh thu cao nhất (Tháng {month_start}-{month_end}, Năm {selected_year})')
                ax_product.set_xlabel('Doanh thu ($)')
                ax_product.set_ylabel('Tên Sản phẩm')
                ax_product.ticklabel_format(style='plain', axis='x')
                
                for container in ax_product.containers:
                    ax_product.bar_label(container, fmt='%.2f', label_type='edge', padding=3)

                st.pyplot(fig_product)
                st.session_state.chart_storage.product_sales_chart = plt.gcf()

                # Thêm combobox để chọn số lượng sản phẩm hiển thị
                st.markdown("---")
                st.subheader('Biểu đồ tùy chỉnh: Top N Sản phẩm')
                
                # Đảm bảo max_value không bị lỗi nếu filtered_df không có Product Name
                if not filtered_df.empty and 'Product Name' in filtered_df.columns:
                    max_products_to_show = min(50, len(filtered_df['Product Name'].unique()))
                else:
                    max_products_to_show = 5 # Giá trị mặc định nếu không có dữ liệu sản phẩm

                n_products_to_show = st.slider(
                    "Chọn số lượng sản phẩm hàng đầu để hiển thị:",
                    min_value=5,
                    max_value=max_products_to_show,
                    value=min(10, max_products_to_show)
                )
                
                # SỬ DỤNG filtered_df Ở ĐÂY
                custom_product_sales = filtered_df.groupby('Product Name')['Revenue'].sum().nlargest(n_products_to_show).reset_index()
                
                if not custom_product_sales.empty:
                    fig_custom_product, ax_custom_product = plt.subplots(figsize=(12, max(6, n_products_to_show * 0.5)))
                    sns.barplot(x='Revenue', y='Product Name', data=custom_product_sales, ax=ax_custom_product, palette='plasma')
                    ax_custom_product.set_title(f'Top {n_products_to_show} Sản phẩm có Doanh thu cao nhất (Tháng {month_start}-{month_end}, Năm {selected_year})')
                    ax_custom_product.set_xlabel('Doanh thu ($)')
                    ax_custom_product.set_ylabel('Tên Sản phẩm')
                    ax_custom_product.ticklabel_format(style='plain', axis='x')
                    
                    for container in ax_custom_product.containers:
                        ax_custom_product.bar_label(container, fmt='%.2f', label_type='edge', padding=3)

                    st.pyplot(fig_custom_product)
                else:
                    st.info(f"Không có dữ liệu sản phẩm để hiển thị với số lượng đã chọn trong khoảng thời gian đã chọn (Tháng {month_start}-{month_end}, Năm {selected_year}).")

            else:
                st.info(f"Không có dữ liệu doanh thu theo sản phẩm để hiển thị trong khoảng thời gian đã chọn (Tháng {month_start}-{month_end}, Năm {selected_year}).")
        else:
            st.warning("Không tìm thấy cột 'Product Name' trong dữ liệu. Vui lòng kiểm tra lại tên cột.")

        # --- Phân tích Tương quan ---
        st.markdown("---")
        # st.header('Phân tích Tương quan giữa các biến')

        # Đảm bảo filtered_df không rỗng trước khi thực hiện phân tích tương quan
        if filtered_df.empty:
            # --- Tương quan Pearson (Biến Định Lượng) ---
            st.subheader('Tương quan Pearson (Biến Định Lượng)')
            st.write("Đo lường mối quan hệ tuyến tính giữa các biến định lượng.")

            quantitative_pairs = [
                ('Revenue', 'Quantity'),
                ('Revenue', 'Discount'),
                ('Revenue', 'Profit'),
                
            ]

            for var1, var2 in quantitative_pairs:
                if var1 in filtered_df.columns and var2 in filtered_df.columns:
                    # Loại bỏ các hàng có giá trị NaN để đảm bảo pearsonr hoạt động
                    temp_df_corr = filtered_df[[var1, var2]].dropna()

                    if not temp_df_corr.empty:
                        correlation, p_value = pearsonr(temp_df_corr[var1], temp_df_corr[var2])
                        st.write(f"**Mối quan hệ giữa {var1} và {var2}:**")
                        st.write(f"Hệ số tương quan Pearson: `{correlation:.2f}`")
                        st.write(f"Giá trị P-value: `{p_value:.3f}`")

                        if p_value < 0.05:
                            if abs(correlation) > 0.7:
                                st.success(f"Có mối tương quan **rất mạnh** và **có ý nghĩa thống kê** giữa **{var1}** và **{var2}** (p < 0.05).")
                            elif abs(correlation) > 0.4:
                                st.success(f"Có mối tương quan **mạnh** và **có ý nghĩa thống kê** giữa **{var1}** và **{var2}** (p < 0.05).")
                            elif abs(correlation) > 0.2:
                                st.success(f"Có mối tương quan **trung bình** và **có ý nghĩa thống kê** giữa **{var1}** và **{var2}** (p < 0.05).")
                            else:
                                st.info(f"Có mối tương quan **yếu** nhưng **có ý nghĩa thống kê** giữa **{var1}** và **{var2}** (p < 0.05).")
                        else:
                            st.info(f"Không có bằng chứng đủ mạnh về mối tương quan tuyến tính có ý nghĩa thống kê giữa **{var1}** và **{var2}** (p >= 0.05).")

                        # Vẽ biểu đồ Scatter Plot
                        fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
                        sns.scatterplot(x=var1, y=var2, data=temp_df_corr, ax=ax_corr, alpha=0.6)
                        ax_corr.set_title(f'Tương quan giữa {var1} và {var2}')
                        ax_corr.set_xlabel(var1)
                        ax_corr.set_ylabel(var2)
                        st.pyplot(fig_corr)
                        
                        st.markdown("---")
                    else:
                        st.info(f"Không có đủ dữ liệu (sau khi loại bỏ NaN) để tính tương quan giữa {var1} và {var2} trong khoảng thời gian đã chọn.")
                else:
                    st.warning(f"Không tìm thấy cột '{var1}' hoặc '{var2}' trong dữ liệu. Vui lòng kiểm tra lại tên cột.")

    else:
        st.warning("Vui lòng tải và xử lý dữ liệu ở 'Trang chính' trước khi xem phân tích.")

categorical_features = ['Sub-Category', 'Category', 'City']
with tab3:
    st.header("Mô hình Dự đoán Doanh thu")
    st.write("Nhập các thông số để nhận dự đoán doanh thu:")
    
    df_for_prediction = None
    if st.session_state.revenue_prediction_model is None or st.session_state.loaded_label_encoders is None:
        st.warning("Mô hình dự đoán hoặc LabelEncoders chưa được tải. Vui lòng kiểm tra các lỗi tải ở trên.")
    else:
        if df_analysis is None:
            df_analysis = pd.read_csv("processed/analyzed_data.csv", encoding="windows-1252")
            
        df_for_prediction = df_analysis.copy()

        if df_for_prediction is not None:
            # Lấy danh sách các giá trị duy nhất từ df_for_prediction (dữ liệu gốc đã được làm sạch)
            # Các LabelEncoder sẽ dùng để mã hóa các giá trị người dùng nhập vào.

            available_sub_categories = []
            if 'Sub-Category' in df_for_prediction.columns and not df_for_prediction['Sub-Category'].empty:
                available_sub_categories = sorted(df_for_prediction['Sub-Category'].unique())
            else:
                st.warning("Không tìm thấy cột 'Sub-Category' hoặc không có dữ liệu trong đó. Vui lòng tải dữ liệu hợp lệ.")
                available_sub_categories = ['(Không có dữ liệu)'] # Fallback
                if not available_sub_categories: # Đảm bảo có ít nhất 1 lựa chọn để tránh lỗi selectbox
                    available_sub_categories = ['N/A']


            available_categories = []
            if 'Category' in df_for_prediction.columns and not df_for_prediction['Category'].empty:
                available_categories = sorted(df_for_prediction['Category'].unique())
            else:
                st.warning("Không tìm thấy cột 'Category' hoặc không có dữ liệu trong đó. Vui lòng tải dữ liệu hợp lệ.")
                available_categories = ['(Không có dữ liệu)'] # Fallback
                if not available_categories:
                    available_categories = ['N/A']


            available_cities = []
            if 'City' in df_for_prediction.columns and not df_for_prediction['City'].empty:
                available_cities = sorted(df_for_prediction['City'].unique())
            else:
                st.warning("Không tìm thấy cột 'City' hoặc không có dữ liệu trong đó. Vui lòng tải dữ liệu hợp lệ.")
                available_cities = ['(Không có dữ liệu)'] # Fallback
                if not available_cities:
                    available_cities = ['N/A']


            st.subheader("Chọn các danh mục:")
            col_cat1, col_cat2, col_cat3 = st.columns(3)
            with col_cat1:
                selected_sub_category = st.selectbox(
                    "Phân loại phụ (Sub-Category):",
                    options=available_sub_categories,
                    index=0 if available_sub_categories else None # Chọn phần tử đầu tiên
                )
            with col_cat2:
                selected_category = st.selectbox(
                    "Danh mục (Category):",
                    options=available_categories,
                    index=0 if available_categories else None
                )
            with col_cat3:
                selected_city = st.selectbox(
                    "Thành phố (City):",
                    options=available_cities,
                    index=0 if available_cities else None
                )

            st.markdown("---")
            st.subheader("Nhập các giá trị định lượng:")

            # Sales (Đơn giá)
            sales_input = st.number_input("Đơn giá (Sales):", min_value=0.0, value=100000.0, format="%.2f", step=1000.0)
            if sales_input <= 0:
                st.error("Đơn giá phải lớn hơn 0. Vui lòng nhập lại.")

            # Quantity (Số lượng)
            quantity_input = st.number_input("Số lượng (Quantity):", min_value=1, value=1, format="%d", step=1)
            if quantity_input <= 0:
                st.error("Số lượng phải lớn hơn 0. Vui lòng nhập lại.")

            st.markdown("---")

            if st.button("Dự đoán Doanh thu"):
                # Kiểm tra các điều kiện đầu vào
                if sales_input <= 0 or quantity_input <= 0:
                    st.error("Vui lòng sửa các lỗi nhập liệu trước khi dự đoán.")
                elif selected_sub_category == '(Không có dữ liệu)' or selected_category == '(Không có dữ liệu)' or selected_city == '(Không có dữ liệu)':
                    st.error("Không có đủ dữ liệu danh mục để dự đoán. Vui lòng tải dữ liệu hợp lệ ở Trang chính.")
                else:
                    input_data_raw = pd.DataFrame([[selected_sub_category, selected_category, selected_city, sales_input, quantity_input]],
                                                columns=['Sub-Category', 'Category', 'City', 'Sales', 'Quantity'])

                    st.write("Dữ liệu đầu vào thô:")
                    st.dataframe(input_data_raw)

                    # Tiền xử lý dữ liệu đầu vào bằng các LabelEncoder đã tải
                    input_data_processed = input_data_raw.copy()
                    # Duyệt qua các đặc trưng định tính đã được định nghĩa
                    for feature in categorical_features:
                        # Lấy LabelEncoder tương ứng
                        le = st.session_state.loaded_label_encoders[feature]
                        # Xử lý trường hợp giá trị mới không có trong dữ liệu huấn luyện
                        input_data_processed[feature] = input_data_processed[feature].apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
                    st.write("Dữ liệu đầu vào sau khi mã hóa:")
                    st.dataframe(input_data_processed)

                    try:
                        predicted_revenue = st.session_state.revenue_prediction_model.predict(input_data_processed)[0]
                        st.success(f"Doanh thu dự đoán là: **{predicted_revenue:,.2f} $**")
                    except Exception as predict_e:
                        st.error(f"Lỗi khi dự đoán: {predict_e}. Vui lòng kiểm tra lại dữ liệu nhập và mô hình.")
                        st.warning("Đảm bảo các cột đầu vào cho mô hình (`Sub-Category`, `Category`, `City`, `Sales`, `Quantity`) khớp với những gì mô hình đã được huấn luyện.")
        else:
            st.warning("Vui lòng tải và xử lý dữ liệu ở 'Trang chính' để có danh sách các danh mục và thành phố.")
  
  
  

                
with tab4:
    st.header("📥 Tai tai lieu")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image("image/pdf.png", width=150)
        
    with col2:
        st.markdown("""
        ### Tai xuong du lieu va bao cao
        Chon loai tai lieu ban muon tai xuong:
        - Bao cao phan tich PDF day du (bao gom bieu do)
        - Du lieu da duoc xu ly va chuan hoa (CSV)
        """)
    
    st.markdown("---")
    
    if 'df_processed' not in st.session_state or st.session_state.df_processed is None:
        st.warning("Vui long tai len va xu ly du lieu truoc o tab 'Trang chinh'")
    else:
        col_download1, col_download2 = st.columns(2)
        
        with col_download1:
            if st.button("📄 Tao bao cao PDF", help="Tao bao cao phan tich day du dang PDF bao gom cac bieu do"):
                
                temp_img_dir = tempfile.mkdtemp()
                
                try:
                    if 'chart_storage' not in st.session_state or st.session_state.chart_storage is None:
                        st.error("Khong tim thay bieu do de tao bao cao. Vui long truy cap tab 'Phan tich du lieu' truoc.")
                        raise ValueError("Chart storage is empty.")
                        
                    storage = st.session_state.chart_storage
                    
                    chart_paths = {}
                    
                    if hasattr(storage, 'monthly_sales_chart') and storage.monthly_sales_chart is not None:
                        monthly_path = os.path.join(temp_img_dir, "monthly_sales.png")
                        storage.monthly_sales_chart.savefig(monthly_path, bbox_inches='tight', dpi=300)
                        chart_paths['monthly'] = monthly_path
                    
                    if hasattr(storage, 'city_sales_chart') and storage.city_sales_chart is not None:
                        city_path = os.path.join(temp_img_dir, "city_sales.png")
                        storage.city_sales_chart.savefig(city_path, bbox_inches='tight', dpi=300)
                        chart_paths['city'] = city_path
                    
                    if hasattr(storage, 'product_sales_chart') and storage.product_sales_chart is not None:
                        product_path = os.path.join(temp_img_dir, "product_sales.png")
                        storage.product_sales_chart.savefig(product_path, bbox_inches='tight', dpi=300)
                        chart_paths['product'] = product_path
                    
                    # Tạo PDF từ lớp PDF tùy chỉnh của bạn (đã được cấu hình để không cần font tiếng Việt)
                    pdf = PDF() 
                    pdf.add_page()
                    
                    # --- Tiêu đề báo cáo ---
                    # Tất cả các chuỗi tiếng Việt sẽ được chuyển thành không dấu
                    pdf.set_font("Arial", 'B', 16) # Dùng Arial in đậm
                    pdf.cell(200, 10, txt=remove_vietnamese_accents("BÁO CÁO PHÂN TÍCH DOANH SỐ BÁN LẺ"), ln=True, align='C')
                    pdf.set_font("Arial", size=12) # Dùng Arial thường
                    pdf.cell(200, 10, txt=remove_vietnamese_accents(f"Ngày tạo: {datetime.now().strftime('%d/%m/%Y %H:%M')}"), ln=True, align='C')
                    pdf.ln(15)
                    
                    # --- Phần 1: Thống kê tổng quan ---
                    pdf.set_font("Arial", 'B', 14)
                    pdf.cell(200, 10, txt=remove_vietnamese_accents("1. THỐNG KÊ TỔNG QUAN"), ln=True)
                    pdf.set_font("Arial", size=12)
                    
                    df = df_analysis.copy()
                    data_analyzer = DataAnalyzer(df)
                    
                    metrics = [
                        ("Tong doanh thu", f"{data_analyzer.get_total_revenue():,.2f}"),
                        ("Tong so luong ban", f"{data_analyzer.get_total_quantity_sold():,}"),
                        ("Tong so giao dich", f"{data_analyzer.get_number_of_transactions():,}"),
                        ("Tong loi nhuan", f"{data_analyzer.get_total_profit():,.2f}")
                    ]
                    
                    for metric, value in metrics:
                        pdf.cell(100, 10, txt=remove_vietnamese_accents(metric) + ":", ln=0)
                        pdf.cell(90, 10, txt=remove_vietnamese_accents(value), ln=1)
                    
                    pdf.ln(10)
                    
                    # --- Phần 2: Biểu đồ doanh thu theo tháng ---
                    if 'monthly' in chart_paths:
                        pdf.set_font("Arial", 'B', 14)
                        pdf.cell(200, 10, txt=remove_vietnamese_accents("2. DOANH THU THEO THỜI GIAN"), ln=True)
                        pdf.set_font("Arial", size=12)
                        
                        pdf.image(chart_paths['monthly'], x=10, w=190)
                        pdf.ln(5)
                        
                        if 'Order Date' in df.columns and 'Revenue' in df.columns:
                            df['Order Date'] = pd.to_datetime(df['Order Date'])
                            df['Month'] = df['Order Date'].dt.month
                            df['Year'] = df['Order Date'].dt.year
                            
                            latest_year = df['Year'].max()
                            if not df[df['Year'] == latest_year].empty:
                                monthly_sales = df[df['Year'] == latest_year].groupby('Month')['Revenue'].sum()
                                
                                if not monthly_sales.empty:
                                    max_month = monthly_sales.idxmax()
                                    min_month = monthly_sales.idxmin()
                                    
                                    pdf.cell(200, 10, txt=remove_vietnamese_accents(f"Nam {latest_year}:"), ln=True)
                                    pdf.cell(200, 10, txt=remove_vietnamese_accents(f"- Thang cao nhat: Thang {max_month} ({monthly_sales[max_month]:,.2f} )"), ln=True)
                                    pdf.cell(200, 10, txt=remove_vietnamese_accents(f"- Thang thap nhat: Thang {min_month} ({monthly_sales[min_month]:,.2f} )"), ln=True)
                                else:
                                    pdf.cell(200, 10, txt=remove_vietnamese_accents("Khong co du lieu doanh thu hang thang cho nam gan nhat."), ln=True)
                            else:
                                pdf.cell(200, 10, txt=remove_vietnamese_accents("Khong co du lieu cho nam gan nhat de phan tich doanh thu hang thang."), ln=True)
                        else:
                            pdf.cell(200, 10, txt=remove_vietnamese_accents("Thieu cot 'Order Date' hoac 'Revenue' de phan tich doanh thu theo thoi gian."), ln=True)

                    pdf.ln(10)
                    
                    # --- Phần 3: Biểu đồ doanh thu theo thành phố ---
                    if 'city' in chart_paths:
                        pdf.set_font("Arial", 'B', 14)
                        pdf.cell(200, 10, txt=remove_vietnamese_accents("3. DOANH THU THEO THANH PHO"), ln=True)
                        pdf.set_font("Arial", size=12)
                        
                        pdf.image(chart_paths['city'], x=10, w=190)
                        pdf.ln(5)
                        
                        if 'City' in df.columns and 'Revenue' in df.columns:
                            top_cities = df.groupby('City')['Revenue'].sum().nlargest(15)
                            if not top_cities.empty:
                                pdf.cell(200, 10, txt=remove_vietnamese_accents("Top 15 thanh pho:"), ln=True)
                                for i, (city, revenue) in enumerate(top_cities.items(), 1):
                                    pdf.multi_cell(0, 7, txt=remove_vietnamese_accents(f"{i}. {city}: {revenue:,.2f}"), align='L')
                            else:
                                pdf.cell(200, 10, txt=remove_vietnamese_accents("Khong co du lieu doanh thu theo thanh pho."), ln=True)
                        else:
                            pdf.cell(200, 10, txt=remove_vietnamese_accents("Thieu cot 'City' hoac 'Revenue' de phan tich doanh thu theo thanh pho."), ln=True)
                    
                    pdf.ln(10)
                    
                    # --- Phần 4: Biểu đồ doanh thu theo sản phẩm ---
                    if 'product' in chart_paths:
                        pdf.set_font("Arial", 'B', 14)
                        pdf.cell(200, 10, txt=remove_vietnamese_accents("4. DOANH THU THEO SAN PHAM"), ln=True)
                        pdf.set_font("Arial", size=12)
                        
                        pdf.image(chart_paths['product'], x=10, w=190)
                        pdf.ln(5)
                        
                        if 'Product Name' in df.columns and 'Revenue' in df.columns:
                            top_products = df.groupby('Product Name')['Revenue'].sum().nlargest(5)
                            if not top_products.empty:
                                pdf.cell(200, 10, txt=remove_vietnamese_accents("Top 5 san pham:"), ln=True)
                                for i, (product, revenue) in enumerate(top_products.items(), 1):
                                    product_name = remove_vietnamese_accents(product) # Chuyển tên sản phẩm sang không dấu
                                    product_name = product_name if len(product_name) < 50 else product_name[:47] + "..."
                                    pdf.multi_cell(0, 7, txt=f"{i}. {product_name}: {revenue:,.2f} ", align='L')
                            else:
                                pdf.cell(200, 10, txt=remove_vietnamese_accents("Khong co du lieu doanh thu theo san pham."), ln=True)
                        else:
                            pdf.cell(200, 10, txt=remove_vietnamese_accents("Thieu cot 'Product Name' hoac 'Revenue' de phan tich doanh thu theo san pham."), ln=True)
                    
                    pdf.ln(10)
                    
                    pdf_path = os.path.join(temp_img_dir, "report.pdf")
                    pdf.output(pdf_path)
                    
                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()
                    
                    st.success("✅ Bao cao da duoc tao thanh cong!")
                    st.download_button(
                        label="📥 Tai xuong bao cao PDF",
                        data=pdf_bytes,
                        file_name=f"BaoCaoDoanhSo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        help="Nhan de tai bao cao day du ve may"
                    )
                    
                except Exception as e:
                    st.error(f"Loi khi tao bao cao: {str(e)}")
                    st.warning("Vui long dam bao ban da tai len va xu ly du lieu, cung nhu da tao cac bieu do o tab 'Phan tich du lieu'.")
                    
                finally:
                    if os.path.exists(temp_img_dir):
                        for file in os.listdir(temp_img_dir):
                            os.remove(os.path.join(temp_img_dir, file))
                        os.rmdir(temp_img_dir)
        
        with col_download2:
            csv_data = st.session_state.df_processed.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📊 Tai du lieu da chuan hoa (CSV)",
                data=csv_data,
                file_name="DuLieuDaChuanHoa.csv",
                mime="text/csv",
                help="Tai xuong file CSV chua du lieu da duoc xu ly va chuan hoa"
            )
            
            st.markdown("""
            **DU LIEU DA CHUAN HOA BAO GOM:**
            - Du lieu goc da duoc lam sach
            - Da xu ly missing values
            - Da loai bo outliers
            - Duoc chuan hoa dinh dang
            - San sang cho cac phan tich tiep theo
            """)

with tab5:
    st.title("Chào mừng đến với Hệ thống Phân Tích Doanh Số Bán Lẻ")
    
    st.markdown("""
    ## Giới thiệu về dự án
    Dự án **Phân Tích Dữ Liệu Doanh Số và Dự Đoán Doanh Thu Trong Lĩnh Vực Bán Lẻ** được phát triển để hỗ trợ các doanh nghiệp bán lẻ tối ưu hóa quy trình kinh doanh thông qua phân tích dữ liệu và dự đoán thông minh. Ứng dụng này sử dụng các kỹ thuật học máy để phân tích xu hướng doanh số và dự đoán doanh thu cũng như số lượng tồn kho cần thiết.

    ### Mục tiêu
    - **Phân tích dữ liệu doanh số**: Khám phá xu hướng bán hàng theo thời gian, khu vực địa lý, danh mục sản phẩm, và các yếu tố khác.
    - **Dự đoán doanh thu**: Dự đoán tổng doanh thu dựa trên số lượng bán ra, giá mỗi đơn vị, và chi phí nhập hàng.
    - **Dự đoán tồn kho**: Ước tính số lượng tồn kho cần thiết dựa trên số lượng bán ra, thành phố, và danh mục sản phẩm.
    - **Trực quan hóa dữ liệu**: Cung cấp các biểu đồ trực quan để hỗ trợ ra quyết định.
    - **Tương tác dễ dàng**: Giao diện web thân thiện, cho phép người dùng nhập liệu và tải xuống dữ liệu.

    ### Tính năng chính
    - **Dự đoán doanh thu**: Nhập **Số lượng (Quantity)**, **Giá mỗi đơn vị (Price per Unit)**, và **Chi phí nhập hàng (Import Cost)** để dự đoán doanh thu.
    - **Dự đoán tồn kho**: Nhập **Số lượng (Quantity)**, **Thành phố (City)**, và **Danh mục sản phẩm (Product Category)** để dự đoán số lượng tồn kho.
    - **Biểu đồ phân phối**: Xem phân phối của doanh thu và tồn kho qua các biểu đồ trực quan.
    - **Tải xuống dữ liệu**: Xuất dữ liệu đã xử lý dưới dạng tệp CSV để phân tích thêm.

    ### Hướng dẫn sử dụng
    1. **Chọn trang từ thanh bên trái**:
       - **Giới thiệu**: Xem thông tin tổng quan về dự án.
       - **Dự đoán tồn kho**: Nhập dữ liệu để dự đoán số lượng tồn kho.
    2. **Nhập thông tin**:
       - Điền các trường như Số lượng, Thành phố, Danh mục sản phẩm.
       - Nhấn nút **Dự đoán số lượng tồn kho** để xem kết quả.
    3. **Xem biểu đồ**:
       - Biểu đồ phân phối tồn kho giúp bạn hiểu rõ hơn về dữ liệu.
    4. **Tải xuống dữ liệu**:
       - Nhấn nút **Tải xuống retail_sales_cleaned.csv** để lấy dữ liệu đã xử lý.

    ### Thông tin liên hệ
    - **Nhóm phát triển**: Đội ngũ phân tích dữ liệu bán lẻ
    - **Email**: doanhieu.11052004@gmail.com
    - **Ngày hoàn thành**: 23/05/2025

    Chúng tôi hy vọng ứng dụng này sẽ giúp bạn tối ưu hóa hoạt động kinh doanh bán lẻ! Nếu có bất kỳ câu hỏi hoặc góp ý nào, vui lòng liên hệ với chúng tôi.
    """)
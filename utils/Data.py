import pandas as pd
import streamlit as st

class Data:
    def __init__(self, file_csv):
        self.file_csv = file_csv
        self.df = None
        self.original_shape = (0, 0)

    def load_data(self):
        data = None
        encodings_to_try = ['utf-8', 'windows-1252', 'latin-1', 'ISO-8859-1'] # Đã thêm ISO-8859-1

        for encoding in encodings_to_try:
            try:
                data = pd.read_csv(self.file_csv, encoding=encoding)
                print(f"Đọc thành công với encoding: {encoding}")
                self.df = data
                self.original_shape = self.df.shape
                return self.df
            except UnicodeDecodeError:
                print(f"Thử encoding {encoding} lỗi, chuyển sang encoding tiếp theo...")
                continue
            except FileNotFoundError:
                print(f"Lỗi: Không tìm thấy tệp tại đường dẫn: {self.file_csv}")
                st.error(f"Lỗi: Không tìm thấy tệp dữ liệu tại đường dẫn: '{self.file_csv}'. Vui lòng kiểm tra lại.")
                return None
            except Exception as e:
                print(f"Đã xảy ra lỗi không mong muốn khi đọc tệp với encoding {encoding}: {e}")
                st.error(f"Đã xảy ra lỗi khi tải dữ liệu với encoding {encoding}: {e}")
                return None

        print(f"Không thể đọc tệp '{self.file_csv}' với các encoding thông thường đã thử.")
        st.error(f"Không thể tải dữ liệu từ tệp '{self.file_csv}'. Vui lòng kiểm tra tệp hoặc encoding.")
        return None

    def process_missing_values(self, df_to_process):
        shape_before = df_to_process.shape
        processed_df = df_to_process.dropna()
        shape_after = processed_df.shape
        rows_removed = shape_before[0] - shape_after[0]
        # st.info(f"Xử lý giá trị thiếu: Đã xóa **{rows_removed}** hàng.")
        return processed_df

    def process_outliers(self, df_to_process):
        shape_before = df_to_process.shape
        numeric_cols = ['Sales', 'Quantity', 'Discount'] # Giả sử các cột này tồn tại
        processed_df = df_to_process.copy()

        initial_rows = processed_df.shape[0]
        rows_removed_total = 0

        for col in numeric_cols:
            if col in processed_df.columns:
                Q1 = processed_df[col].quantile(0.25)
                Q3 = processed_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Lọc tại chỗ cho lần lặp hiện tại
                processed_df = processed_df[(processed_df[col] >= lower_bound) & (processed_df[col] <= upper_bound)]
        
        rows_removed_total = initial_rows - processed_df.shape[0]
        # st.info(f"Xử lý giá trị ngoại lai: Đã xóa **{rows_removed_total}** hàng.")
        return processed_df
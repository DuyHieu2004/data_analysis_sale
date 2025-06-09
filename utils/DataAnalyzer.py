import pandas as pd

class DataAnalyzer:
    def __init__(self, dataframe):
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("Đầu vào phải là một pandas DataFrame.")
        self.df = dataframe

    def get_total_sales(self):
        if 'Sales' in self.df.columns:
            return self.df['Sales'].sum()
        return 0

    def get_total_quantity_sold(self):
        if 'Quantity' in self.df.columns:
            return self.df['Quantity'].sum()
        return 0

    def get_number_of_transactions(self):
        if 'Order ID' in self.df.columns:
            return self.df['Order ID'].count()
        return len(self.df)

    def get_total_profit(self):
        if 'Profit' in self.df.columns:
            return self.df['Profit'].sum()
        return 0
    
    

    def get_total_revenue(self):
        required_cols = ['Profit', 'Discount', 'Quantity', 'Sales']
        
        # Kiểm tra xem tất cả các cột cần thiết có tồn tại không
        if not all(col in self.df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            raise ValueError(f"Không tìm thấy (các) cột cần thiết: {', '.join(missing_cols)} để tính Revenue.")
        
        # Tạo một cột tạm thời cho công thức bạn muốn
        # Đảm bảo các cột là kiểu số để thực hiện phép toán
        temp_df = self.df.copy()
        for col in required_cols:
            temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce').fillna(0) # Chuyển đổi sang số, xử lý giá trị lỗi

        # Tính toán giá trị Revenue cho mỗi dòng
        # Giả định 'Sales' là giá bán của một sản phẩm, và 'Quantity' là số lượng sản phẩm đó.
        # Nếu 'Sales' đã là tổng doanh số của dòng, thì Quantity * Sales sẽ là tính toán dư thừa.
        # Tuy nhiên, tuân thủ công thức bạn đã cho:
        temp_df['Calculated_Revenue_Per_Row'] = temp_df['Profit'] + temp_df['Discount']*temp_df['Sales'] + (temp_df['Quantity'] * temp_df['Sales'])
        
        return temp_df['Calculated_Revenue_Per_Row'].sum()
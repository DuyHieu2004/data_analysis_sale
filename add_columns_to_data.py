import pandas as pd # type: ignore
import numpy as np # type: ignore
from datetime import datetime, timedelta
import random

# Đọc dữ liệu gốc.
df = pd.read_csv('data/org_data/retail_sales_dataset.csv')

# Thiết lập seed để đảm bảo kết quả tái lập.
np.random.seed(42)
random.seed(42)

# 1. Thêm cột City.
cities = ['Hà Nội', 'TP.HCM', 'Đà Nẵng', 'Cần Thơ', 'Hải Phòng']
city_weights = [0.35, 0.4, 0.15, 0.05, 0.05] # Xác xuất dựa trên dân số.
df['City'] = np.random.choice(cities, size=len(df), p=city_weights)

# 2. Thêm cột Time.
def generate_random_time():
    # Giả lập giờ giao dịch từ 8h đến 22h.
    hour = np.random.choice(
        [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
        p=[0.05, 0.07, 0.10, 0.10, 0.08, 0.06, 0.05, 0.05, 0.06, 0.10, 0.10, 0.06, 0.04, 0.05, 0.03]
    )
    minute = random.randint(0, 59)
    return f'{hour:02d}:{minute:02d}'

df['Time'] = [generate_random_time() for _ in range(len(df))]

# 3. Thêm cột Inventory.
def generate_inventory(category):
    if category == 'Beauty':
        return random.randint(50, 200)
    elif category == 'Clothing':
        return random.randint(100, 300)
    else: # Electronics
        return random.randint(20, 100)
    
df['Inventory'] = df['Product Category'].apply(generate_inventory)

# 4. Thêm cột Import Cost.
df['Import Cost'] = df['Price per Unit'] * np.random.uniform(0.6, 0.8, size=len(df))
df['Import Cost'] = df['Import Cost'].round(2)

# 5. Thêm cột Promotion.
df['Promotion'] = np.random.choice(['Yes', 'No'], size=len(df), p=[0.3, 0.7])

# 6. Danh sách ngày lễ lớn ở Việt Nam (năm 2023).
holidays = [
    '2023-01-21', '2023-01-22', '2023-01-23', '2023-01-24',     # Tết Nguyên Đán
    '2023-04-30',       # Ngày Giải phóng miền Nam
    '2023-05-01',       # Ngày Quốc tế Lao động
    '2023-09-02',       # Ngày Quốc khánh
    '2023-12-24', '2023-12-25'  # Giáng sinh
]
df['Holiday'] = df['Date'].apply(lambda x: 'Yes' if x in holidays else 'No')

# Sắp xếp lại các cột
df = df[['Transaction ID', 'Date', 'Time', 'City', 'Customer ID', 'Gender', 'Age', 
         'Product Category', 'Quantity', 'Price per Unit', 'Total Amount', 
         'Inventory', 'Import Cost', 'Promotion', 'Holiday']]

# Lưu dữ liệu vào tệp CSV mới
df.to_csv('retail_sales_dataset_extended.csv', index=False)
print("Dữ liệu đã được bổ sung và lưu vào 'retail_sales_dataset_extended.csv'")

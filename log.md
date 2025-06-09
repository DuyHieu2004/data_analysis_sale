# LOG THỰC HIỆN.

## Chuẩn bị dữ liệu.
Dữ liệu được lấy từ nguồn Kaggle, tên retail_sales_dataset.csv.

Thực hiện giả định thêm các thuộc tính cho phù hợp với dự án.
- **City**: Phân bổ ngẫu nhiên 5 thành phố (Hà Nội, TP.HCM, Đà Nẵng, Cần Thơ, Hải Phòng) với xác suất dựa trên dân số (ví dụ: TP.HCM và Hà Nội có xác suất cao hơn).
- **Time**: Phân bổ ngẫu nhiên giờ giao dịch trong khoảng 8:00–22:00, với phân phối tập trung vào các khung giờ cao điểm (10:00–12:00, 17:00–20:00).
- **Inventory**: Giả định số lượng tồn kho ban đầu cho mỗi danh mục sản phẩm (Beauty: 50–200, Clothing: 100–300, Electronics: 20–100) và giảm dần dựa trên số lượng bán ra.
- **Import Cost**: Giả định chi phí nhập hàng là 60–80% giá bán (Price per Unit) để đảm bảo lợi nhuận hợp lý.
- **Promotion**: Phân bổ ngẫu nhiên Yes/No, với 30% giao dịch có khuyến mãi.
- **Holiday**: Dựa trên cột Date, kiểm tra xem ngày giao dịch có rơi vào các ngày lễ lớn của Việt Nam (Tết Nguyên Đán, Quốc khánh 2/9, Giáng sinh, v.v.) hay không.

## Kiểm tra và làm sạch dữ liệu.
1. Kiểm tra dữ liệu.
    
    * Kiểm tra giá trị thiếu.
    * Kiểm tra giá trị bất thường.
        
        * Cột `Total Amount` có khớp với `Quantity * Price per Unit` không?
        
2. chuẩn hóa dữ liệu.
    
    * Chuyển cột `Date` sang đinh dạng `datetime` (nếu chưa).
    * Chuyển cột `Time` thành định dạng thời gian (hoặc trích xuất giờ để phân tích).
    
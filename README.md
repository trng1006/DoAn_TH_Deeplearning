# 🥗 Phân Loại 36 Loại Trái Cây & Rau Củ bằng Custom CNN (PyTorch)

Dự án này ứng dụng Deep Learning (Computer Vision) để nhận diện tự động 36 loại trái cây và rau củ khác nhau. Mô hình được xây dựng hoàn toàn từ đầu (from scratch) bằng framework **PyTorch**, đạt độ chính xác ấn tượng **93%** trên tập Validation nhờ chiến lược kết hợp dữ liệu (Domain Adaptation) thông minh.

## 📊 1. Về Tập Dữ Liệu (Datasets)
Dự án không chỉ sử dụng một mà kết hợp hai bộ dataset lớn từ Kaggle để giải quyết bài toán "Học vẹt phông nền" (Background Bias), giúp mô hình có thể nhận diện tốt trong môi trường thực tế:

1. **In-the-wild Dataset (`kritikseth/fruit-and-vegetable-image-recognition`):** Cung cấp sự đa dạng về môi trường, ánh sáng, góc chụp thực tế.
2. **Lab-quality Dataset (`moltean/fruits` - Fruits 360):** Được lọc khắt khe và "bơm" thêm vào 36 lớp tương ứng, cung cấp số lượng ảnh khổng lồ (hơn 65,000 ảnh) với độ phân giải cao trên phông nền trắng, giúp mô hình trích xuất đường nét và vân vỏ cực chuẩn.

**Chiến thuật phân chia (Data Splitting):**
Toàn bộ dữ liệu được gộp lại (ConcatDataset) và chia ngẫu nhiên theo tỷ lệ chuẩn:
* **80% Training:** Dành cho huấn luyện (Có áp dụng Data Augmentation).
* **20% Validation:** Dành cho kiểm thử (Ảnh nguyên bản, không méo/nhiễu).

## 🛠 2. Tiền xử lý & Data Augmentation
Để chống Overfitting và giúp mô hình "lì đòn" hơn, tập Train được đi qua một pipeline Augmentation mạnh mẽ:
* **Resize:** Cố định kích thước ảnh về `128x128` pixel.
* **Xoay lật:** `RandomHorizontalFlip` (50%), `RandomVerticalFlip` (20%), và xoay nghiêng ngẫu nhiên tối đa `45 độ`.
* **Nhiễu màu (Color Jitter):** Điều chỉnh độ sáng, độ tương phản và độ bão hòa (30%) cùng dải màu (10%).
* **Chuẩn hóa (Normalize):** Đưa về mean/std chuẩn của ImageNet `[0.485, 0.456, 0.406]`.

## 🧠 3. Kiến trúc Mô hình (FruitCNN)
Mô hình `FruitCNN` được thiết kế tối ưu với số lượng tham số vừa đủ, không quá cồng kềnh nhưng vẫn trích xuất đặc trưng sâu rất tốt.
* **Feature Extractor:** Gồm 3 khối Tích chập (Convolutional Blocks). Mỗi khối chứa:
  * 2 lớp `Conv2d` (kernel size 3x3) kết hợp `BatchNorm2d` và `ReLU`.
  * `MaxPool2d` để giảm kích thước ma trận.
  * `Dropout2d` (tăng dần 0.1 -> 0.2 -> 0.3) để ngắt kết nối ngẫu nhiên, chống học vẹt.
* **Global Average Pooling:** Dùng `AdaptiveAvgPool2d(1)` thay vì Flatten toàn bộ để giảm số lượng tham số khổng lồ, chống Overfitting hiệu quả.
* **Classifier:** Lớp Fully Connected với `Linear(128, 256)` -> `Dropout(0.5)` -> `Linear(256, 36)`.

## ⚙️ 4. Thông số Huấn luyện (Hyperparameters)
* **Epochs:** 20
* **Batch Size:** 64
* **Learning Rate:** 0.001
* **Optimizer:** AdamW (kèm weight decay 1e-4) - Hoạt động tốt hơn Adam thường với dữ liệu lớn.
* **Loss Function:** CrossEntropyLoss
* **Scheduler:** `ReduceLROnPlateau` (Giảm một nửa tốc độ học nếu Acc không tăng sau 2 epoch).

## 🏆 5. Kết quả (Results)
* **Độ chính xác (Accuracy):** Đạt **93%** trên tập Validation.
* Mô hình hội tụ tốt, biểu đồ Loss của Train và Validation bám sát nhau, chứng tỏ hiện tượng Overfitting đã được kiểm soát thành công bằng Dropout và Augmentation.
* Trọng số mô hình tốt nhất được lưu tại: `best_model_v2.pth`

## 🚀 6. Hướng dẫn chạy code (How to Run)

Dự án được setup để chạy trực tiếp trên **Google Colab** với môi trường GPU (T4).

**Bước 1: Chuẩn bị Kaggle API**
Lấy file `kaggle.json` từ tài khoản Kaggle của bạn (Profile -> Settings -> Create New API Token).

**Bước 2: Chạy tuần tự trên Colab**
1. Mở file notebook. Đảm bảo đã bật GPU: `Runtime` -> `Change runtime type` -> Chọn `T4 GPU`.
2. Chạy **Cell 1**: Tải dữ liệu. Hệ thống sẽ yêu cầu bạn upload file `kaggle.json` lên.
3. Chạy **Cell 2**: Lọc ảnh và gộp chung 2 bộ dataset.
4. Chạy **Cell 3**: Thiết lập Config, Data Augmentation và chia DataLoader (80-20).
5. Chạy **Cell 4**: Xem trước (Preview) một batch ảnh đã được Augment.
6. Chạy **Cell 5**: Khởi tạo kiến trúc `FruitCNN`.
7. Chạy **Cell 6**: Bắt đầu quá trình Training. File `best_model_v2.pth` sẽ tự động tải về máy bạn khi hoàn tất.
8. Chạy **Cell 7**: Vẽ biểu đồ đánh giá (Accuracy & Loss).

## 🚀 7. Hướng Dẫn Cài Đặt & Khởi Chạy Dự Án

### Cấu Trúc Thư Mục Cần Thiết
```text
Fruit_AI_Project/
│
├── main.py                # Mã nguồn Backend API (FastAPI)
├── index.html             # Mã nguồn Frontend (Giao diện Web)
├── best_model_v2.pth      # File trọng số mô hình đã train
└── README.md              # File hướng dẫn này

Bước 1: Chuẩn bị môi trường & Thư viện
Đảm bảo máy tính đã cài đặt Python. Mở Terminal (Command Prompt/PowerShell) tại thư mục Fruit_AI_Project và chạy lệnh:

Bash
pip install fastapi uvicorn torch torchvision pillow python-multipart
Bước 2: Bật Backend (Máy chủ AI)
Tại Terminal vừa cài đặt thư viện, gõ lệnh sau để khởi động hệ thống Backend:

Bash
python -m uvicorn main:app --reload
Hệ thống báo ✅ Đã load model thành công! và chạy tại http://127.0.0.1:8000 là hoàn tất. (Giữ nguyên cửa sổ Terminal này).

Bước 3: Bật Frontend (Giao diện Web)
Mở một cửa sổ Terminal mới (song song với cửa sổ Backend), vẫn trỏ về thư mục dự án và gõ lệnh tạo Local Server:

Bash
python -m http.server 5500
Bước 4: Truy cập & Trải nghiệm
Mở trình duyệt web (Chrome, Edge, Safari,...) và truy cập vào đường link sau:
👉 http://localhost:5500

Giao diện sẽ hiện lên, bạn chỉ cần kéo thả hình ảnh thực phẩm vào, AI sẽ phân tích và trả về Top 3 kết quả tốt nhất đi kèm biểu đồ độ tin cậy.


## 👨‍💻 Tác giả
* **Lê Hồng Quốc** 
* Sinh viên - Trường Đại học Công Thương TP.HCM (HUIT)

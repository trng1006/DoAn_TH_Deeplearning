# 🥗 Phân Loại 36 Loại Trái Cây & Rau Củ bằng Custom CNN (PyTorch)

Dự án này ứng dụng Deep Learning (Computer Vision) để nhận diện tự động 36 loại trái cây và rau củ khác nhau. Mô hình được xây dựng hoàn toàn từ đầu (from scratch) bằng framework **PyTorch**, đạt độ chính xác **77.26%** trên tập Validation nhờ chiến lược kết hợp dữ liệu thông minh (Smart Data Fusion) và kiến trúc `FruitCNN_V3` tùy chỉnh.

## 📊 1. Về Tập Dữ Liệu (Datasets)
Dự án không chỉ sử dụng một mà kết hợp hai bộ dataset lớn từ Kaggle để giải quyết bài toán "Học vẹt phông nền" (Background Bias), giúp mô hình nhận diện tốt trong môi trường thực tế:

1. **In-the-wild Dataset (`kritikseth/fruit-and-vegetable-image-recognition`):** Cung cấp sự đa dạng về môi trường, ánh sáng, góc chụp thực tế.
2. **Lab-quality Dataset (`moltean/fruits` - Fruits 360):** Bổ sung ảnh có phông nền trắng, độ phân giải cao giúp mô hình trích xuất đường nét cực chuẩn. 
   * **Chiến thuật Smart Fusion:** Hệ thống được thiết lập các "Luật cấm" và "Luật khớp tên" để chống nhiễu (ngăn *pineapple* lọt vào *apple*). Đồng thời, giới hạn bơm tối đa **300 ảnh/lớp** để ngăn chặn hiện tượng mất cân bằng dữ liệu (Data Imbalance).

**Chiến thuật phân chia (Data Splitting):**
Toàn bộ dữ liệu được gộp lại (ConcatDataset) và chia ngẫu nhiên bằng PyTorch theo tỷ lệ chuẩn:
* **80% Training:** Dành cho huấn luyện (Có áp dụng Data Augmentation).
* **20% Validation:** Dành cho kiểm thử (Ảnh nguyên bản, không biến dạng).

## 🛠 2. Tiền xử lý & Data Augmentation
Để chống Overfitting và giúp mô hình "lì đòn" hơn, tập Train được đi qua một pipeline Augmentation mạnh mẽ:
* **Resize & Crop:** Resize về `144x144` và CenterCrop chuẩn `128x128` pixel.
* **Xoay lật:** `RandomHorizontalFlip` (50%), `RandomVerticalFlip` (20%), và `RandomRotation` (30 độ).
* **Nhiễu màu & Che khuất:** `ColorJitter` (20% sáng/tương phản/bão hòa, 10% dải màu) và `RandomErasing` (30% xác suất che khuất ngẫu nhiên phần tử).
* **Chuẩn hóa (Normalize):** Đưa về mean/std chuẩn của ImageNet `[0.485, 0.456, 0.406]`.

## 🧠 3. Kiến trúc Mô hình (FruitCNN_V3)
Mô hình `FruitCNN_V3` được thiết kế nâng cấp với 4 khối trích xuất đặc trưng sâu, xử lý tốt lượng dữ liệu phức tạp.
* **Feature Extractor:** Gồm 4 khối Tích chập (Convolutional Blocks) với số kênh tăng dần (32 -> 64 -> 128 -> 256). Mỗi khối chứa:
  * 2 lớp `Conv2d` (kernel 3x3) kết hợp `BatchNorm2d` và `LeakyReLU` (chống chết nơ-ron).
  * `MaxPool2d` để giảm kích thước ma trận.
  * `Dropout2d` tăng dần (0.1 -> 0.1 -> 0.2 -> 0.3) để ngắt kết nối ngẫu nhiên.
* **Global Average Pooling:** Dùng `AdaptiveAvgPool2d(1)` thay vì Flatten toàn bộ, tối ưu hóa triệt để số lượng tham số.
* **Classifier:** Lớp Fully Connected mạnh mẽ với `Linear(256, 512)` -> `Dropout(0.4)` -> `Linear(512, 36)`.

## ⚙️ 4. Thông số Huấn luyện (Hyperparameters)
* **Epochs:** 50
* **Batch Size:** 64
* **Learning Rate:** 0.001
* **Optimizer:** AdamW (kèm weight decay 5e-4) - Hoạt động tốt hơn Adam thường với dữ liệu lớn.
* **Loss Function:** CrossEntropyLoss (với `label_smoothing=0.1` giúp tăng độ tổng quát hóa).
* **Scheduler:** `ReduceLROnPlateau` (Giảm một nửa tốc độ học nếu Acc không tăng sau 4 epoch).

## 🏆 5. Kết quả (Results)
* **Độ chính xác (Accuracy):** Đạt **77.26%** trên tập Validation tại Epoch 50.
* Biểu đồ Loss và Accuracy tiến triển cực kỳ ổn định, hội tụ mượt mà, chứng tỏ chiến lược Augmentation và Dropout đã triệt tiêu hoàn toàn Overfitting.
* Trọng số mô hình tốt nhất được lưu tại: `best_model_v5.pth`

## 🚀 6. Hướng dẫn chạy code (How to Run)

Dự án được setup để chạy trực tiếp trên **Google Colab** với môi trường GPU (T4).

**Bước 1: Chuẩn bị Kaggle API**
Lấy file `kaggle.json` từ tài khoản Kaggle của bạn (Profile -> Settings -> Create New API Token).

**Bước 2: Chạy tuần tự trên Colab**
1. Mở file notebook. Đảm bảo đã bật GPU: `Runtime` -> `Change runtime type` -> Chọn `T4 GPU`.
2. Chạy **Cell 1 & 2**: Tải dữ liệu và giải nén (Hệ thống sẽ yêu cầu upload file `kaggle.json`).
3. Chạy **Cell 3**: Kích hoạt bộ lọc Smart Data Fusion (Gộp tối đa 300 ảnh/lớp).
4. Chạy **Cell 4**: Thiết lập Config, Data Augmentation và chia DataLoader (80-20).
5. Chạy **Cell 5**: Xem trước (Preview) một batch ảnh đã được Augment.
6. Chạy **Cell 6**: Khởi tạo kiến trúc `FruitCNN_V3`.
7. Chạy **Cell 7**: Bắt đầu quá trình Training. File `best_model_v5.pth` sẽ tự động tải về máy bạn khi hoàn tất.
8. Chạy **Cell 8**: Vẽ biểu đồ đánh giá (Accuracy & Loss).

## 🚀 7. Hướng Dẫn Cài Đặt & Khởi Chạy Dự Án (Bản Web)

### Cấu Trúc Thư Mục Cần Thiết
```text
Fruit_AI_Project/
│
├── main.py                # Mã nguồn Backend API (FastAPI)
├── index.html             # Mã nguồn Frontend (Giao diện Web)
├── best_model_v5.pth      # File trọng số mô hình đã train
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

👨‍💻 Tác giả
Lê Hồng Quốc * Sinh viên - Trường Đại học Công Thương TP.HCM (HUIT)
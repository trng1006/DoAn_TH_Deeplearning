from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io

app = FastAPI()

# Cho phép Frontend (chạy trên trình duyệt) gọi API đến Backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. ĐỊNH NGHĨA LẠI KIẾN TRÚC MODEL V3 (CHÍNH XÁC NHƯ TRÊN COLAB)
class FruitCNN_V3(nn.Module):
    def __init__(self, num_classes):
        super(FruitCNN_V3, self).__init__()
        
        # Khối 1: 3 -> 32 channels
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.1),
        )
        # Khối 2: 32 -> 64 channels
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),
        )
        # Khối 3: 64 -> 128 channels
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.3),
        )
        # Khối 4: 128 -> 256 channels
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.4),
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512), nn.LeakyReLU(0.1, inplace=True), nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


# 2. CẤU HÌNH VÀ LOAD MODEL
CLASS_NAMES = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']

# Khởi tạo model với class V3
model = FruitCNN_V3(num_classes=36)

try:
    # Load file best_model_v5.pth
    model.load_state_dict(torch.load('best_model_v5.pth', map_location=torch.device('cpu')))
    model.eval()
    print("✅ Đã load model V5 thành công!")
except Exception as e:
    print(f"⚠️ Chưa load được model. Lỗi: {e}")

# Cấu hình tiền xử lý ảnh chuẩn xác (chống méo ảnh)
inference_transform = transforms.Compose([
    transforms.Resize(144),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. TẠO API DỰ ĐOÁN
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Đọc ảnh người dùng gửi lên
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # ĐÃ SỬA LỖI TÊN BIẾN: transform -> inference_transform
        img_tensor = inference_transform(image).unsqueeze(0)
        
        # Chạy model
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            
            # Lấy Top 3 dự đoán cao nhất
            top3_prob, top3_indices = torch.topk(probabilities, 3)
            
        # Format kết quả trả về
        results = []
        for i in range(3):
            results.append({
                "class_name": CLASS_NAMES[top3_indices[i].item()].capitalize(),
                "confidence": round(top3_prob[i].item() * 100, 2)
            })
            
        return {"success": True, "predictions": results}
        
    except Exception as e:
        return {"success": False, "error": str(e)}
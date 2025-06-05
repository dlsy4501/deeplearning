# predict.py
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models  # 1. EfficientNet 임포트

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. EfficientNet-B6에 맞는 전처리
transform = transforms.Compose([
    transforms.Resize(256),  # 학습시 사용한 입력 크기로 변경
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 3. EfficientNet-B6 모델 초기화
model = models.efficientnet_b6(weights=None)  # 사전학습 가중치 미사용
model.classifier[1] = torch.nn.Linear(
    model.classifier[1].in_features, 
    100  # CIFAR-100 클래스 수
)
model.load_state_dict(torch.load("weight_나반1조_0602_1852.pth", map_location=device))
model = model.to(device).eval()  # 4. 디바이스 이동과 평가 모드

# 결과 저장용 리스트
output_lines = ["number,label"]

with torch.no_grad():
    for i in range(1, 3001):
        fname = f"{i}.jpg"  # 1.jpg, 2.jpg, ...
        path = os.path.join("./Dataset/CImages", fname)
        image = Image.open(path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        outputs = model(image)
        _, pred = torch.max(outputs, 1)
        label = pred.item()
        # 출력: 4자리 number, 2자리 label
        output_lines.append(f"{i:04d},{label:02d}")

# 결과 파일 저장
with open("result_나반1조_0602_1852.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))

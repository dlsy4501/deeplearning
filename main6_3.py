import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
from datetime import datetime

# 1. 팀 정보 설정
TEAM_NAME = "나반1조"
current_time = datetime.now().strftime("%m%d_%H%M")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 데이터 전처리 (훈련 시와 동일한 설정)
test_transform = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(320),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. 개선된 모델 구조 (훈련 시와 동일)
class AdvancedEfficientNet(nn.Module):
    def __init__(self, num_classes=100, dropout_rate=0.3):
        super().__init__()
        self.base = models.efficientnet_b6(weights='IMAGENET1K_V1')
        in_features = self.base.classifier[1].in_features
        
        # 기존 classifier 제거
        self.base.classifier = nn.Identity()
        
        # 개선된 분류기 구조
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.base.features(x)
        x = self.classifier(features)
        return x

# 4. 모델 로드 (사전 학습된 가중치 사용)
model = AdvancedEfficientNet().to(device)

# 체크포인트 로드 및 키 확인
checkpoint = torch.load("EfficientNetB6_best.pth", map_location=device)

# 체크포인트 구조 확인
print("체크포인트 키들:", list(checkpoint.keys()))

# 모델 state_dict 로드 (키 구조에 따라 선택)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
elif 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'])
else:
    # 체크포인트가 직접 state_dict인 경우
    model.load_state_dict(checkpoint)

model.eval()

# 5. CIFAR-100 테스트셋 정확도 계산
def get_cifar100_accuracy():
    from torchvision.datasets import CIFAR100
    test_set = CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=False)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# 6. 외부 테스트셋 추론 및 결과 저장
def generate_results():
    image_dir = './Dataset/CImages/'
    results = []
    
    # 1.jpg ~ 3000.jpg 순차 처리
    for i in range(1, 3001):
        img_path = os.path.join(image_dir, f"{i}.jpg")
        
        try:
            image = Image.open(img_path).convert('RGB')
            image = test_transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(image)
                _, pred = torch.max(output, 1)
            
            results.append(f"{i:04d}, {pred.item()}")
            
            # 100개마다 진행상황 출력
            if i % 100 == 0:
                print(f"처리 완료: {i}/3000")
                
        except Exception as e:
            print(f"이미지 {i}.jpg 처리 중 오류: {e}")
            results.append(f"{i:04d}, 0")  # 오류 시 기본값
    
    # 결과 파일 저장
    result_file = f'result_{TEAM_NAME}_{current_time}.txt'
    with open(result_file, 'w') as f:
        f.write('\n'.join(results))
    print(f"결과 파일 저장 완료: {result_file}")

if __name__ == "__main__":
    # 가중치 파일명 변경 저장
    weight_file = f'weight_{TEAM_NAME}_{current_time}.pth'
    torch.save(model.state_dict(), weight_file)
    print(f"가중치 파일 저장 완료: {weight_file}")
    
    # 정확도 계산
    train_acc = "N/A (학습 데이터 사용 안 함)"  # 실제 코드에서 계산 필요시 추가
    test_acc = get_cifar100_accuracy()
    print(f"CIFAR-100 Test Accuracy: {test_acc:.2f}%")
    
    # 추론 실행
    generate_results()

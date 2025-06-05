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

# 2. 데이터 전처리 (CIFAR-100 표준)
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 3. 모델 로드 (사전 학습된 가중치 사용)
model = models.efficientnet_b6(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 100)
model.load_state_dict(torch.load("CIFAR100_EfficientNetB7.pth", map_location=device))
model = model.to(device)
model.eval()

# 4. CIFAR-100 테스트셋 정확도 계산
def get_cifar100_accuracy():
    from torchvision.datasets import CIFAR100
    test_set = CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=8, shuffle=False)
   
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

# 5. 외부 테스트셋 추론 및 결과 저장
def generate_results():
    image_dir = './Dataset/CImages/'
    results = []
   
    # 0001.jpg ~ 3000.jpg 순차 처리
    for i in range(1, 3001):
        img_path = os.path.join(image_dir, f"{i:04d}.jpg")
        image = Image.open(img_path).convert('RGB')
        image = test_transform(image).unsqueeze(0).to(device)
       
        with torch.no_grad():
            output = model(image)
            _, pred = torch.max(output, 1)
       
        results.append(f"{i:04d}, {pred.item()}")
   
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

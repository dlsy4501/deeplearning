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

# 2. 데이터 전처리 (학습시와 동일)
test_transform = transforms.Compose([
    transforms.Resize(528),
    transforms.CenterCrop(528),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 3. 모델 로드 함수 (체크포인트 호환성 처리)
def load_model(weight_path):
    # 체크포인트 로드
    checkpoint = torch.load(weight_path, map_location=device)
    
    # 모델 초기화
    model = models.efficientnet_b6(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 100)
    
    # 체크포인트 키 확인
    if 'model_state_dict' in checkpoint:  # 전체 체크포인트인 경우
        model.load_state_dict(checkpoint['model_state_dict'])
    else:  # 모델 가중치만 저장된 경우
        model.load_state_dict(checkpoint)
    
    return model.to(device).eval()

# 4. 외부 데이터 추론 함수
def generate_results():
    model = load_model("CIFAR100_EfficientNetB7.pth")  # 파일명은 실제 학습 코드와 일치
    
    image_dir = './Dataset/CImages/'
    results = []
    
    for i in range(1, 3001):
        try:
            img_path = os.path.join(image_dir, f"{i:04d}.jpg")
            image = Image.open(img_path).convert('RGB')
            image = test_transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(image)
                _, pred = torch.max(output, 1)
                
            results.append(f"{i:04d}, {pred.item()}")
            
        except Exception as e:
            print(f"Error processing {i:04d}.jpg: {str(e)}")
            results.append(f"{i:04d}, -1")
    
    return results

# 5. 결과 저장 및 실행
if __name__ == "__main__":
    # 결과 생성
    results = generate_results()
    
    # 파일 저장
    result_file = f'result_{TEAM_NAME}_{current_time}.txt'
    with open(result_file, 'w') as f:
        f.write('\n'.join(results))
    print(f"결과 파일 저장 완료: {result_file}")

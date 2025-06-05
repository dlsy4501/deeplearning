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

# 2. 데이터 전처리 (학습시와 동일한 전처리)
test_transform = transforms.Compose([
    transforms.Resize(528),  # EfficientNet-B6 표준 입력 크기
    transforms.CenterCrop(528),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 3. 모델 로드 (아키텍처-가중치 일치화)
def load_model(weight_path):
    model = models.efficientnet_b6(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 100)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    return model.to(device).eval()

model = load_model("CIFAR100_EfficientNetB6.pth")  # 파일명 정정

# 4. 외부 데이터 추론 함수 (강화된 에러 처리)
def predict_external(image_dir='./Dataset/CImages/'):
    results = ["number,label"]
    
    for i in range(1, 3001):
        try:
            # 파일명 포맷: 1.jpg ~ 3000.jpg
            img_path = os.path.join(image_dir, f"{i}.jpg")
            
            # 이미지 로드 및 검증
            image = Image.open(img_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            # 전처리 및 추론
            tensor = test_transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(tensor)
                pred = torch.argmax(output).item()
                
            results.append(f"{i:04d},{pred:02d}")  # 4자리 숫자 + 2자리 라벨
            
        except Exception as e:
            print(f"[에러] {i:04d}.jpg 처리 실패: {str(e)}")
            results.append(f"{i:04d},-1")  # 에러 표시
            
    return results

# 5. 결과 파일 저장 (UTF-8 인코딩)
def save_results(results, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(results))
    print(f"결과 파일 저장 완료: {os.path.abspath(filename)}")

if __name__ == "__main__":
    # 추론 실행
    predictions = predict_external()
    
    # 파일 저장
    result_file = f'result_{TEAM_NAME}_{current_time}.txt'
    save_results(predictions, result_file)
    
    # 추가 정보 출력
    print(f"처리 완료: {len(predictions)-1}개 중 {sum([1 for r in predictions if r.endswith(',-1')])}개 실패")

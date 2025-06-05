import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR

# 1. 환경 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = GradScaler()

# 2. 데이터 전처리 최적화 (입력 크기 320x320)
transform = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(320),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. 데이터셋 로드 (전체 데이터셋 사용)
train_set = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
test_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

# 4. 배치 크기 증가 (GPU 메모리 허용시 64로 변경)
train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

# 5. 모델 준비 (간소화된 구조)
class CustomEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.efficientnet_b6(weights='IMAGENET1K_V1')
        in_features = self.base.classifier[1].in_features  # shape 에러 핵심 수정 부분
        
        # 분류기 단순화
        self.base.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 100)
        )
        
    def forward(self, x):
        return self.base(x)

model = CustomEfficientNet().to(device)

# 6. 최적화 설정
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.005)
scheduler = OneCycleLR(optimizer, 
                      max_lr=1e-3,
                      total_steps=len(train_loader)*10,
                      pct_start=0.3)

# 7. 학습 루프 최적화
num_epochs = 10
best_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    train_iter = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
    for inputs, labels in train_iter:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # 혼합 정밀도 연산
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # 메트릭 계산
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        train_iter.set_postfix(
            loss=loss.item(),
            acc=correct/total,
            lr=scheduler.get_last_lr()[0]
        )

    # 에폭별 통계
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    print(f'Epoch {epoch+1} - Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}')

    # 베스트 모델 저장
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        torch.save(model.state_dict(), "EfficientNetB2_optimized.pth")

# 8. 테스트 평가
model.eval()
total_correct = 0
total_samples = 0
with torch.no_grad():
    test_iter = tqdm(test_loader, desc='Testing')
    for data, labels in test_iter:
        data, labels = data.to(device), labels.to(device)
        
        with autocast():
            outputs = model(data)
            
        _, predicted = outputs.max(1)
        total_samples += labels.size(0)
        total_correct += predicted.eq(labels).sum().item()
        test_iter.set_postfix(acc=total_correct/total_samples)

print(f'최종 테스트 정확도: {total_correct/total_samples:.4f}')

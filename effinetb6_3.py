import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
import random
import numpy as np

# 시드 설정 (재현성)
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# 1. 환경 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = GradScaler()

# 2. 강화된 데이터 전처리 (훈련/테스트 분리)
train_transform = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(320),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1),  # 추가
    transforms.RandomRotation(degrees=10),  # 추가
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),  # 추가
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(320),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. 데이터셋 로드 (훈련/테스트 transform 분리)
train_set = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
test_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)

# 4. 데이터로더 (동일한 배치 크기 유지)
train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

# 5. 개선된 모델 구조
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
        
        # 분류기 가중치 초기화
        self._initialize_classifier()
        
    def _initialize_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        features = self.base.features(x)
        x = self.classifier(features)
        return x

model = AdvancedEfficientNet().to(device)

# 6. 고급 최적화 설정
# Label Smoothing과 Focal Loss 결합
class LabelSmoothingFocalLoss(nn.Module):
    def __init__(self, num_classes=100, alpha=0.25, gamma=2.0, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        # Label smoothing
        with torch.no_grad():
            true_dist = torch.zeros_like(inputs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.data.unsqueeze(1), 1.0 - self.smoothing)
        
        smooth_loss = torch.sum(-true_dist * torch.log_softmax(inputs, dim=1), dim=1)
        
        return 0.7 * focal_loss.mean() + 0.3 * smooth_loss.mean()

criterion = LabelSmoothingFocalLoss(smoothing=0.1)

# 차별적 학습률 설정
backbone_params = []
classifier_params = []

for name, param in model.named_parameters():
    if 'classifier' in name:
        classifier_params.append(param)
    else:
        backbone_params.append(param)

optimizer = optim.AdamW([
    {'params': backbone_params, 'lr': 5e-5, 'weight_decay': 1e-4},
    {'params': classifier_params, 'lr': 1e-3, 'weight_decay': 5e-4}
])

# Cosine Annealing with Warm Restarts
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=5, T_mult=2, eta_min=1e-6
)

# 7. 점진적 학습 전략
def set_backbone_trainable(model, trainable):
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = trainable

# 8. 학습 루프 (10 에포크 유지)
num_epochs = 10
best_acc = 0.0
patience = 4
patience_counter = 0

for epoch in range(num_epochs):
    # 점진적 unfreezing
    if epoch < 3:
        set_backbone_trainable(model, False)
    else:
        set_backbone_trainable(model, True)
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    train_iter = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
    for batch_idx, (inputs, labels) in enumerate(train_iter):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # 혼합 정밀도 연산
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  # gradient clipping 전 unscale
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # 메트릭 계산
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        train_iter.set_postfix(
            loss=loss.item(),
            acc=correct/total,
            lr=optimizer.param_groups[0]['lr']
        )
    
    scheduler.step()

    # 에폭별 통계
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    
    # 검증 정확도 계산
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0.0
    
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            with autocast():
                outputs = model(data)
                loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    val_acc = val_correct / val_total
    val_loss = val_loss / len(test_loader)
    
    print(f'Epoch {epoch+1} - Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

    # 베스트 모델 저장 (검증 정확도 기준)
    if val_acc > best_acc:
        best_acc = val_acc
        patience_counter = 0
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
            'epoch': epoch
        }, "EfficientNetB6_best.pth")
        print(f'새로운 최고 검증 정확도: {best_acc:.4f} - 모델 저장')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'조기 종료: {patience} 에포크 동안 개선 없음')
            break

# 9. 최종 테스트 (TTA 적용)
# 최고 모델 로드
checkpoint = torch.load("EfficientNetB6_best.pth")
model.load_state_dict(checkpoint['model_state_dict'])

def test_time_augmentation(model, test_loader, device, num_tta=5):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in tqdm(test_loader, desc='TTA Testing'):
            data, labels = data.to(device), labels.to(device)
            
            # 원본 예측
            outputs = model(data)
            batch_predictions = torch.softmax(outputs, dim=1)
            
            # TTA 예측들
            for _ in range(num_tta - 1):
                # 수평 flip
                flipped_data = torch.flip(data, dims=[3])
                outputs_tta = model(flipped_data)
                batch_predictions += torch.softmax(outputs_tta, dim=1)
            
            batch_predictions /= num_tta
            predicted = torch.argmax(batch_predictions, dim=1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_labels)
    return accuracy

# TTA 테스트
tta_accuracy = test_time_augmentation(model, test_loader, device, num_tta=3)
print(f'TTA 테스트 정확도: {tta_accuracy:.4f}')

# 일반 테스트
model.eval()
total_correct = 0
total_samples = 0
with torch.no_grad():
    test_iter = tqdm(test_loader, desc='Final Testing')
    for data, labels in test_iter:
        data, labels = data.to(device), labels.to(device)
        
        with autocast():
            outputs = model(data)
            
        _, predicted = outputs.max(1)
        total_samples += labels.size(0)
        total_correct += predicted.eq(labels).sum().item()
        test_iter.set_postfix(acc=total_correct/total_samples)

print(f'최종 테스트 정확도: {total_correct/total_samples:.4f}')
print(f'최고 검증 정확도: {best_acc:.4f}')

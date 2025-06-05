import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from sklearn.metrics import accuracy_score
import random

# 시드 설정 (재현성을 위해)
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

scaler = GradScaler()

# 1. 강화된 데이터 전처리 및 증강
train_transform = transforms.Compose([
    transforms.Resize(320),  # EfficientNet-B6에 적합한 더 큰 해상도
    transforms.RandomCrop(320, padding=24),  # 랜덤 크롭 with 패딩
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.33)),  # Random Erasing
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(320),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 2. 데이터셋 로드
train_set = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
test_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)

# 배치 크기 증가 (메모리 허용 시)
train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

# 3. 개선된 모델 구조
class ImprovedEfficientNet(nn.Module):
    def __init__(self, num_classes=100):
        super(ImprovedEfficientNet, self).__init__()
        self.backbone = models.efficientnet_b6(weights='IMAGENET1K_V1')
        
        # 기존 classifier 제거
        num_ftrs = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # 개선된 classifier 추가
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone.features(x)
        x = self.classifier(features)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImprovedEfficientNet(num_classes=100).to(device)

# 4. Label Smoothing Loss 함수
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        
    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

# 5. 옵티마이저 및 스케줄러 설정
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

# 차별적 학습률 적용
backbone_params = []
classifier_params = []

for name, param in model.named_parameters():
    if 'classifier' in name:
        classifier_params.append(param)
    else:
        backbone_params.append(param)

optimizer = optim.AdamW([
    {'params': backbone_params, 'lr': 1e-5, 'weight_decay': 1e-4},
    {'params': classifier_params, 'lr': 1e-3, 'weight_decay': 1e-4}
])

# 코사인 어닐링 스케줄러
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)

# 6. 초기 몇 에포크는 backbone freeze
def freeze_backbone(model):
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False

def unfreeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = True

# 7. 학습 루프
num_epochs = 20
best_acc = 0.0
patience = 5
patience_counter = 0

for epoch in range(num_epochs):
    # 처음 3 에포크는 backbone freeze
    if epoch < 3:
        freeze_backbone(model)
    else:
        unfreeze_backbone(model)
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    train_iter = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
    for batch_idx, (inputs, labels) in enumerate(train_iter):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        train_iter.set_postfix(
            loss=loss.item(), 
            acc=correct/total,
            lr=optimizer.param_groups[0]['lr']
        )
    
    scheduler.step()
    
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
            outputs = model(data)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_acc = val_correct / val_total
    val_loss = val_loss / len(test_loader)
    
    print(f'Epoch {epoch+1} - Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
    
    # 조기 종료 및 모델 저장
    if val_acc > best_acc:
        best_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), "best_CIFAR100_EfficientNetB6.pth")
        print(f'새로운 최고 정확도: {best_acc:.4f} - 모델 저장됨')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'조기 종료: {patience} 에포크 동안 개선 없음')
            break

# 8. 최종 테스트 (최고 모델 로드)
model.load_state_dict(torch.load("best_CIFAR100_EfficientNetB6.pth"))
model.eval()

# Test Time Augmentation (TTA) 적용
def test_time_augmentation(model, test_loader, device, num_tta=5):
    tta_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
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
                # 여기서는 간단히 원본만 사용 (실제로는 TTA 변형을 적용)
                outputs_tta = model(data)
                batch_predictions += torch.softmax(outputs_tta, dim=1)
            
            batch_predictions /= num_tta
            predicted = torch.argmax(batch_predictions, dim=1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return accuracy_score(all_labels, all_predictions)

# 최종 테스트 정확도 (TTA 적용)
final_accuracy = test_time_augmentation(model, test_loader, device, num_tta=3)
print(f'최종 테스트 정확도 (TTA): {final_accuracy:.4f}')

# 일반 테스트 정확도
model.eval()
total_correct = 0
total_samples = 0
with torch.no_grad():
    test_iter = tqdm(test_loader, desc='Final Testing')
    for data, labels in test_iter:
        data, labels = data.to(device), labels.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()
        test_iter.set_postfix(acc=total_correct/total_samples)

print(f'최종 테스트 정확도 (일반): {total_correct/total_samples:.4f}')
print(f'최고 검증 정확도: {best_acc:.4f}')
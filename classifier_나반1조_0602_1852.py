import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()


# 1. 데이터 전처리
transform = transforms.Compose([
    transforms.Resize(256),            # EfficientNet-B6 입력 크기
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 2. 데이터셋 로드 (전체 데이터셋)
train_set = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
test_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

# # (테스트용 소규모 데이터셋 사용시)
# train_set = Subset(train_set, range(500))
# test_set = Subset(test_set, range(100))

train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=2)
test_loader = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=2)

# 3. EfficientNet-B6 모델 준비
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_b6(weights='IMAGENET1K_V1')
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 100)  # CIFAR-100 클래스 수에 맞게 수정
model = model.to(device)

# 4. 손실함수, 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scaler = GradScaler()

# 5. 학습 루프
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    train_iter = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
    for inputs, labels in train_iter:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        train_iter.set_postfix(loss=loss.item(), acc=correct/total)
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    print(f'Epoch {epoch+1} - Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}')

# 6. 모델 저장
torch.save(model.state_dict(), "CIFAR100_EfficientNetB7.pth")

# 7. 테스트 정확도 평가
model.eval()
total_correct = 0
total_samples = 0
with torch.no_grad():
    test_iter = tqdm(test_loader, desc='Testing')
    for data, labels in test_iter:
        data, labels = data.to(device), labels.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()
        test_iter.set_postfix(acc=total_correct/total_samples)
print(f'최종 테스트 정확도: {total_correct/total_samples:.4f}')
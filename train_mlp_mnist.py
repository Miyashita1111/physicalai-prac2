import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
import wandb

# MLPモデルの定義
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)


# WandB初期化
config = {
    "epochs": 5,
    "batch_size": 64,
    "learning_rate": 0.001,
    "subset_size": 1000
}
wandb.init(project="mnist-mlp", name="subset-training", config=config)

# デバイス設定（GPUがあれば利用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# データ前処理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# データセットのダウンロード
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# 学習データの一部のみ使用
subset_indices = list(range(wandb.config.subset_size))
train_dataset = Subset(train_dataset, subset_indices)
train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=wandb.config.batch_size, shuffle=False)

# モデル・損失関数・オプティマイザの定義
model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
epochs = wandb.config.epochs

# トレーニングループ
for epoch in range(epochs):
    total_loss = 0
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # 評価ループ
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    test_accuracy = correct / total
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # WandBでログを送信
    wandb.log({
        "epoch": epoch + 1,
        "loss": avg_loss,
        "test_accuracy": test_accuracy
    })

# モデルの保存
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "./model/mnist_mlp_model.pt")
wandb.finish()


import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from CNN_net import CNN
from torchvision import datasets, transforms

BATCH_SIZE = 32
RANDOM_SEED = 42

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
class CNNTrainer:
    def __init__(self, model, train_loader, test_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(self.train_loader):.4f}")

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")
# 加载数据集
# 使用 __file__ 确定项目路径，从项目根目录或 CNN 目录运行都可以。
project_root = Path(__file__).resolve().parents[1]
dataset_path = project_root / "dataset_2026_updated" / "dataset_2026"

if not dataset_path.is_dir():
    raise FileNotFoundError(f"找不到数据集目录：{dataset_path}")

# 不使用旋转、平移、翻转或颜色增强。
# ImageFolder 默认以 RGB 读取，因此显式转为单通道灰度图。
image_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

full_dataset = datasets.ImageFolder(
    root=str(dataset_path),
    transform=image_transform,
)

class_names = full_dataset.classes
class_to_index = full_dataset.class_to_idx
number_of_classes = len(class_names)

# 每个类别分别划分 75% 训练集和 25% 测试集。
# 固定随机种子，保证每次运行得到相同的划分。
targets = np.asarray(full_dataset.targets)
random_generator = np.random.default_rng(RANDOM_SEED)

train_indices = []
test_indices = []

for class_index in range(number_of_classes):
    class_indices = np.flatnonzero(targets == class_index)
    random_generator.shuffle(class_indices)

    split_position = int(len(class_indices) * 0.75)
    train_indices.extend(class_indices[:split_position].tolist())
    test_indices.extend(class_indices[split_position:].tolist())

train_dataset = Subset(full_dataset, train_indices)
test_dataset = Subset(full_dataset, test_indices)

# DataLoader 负责按 batch 读取数据。Windows 上使用
# num_workers=0 和 pin_memory=False，兼容当前的 PyTorch/CUDA 环境。
shuffle_generator = torch.Generator().manual_seed(RANDOM_SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    generator=shuffle_generator,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)

print(f"运行设备：{device}")
print(f"类别顺序：{class_names}")
print(f"类别编号：{class_to_index}")
print(f"完整数据集：{len(full_dataset)} 张")
print(f"训练集：{len(train_dataset)} 张")
print(f"测试集：{len(test_dataset)} 张")

# 读取一个 batch，检查数据形状。
sample_images, sample_labels = next(iter(train_loader))
print(f"图像 batch 形状：{sample_images.shape}")
print(f"标签 batch 形状：{sample_labels.shape}")

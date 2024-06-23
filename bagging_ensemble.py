import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from sklearn.utils import resample
import time

# 设置随机种子
torch.manual_seed(2023)
np.random.seed(2023)

# 加载CIFAR10数据集并进行预处理
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=ToTensor())
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=ToTensor())

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 定义基本模型的架构和训练过程
class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 28, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(28, 180, 3, padding=1)
        self.conv3 = nn.Conv2d(180, 320, 3, padding=1)
        self.conv4 = nn.Conv2d(320, 640, 3, padding=1)

        self.fc1 = nn.Linear(640 * 2 * 2, 300)  # 32 / 2 = 16, 16 / 2 = 8
        self.fc2 = nn.Linear(300, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        out = self.conv1(x)   
        out = f.relu(out)
        out = self.pool(out)

        out = self.pool(f.relu(self.conv2(out)))
        out = self.pool(f.relu(self.conv3(out)))
        out = self.pool(f.relu(self.conv4(out)))

        out = out.view(-1, 640 * 2 * 2)
        out = f.relu(self.fc1(out))
        out = f.relu(self.fc2(out))
        out = f.relu(self.fc3(out))
        out = self.fc4(out)

        # 添加L2正则化
        l2_reg = None
        for param in self.parameters():
            if l2_reg is None:
                l2_reg = param.norm(2)
            else:
                l2_reg = l2_reg + param.norm(2)
        out = out + 0.001 * l2_reg
        return out

num_epoches = 5
def train_base_model(model, train_loader, num_epoches):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.to(device)  # 将模型移动到设备
    for epoch in range(num_epoches):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)  # 将输入数据移动到设备
            labels = labels.to(device)  # 将标签移动到设备
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Training Loss: {running_loss / len(train_loader)}")

# 定义袋装集成的函数
def bagging_ensemble(train_loader, n_models, sample_size, batch_size=128):
    models = []

    for _ in range(n_models):
        # 对训练数据进行有放回的采样
        train_subset = resample(train_loader.dataset, n_samples=sample_size)

        # 创建基本模型并训练
        model = BaseModel()
        train_base_model(model, DataLoader(train_subset, batch_size=batch_size, shuffle=True),
                         num_epoches=num_epoches)
        models.append(model)

    return models

# 进行袋装集成的训练和预测
n_models = 6  # 集成的模型数量
batch_size = 128
sample_size = len(train_dataset)  # 采样的训练样本数量

# 预测测试数据
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start = time.time()
ensemble_models = bagging_ensemble(train_loader, n_models, sample_size, batch_size)


predictions = []

for model in ensemble_models:
    model.to(device)
    model.eval()
    model_predictions = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            model_predictions.append(predicted.cpu().numpy())

    predictions.append(np.concatenate(model_predictions))

predictions = np.array(predictions)
predictions = torch.tensor(predictions)

# 简单多数投票
def majority_voting(*predictions):
    # 沿着列方向计算每个类别出现的次数
    stacked = torch.stack(predictions)
    majority, _ = torch.mode(stacked, dim=0)
    return majority.flatten()

# 使用简单多数投票进行预测
majority_result = majority_voting(*predictions)

testlabel = torch.utils.data.DataLoader(
    dataset = test_dataset,
    batch_size = 10000,
    shuffle=False)

labels = []
for features, labels in testlabel:
    labels = labels

num_correct = (majority_result == labels).sum().item()
num_samples = len(labels)

print(f"模型的准确率:\t{(num_correct / num_samples):.2%}")
end = time.time()
exe_time = end - start
minute = int(exe_time / 60)
msc = exe_time % 60
print(f"Execution time:  {minute}min {msc:.2f}msc")
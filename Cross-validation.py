import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import KFold

# 设置随机种子以确保实验的可重复性
torch.manual_seed(42)

# 网络模型的建立
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # 神经网络的输入为 三个通道
        # Conv2d 参数：
        # （1）in_channels(int)输入特征矩阵的深度（图片通道数）
        # （2）out_channels(int)为卷积核的个数
        # （3）kerner_size(int or tuple)为卷积核的尺寸
        self.conv1 = nn.Conv2d(3, 28, 3, padding=1)
        self.conv2 = nn.Conv2d(28, 180, 3, padding=1)
        self.conv3 = nn.Conv2d(180, 320, 3, padding=1)
        self.conv4 = nn.Conv2d(320, 640, 3, padding=1)

        # 批归一化
        self.bn1 = nn.BatchNorm2d(28)
        self.bn2 = nn.BatchNorm2d(180)
        self.bn3 = nn.BatchNorm2d(320)
        self.bn4 = nn.BatchNorm2d(640)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(640 * 2 * 2, 300)
        self.fc2 = nn.Linear(300, 64)
        self.fc3 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(0.5)



    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out) 
        out = f.relu(out)
        out = self.pool(out)
        out = self.dropout(out)

        out = self.pool(f.relu(self.bn2(self.conv2(out))))
        # out = self.dropout(out)

        out = self.pool(f.relu(self.bn3(self.conv3(out))))
        # out = self.dropout(out)

        out = self.pool(f.relu(self.bn4(self.conv4(out))))
        out = self.dropout(out)

        out = out.view(-1, 640 * 2 * 2)
        out = f.relu(self.fc1(out))
        # out = self.dropout(out)
        out = f.relu(self.fc2(out))
        # out = self.dropout(out)
        # out = self.dropout(out)
        out = self.fc3(out)
        
        # 添加L2正则化
        l2_reg = None
        for param in self.parameters():
            if l2_reg is None:
                l2_reg = param.norm(2)
            else:
                l2_reg = l2_reg + param.norm(2)
        out = out + 0.001 * l2_reg

        return out

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载CIFAR数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# 定义交叉验证的折数
num_folds = 5
num_epoch = 10

# 进行交叉验证
kf = KFold(n_splits=num_folds, shuffle=True)

models = []

for fold, (train_indices, val_indices) in enumerate(kf.split(trainset)):
    print(f"Fold: {fold+1}")

    # 创建训练集和验证集的数据加载器
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=80, sampler=train_sampler)
    valloader = torch.utils.data.DataLoader(trainset, batch_size=80, sampler=val_sampler)

    # 创建CNN模型实例
    model = CNN().to(device)
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)

    # 训练模型
    for epoch in range(num_epoch):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs = data[0].to(device)
            labels = data[1].to(device)

            optimizer.zero_grad()

            # 前向传播、反向传播和优化
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f'Epoch: {epoch+1}, Batch: {i+1}, Loss: {running_loss / 100}')
                running_loss = 0.0

    # 在验证集上评估模型
    correct = 0
    total = 0
    with torch.no_grad():
        for data in valloader:
            images = data[0].to(device)
            labels = data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Validation Accuracy: {accuracy}%')

    models.append((model, accuracy))

best = 0    # 正确率最高的模型索引
acc = 0
for i in range(len(models)):
    model = models[i]
    if model[1] > acc:
        best = i
        acc = model[1]

testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform)

testloader = torch.utils.data.DataLoader(
    dataset = testset,
    shuffle=False)

model = models[best][0]

with torch.no_grad():
    num_correct = 0
    num_samples = 0
    for features, labels in testloader:
        # 与全连接神经网络要求扁平数据不同，CNN对3个通道的数据进行卷积
        # 该步骤省略：  features.cc = features.view(-1, 32 * 32)
        features = features.to(device)
        labels = labels.to(device)
        pred = model(features)
        # 获取最大的角标，表示的就是哪个数字
        values, indexes = torch.max(pred, axis=1)
        # 统计正确的结果
        num_correct += (indexes == labels).sum().item()
        num_samples += len(labels)

    print(f"模型的准确率:\t{(num_correct / num_samples):.2%}")
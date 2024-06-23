import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True,
    download=True,
    transform=transform)
testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform)

batch_size = 80
trainloader = torch.utils.data.DataLoader(
    dataset = trainset,
    batch_size=batch_size,
    shuffle=True)
testloader = torch.utils.data.DataLoader(
    dataset = testset,
    batch_size=batch_size,
    shuffle=False)
        

# 网络模型的建立
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 28, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(28, 180, 3, padding=1)
        self.conv3 = nn.Conv2d(180, 320, 3, padding=1)
        self.conv4 = nn.Conv2d(320, 640, 3, padding=1)
        self.conv5 = nn.Conv2d(640, 1280, 3, padding=1)

        self.fc1 = nn.Linear(1280 * 2 * 2, 300)  # 32 / 2 = 16, 16 / 2 = 8
        self.fc2 = nn.Linear(300, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        # 批归一化
        self.bn1 = nn.BatchNorm2d(28)
        self.bn2 = nn.BatchNorm2d(180)
        self.bn3 = nn.BatchNorm2d(320)
        self.bn4 = nn.BatchNorm2d(640)
        self.bn5 = nn.BatchNorm2d(1280)

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

        out = f.relu(self.bn5(self.conv5(out)))

        out = out.view(-1, 1280 * 2 * 2)
        out = f.relu(self.fc1(out))
        # out = self.dropout(out)
        out = f.relu(self.fc2(out))
        # out = self.dropout(out)
        out = f.relu(self.fc3(out))
        # out = self.dropout(out)
        out = self.fc4(out)
        
        # 添加L2正则化
        l2_reg = None
        for param in self.parameters():
            if l2_reg is None:
                l2_reg = param.norm(2)
            else:
                l2_reg = l2_reg + param.norm(2)
        out = out + 0.001 * l2_reg

        # 使用Softmax函数将输出转换为概率分布
        # out = f.softmax(out, dim=1)
        return out

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
learning_rate = 0.001
epoches = 20

model = ConvNet().to(device)
lossFun = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

losses = []
start = time.time()
for epoch in range(epoches):
    running_loss = 0.0
    running_acc = 0.0
    epoches_loss = []
    for i,data in enumerate(trainloader):
        features = data[0].to(device)
        labels = data[1].to(device)

        preds = model(features)
        loss = lossFun(preds, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()

        total = labels.shape[0]  # labels 的长度
        _, predicted = torch.max(preds.data, 1)
        # 预测正确的数目
        correct = (predicted == labels).sum().item()  
        accuracy =  correct / total
        running_acc += accuracy

        if i % 100 == 99:
            print(f'{epoch+1},{i+1},\
                    {(running_loss / 100):.6f},\
                    {(running_acc / 100):.2%}'
            )
            running_loss = 0.0
            running_acc = 0.0

        losses.append(loss.item())

# model.cpu()
with torch.no_grad():
    num_correct = 0
    num_samples = 0
    for features, labels in testloader:
        features = features.to(device)  # 将输入数据移动到设备
        labels = labels.to(device)  # 将标签移动到设备
        pred = model(features)
        # 获取最大的角标，表示的就是哪个数字
        values, indexes = torch.max(pred, axis=1)
        # 统计正确的结果
        num_correct += (indexes == labels).sum().item()
        num_samples += len(labels)


    end = time.time()
    exe_time = end - start
    minute = int(exe_time / 60)
    msc = exe_time % 60
    print(f"模型的准确率:\t{(num_correct / num_samples):.2%}")
    print(f"Execution time:  {minute}min {msc:.2f}msc")



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


        # L2归一化
        # self.conv1 = nn.utils.weight_norm(self.conv1, dim=None)
        # self.conv2 = nn.utils.weight_norm(self.conv2, dim=None)
        # self.fc1 = nn.utils.weight_norm(self.fc1, dim=None)
        # self.fc2 = nn.utils.weight_norm(self.fc2, dim=None)
        # self.fc3 = nn.utils.weight_norm(self.fc3, dim=None)
        

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

        # 添加L1正则化
        l1_reg = torch.tensor(0.0)
        l1_reg = l1_reg.to('cuda')
        for param in self.parameters():
            l1_reg += torch.norm(param, 1)
        out = out + 0.001 * l1_reg
        
        # 添加L2正则化
        # l2_reg = None
        # for param in self.parameters():
        #     if l2_reg is None:
        #         l2_reg = param.norm(2)
        #     else:
        #         l2_reg = l2_reg + param.norm(2)
        # out = out + 0.001 * l2_reg

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


    end = time.time()
    exe_time = end - start
    minute = int(exe_time / 60)
    msc = exe_time % 60
    print(f"模型的准确率:\t{(num_correct / num_samples):.2%}")
    print(f"Execution time:  {minute}min {msc:.2f}msc")



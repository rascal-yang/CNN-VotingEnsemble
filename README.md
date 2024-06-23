# 卷积神经网络图像分类项目

## 项目概述
本项目是一系列使用PyTorch框架实现的神经网络模型，用于图像分类任务，特别是针对CIFAR-10数据集。项目中包含了多种不同的网络结构，如卷积神经网络(CNN)以及集成学习方法。

## 环境要求
- Python 3.9
- PyTorch
- torchvision
- numpy
- sklearn (仅限集成学习方法)
- 其他可能的依赖项（根据实际运行情况可能需要安装）

## 文件结构
```
neural_network_image_classification/
│
├── votingEnsemble.py          # 多数投票集成方法
├── cnn_2.py                  # 包含批归一化的CNN模型
├── Cross-validation.py       # 交叉验证方法
├── cnn.py                    # 基本CNN模型，包含L1和L2正则化
└── bagging_ensemble.py       # 基于Bagging的集成学习方法
```

## 如何运行
1. 确保Python环境已安装，并且安装了所需的依赖包。
2. 将项目代码克隆或下载到本地环境。
3. https://www.cs.toronto.edu/~kriz/cifar.html 下载数据集
4. 运行特定Python文件以训练和测试模型。

## 模型说明
- `ConvNet`：基础的卷积神经网络模型。
- 集成方法：包括多数投票和Bagging集成，用于提高模型的泛化能力和准确率。

## 性能评估
- 所有模型在CIFAR-10数据集上进行训练和测试。
- 准确率是衡量模型性能的主要指标。

## 贡献与反馈
欢迎提交Pull Request或创建Issue来改进项目。

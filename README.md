# 图像蒸馏演示项目 (Distillation Demo)

## 项目简介

本项目是一个基于PyTorch的图像蒸馏（Distillation）演示，主要展示了如何使用编码器-解码器（Encoder-Decoder）架构进行图像处理和特征提取。

## 项目结构

```
distillation_demo/
│
├── data/                # 数据目录
│   └── 1.png            # 示例图像
│
├── dataset.py            # 数据集处理脚本
├── model.py              # 主要模型定义
├── model_mini.py         # 蒸馏版模型
├── train.py              # 主要训练脚本
├── train_mini.py         # 蒸馏版训练脚本
└── README.md             # 项目说明文档
```

## 模型架构

项目实现了一个简单的图像编码-解码模型：

- `Encoder`：
  - 输入通道：3（RGB图像）
  - 第一层卷积：3 -> 64通道
  - 使用BatchNorm和ReLU激活函数
  - 第二层卷积：64 -> 3通道

- `Decoder`：
  - 对称的解码结构
  - 使用卷积层
  - Sigmoid激活函数，将输出映射到[0, 1]范围

## 依赖环境

- Python 3.7+
- PyTorch
- NumPy

## 安装步骤

1. 克隆仓库
```bash
git clone https://github.com/yourusername/distillation_demo.git
cd distillation_demo
```


## 训练模型

使用以下命令训练模型：

```bash
python train.py
```

对于蒸馏版模型：

```bash
python train_mini.py
```

## 注意事项

- 项目为演示目的，可能需要根据实际使用场景进行调整
- 建议在GPU环境下训练以获得更好的性能

## 许可证

[在此添加您的许可证信息]

## 贡献

欢迎提交issues和pull requests来改进这个项目。

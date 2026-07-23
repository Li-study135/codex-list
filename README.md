# RNN MNIST 分类器

使用循环神经网络（RNN）对 MNIST 手写数字数据集进行分类。

## 核心思路

将每张 28x28 的灰度图像看作一个**序列**：

- 时间步（seq_len）= 28（图像的每一行）
- 输入特征（input_size）= 28（每行的 28 个像素值）

模型按行读取图像像素，每一行作为一个时间步输入 RNN，最后取最后一个时间步的隐藏状态接全连接层输出 10 类概率。

```
输入: (batch, 1, 28, 28)
  │ squeeze(channel)
输入: (batch, 28, 28)
  │ RNN / LSTM / GRU
输出: (batch, hidden_size)
  │ Linear
输出: (batch, 10)
```

## 支持的模型类型

| 参数             | 默认值  | 说明                           |
|------------------|---------|--------------------------------|
| --rnn-type       | lstm    | rnn / lstm / gru               |
| --hidden-size    | 128     | 隐藏层维度                     |
| --num-layers     | 2       | RNN 层数                       |
| --bidirectional  | 否      | 加 --bidirectional 启用双向    |
| --dropout        | 0.3     | Dropout 概率                   |

## 快速开始

```bash
pip install torch torchvision matplotlib

# 默认配置（LSTM-128 双层，10 个 epoch）
python mnist_rnn.py

# 使用 GRU
python mnist_rnn.py --rnn-type gru

# 使用双向 LSTM
python mnist_rnn.py --bidirectional

# 自定义参数
python mnist_rnn.py --rnn-type lstm --hidden-size 256 --num-layers 3 --epochs 20

# 使用 CPU（即使有 GPU）
python mnist_rnn.py --no-cuda
```

首次运行会自动下载 MNIST 数据集（约 11 MB）。

## 训练输出示例

```
设备：cuda
模型：LSTM | 参数量：214,282
Epoch  1/10 | Loss: 0.5736 | Train Acc: 0.8258 | Val Acc: 0.9428
  → 新最佳模型已保存 (acc=0.9428)
Epoch  2/10 | Loss: 0.1250 | Train Acc: 0.9657 | Val Acc: 0.9698
...
Epoch 10/10 | Loss: 0.0174 | Train Acc: 0.9970 | Val Acc: 0.9854

测试集准确率：0.9854
```

训练完成后，outputs/ 目录下会生成：

| 文件                | 说明                                   |
|---------------------|----------------------------------------|
| best_model.pt       | 验证集上准确率最高的模型权重            |
| curves.png          | Loss 和 Accuracy 变化曲线              |
| predictions.png     | 测试集预测结果可视化（绿色=正确，红色=错误）|

## 全部参数

| 参数             | 默认值    | 说明                       |
|------------------|-----------|----------------------------|
| --rnn-type       | lstm      | RNN 类型                   |
| --hidden-size    | 128       | 隐藏层维度                 |
| --num-layers     | 2         | RNN 层数                   |
| --bidirectional  | False     | 是否启用双向 RNN           |
| --dropout        | 0.3       | Dropout 概率               |
| --epochs         | 10        | 训练轮数                   |
| --batch-size     | 64        | 批次大小                   |
| --lr             | 1e-3      | 学习率                     |
| --no-cuda        | False     | 强制使用 CPU               |
| --output-dir     | outputs   | 输出目录                   |

## 环境要求

- Python 3.8+
- PyTorch 1.10+
- torchvision
- matplotlib
- requests

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path


# ── 模型定义 ──────────────────────────────────────────────────────────────

class RNNClassifier(nn.Module):
    """用 RNN 做 MNIST 分类。

    将 28×28 的图片看作 28 个时间步（每一行一个 step），
    每个时间步输入维度为 28（一行有 28 个像素）。

    Args:
        input_size: 每个时间步的输入特征数（MNIST 一行像素数）
        hidden_size: 隐藏层神经元数量
        num_layers: RNN 堆叠层数
        num_classes: 分类类别数（MNIST 为 0-9）
        rnn_type: 循环单元类型 — "rnn", "lstm", "gru"
        bidirectional: 是否使用双向 RNN
        dropout: 除最后一层外，各层之间的 dropout 概率
    """

    def __init__(
        self,
        input_size: int = 28,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 10,
        rnn_type: str = "rnn",
        bidirectional: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        # 支持的 RNN 类型映射表
        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}
        # 校验传入的 rnn_type 是否合法
        if rnn_type not in rnn_cls:
            raise ValueError(f"rnn_type 必须为 {list(rnn_cls.keys())}，传入：{rnn_type}")

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        # 构造 RNN 层：batch_first=True 表示输入形状为 (batch, seq, feature)
        self.rnn = rnn_cls[rnn_type](
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            # PyTorch 要求 num_layers=1 时 dropout 必须为 0
            dropout=dropout if num_layers > 1 else 0,
        )
        # 全连接输出层：将 RNN 最后一层的隐状态映射到类别分数
        self.fc = nn.Linear(hidden_size * num_directions, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 可能来自 DataLoader，形状为 (batch, 1, 28, 28)，
        # 需要压缩通道维变为 (batch, 28, 28)
        if x.dim() == 4:
            x = x.squeeze(1)
        # 前向传播，out 是各时间步输出，hn 是最后一层隐状态
        out, hn = self.rnn(x)

        if self.rnn_type == "lstm":
            # LSTM 返回 (h_n, c_n) 元组，分类只需要 h_n
            hn = hn[0]

        # hn shape: (num_layers * num_directions, batch, hidden_size)
        # 取最后一层的隐状态用于分类
        # 单向：用 hn[-1]；双向：拼接最后前向层 hn[-2] 和最后反向层 hn[-1]
        h_last = hn[-1] if not self.bidirectional else torch.cat([hn[-2], hn[-1]], dim=-1)
        # 经过全连接层得到各类别 logits，形状 (batch, num_classes)
        logits = self.fc(h_last)
        return logits


# ── 训练 / 评估 ──────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.MSELoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += criterion(logits, y).item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)
    return correct / total, loss_sum / total


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


# ── 结果可视化 ────────────────────────────────────────────────────────────

def plot_curves(
    train_losses: list[float],
    train_accs: list[float],
    val_accs: list[float],
    save_path: str,
):
    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_losses, marker="o", label="Train Loss")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(epochs, train_accs, marker="o", label="Train Acc")
    axes[1].plot(epochs, val_accs, marker="s", label="Val Acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[结果图已保存] {save_path}")


def visualize_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_path: str,
    num_samples: int = 16,
):
    model.eval()
    images, true_labels, pred_labels = [], [], []
    for x, y in loader:
        logits = model(x.to(device))
        preds = logits.argmax(dim=1).cpu()
        for i in range(x.size(0)):
            images.append(x[i].squeeze())
            true_labels.append(y[i].item())
            pred_labels.append(preds[i].item())
            if len(images) >= num_samples:
                break
        if len(images) >= num_samples:
            break

    cols = 8
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    axes = axes.flatten()
    for i in range(num_samples):
        axes[i].imshow(images[i], cmap="gray")
        color = "green" if true_labels[i] == pred_labels[i] else "red"
        axes[i].set_title(f"T:{true_labels[i]} P:{pred_labels[i]}", color=color, fontsize=9)
        axes[i].axis("off")
    for i in range(num_samples, len(axes)):
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[预测可视化已保存] {save_path}")


# ── 主流程 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="用 RNN 分类 MNIST")
    parser.add_argument("--rnn-type", choices=["rnn", "lstm", "gru"], default="lstm",
                        help="RNN 类型（默认 lstm）")
    parser.add_argument("--hidden-size", type=int, default=128,
                        help="隐藏层维度（默认 128）")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="RNN 层数（默认 2）")
    parser.add_argument("--bidirectional", action="store_true",
                        help="是否使用双向 RNN")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout 概率（默认 0.3）")
    parser.add_argument("--epochs", type=int, default=10,
                        help="训练轮数（默认 10）")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="批次大小（默认 64）")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="学习率（默认 1e-3）")
    parser.add_argument("--no-cuda", action="store_true",
                        help="强制使用 CPU")
    parser.add_argument("--output-dir", default="outputs",
                        help="输出目录（默认 outputs）")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"设备：{device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 数据加载 ────────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 从训练集分一部分做验证
    val_size = 5000
    train_sub, val_sub = torch.utils.data.random_split(
        train_dataset, [len(train_dataset) - val_size, val_size],
    )
    val_loader = DataLoader(val_sub, batch_size=args.batch_size, shuffle=False)
    # 用剩下的训练
    train_loader_full = DataLoader(train_sub, batch_size=args.batch_size, shuffle=True)

    # ── 模型初始化 ──────────────────────────────────────────────────────
    model = RNNClassifier(
        input_size=28,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=10,
        rnn_type=args.rnn_type,
        bidirectional=args.bidirectional,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型：{args.rnn_type.upper()} | 参数量：{total_params:,}")
    print(f"       hidden_size={args.hidden_size}, num_layers={args.num_layers}, "
          f"bidirectional={args.bidirectional}, dropout={args.dropout}")
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    best_acc = 0.0

    train_losses, train_accs, val_accs = [], [], []

    # ── 训练循环 ────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader_full, optimizer, device)
        train_acc, _ = evaluate(model, train_loader_full, device)
        val_acc, val_loss = evaluate(model, val_loader, device)

        train_losses.append(loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:2d}/{args.epochs} | Loss: {loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | LR: {lr_now:.2e}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"  → 新最佳模型已保存 (acc={best_acc:.4f})")

        scheduler.step()

    # ── 测试集评估 ──────────────────────────────────────────────────────
    model.load_state_dict(torch.load(output_dir / "best_model.pt", weights_only=True))
    test_acc, test_loss = evaluate(model, test_loader, device)
    print(f"\n测试集准确率：{test_acc:.4f} | 测试集 Loss：{test_loss:.4f}")

    # ── 可视化 ──────────────────────────────────────────────────────────
    plot_curves(train_losses, train_accs, val_accs, str(output_dir / "curves.png"))
    visualize_predictions(model, test_loader, device, str(output_dir / "predictions.png"))


if __name__ == "__main__":
    main()

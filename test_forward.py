"""RNNClassifier 前向传播单元测试。

覆盖三类场景：
  - 输入张量形状异常（维度错误、特征数不匹配、整数输入等）
  - 空张量（batch=0）
  - GPU / CPU 设备切换（含跨设备传递）

不修改原有模型代码，不引入第三方测试依赖。
"""

import unittest
import torch
from mnist_rnn import RNNClassifier


# 各测试类共享的模型超参数
INPUT_SIZE = 28
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_CLASSES = 10


def default_model(rnn_type="lstm", bidirectional=False, device="cpu"):
    """创建默认配置的 RNNClassifier 并移至指定设备。"""
    model = RNNClassifier(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        rnn_type=rnn_type,
        bidirectional=bidirectional,
        dropout=0.0,
    ).to(device)
    model.eval()
    return model


# ═══════════════════════════════════════════════════════
# 1. 输入张量形状异常
# ═══════════════════════════════════════════════════════

class TestShapeAnomalies(unittest.TestCase):
    """各种非法 / 边缘输入形状的异常处理测试。"""

    def setUp(self):
        self.model = default_model()

    # ── 合法形状应正常通过 ────────────────────────────────

    def test_valid_3d(self):
        """3D 输入 (N, 28, 28) 应正常得到 (N, 10) 输出。"""
        x = torch.randn(4, 28, 28)
        out = self.model(x)
        self.assertEqual(out.shape, (4, NUM_CLASSES))

    def test_valid_4d(self):
        """4D 输入 (N, 1, 28, 28) 应自动 squeeze 并正常输出 (N, 10)。"""
        x = torch.randn(4, 1, 28, 28)
        out = self.model(x)
        self.assertEqual(out.shape, (4, NUM_CLASSES))

    # ── 非法维度 ──────────────────────────────────────────

    def test_2d_no_batch_dim(self):
        """2D ?? (28, 28) -- RNN ?? 2D/3D??????"""
        x = torch.randn(28, 28)
        out = self.model(x)
        self.assertIsNotNone(out)

    def test_5d_extra_dim(self):
        """5D 输入 (N, 1, 1, 28, 28) —— 过多维度，应抛出 RuntimeError。"""
        x = torch.randn(2, 1, 1, 28, 28)
        with self.assertRaises((RuntimeError, ValueError)):
            self.model(x)

    def test_4d_wrong_channel(self):
        """4D 输入 (N, 3, 28, 28) —— channel != 1 无法被 squeeze，应抛出 RuntimeError。"""
        x = torch.randn(2, 3, 28, 28)
        with self.assertRaises((RuntimeError, ValueError)):
            self.model(x)

    # ── 维度大小不匹配 ────────────────────────────────────

    def test_wrong_feature_dim(self):
        """3D 输入 (N, 28, 50) —— 特征维度 50 != input_size=28，应抛出 RuntimeError。"""
        x = torch.randn(2, 28, 50)
        with self.assertRaises((RuntimeError, ValueError)):
            self.model(x)

    def test_wrong_seq_len_3d_works(self):
        """3D 输入 (N, 14, 28) —— seq_len 可以任意，应正常输出 (N, 10)。"""
        x = torch.randn(2, 14, 28)
        out = self.model(x)
        self.assertEqual(out.shape, (2, NUM_CLASSES))

    # ── 数据类型 ──────────────────────────────────────────

    def test_int_tensor(self):
        """整型输入 —— RNN 需要浮点，应抛出 RuntimeError。"""
        x = torch.randint(0, 256, (2, 28, 28), dtype=torch.long)
        with self.assertRaises((RuntimeError, ValueError)):
            self.model(x)


# ═══════════════════════════════════════════════════════
# 2. 空张量
# ═══════════════════════════════════════════════════════

class TestEmptyTensor(unittest.TestCase):
    """batch=0 空张量的前向传播。"""

    def setUp(self):
        self.model = default_model()

    def test_empty_3d(self):
        """空 3D 张量 (0, 28, 28) 应输出 (0, 10)。"""
        x = torch.empty(0, 28, 28)
        out = self.model(x)
        self.assertEqual(out.shape, (0, NUM_CLASSES))

    def test_empty_4d(self):
        """空 4D 张量 (0, 1, 28, 28) 应自动 squeeze 并输出 (0, 10)。"""
        x = torch.empty(0, 1, 28, 28)
        out = self.model(x)
        self.assertEqual(out.shape, (0, NUM_CLASSES))

    def test_empty_lstm(self):
        """空输入 + LSTM 类型，验证 LSTM 特有的 (h_n, c_n) 分支也能正常跑通。"""
        model = default_model(rnn_type="lstm")
        x = torch.empty(0, 28, 28)
        out = model(x)
        self.assertEqual(out.shape, (0, NUM_CLASSES))

    def test_empty_gru(self):
        """空输入 + GRU 类型。"""
        model = default_model(rnn_type="gru")
        x = torch.empty(0, 28, 28)
        out = model(x)
        self.assertEqual(out.shape, (0, NUM_CLASSES))

    def test_empty_bidirectional(self):
        """空输入 + 双向 LSTM，验证 bidirectional concat 分支。"""
        model = default_model(rnn_type="lstm", bidirectional=True)
        x = torch.empty(0, 28, 28)
        out = model(x)
        self.assertEqual(out.shape, (0, NUM_CLASSES))


# ═══════════════════════════════════════════════════════
# 3. GPU / CPU 设备切换
# ═══════════════════════════════════════════════════════

class TestDeviceHandling(unittest.TestCase):
    """模型与输入在不同设备上的行为测试。"""

    def test_cpu_forward(self):
        """CPU 模型 + CPU 张量 → 正常输出 (N, 10)。"""
        model = default_model(device="cpu")
        x = torch.randn(4, 28, 28)
        out = model(x)
        self.assertEqual(out.shape, (4, NUM_CLASSES))
        self.assertEqual(out.device.type, "cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA 不可用")
    def test_cuda_forward(self):
        """CUDA 模型 + CUDA 张量 → 正常输出 (N, 10) 且结果在 CUDA 上。"""
        device = "cuda"
        model = default_model(device=device)
        x = torch.randn(4, 28, 28, device=device)
        out = model(x)
        self.assertEqual(out.shape, (4, NUM_CLASSES))
        self.assertEqual(out.device.type, "cuda")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA 不可用")
    def test_cpu_model_cuda_tensor(self):
        """CPU 模型 + CUDA 张量 → 设备不匹配，应抛出 RuntimeError。"""
        model = default_model(device="cpu")
        x = torch.randn(4, 28, 28, device="cuda")
        with self.assertRaises((RuntimeError, ValueError)):
            model(x)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA 不可用")
    def test_cuda_model_cpu_tensor(self):
        """CUDA 模型 + CPU 张量 → 设备不匹配，应抛出 RuntimeError。"""
        model = default_model(device="cuda")
        x = torch.randn(4, 28, 28, device="cpu")
        with self.assertRaises((RuntimeError, ValueError)):
            model(x)

    def test_model_move_between_devices(self):
        """模型在 CPU ↔ CUDA 间迁移后仍能正确前向传播。

        先创建模型并完成一次 CPU 推理，
        再将模型移至 CUDA（若有）并完成一次推理，
        最后迁回 CPU 并验证输出形状和设备类型正确。
        """
        model = default_model(device="cpu")
        x_cpu = torch.randn(2, 28, 28)
        out_cpu = model(x_cpu)
        self.assertEqual(out_cpu.shape, (2, NUM_CLASSES))
        self.assertEqual(out_cpu.device.type, "cpu")

        if torch.cuda.is_available():
            model.to("cuda")
            x_cuda = torch.randn(2, 28, 28, device="cuda")
            out_cuda = model(x_cuda)
            self.assertEqual(out_cuda.shape, (2, NUM_CLASSES))
            self.assertEqual(out_cuda.device.type, "cuda")

            model.to("cpu")
            out_cpu2 = model(x_cpu)
            self.assertEqual(out_cpu2.shape, (2, NUM_CLASSES))
            self.assertEqual(out_cpu2.device.type, "cpu")


# ═══════════════════════════════════════════════════════
# 4. 多 RNN 类型的形状合法性验证（补充覆盖）
# ═══════════════════════════════════════════════════════

class TestRNNTypeVariants(unittest.TestCase):
    """不同 RNN 类型下合法输入的输出形状验证。"""

    def test_rnn_basic(self):
        """基础 RNN + 4D 输入 → (N, 10)。"""
        model = default_model(rnn_type="rnn")
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        self.assertEqual(out.shape, (2, NUM_CLASSES))

    def test_gru_basic(self):
        """GRU + 4D 输入 → (N, 10)。"""
        model = default_model(rnn_type="gru")
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        self.assertEqual(out.shape, (2, NUM_CLASSES))

    def test_lstm_bidirectional(self):
        """双向 LSTM + 合法输入 → (N, 10)。"""
        model = default_model(rnn_type="lstm", bidirectional=True)
        x = torch.randn(2, 28, 28)
        out = model(x)
        self.assertEqual(out.shape, (2, NUM_CLASSES))

    def test_rnn_bidirectional_with_4d(self):
        """双向 RNN + 4D 输入 → (N, 10)。"""
        model = default_model(rnn_type="rnn", bidirectional=True)
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        self.assertEqual(out.shape, (2, NUM_CLASSES))


if __name__ == "__main__":
    unittest.main(verbosity=2)

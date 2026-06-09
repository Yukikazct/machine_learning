"""
机器学习上机实验12：神经网络-2
基于PyTorch的深层神经网络训练机制研究

实验一：初始化方法对梯度传播的影响（Normal vs Xavier）
实验二：残差连接对深层网络训练的影响（普通MLP vs 残差MLP）
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免plt.show()阻塞
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 配置中文字体支持
# 在macOS上优先使用 Heiti TC（黑体），次选 Arial Unicode MS
_chinese_fonts = [f.name for f in fm.fontManager.ttflist]
if 'Heiti TC' in _chinese_fonts:
    plt.rcParams['font.family'] = 'Heiti TC'
elif 'Arial Unicode MS' in _chinese_fonts:
    plt.rcParams['font.family'] = 'Arial Unicode MS'
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from collections import OrderedDict

# ============================================================
# 全局配置
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
INPUT_DIM = 784       # MNIST: 28x28
HIDDEN_DIM = 256
NUM_HIDDEN_LAYERS = 20
OUTPUT_DIM = 10       # 10个类别
TRAIN_SIZE = 5000
TEST_SIZE = 1000

print(f"使用设备: {DEVICE}")
print(f"PyTorch 版本: {torch.__version__}")


# ============================================================
# 数据加载：MNIST数据集
# ============================================================
def load_mnist_data(train_size=5000, test_size=1000):
    """
    加载MNIST数据集，并随机抽取指定数量的训练/测试样本。

    参数:
        train_size: 训练集样本数量
        test_size:  测试集样本数量

    返回:
        train_loader, test_loader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST 均值与标准差
    ])

    # 下载/加载完整MNIST
    full_train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    full_test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # 随机抽取子集（固定随机种子以保证可复现）
    generator = torch.Generator().manual_seed(42)
    train_indices = torch.randperm(len(full_train_dataset), generator=generator)[:train_size]
    test_indices = torch.randperm(len(full_test_dataset), generator=generator)[:test_size]

    train_subset = Subset(full_train_dataset, train_indices)
    test_subset = Subset(full_test_dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"训练集样本数: {len(train_subset)}, 测试集样本数: {len(test_subset)}")
    return train_loader, test_loader


# ============================================================
# 实验一：初始化方法对梯度传播的影响
# ============================================================

class DeepMLP_Tanh(nn.Module):
    """
    深层MLP网络（使用Tanh激活函数）

    结构: Input(784) → 20个隐藏层(256, Tanh) → Output(10)
    用于实验一：比较不同初始化方法对梯度传播的影响。
    """

    def __init__(self, init_method="normal"):
        """
        参数:
            init_method: "normal" 表示 N(0,1) 随机初始化,
                         "xavier" 表示 Xavier 初始化
        """
        super(DeepMLP_Tanh, self).__init__()

        self.layers = nn.ModuleList()

        # 输入层 → 第一个隐藏层
        layer = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.layers.append(layer)

        # 20个隐藏层
        for _ in range(NUM_HIDDEN_LAYERS):
            layer = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
            self.layers.append(layer)

        # 最后一个隐藏层 → 输出层
        output_layer = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)
        self.layers.append(output_layer)

        # 根据指定方法初始化参数
        self._init_weights(init_method)

    def _init_weights(self, method):
        """按指定方法初始化所有权重"""
        for m in self.layers:
            if method == "normal":
                init.normal_(m.weight, mean=0.0, std=1.0)
            elif method == "xavier":
                init.xavier_normal_(m.weight)
            # 偏置初始化为0
            if m.bias is not None:
                init.constant_(m.bias, 0.0)

    def forward(self, x):
        """
        前向传播：依次通过各层，前N-1层后接Tanh激活，最后一层直接输出。

        返回:
            out: 网络输出
            activations: 每层激活值列表（用于梯度分析）
        """
        activations = []
        x = x.view(x.size(0), -1)  # 展平为 (batch, 784)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # 除最后一层外，都经过Tanh
                x = torch.tanh(x)
            activations.append(x)

        return x, activations

    def get_gradient_norms(self):
        """获取每层权重的梯度范数（L2范数）"""
        grad_norms = []
        for i, layer in enumerate(self.layers):
            if layer.weight.grad is not None:
                norm = layer.weight.grad.norm(2).item()
            else:
                norm = 0.0
            grad_norms.append(norm)
        return grad_norms


def experiment_1_gradient_propagation():
    """
    实验一：初始化方法对梯度传播的影响

    比较 Normal(0,1) 与 Xavier 初始化下：
    - 一次前向传播 + 一次反向传播后，各层梯度范数的分布
    """
    print("\n" + "=" * 60)
    print("实验一：初始化方法对梯度传播的影响")
    print("=" * 60)

    # 创建两个网络：分别使用 Normal 和 Xavier 初始化
    model_normal = DeepMLP_Tanh(init_method="normal").to(DEVICE)
    model_xavier = DeepMLP_Tanh(init_method="xavier").to(DEVICE)

    # 使用一个随机batch模拟前向+反向传播
    dummy_input = torch.randn(BATCH_SIZE, INPUT_DIM).to(DEVICE)
    dummy_target = torch.randint(0, OUTPUT_DIM, (BATCH_SIZE,)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    results = {}
    for name, model in [("Normal(0,1)", model_normal), ("Xavier", model_xavier)]:
        model.zero_grad()
        output, _ = model(dummy_input)
        loss = criterion(output, dummy_target)
        loss.backward()

        grad_norms = model.get_gradient_norms()
        results[name] = grad_norms
        print(f"\n{name} 初始化 - 各层梯度范数:")
        for i, norm in enumerate(grad_norms):
            layer_type = "Input" if i == 0 else ("Hidden" + str(i) if i <= NUM_HIDDEN_LAYERS else "Output")
            print(f"  Layer {i:2d} ({layer_type:8s}): {norm:.6f}")

    # ---- 绘图 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (name, grad_norms) in zip(axes, results.items()):
        layers = list(range(len(grad_norms)))
        ax.plot(layers, grad_norms, 'o-', color='steelblue', markersize=4, linewidth=1.2)
        ax.set_xlabel("Layer Index", fontsize=12)
        ax.set_ylabel("Gradient Norm (L2)", fontsize=12)
        ax.set_title(f"Gradient Norm vs Layer Index\n({name} Initialization)", fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # 对数坐标便于观察梯度消失/爆炸

    plt.suptitle("实验一：初始化方法对梯度传播的影响", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("experiment_1_gradient_norm.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("\n[实验一] 图像已保存为 experiment_1_gradient_norm.png")

    return results


# ============================================================
# 实验二：残差连接对深层网络训练的影响
# ============================================================

class ResidualBlock(nn.Module):
    """
    残差块结构:
        y = F(x) + x
        其中 F(x) = Linear(256,256) → ReLU

    残差连接 (Skip Connection) 缓解了深层网络的梯度消失问题。

    注意：线性层被初始化为接近零的小权重，使得训练初期 F(x)≈0，
    整个块近似恒等映射，有助于深层残差网络的稳定训练。
    """

    def __init__(self, dim=256):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(dim, dim)
        self.relu = nn.ReLU()

        # 将残差分支初始化为接近零：训练初期 F(x)≈0，整个块近似恒等映射
        # 这是深度ResNet的标准初始化策略
        init.normal_(self.linear.weight, mean=0.0, std=0.001)
        if self.linear.bias is not None:
            init.constant_(self.linear.bias, 0.0)

    def forward(self, x):
        identity = x                          # Skip Connection: 保存输入
        out = self.linear(x)                  # Linear变换
        out = self.relu(out)                  # ReLU激活
        out = out + identity                  # 残差连接: F(x) + x
        return out


class PlainMLP(nn.Module):
    """
    网络A：普通MLP（无残差连接）

    结构: Input(784) → Linear(784,256) → 20个隐藏层(256, ReLU) → Output(10)
    """

    def __init__(self):
        super(PlainMLP, self).__init__()

        self.input_layer = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.relu = nn.ReLU()

        # 20个普通隐藏层
        self.hidden_layers = nn.ModuleList()
        for _ in range(NUM_HIDDEN_LAYERS):
            self.hidden_layers.append(nn.Linear(HIDDEN_DIM, HIDDEN_DIM))

        self.output_layer = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)
        self._init_weights()

    def _init_weights(self):
        """使用Xavier初始化所有层"""
        for m in [self.input_layer] + list(self.hidden_layers) + [self.output_layer]:
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        x = self.output_layer(x)
        return x


class ResidualMLP(nn.Module):
    """
    网络B：残差MLP（含Skip Connection）

    结构: Input(784) → Linear(784,256) → 20个Residual Block → Output(10)
    """

    def __init__(self):
        super(ResidualMLP, self).__init__()

        self.input_layer = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.relu = nn.ReLU()

        # 20个残差块
        self.residual_blocks = nn.ModuleList()
        for _ in range(NUM_HIDDEN_LAYERS):
            self.residual_blocks.append(ResidualBlock(HIDDEN_DIM))

        self.output_layer = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)

        # Xavier初始化输入层和输出层
        init.xavier_normal_(self.input_layer.weight)
        init.constant_(self.input_layer.bias, 0.0)
        init.xavier_normal_(self.output_layer.weight)
        init.constant_(self.output_layer.bias, 0.0)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.input_layer(x))
        for block in self.residual_blocks:
            x = block(x)
        x = self.output_layer(x)
        return x


def compute_gradient_norms(model):
    """计算模型所有层的梯度范数"""
    norms = []
    for name, param in model.named_parameters():
        if param.grad is not None and 'weight' in name:
            norms.append(param.grad.norm(2).item())
    return norms


def train_one_epoch(model, train_loader, optimizer, criterion, clip_grad=True):
    """训练一个epoch，返回平均loss和准确率

    参数:
        clip_grad: 是否使用梯度裁剪（默认True，防止深层网络梯度爆炸）
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        # 梯度裁剪：防止深层网络梯度爆炸
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, test_loader, criterion):
    """评估模型在测试集上的表现"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def record_gradient_norms_after_batch(model, train_loader):
    """在一个batch后记录各层梯度范数"""
    model.train()
    data, target = next(iter(train_loader))
    data, target = data.to(DEVICE), target.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    model.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()

    return compute_gradient_norms(model)


def experiment_2_residual_connection(train_loader, test_loader):
    """
    实验二：残差连接对深层网络训练的影响

    比较普通MLP与残差MLP在：
    - 训练Loss变化
    - 训练/测试准确率变化
    - 梯度范数变化
    """
    print("\n" + "=" * 60)
    print("实验二：残差连接对深层网络训练的影响")
    print("=" * 60)

    num_epochs = 20

    # 创建两个模型
    plain_model = PlainMLP().to(DEVICE)
    residual_model = ResidualMLP().to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    # SGD优化器: lr=0.05, momentum=0.9
    plain_optimizer = torch.optim.SGD(
        plain_model.parameters(), lr=0.05, momentum=0.9
    )
    residual_optimizer = torch.optim.SGD(
        residual_model.parameters(), lr=0.05, momentum=0.9
    )

    # 记录训练历史
    history = {
        "plain": {"train_loss": [], "train_acc": [], "test_acc": [], "grad_norms": []},
        "residual": {"train_loss": [], "train_acc": [], "test_acc": [], "grad_norms": []},
    }

    print(f"\n开始训练 ({num_epochs} epochs)...")
    for epoch in range(num_epochs):
        # ---- 训练普通MLP ----
        plain_loss, plain_train_acc = train_one_epoch(
            plain_model, train_loader, plain_optimizer, criterion
        )
        _, plain_test_acc = evaluate(plain_model, test_loader, criterion)

        history["plain"]["train_loss"].append(plain_loss)
        history["plain"]["train_acc"].append(plain_train_acc)
        history["plain"]["test_acc"].append(plain_test_acc)

        # ---- 训练残差MLP ----
        residual_loss, residual_train_acc = train_one_epoch(
            residual_model, train_loader, residual_optimizer, criterion
        )
        _, residual_test_acc = evaluate(residual_model, test_loader, criterion)

        history["residual"]["train_loss"].append(residual_loss)
        history["residual"]["train_acc"].append(residual_train_acc)
        history["residual"]["test_acc"].append(residual_test_acc)

        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Plain MLP - Loss: {plain_loss:.4f}, Train Acc: {plain_train_acc:.4f}, Test Acc: {plain_test_acc:.4f} | "
              f"Residual MLP - Loss: {residual_loss:.4f}, Train Acc: {residual_train_acc:.4f}, Test Acc: {residual_test_acc:.4f}")

    # 记录最终梯度范数（每个模型在一个batch后的梯度范数）
    history["plain"]["grad_norms"] = record_gradient_norms_after_batch(plain_model, train_loader)
    history["residual"]["grad_norms"] = record_gradient_norms_after_batch(residual_model, train_loader)

    # ---- 绘图 ----
    epochs = list(range(1, num_epochs + 1))

    # 图1：训练Loss对比
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(epochs, history["plain"]["train_loss"], 'o-', color='coral', label='Plain MLP',
             markersize=4, linewidth=1.5)
    ax1.plot(epochs, history["residual"]["train_loss"], 's-', color='steelblue', label='Residual MLP',
             markersize=4, linewidth=1.5)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Train Loss", fontsize=12)
    ax1.set_title("图1：普通MLP vs 残差MLP — 训练Loss对比", fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiment_2_train_loss.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 图2：测试准确率对比
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(epochs, history["plain"]["test_acc"], 'o-', color='coral', label='Plain MLP',
             markersize=4, linewidth=1.5)
    ax2.plot(epochs, history["residual"]["test_acc"], 's-', color='steelblue', label='Residual MLP',
             markersize=4, linewidth=1.5)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Test Accuracy", fontsize=12)
    ax2.set_title("图2：普通MLP vs 残差MLP — 测试准确率对比", fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiment_2_test_accuracy.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 图3：梯度范数对比
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    plain_norms = history["plain"]["grad_norms"]
    residual_norms = history["residual"]["grad_norms"]
    ax3.plot(range(len(plain_norms)), plain_norms, 'o-', color='coral', label='Plain MLP',
             markersize=4, linewidth=1.2)
    ax3.plot(range(len(residual_norms)), residual_norms, 's-', color='steelblue', label='Residual MLP',
             markersize=4, linewidth=1.2)
    ax3.set_xlabel("Layer Index (weight parameters)", fontsize=12)
    ax3.set_ylabel("Gradient Norm (L2)", fontsize=12)
    ax3.set_title("图3：普通MLP vs 残差MLP — 梯度范数对比", fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    plt.tight_layout()
    plt.savefig("experiment_2_gradient_norm.png", dpi=150, bbox_inches='tight')
    plt.close()

    print("\n[实验二] 图像已保存:")
    print("  - experiment_2_train_loss.png")
    print("  - experiment_2_test_accuracy.png")
    print("  - experiment_2_gradient_norm.png")

    # 打印汇总结果
    print("\n" + "-" * 50)
    print("实验结果汇总:")
    print(f"  普通MLP   - 最终训练Loss: {history['plain']['train_loss'][-1]:.4f}, "
          f"最终测试准确率: {history['plain']['test_acc'][-1]:.4f}")
    print(f"  残差MLP   - 最终训练Loss: {history['residual']['train_loss'][-1]:.4f}, "
          f"最终测试准确率: {history['residual']['test_acc'][-1]:.4f}")

    return history


# ============================================================
# 主函数
# ============================================================
def main():
    """主函数：依次运行实验一和实验二"""
    print("=" * 60)
    print("机器学习上机实验12：神经网络-2")
    print("基于PyTorch的深层神经网络训练机制研究")
    print("=" * 60)

    # 加载数据
    train_loader, test_loader = load_mnist_data(TRAIN_SIZE, TEST_SIZE)

    # 实验一：初始化方法对梯度传播的影响
    experiment_1_gradient_propagation()

    # 实验二：残差连接对深层网络训练的影响
    experiment_2_residual_connection(train_loader, test_loader)

    print("\n" + "=" * 60)
    print("全部实验完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

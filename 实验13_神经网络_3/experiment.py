"""
机器学习上机实验13-14：神经网络-3
基于PyTorch的Transformer Encoder图像分类实现

实验内容：
  任务1：数据加载 — MNIST数据集
  任务2：构建TransformerClassifier
  任务3：完成Forward函数
  任务4：模型训练
  任务5：查看模型结构
  任务6：Head数量实验 (nhead=1,2,4,8)
  任务7：Position Encoding消融实验
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 中文字体支持
_chinese_fonts = [f.name for f in fm.fontManager.ttflist]
if 'Heiti TC' in _chinese_fonts:
    plt.rcParams['font.family'] = 'Heiti TC'
elif 'Arial Unicode MS' in _chinese_fonts:
    plt.rcParams['font.family'] = 'Arial Unicode MS'
plt.rcParams['axes.unicode_minus'] = False

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ============================================================
# 全局配置
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 3
LEARNING_RATE = 0.001

print(f"使用设备: {DEVICE}")
print(f"PyTorch 版本: {torch.__version__}")


# ============================================================
# 任务1：数据加载
# ============================================================
def load_mnist_data(batch_size=128):
    """
    加载MNIST数据集并执行标准化。

    预处理: ToTensor() + Normalize((0.5,), (0.5,))
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"训练集样本数: {len(train_dataset)}, 测试集样本数: {len(test_dataset)}")
    print(f"Batch Size: {batch_size}")
    return train_loader, test_loader


# ============================================================
# 任务2&3：构建TransformerClassifier + Forward
# ============================================================
class TransformerClassifier(nn.Module):
    """
    Transformer Encoder 图像分类器

    网络结构:
      Input (batch,1,28,28)
        → reshape → (batch,28,28)
        → Linear Embedding (28→64) → (batch,28,64)
        → + Position Encoding (1,28,64)
        → Transformer Encoder × 2 (d_model=64, nhead=4)
        → Mean Pooling → (batch,64)
        → Linear (64→10) → (batch,10)
    """

    def __init__(self, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, num_classes=10):
        """
        参数:
            d_model:          Token维度 (默认64)
            nhead:            多头注意力头数 (默认4)
            num_layers:       Transformer Encoder层数 (默认2)
            dim_feedforward:  前馈网络隐藏层维度 (默认128)
            num_classes:      分类类别数 (默认10)
        """
        super(TransformerClassifier, self).__init__()

        self.d_model = d_model

        # Step 1: Embedding层 — 将每行28维像素映射到64维特征
        self.embedding = nn.Linear(28, d_model)

        # Step 2: 可学习位置编码 (1, 28, 64)
        self.pos_embedding = nn.Parameter(torch.randn(1, 28, d_model))

        # Step 3: Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Step 4: 分类头
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        前向传播:
          输入图片 → 去掉通道维 → Embedding → +Position Encoding
          → Transformer → Mean Pooling → 分类器

        参数:
            x: (batch, 1, 28, 28)  输入图像
        返回:
            out: (batch, 10)  分类logits
        """
        # (batch, 1, 28, 28) → (batch, 28, 28)
        x = x.squeeze(1)

        # (batch, 28, 28) → (batch, 28, 64)  Embedding
        x = self.embedding(x)

        # + Position Encoding
        x = x + self.pos_embedding

        # Transformer Encoder: (batch, 28, 64) → (batch, 28, 64)
        x = self.transformer_encoder(x)

        # Mean Pooling: (batch, 28, 64) → (batch, 64)
        x = x.mean(dim=1)

        # 分类器: (batch, 64) → (batch, 10)
        x = self.classifier(x)

        return x


# ============================================================
# 训练与评估函数
# ============================================================
def train_one_epoch(model, train_loader, optimizer, criterion):
    """训练一个epoch，返回平均loss和准确率"""
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
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, test_loader, criterion):
    """评估模型在测试集上的表现"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, target in test_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        output = model(data)
        loss = criterion(output, target)

        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)

    return total_loss / total, correct / total


def train_model(model, train_loader, test_loader, epochs=EPOCHS, lr=LEARNING_RATE, verbose=True):
    """
    完整训练流程，返回训练历史。

    参数:
        model:       模型实例
        train_loader: 训练数据加载器
        test_loader:  测试数据加载器
        epochs:      训练轮数
        lr:          学习率
        verbose:     是否打印每轮信息

    返回:
        history: dict, 包含 train_loss, train_acc, test_acc 列表
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        test_loss, test_acc = evaluate(model, test_loader, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        if verbose:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Test Acc: {test_acc:.4f}")

    return history


# ============================================================
# 任务4：模型训练 (基础配置)
# ============================================================
def task_4_train_baseline(train_loader, test_loader):
    """
    任务4：使用默认配置训练Transformer分类器
    配置: d_model=64, nhead=4, num_layers=2, epochs=3, lr=0.001
    """
    print("\n" + "=" * 60)
    print("任务4：Transformer Encoder 模型训练")
    print("=" * 60)
    print(f"配置: d_model=64, nhead=4, num_layers=2")
    print(f"训练参数: epochs={EPOCHS}, lr={LEARNING_RATE}, batch_size={BATCH_SIZE}")

    model = TransformerClassifier(
        d_model=64, nhead=4, num_layers=2, dim_feedforward=128
    ).to(DEVICE)

    # 任务5：查看模型结构
    print(f"\n【任务5】模型结构:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n可训练参数总数: {total_params:,}")

    # 打印参数详情
    print("\n参数详情:")
    for name, param in model.named_parameters():
        print(f"  {name}: shape={list(param.shape)}, 参数量={param.numel():,}")

    history = train_model(model, train_loader, test_loader, epochs=EPOCHS, lr=LEARNING_RATE)

    print(f"\n训练结果:")
    print(f"  最终训练Loss: {history['train_loss'][-1]:.4f}")
    print(f"  最终训练准确率: {history['train_acc'][-1]:.4f}")
    print(f"  最终测试准确率: {history['test_acc'][-1]:.4f}")

    return model, history


# ============================================================
# 任务6：Head数量实验
# ============================================================
def task_6_head_experiment(train_loader, test_loader):
    """
    任务6：Head数量实验

    分别测试 nhead = 1, 2, 4, 8
    每个配置训练1个Epoch，记录Test Accuracy
    """
    print("\n" + "=" * 60)
    print("任务6：Head数量实验")
    print("=" * 60)

    head_list = [1, 2, 4, 8]
    results = {}

    for nhead in head_list:
        print(f"\n--- 训练 nhead={nhead} (1 Epoch) ---")
        model = TransformerClassifier(
            d_model=64, nhead=nhead, num_layers=2, dim_feedforward=128
        ).to(DEVICE)

        history = train_model(model, train_loader, test_loader, epochs=1, lr=LEARNING_RATE)
        results[nhead] = history
        print(f"  nhead={nhead}: Train Loss={history['train_loss'][-1]:.4f}, "
              f"Test Acc={history['test_acc'][-1]:.4f}")

    # 绘图
    fig, ax = plt.subplots(figsize=(8, 5))
    heads_str = [str(h) for h in head_list]
    test_accs = [results[h]["test_acc"][-1] for h in head_list]
    train_losses = [results[h]["train_loss"][-1] for h in head_list]

    x = np.arange(len(head_list))
    width = 0.35

    bars1 = ax.bar(x - width/2, test_accs, width, color='steelblue', label='Test Accuracy')
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, train_losses, width, color='coral', label='Train Loss')

    ax.set_xlabel("nhead", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12, color='steelblue')
    ax2.set_ylabel("Train Loss", fontsize=12, color='coral')
    ax.set_title("Head数量对模型性能的影响 (1 Epoch)", fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(heads_str)

    # 在柱状图上标注数值
    for bar, val in zip(bars1, test_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, train_losses):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig("experiment_head_count.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[任务6] 图像已保存为 experiment_head_count.png")

    return results


# ============================================================
# 任务7：Position Encoding 消融实验
# ============================================================
class TransformerClassifierNoPE(nn.Module):
    """不含Position Encoding的Transformer分类器（用于消融实验）"""

    def __init__(self, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, num_classes=10):
        super(TransformerClassifierNoPE, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(28, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = x.squeeze(1)
        x = self.embedding(x)
        # 不加 Position Encoding
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x


def task_7_pe_ablation(train_loader, test_loader):
    """
    任务7：Position Encoding 消融实验

    实验A: 带 Position Encoding (x = x + self.pos_embedding)
    实验B: 不带 Position Encoding
    """
    print("\n" + "=" * 60)
    print("任务7：Position Encoding 消融实验")
    print("=" * 60)

    # 实验A：带PE
    print("\n--- 实验A: 带 Position Encoding (3 Epochs) ---")
    model_with_pe = TransformerClassifier(
        d_model=64, nhead=4, num_layers=2, dim_feedforward=128
    ).to(DEVICE)
    history_with_pe = train_model(model_with_pe, train_loader, test_loader,
                                  epochs=EPOCHS, lr=LEARNING_RATE)

    # 实验B：不带PE
    print("\n--- 实验B: 不带 Position Encoding (3 Epochs) ---")
    model_no_pe = TransformerClassifierNoPE(
        d_model=64, nhead=4, num_layers=2, dim_feedforward=128
    ).to(DEVICE)
    history_no_pe = train_model(model_no_pe, train_loader, test_loader,
                                epochs=EPOCHS, lr=LEARNING_RATE)

    # ---- 绘图 ----
    epochs_range = list(range(1, EPOCHS + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 图1：Test Accuracy 对比
    ax1 = axes[0]
    ax1.plot(epochs_range, history_with_pe["test_acc"], 's-', color='steelblue',
             markersize=6, linewidth=2, label='With Position Encoding')
    ax1.plot(epochs_range, history_no_pe["test_acc"], 'o-', color='coral',
             markersize=6, linewidth=2, label='Without Position Encoding')
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Test Accuracy", fontsize=12)
    ax1.set_title("Position Encoding 对测试准确率的影响", fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 图2：Train Loss 对比
    ax2 = axes[1]
    ax2.plot(epochs_range, history_with_pe["train_loss"], 's-', color='steelblue',
             markersize=6, linewidth=2, label='With Position Encoding')
    ax2.plot(epochs_range, history_no_pe["train_loss"], 'o-', color='coral',
             markersize=6, linewidth=2, label='Without Position Encoding')
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Train Loss", fontsize=12)
    ax2.set_title("Position Encoding 对训练Loss的影响", fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("experiment_pe_ablation.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[任务7] 图像已保存为 experiment_pe_ablation.png")

    # 打印汇总
    print("\n" + "-" * 50)
    print("Position Encoding 消融实验结果汇总:")
    print(f"  带PE    - 最终 Test Acc: {history_with_pe['test_acc'][-1]:.4f}, "
          f"Train Loss: {history_with_pe['train_loss'][-1]:.4f}")
    print(f"  不带PE  - 最终 Test Acc: {history_no_pe['test_acc'][-1]:.4f}, "
          f"Train Loss: {history_no_pe['train_loss'][-1]:.4f}")

    return history_with_pe, history_no_pe


# ============================================================
# 主函数
# ============================================================
def main():
    """主函数：依次完成所有实验任务"""
    print("=" * 60)
    print("机器学习上机实验13-14：神经网络-3")
    print("Transformer Encoder 图像分类实现")
    print("=" * 60)

    # 任务1：数据加载
    print("\n" + "=" * 60)
    print("任务1：MNIST 数据加载")
    print("=" * 60)
    train_loader, test_loader = load_mnist_data(BATCH_SIZE)

    # 任务4 & 5：模型训练 + 查看模型结构
    baseline_model, baseline_history = task_4_train_baseline(train_loader, test_loader)

    # 任务6：Head数量实验
    head_results = task_6_head_experiment(train_loader, test_loader)

    # 任务7：Position Encoding消融实验
    pe_results = task_7_pe_ablation(train_loader, test_loader)

    # ============================================================
    # 思考题回答
    # ============================================================
    print("\n" + "=" * 60)
    print("思考题回答")
    print("=" * 60)

    print("""
【任务5 - 模型结构】
  Q: 模型中共有几个Transformer Encoder层？
  A: 2层 (num_layers=2)
  Q: 每个Token最终维度是多少？
  A: 64维 (d_model=64)

【任务6 - Head数量实验】
  Q: Head数增加后性能是否一定提升？为什么？
  A: 不一定。Head数增加意味着每个Head处理的维度更小(d_model/nhead)，
     更多的Head可以关注不同子空间的特征，理论上有助于捕获更丰富的模式。
     但Head数并非越大越好：
     1. Head数过多会导致每个Head维度过小，表达能力下降
     2. 计算量增加，可能过拟合
     3. 对于简单任务(如MNIST)，少量Head可能已经足够
     最佳Head数需要通过实验确定，与本实验中d_model=64匹配的常见选择是nhead=4或8。

【任务7 - Position Encoding消融实验】
  Q: 为什么要加位置编码？
  A: Transformer的Self-Attention机制本身是置换不变的(permutation-invariant)，
     即它对输入序列的顺序不敏感。对于图像而言，行的顺序非常重要——
     第1行(图片顶部)和第28行(图片底部)包含完全不同的空间信息。
     Position Encoding为每个Token添加了位置信息，使模型能够感知Token的顺序，
     从而理解图像的空间结构。没有PE，模型无法区分不同位置的行。

     消融实验中，不加PE通常会导致性能下降，因为模型失去了空间位置信息。
""")

    print("\n" + "=" * 60)
    print("全部实验完成！")
    print("生成文件:")
    print("  - experiment_head_count.png   (Head数量实验)")
    print("  - experiment_pe_ablation.png  (PE消融实验)")
    print("=" * 60)


if __name__ == "__main__":
    main()

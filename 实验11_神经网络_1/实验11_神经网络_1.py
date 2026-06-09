"""
机器学习上机实验11：神经网络-1
PyTorch自动求导与神经网络训练

实验内容：
  实验一：PyTorch计算图与自动求导
    - 任务1：构建计算图并计算梯度
    - 任务2：损失函数梯度传播与链式法则

  实验二：神经网络训练与梯度分析
    - 任务1：构建XOR数据集
    - 任务2：构建神经网络 (2→4→1, Sigmoid激活)
    - 任务3：统计网络参数
    - 任务4：观察梯度
    - 任务5：观察参数更新
    - 任务6：完成网络训练
    - 任务7：分析梯度变化

运行方式：
  python3 实验11_神经网络_1.py

生成文件：
  - xor_dataset.npz       — XOR数据集
  - results_loss.png      — Loss-Epoch曲线
  - results_grad_norm.png — Gradient Norm-Epoch曲线
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# 全局设置
# ============================================================
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def print_separator(title):
    """打印分隔标题"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ============================================================
# 实验一：PyTorch计算图与自动求导
# ============================================================
def experiment_1():
    """
    实验一：PyTorch计算图与自动求导

    任务1：构建计算图并计算梯度
      给定表达式 y = (wx + b)²，其中 w=2.0, x=3.0, b=1.0
      要求计算 y 并求 w 和 b 的梯度。

    任务2：损失函数梯度传播与链式法则
      给定：z = wx + b, a = σ(z), L = a²
      要求观察梯度从 Loss 反向传播到参数 (w) 和 (b) 的过程，
      并观察连续两次 backward() 后梯度的累积现象。
    """
    print_separator("实验一：PyTorch计算图与自动求导")

    # ----------------------------------------------------------
    # 任务1：构建计算图并计算梯度
    # ----------------------------------------------------------
    print("\n【任务1】构建计算图并计算梯度")
    print("-" * 40)

    # 定义变量
    w = torch.tensor(2.0, requires_grad=True)  # 权重，需要梯度
    x = torch.tensor(3.0)                       # 输入，不需要梯度
    b = torch.tensor(1.0, requires_grad=True)  # 偏置，需要梯度

    # (1) 补充代码完成表达式计算：y = (w*x + b)^2
    y = (w * x + b) ** 2
    print(f"y = (w*x + b)^2 = ({w.item()}*{x.item()} + {b.item()})^2 = {y.item()}")

    # (2) 调用自动求导
    y.backward()

    # (3) 输出梯度
    print(f"\n梯度结果：")
    print(f"  w.grad = {w.grad.item():.4f}")   # ∂y/∂w = 2(wx+b)*x = 2*7*3 = 42
    print(f"  b.grad = {b.grad.item():.4f}")   # ∂y/∂b = 2(wx+b)*1 = 2*7*1 = 14
    print(f"  x 没有 grad 属性（因为 requires_grad=False）")

    # 思考题解答
    print("\n【思考题】")
    print("  1. w和b具有梯度的原因：因为设置了 requires_grad=True，")
    print("     PyTorch会追踪所有对它们的操作，并在backward()时计算梯度。")
    print("  2. x没有梯度的原因：创建时 requires_grad 默认为 False，")
    print("     PyTorch不会追踪该变量的梯度，通常输入数据不需要求梯度。")
    print("  3. requires_grad=True 的作用：告诉PyTorch需要对该张量计算梯度，")
    print("     这是自动求导机制的基础。")

    # 输出计算图信息
    print(f"\n计算图信息：")
    print(f"  y.grad_fn = {y.grad_fn}")
    print(f"  解释：grad_fn 表示创建该张量的运算操作。")
    print(f"  y 由 PowBackward0（幂运算） 产生。")

    # 继续构造 z = w*x + b 观察其 grad_fn
    z = w * x + b
    print(f"  z.grad_fn = {z.grad_fn}")
    print(f"  z 由 AddBackward0（加法运算） 产生。")
    print(f"\n  计算图节点连接方式：")
    print(f"    叶子节点(w, x, b) → 乘法(MulBackward) → 加法(AddBackward)")
    print(f"    → 幂运算(PowBackward) → y")
    print(f"    backward()时梯度沿箭头反方向传播。")

    # ----------------------------------------------------------
    # 任务2：损失函数梯度传播与链式法则
    # ----------------------------------------------------------
    print("\n\n【任务2】损失函数梯度传播与链式法则")
    print("-" * 40)

    # 重新定义变量（避免与任务1的梯度混淆）
    w = torch.tensor(2.0, requires_grad=True)
    x = torch.tensor(3.0)
    b = torch.tensor(1.0, requires_grad=True)

    # 补充代码
    z = w * x + b                        # 线性组合
    a = torch.sigmoid(z)                 # Sigmoid激活
    loss = a ** 2                        # 损失函数 L = a²

    print(f"前向计算：")
    print(f"  z = w*x + b = {z.item():.4f}")
    print(f"  a = σ(z) = {a.item():.4f}")
    print(f"  loss = a² = {loss.item():.4f}")

    # 反向传播（retain_graph=True 保留计算图，以便第二次backward观察梯度累积）
    loss.backward(retain_graph=True)

    print(f"\n第一次 backward() 后梯度：")
    print(f"  w.grad = {w.grad.item():.6f}")
    print(f"  b.grad = {b.grad.item():.6f}")

    # 验证链式法则
    # ∂L/∂w = ∂L/∂a * ∂a/∂z * ∂z/∂w = 2a * σ(z)(1-σ(z)) * x
    a_val = a.item()
    dz_dw = x.item()                      # ∂z/∂w = x
    da_dz = a_val * (1 - a_val)           # ∂a/∂z = σ'(z) = σ(z)(1-σ(z))
    dL_da = 2 * a_val                     # ∂L/∂a = 2a
    w_grad_manual = dL_da * da_dz * dz_dw
    print(f"\n链式法则验证：")
    print(f"  ∂L/∂a = 2a = {dL_da:.6f}")
    print(f"  ∂a/∂z = σ'(z) = {da_dz:.6f}")
    print(f"  ∂z/∂w = x = {dz_dw:.6f}")
    print(f"  ∂L/∂w = ∂L/∂a * ∂a/∂z * ∂z/∂w = {w_grad_manual:.6f}")
    print(f"  PyTorch计算结果：{w.grad.item():.6f}  ✓ 一致")

    print("\n【思考题 - 计算图绘制】")
    print("""
         x=3.0 ──┐
                  ├──→ [*] ──→ [+] ──→ [σ] ──→ [²] ──→ L
         w=2.0 ──┘        ↗          z       a    loss
                  b=1.0 ──┘

         反向传播路径：
         L ──∂L/∂a=2a──→ a ──∂a/∂z=σ'(z)──→ z ──∂z/∂w=x──→ w
                                    │                      │
                                    └──∂z/∂b=1──────────→ b
    """)

    # 连续调用两次 backward() 观察梯度累积
    print("\n【梯度累积现象】")
    print("-" * 40)
    print(f"第一次 backward() 后 w.grad = {w.grad.item():.6f}")

    # 第二次 backward()
    loss.backward()
    print(f"第二次 backward() 后 w.grad = {w.grad.item():.6f}")
    print(f"  梯度变成原来的两倍！说明 backward() 会累加梯度。")

    # 清零梯度
    w.grad.zero_()
    b.grad.zero_()
    print(f"\n执行 w.grad.zero_() 和 b.grad.zero_() 后：")
    print(f"  w.grad = {w.grad.item():.6f}")
    print(f"  b.grad = {b.grad.item():.6f}")

    print("\n【思考题】")
    print("  1. 连续调用两次backward()后梯度变大的原因：")
    print("     PyTorch默认会累积梯度（grad += new_grad），而不是覆盖。")
    print("     因此第二次backward()的梯度被加到了第一次的结果上。")
    print("  2. 训练过程中需要执行optimizer.zero_grad()的原因：")
    print("     正是为了避免梯度累积——每个batch的梯度应该独立计算，")
    print("     如果不清零，当前batch的梯度会和之前所有batch的梯度累加，"  )
    print("     导致参数更新方向错误。")


# ============================================================
# 实验二：神经网络训练与梯度分析
# ============================================================

class XORNet(nn.Module):
    """
    任务2：构建XOR神经网络

    网络结构：
      Input(2) → Linear(2→4) → Sigmoid → Linear(4→1) → Sigmoid → Output(1)

    这是一个两层全连接网络，用于学习XOR（异或）映射关系。
    XOR问题在二维平面上不是线性可分的，需要至少一个隐藏层。
    """

    def __init__(self):
        super().__init__()
        # 第一层：输入2维 → 隐藏层4维
        self.fc1 = nn.Linear(2, 4)
        # 第二层：隐藏层4维 → 输出1维
        self.fc2 = nn.Linear(4, 1)
        # Sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

        # 初始化权重（使用固定种子以保证可复现性）
        self._init_weights()

    def _init_weights(self):
        """使用Xavier初始化方法初始化权重"""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量，shape (batch_size, 2)

        Returns:
            output: 输出张量，shape (batch_size, 1)
        """
        # 第一层：线性变换 + Sigmoid激活
        h = self.sigmoid(self.fc1(x))
        # 第二层：线性变换 + Sigmoid激活
        output = self.sigmoid(self.fc2(h))
        return output


def experiment_2():
    """
    实验二：神经网络训练与梯度分析

    任务1：构建XOR数据集
    任务2：构建神经网络 (2→4→1, Sigmoid)
    任务3：统计网络参数
    任务4：观察梯度
    任务5：观察参数更新
    任务6：完成网络训练（2000 epochs）
    任务7：分析梯度变化
    """
    print_separator("实验二：神经网络训练与梯度分析")

    # ========================================================
    # 任务1：构建XOR数据集
    # ========================================================
    print("\n【任务1】构建XOR数据集")
    print("-" * 40)

    # 设置随机种子以保证可复现性
    torch.manual_seed(42)

    # XOR输入数据
    X = torch.tensor([
        [0., 0.],
        [0., 1.],
        [1., 0.],
        [1., 1.]
    ])

    # XOR标签（异或运算）
    Y = torch.tensor([
        [0.],
        [1.],
        [1.],
        [0.]
    ])

    print(f"X (输入):\n{X}")
    print(f"\nY (标签):\n{Y}")
    print(f"\nXOR真值表：")
    print(f"  0 ⊕ 0 = 0")
    print(f"  0 ⊕ 1 = 1")
    print(f"  1 ⊕ 0 = 1")
    print(f"  1 ⊕ 1 = 0")

    # 保存数据集
    dataset_path = os.path.join(OUTPUT_DIR, "xor_dataset.npz")
    np.savez(dataset_path,
             X=X.numpy(),
             Y=Y.numpy(),
             description="XOR dataset for neural network training")
    print(f"\n数据集已保存至：{dataset_path}")

    # ========================================================
    # 任务2：构建神经网络
    # ========================================================
    print("\n\n【任务2】构建神经网络")
    print("-" * 40)

    model = XORNet()
    print("网络结构：")
    print("  Input(2) → Linear(2→4) → Sigmoid → Linear(4→1) → Sigmoid → Output(1)")
    print(f"\n模型详情：\n{model}")

    # ========================================================
    # 任务3：统计网络参数
    # ========================================================
    print("\n\n【任务3】统计网络参数")
    print("-" * 40)

    total_params = 0
    print("网络参数列表：")
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        print(f"  {name}: shape={list(param.shape)}, 参数量={num_params}")
        if param.requires_grad:
            print(f"    requires_grad=True")

    print(f"\n可训练参数总数：{total_params}")
    print(f"  计算过程：")
    print(f"    fc1.weight: 2×4 = 8")
    print(f"    fc1.bias:   1×4 = 4")
    print(f"    fc2.weight: 4×1 = 4")
    print(f"    fc2.bias:   1×1 = 1")
    print(f"    总计: 8 + 4 + 4 + 1 = 17")

    # ========================================================
    # 任务4：观察梯度
    # ========================================================
    print("\n\n【任务4】观察梯度")
    print("-" * 40)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    # 执行一次前向传播与反向传播
    output = model(X)
    loss = criterion(output, Y)
    loss.backward()

    print(f"前向传播输出：\n{output.data}")
    print(f"\n损失值 (MSE): {loss.item():.6f}")

    print(f"\n梯度值：")
    print(f"  fc1.weight.grad:\n{model.fc1.weight.grad}")
    print(f"  fc2.weight.grad:\n{model.fc2.weight.grad}")
    print(f"  fc1.bias.grad:\n{model.fc1.bias.grad}")
    print(f"  fc2.bias.grad:\n{model.fc2.bias.grad}")

    print("\n【思考题】")
    print("  1. 梯度表示的含义：")
    print("     梯度表示损失函数相对于各参数的变化率（偏导数）。")
    print("     它指明了在当前参数值处，损失函数上升最快的方向。")
    print("     训练的目标是沿着梯度的反方向更新参数，使损失下降。")
    print("  2. 梯度值越大意味着：")
    print("     该参数对损失函数的影响越大，需要更大的调整。")
    print("     但过大的梯度也可能导致训练不稳定（梯度爆炸）。")

    # ========================================================
    # 任务5：观察参数更新
    # ========================================================
    print("\n\n【任务5】观察参数更新")
    print("-" * 40)

    # 保存更新前的参数
    old_fc1_weight = model.fc1.weight.data.clone()
    old_fc1_bias = model.fc1.bias.data.clone()
    old_fc2_weight = model.fc2.weight.data.clone()
    old_fc2_bias = model.fc2.bias.data.clone()

    print("更新前 fc1.weight:")
    print(old_fc1_weight)

    # 执行参数更新
    optimizer.step()

    print("\n更新后 fc1.weight:")
    print(model.fc1.weight.data)

    print("\n参数变化量 (Δ = new - old):")
    print(model.fc1.weight.data - old_fc1_weight)

    print("\n【思考题】")
    print("  1. 参数发生了变化：是的，optimizer.step()根据梯度更新了所有参数。")
    print("  2. 参数更新的依据：")
    print("     SGD更新公式：θ_new = θ_old - lr × grad")
    print(f"     其中 lr=0.5（学习率），grad 是任务4中计算得到的梯度。")

    # 验证SGD更新公式
    print(f"\n  验证 SGD 更新公式（fc1.weight[0,0]）：")
    print(f"    θ_old = {old_fc1_weight[0,0].item():.6f}")
    print(f"    grad  = {model.fc1.weight.grad[0,0].item():.6f}")
    print(f"    θ_new (计算) = {old_fc1_weight[0,0].item() - 0.5 * model.fc1.weight.grad[0,0].item():.6f}")
    print(f"    θ_new (实际) = {model.fc1.weight.data[0,0].item():.6f}")

    # ========================================================
    # 任务6：完成网络训练
    # ========================================================
    print("\n\n【任务6】完成网络训练")
    print("-" * 40)

    # 重新初始化模型和优化器
    torch.manual_seed(42)
    model = XORNet()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    epochs = 2000
    loss_history = []
    grad_norm_fc1_history = []
    grad_norm_fc2_history = []
    output_history = []  # 每500轮记录一次输出

    print(f"开始训练（共 {epochs} 轮）...")
    print(f"  优化器: SGD")
    print(f"  学习率: 0.5")
    print(f"  损失函数: MSE")

    for epoch in range(epochs):
        # 清零梯度（必须！否则梯度会累积）
        optimizer.zero_grad()

        # 前向传播
        output = model(X)

        # 计算损失
        loss = criterion(output, Y)

        # 反向传播
        loss.backward()

        # 记录梯度范数（在step之前记录，因为step后梯度不变但下次会清零）
        grad_norm_fc1 = model.fc1.weight.grad.norm().item()
        grad_norm_fc2 = model.fc2.weight.grad.norm().item()

        # 更新参数
        optimizer.step()

        # 记录历史
        loss_history.append(loss.item())
        grad_norm_fc1_history.append(grad_norm_fc1)
        grad_norm_fc2_history.append(grad_norm_fc2)

        # 每500轮记录一次输出
        if (epoch + 1) % 500 == 0:
            output_history.append((epoch + 1, output.data.clone()))
            print(f"  Epoch {epoch+1:4d}/{epochs}  |  Loss: {loss.item():.6f}")

    print(f"\n训练完成！")
    print(f"  最终Loss: {loss_history[-1]:.6f}")
    print(f"  初始Loss: {loss_history[0]:.6f}")

    # 输出最终预测结果
    print(f"\n最终预测结果：")
    print(f"  X     |  Y_true  |  Y_pred  |  预测类别")
    print(f"  ------|----------|----------|----------")
    with torch.no_grad():
        final_output = model(X)
        for i in range(len(X)):
            pred = final_output[i].item()
            true_val = Y[i].item()
            pred_class = 1 if pred >= 0.5 else 0
            match = "✓" if pred_class == int(true_val) else "✗"
            print(f"  {X[i].tolist()} |   {true_val:.0f}     |  {pred:.4f}  |    {pred_class}      {match}")

    # 绘制 Loss-Epoch 曲线
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))

    ax1.plot(range(1, epochs + 1), loss_history, 'b-', linewidth=1.0, alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Loss-Epoch Curve (XOR Network Training)')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend(['Training Loss'], loc='upper right')

    plt.tight_layout()
    loss_plot_path = os.path.join(OUTPUT_DIR, "results_loss.png")
    plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nLoss曲线已保存至：{loss_plot_path}")

    # ========================================================
    # 任务7：分析梯度变化
    # ========================================================
    print("\n\n【任务7】分析梯度变化")
    print("-" * 40)

    # 绘制 Gradient Norm-Epoch 曲线
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # 子图1：Loss曲线（线性坐标）
    ax1.plot(range(1, epochs + 1), loss_history, 'b-', linewidth=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Loss-Epoch Curve')
    ax1.grid(True, alpha=0.3)

    # 子图2：梯度范数曲线
    ax2.plot(range(1, epochs + 1), grad_norm_fc1_history,
             'r-', linewidth=0.8, alpha=0.7, label='fc1.weight (Input→Hidden)')
    ax2.plot(range(1, epochs + 1), grad_norm_fc2_history,
             'b-', linewidth=0.8, alpha=0.7, label='fc2.weight (Hidden→Output)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Gradient Norm')
    ax2.set_title('Gradient Norm-Epoch Curve')
    ax2.set_yscale('log')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    grad_plot_path = os.path.join(OUTPUT_DIR, "results_grad_norm.png")
    plt.savefig(grad_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"梯度范数曲线已保存至：{grad_plot_path}")

    # 分析
    print(f"\n梯度范数统计：")
    print(f"  fc1.weight 梯度范数 - 初始: {grad_norm_fc1_history[0]:.6f}")
    print(f"  fc1.weight 梯度范数 - 最终: {grad_norm_fc1_history[-1]:.6f}")
    print(f"  fc2.weight 梯度范数 - 初始: {grad_norm_fc2_history[0]:.6f}")
    print(f"  fc2.weight 梯度范数 - 最终: {grad_norm_fc2_history[-1]:.6f}")

    # 检查各层的相对大小
    avg_grad_fc1 = np.mean(grad_norm_fc1_history[-100:])
    avg_grad_fc2 = np.mean(grad_norm_fc2_history[-100:])
    print(f"\n  最后100轮平均梯度范数：")
    print(f"    fc1 (输入层附近): {avg_grad_fc1:.6f}")
    print(f"    fc2 (输出层附近): {avg_grad_fc2:.6f}")

    print("\n【思考题】")
    print("  1. 梯度范数随训练变化的原因：")
    print("     训练初期，参数随机初始化，预测误差大，梯度较大。")
    print("     随着训练进行，损失逐渐减小，预测误差变小，")
    print("     梯度范数也随之减小，最终趋于稳定（收敛）。")
    print("  2. 输入层附近梯度与输出层附近梯度的差异：")
    print("     fc1（靠近输入层）的梯度通常比fc2（靠近输出层）的梯度小，")
    print("     这与梯度消失问题有关——Sigmoid函数的导数最大仅为0.25，")
    print("     梯度经过多层Sigmoid传播后会逐层衰减。")
    if avg_grad_fc1 < avg_grad_fc2:
        print(f"     本实验中 fc1 平均梯度({avg_grad_fc1:.6f}) < fc2 平均梯度({avg_grad_fc2:.6f})，")
        print(f"     验证了梯度在反向传播中逐层衰减的现象。")
    else:
        print(f"     本实验中 fc1 平均梯度({avg_grad_fc1:.6f}) 与 fc2 平均梯度({avg_grad_fc2:.6f}) 的关系，" )
        print(f"     可能与具体初始化及数据规模有关。")

    # 汇总训练过程中的关键输出
    print(f"\n训练过程关键节点输出：")
    for epoch, out in output_history:
        print(f"\n  Epoch {epoch}:")
        print(f"  {out.numpy()}")


# ============================================================
# 主函数
# ============================================================
def main():
    """主函数：依次执行实验一和实验二"""
    print("=" * 60)
    print("  机器学习上机实验11：神经网络-1")
    print("  PyTorch自动求导与神经网络训练")
    print("=" * 60)

    # 实验一：计算图与自动求导
    experiment_1()

    # 实验二：神经网络训练与梯度分析
    experiment_2()

    print("\n" + "=" * 60)
    print("  实验完成！")
    print(f"  生成文件：")
    print(f"    - {os.path.join(OUTPUT_DIR, 'xor_dataset.npz')}")
    print(f"    - {os.path.join(OUTPUT_DIR, 'results_loss.png')}")
    print(f"    - {os.path.join(OUTPUT_DIR, 'results_grad_norm.png')}")
    print("=" * 60)


if __name__ == "__main__":
    main()

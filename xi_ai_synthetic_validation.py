"""
Ξ-AI 范式验证 - 合成数据测试
避开MNIST下载问题，验证ζ-Attention核心机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ZETA_ATTENTION_VALIDATION = True


class XiAttention_v2(nn.Module):
    """
    ξ-Attention v2: 修复的显式公式实现

    核心数学：
    A_ij = Re log(1/(1 - χ_{ij} n_{ij}^{-s}))  , s = 1/2 + it
    零点修正：V - Σ_ρ K^ρ / ρ
    """
    def __init__(self, dim=64, n_head=4, n_zero=8, im_min=14.0, im_max=50.0):
        super().__init__()
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.n_zero = n_zero

        # 投影层
        self.Wq = nn.Linear(dim, dim)
        self.Wk = nn.Linear(dim, dim)
        self.Wv = nn.Linear(dim, dim)

        # 可学零点：rho = re + i*im
        self.rho_re = nn.Parameter(torch.zeros(n_head, n_zero))
        self.rho_im = nn.Parameter(torch.zeros(n_head, n_zero))

        # 初始化：使用谱映射
        self._init_zeros()

        # 参数
        self.temp = nn.Parameter(torch.tensor(1.0))  # 温度参数
        self.zero_scale = nn.Parameter(torch.tensor(0.01))  # 零点修正缩放

    def _init_zeros(self):
        """
        初始化零点：实部靠近0.5，虚部来自谱
        """
        # 虚部：分段线性分布，模拟临界零点
        im_min, im_max = 14.0, 50.0
        ts = torch.linspace(im_min, im_max, self.n_zero)
        self.rho_im.data[...] = ts.unsqueeze(0).repeat(self.n_head, 1)

        # 实部：初始在0.5附近
        self.rho_re.data[...] = 0.5 + torch.randn(self.n_head, self.n_zero) * 0.05

    def forward(self, x):
        B, N, D = x.shape
        H, hd = self.n_head, self.head_dim

        # 线性投影并reshape
        Q = self.Wq(x).view(B, N, H, hd).transpose(1, 2)  # [B, H, N, hd]
        K = self.Wk(x).view(B, N, H, hd).transpose(1, 2)
        V = self.Wv(x).view(B, N, H, hd).transpose(1, 2)

        # --- 计算注意力权重 ---
        idx = torch.arange(N, device=x.device)
        distance = (idx[:, None] - idx[None, :]).abs().float() + 1.0

        qk = (Q @ K.transpose(-1, -2)) / np.sqrt(hd)
        chi = torch.tanh(qk)

        # 欧拉局部因子: 1/(1 - χ * n^{-s})
        power = -0.5 + 0.5 * distance
        denom = 1.0 - chi * distance.unsqueeze(0).unsqueeze(0).pow(power)
        denom = denom.clamp(min=1e-6, max=1-1e-6)
        log_factor = torch.log(1.0 / denom)

        logits = self.temp * log_factor.real
        A = torch.softmax(logits, dim=-1)

        out = A @ V

        # --- 零点修正项 (显式公式) ---
        rho = torch.complex(self.rho_re, self.rho_im)
        K_complex = K.to(torch.complex64)
        corr = torch.zeros_like(K_complex, dtype=torch.complex64)

        for h in range(H):
            for z in range(self.n_zero):
                k = K_complex[:, h]
                correction = torch.pow(k, rho[h, z]) / rho[h, z]
                corr[:, h] += correction.real

        corr = corr * self.zero_scale
        out = out - corr

        out = out.transpose(1, 2).reshape(B, N, D)
        return out

    def rh_loss(self):
        return ((self.rho_re - 0.5).abs().mean())


class XiTransformer(nn.Module):
    """更简单的 Transformer 变体 (支持二维输入)"""
    def __init__(self, input_dim, d_model=64, n_layers=2, n_heads=2, n_zeros=4, seq_len=1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_layers = n_layers
        self.seq_len = seq_len

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = self._init_pos_encoding(seq_len)

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                XiAttention_v2(d_model, n_heads, n_zeros),
                nn.Dropout(0.1)
            ) for _ in range(n_layers)
        ])

        self.output_layer = nn.Linear(d_model, 2)  # 二分类

    def _init_pos_encoding(self, max_len=1000):
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if self.d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        # 支持二维或三维输入
        if x.dim() == 2:
            B, D = x.shape
            T = 1
            x = x.unsqueeze(1)  # [B, 1, D]
        else:
            B, T, D = x.shape

        assert D == self.input_dim, f"Input dim mismatch: {D} vs {self.input_dim}"

        # 线性投影
        x = self.input_proj(x)

        # 添加位置编码
        x = x + self.pos_encoding[:T, :]

        # Transformer 层
        for layer in self.layers:
            x = x + layer(x)

        # 使用最后一个位置的输出
        return self.output_layer(x[:, -1, :])

    def rh_loss(self):
        return sum(layer[1].rh_loss() for layer in self.layers) / len(self.layers)


def generate_synthetic_data(n_samples=1000, n_features=20, chips=2):
    """
    生成交织线性/非线性可分的数据
    chips: 类别间的最小欧氏距离
    """
    X1 = np.random.randn(n_samples // 2, n_features)
    X2 = np.random.randn(n_samples // 2, n_features) + chips * np.ones((n_samples // 2, n_features))
    X = np.vstack([X1, X2])
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    return X, y


def generate_complex_pattern_data(n_samples=1000, n_features=20, clusters=3):
    """生成复杂模式数据"""
    X = []
    y = []
    cluster_size = n_samples // clusters
    for i in range(clusters):
        center = np.random.randn(n_features) * 2.0
        X.append(np.random.randn(cluster_size, n_features) + center)
        y.extend([i] * cluster_size)
    X = np.vstack(X)
    y = np.array(y)
    return X, y


def train_model(model, X_train, y_train, X_val, y_val, epochs=100, lr=1e-3):
    """训练模型"""
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.LongTensor(y_val)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val_acc = 0
    best_weights = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        ce = F.cross_entropy(outputs, y_train)
        rh = model.rh_loss()
        loss = ce + 0.01 * rh
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_acc = (val_outputs.argmax(1) == y_val).float().mean().item()
                train_acc = (outputs.argmax(1) == y_train).float().mean().item()
                print(f"Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | RH Loss: {rh.item():.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_weights, best_val_acc


def compare_regularization_modes(model_class, X_train, y_train, X_val, y_val, train_epochs=50):
    """比较不同正则化强度的效果"""
    print("\n🔬 比较不同 RH 正则化系数...")

    results = []
    for lambda_rh in [0.0, 0.001, 0.01, 0.1]:
        if lambda_rh == 0.0:
            model = model_class(X_train.shape[1], d_model=32, n_layers=2, n_heads=2, n_zeros=4)
            # 特殊处理：没有rh_loss
            def dummy_rh_loss(self):
                return torch.tensor(0.0)
            model.rh_loss = dummy_rh_loss.__get__(model)
        else:
            model = model_class(X_train.shape[1], d_model=32, n_layers=2, n_heads=2, n_zeros=4)
            for layer in model.layers:
                layer[1].temp.data.fill_(1.0)
                layer[1].zero_scale.data.fill_(0.01)
                layer[1].rho_re.data.fill_(0.5 + 0.05 * np.random.randn(2, 4))
                layer[1].rho_im.data.fill_(np.linspace(14.0, 40.0, 4))

        best_weights, val_acc = train_model(model, X_train, y_train, X_val, y_val, train_epochs, lr=0.002)

        results.append({
            'lambda_rh': lambda_rh,
            'val_acc': val_acc
        })

        print(f"  λ_RH={__import__('numpy').string(lamba_rh):.3f} -> Val Acc: {val_acc:.4f}")

    return results


def run_validation():
    """主验证流程"""
    print("=" * 60)
    print("Ξ-AI 范式验证 - 合成数据测试")
    print("验证 ζ-Attention 核心机制")
    print("=" * 60)

    # 1. 生成数据
    print("\n📊 生成合成数据...")
    X, y = generate_complex_pattern_data(n_samples=2000, n_features=30, clusters=4)

    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 划分训练/验证/测试
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print(f"  训练集大小: {len(X_train)}")
    print(f"  验证集大小: {len(X_val)}")
    print(f"  测试集大小: {len(X_test)}")

    # 2. 训练原始模型 (无正则化)
    print("\n📈 训练原始模型 (无λ_RH)...")
    model_0 = XiTransformer(X_train.shape[1], d_model=32, n_layers=2, n_heads=2, n_zeros=4)
    # 禁用 rh_loss
    model_0.rh_loss = lambda: torch.tensor(0.0)
    best_weights_0, _ = train_model(model_0, X_train, y_train, X_val, y_val, epochs=100)

    # 测试原始模型
    model_0.eval()
    with torch.no_grad():
        test_acc_0 = (model_0(X_test).argmax(1) == y_test).float().mean().item()
    print(f"  最终测试准确率: {test_acc_0:.4f}")

    # 3. 训练有正则化的模型
    print("\n📈 训练正则化模型 (λ_RH=0.01)...")
    model_reg = XiTransformer(X_train.shape[1], d_model=32, n_layers=2, n_heads=2, n_zeros=4)
    best_weights_reg, _ = train_model(model_reg, X_train, y_train, X_val, y_val, epochs=100)

    # 测试正则化模型
    model_reg.load_state_dict(best_weights_reg)
    model_reg.eval()
    with torch.no_grad():
        test_acc_reg = (model_reg(X_test).argmax(1) == y_test).float().mean().item()
    print(f"  最终测试准确率: {test_acc_reg:.4f}")


    # 4. 分析零点分布
    print("\n📊 分析零点分布...")
    if not model_0.rh_loss.__module__ != "no_rh":
        zero_re_model0 = model_0.layers[0][1].rho_re.detach().numpy()
        zero_re_model_reg = model_reg.layers[0][1].rho_re.detach().numpy()

        print("  原始模型零点实部分布:")
        print(f"    均值: {zero_re_model0.mean():.4f}")
        print(f"    标准差: {zero_re_model0.std():.4f}")

        print("  正则化模型零点实部分布:")
        print(f"    均值: {zero_re_model_reg.mean():.4f}")
        print(f"    标准差: {zero_re_model_reg.std():.4f}")

    # 5. 结果总结
    print("\n" + "=" * 60)
    print("验证结果总结")
    print("=" * 60)
    print(f"原始模型测试准确率: {test_acc_0:.4f}")
    print(f"正则化模型测试准确率: {test_acc_reg:.4f}")
    print(f"差异: {test_acc_reg - test_acc_0:+.4f}")

    # 6. 稳定性测试
    print("\n🔬 稳定性测试：在不同初始化和数据划分下的性能")
    stabilities = []
    for seed in range(5):
        np.random.seed(seed)
        X_seed, y_seed = generate_complex_pattern_data(n_samples=2000, n_features=30, clusters=4)
        scaler_seed = StandardScaler()
        X_seed = scaler_seed.fit_transform(X_seed)
        X_tr, X_te, y_tr, y_te = train_test_split(X_seed, y_seed, test_size=0.4, random_state=seed, stratify=y_seed)

        model_test = XiTransformer(X_tr.shape[1], d_model=32, n_layers=2, n_heads=2, n_zeros=4)
        model_test.rh_loss = lambda: torch.tensor(0.0)
        _, val_acc = train_model(model_test, X_tr, y_tr, X_tr, y_tr, epochs=50, lr=0.002)
        stabilities.append(val_acc)

    import numpy as np
    print(f"  Lambda=0 的稳定性 (5 次): {np.mean(stabilities):.4f} ± {np.std(stabilities):.4f}")

    # 7. 对比 λ_RH 的影响
    print("\n🔬 对比不同 λ_RH 的效果")
    comparison = compare_regularization_modes(
        XiTransformer,
        X_train, y_train, X_val, y_val
    )

    print("\n✅ 验证完成！")


if __name__ == "__main__":
    run_validation()

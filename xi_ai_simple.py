"""
Ξ-AI 范式验证 - 简化版
直接验证零点正则化的效果
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class XiMLP(nn.Module):
    """
    简单 MLP + 零点修正项
    模拟 ξ-Transformer 的核心结构
    """
    def __init__(self, input_dim, hidden_dim=64, n_zeros=8):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)
        self.act = nn.ReLU()

        # 可学零点
        self.n_zeros = n_zeros
        self.rho_re = nn.Parameter(torch.randn(n_zeros) * 0.05 + 0.5)
        self.rho_im = nn.Parameter(torch.linspace(14.0, 50.0, n_zeros))

        # 零点修正缩放
        self.zero_scale = nn.Parameter(torch.tensor(0.001))

    def zero_correction(self, x):
        """应用零点修正 x -> x - Σ_ρ x^ρ / ρ"""
        corr = 0.0
        for k in range(self.n_zeros):
            rho = torch.complex(self.rho_re[k], self.rho_im[k])
            # 主要项：x^ρ / ρ 的实部
            corr += torch.pow(x, rho) / rho
        return (self.zero_scale * corr.real).sum()

    def forward(self, x):
        x = self.act(self.fc1(x))
        # 主输出
        main_out = self.fc2(x)
        # 加上零点修正
        correction = self.zero_correction(x)
        return main_out + correction

    def rh_loss(self):
        """RH Loss: 零点实部到 0.5 的惩罚"""
        return ((self.rho_re - 0.5).abs().mean())


def generate_data():
    """生成数据"""
    X, y = make_classification(
        n_samples=2000, n_features=30, n_informative=15,
        n_redundant=5, n_clusters_per_class=2, random_state=42
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y)
    return X_train, X_test, y_train, y_test


def train_model(model, X_train, y_train, X_val, y_val, lambda_rh=0.01, epochs=100, lr=1e-3):
    """训练模型"""
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.LongTensor(y_val)

    best_val_acc = 0.0
    best_weights = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # 训练损失
        pred = model(X_train)
        ce = F.cross_entropy(pred, y_train)
        rh = model.rh_loss()
        loss = ce + lambda_rh * rh
        loss.backward()
        optimizer.step()

        # 评估
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_acc = (val_pred.argmax(1) == y_val).float().mean().item()
                train_acc = (pred.argmax(1) == y_train).float().mean().item()
                print(f"Epoch {epoch+1:02d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | RH Loss: {rh.item():.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_weights, best_val_acc

def test_model(model, X_test, y_test):
    """测试模型"""
    model.eval()
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)
    with torch.no_grad():
        pred = model(X_test)
        acc = (pred.argmax(1) == y_test).float().mean().item()
    return acc


def test_model(model, X_test, y_test):
    """测试模型"""
    model.eval()
    with torch.no_grad():
        pred = model(X_test)
        acc = (pred.argmax(1) == y_test).float().mean().item()
    return acc


def main():
    print("=" * 60)
    print("Xi-AI 范式验证 - MLP 版")
    print("验证零点正则化的泛化效果")
    print("=" * 60)

    X_train, X_test, y_train, y_test = generate_data()
    print(f"数据集: {len(X_train)} train, {len(X_test)} test")

    # 对比不同 λ_RH
    results = []
    for lambda_rh in [0.0, 0.001, 0.01, 0.1]:
        print(f"\n🔧 训练 λ_RH={lambda_rh:.3f} 的模型...")
        model = XiMLP(X_train.shape[1], hidden_dim=64, n_zeros=8)
        best_weights, val_acc = train_model(
            model, X_train, y_train, X_test, y_test,
            lambda_rh=lambda_rh, epochs=50, lr=0.001
        )

        # 加载最佳权重并测试
        model.load_state_dict(best_weights)
        test_acc = test_model(model, X_test, y_test)
        results.append({
            'lambda_rh': lambda_rh,
            'val_acc': val_acc,
            'test_acc': test_acc
        })

        # 打印零点状态
        zero_re = model.rho_re.detach().numpy()
        print(f"  Best Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")
        print(f"  Final Zero Re: mean={zero_re.mean():.4f}, std={zero_re.std():.4f}")

    # 总结
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    for res in results:
        print(f"λ_RH={res['lambda_rh']:.3f}: Val@{res['val_acc']:.3f}, Test@{res['test_acc']:.3f}")


if __name__ == "__main__":
    main()

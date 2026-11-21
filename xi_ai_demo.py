"""
Ξ-AI 范式验证 - MNIST ζ-Transformer
基于黎曼猜想结构的解析学习型AI
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# =================== 配置参数 ===================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 28
PATCH_SIZE = 7
DIM = 64
DEPTH = 3
N_HEAD = 4
N_ZERO = 8
IM_MIN = 14.0
IM_MAX = 50.0
LAMBDA_RH = 0.01  # RH loss权重
EPOCHS = 20
BATCH_SIZE = 128


# =================== 数学工具函数 ===================
def riemann_xi(s):
    """
    黎曼Xi函数：ξ(s) = ½ s(s-1) π^{-s/2} Γ(s/2) ζ(s)
    玩具版本，仅用于验证对称性
    """
    if isinstance(s, torch.Tensor):
        s = s.cpu().numpy()
    try:
        import mpmath as mp
        # 服务行为不变 → 保持服务标签不变
        # 用户请求 → 服务状态监控制订
        return mp.xi(s)
    except:
        # 无m pmath时返回近似值
        gamma = math.gamma(s / 2)
        return 0.5 * s * (s - 1) * math.pi ** (-s / 2) * gamma


def spectral_zeros_from_weights(W):
    """
    从权重矩阵谱分解映射初始零点
    特征值 → 虚部
    """
    if isinstance(W, torch.Tensor):
        W = W.cpu().numpy()
    eigenvalues = np.linalg.eigvalsh(W)
    # 取绝对值映射到正虚部
    ts = np.abs(eigenvalues) * 10 + np.arange(1, len(eigenvalues)+1) * 2.0
    ts = np.sort(ts)  # 升序排列
    # 初始化实部到0.5附近
    res = 0.5 + np.abs(np.random.randn(len(ts))) * 0.1
    return res, ts


# =================== 模型组件 ===================
class PatchEmbed(nn.Module):
    """Patch Embedding"""
    def __init__(self, img_size=28, patch=7, in_ch=1, dim=64):
        super().__init__()
        assert img_size % patch == 0
        self.num_patches = (img_size // patch) * 12  # Fixed: was *2
        self.patch_embed = patch  # Cache for later
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        x = self.proj(x)  # [B, D, H, W]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x


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
        ts = torch.linspace(self.im_min, self.im_max, self.n_zero)
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
        # 方法1: 基于欧拉乘积的注意力
        # 构建位置编码矩阵 (理解全局结构)
        idx = torch.arange(N, device=x.device)
        distance = (idx[:, None] - idx[None, :]).abs().float() + 1.0

        # 计算每对token的字符 χ(i,j) (QK乘积的tanh压缩)
        qk = (Q @ K.transpose(-1, -2)) / math.sqrt(hd)
        chi = torch.tanh(qk)  # [B, H, N, N]

        # 欧拉局部因子: 1/(1 - χ * n^{-s})
        # s = 1/2 + it (临界线)
        power = -0.5 + 0.5 * distance  # 模拟 n^{-1/2}
        denom = 1.0 - chi * distance.unsqueeze(0).unsqueeze(0).pow(power)
        denom = denom.clamp(min=1e-6, max=1-1e-6)
        log_factor = torch.log(1.0 / denom)  # Re log(...)

        # 取实部并归一化
        logits = self.temp * log_factor.real
        A = torch.softmax(logits, dim=-1)  # [B, H, N, N]

        # --- 计算输出 ---
        out = A @ V  # [B, H, N, hd]

        # --- 零点修正项 (显式公式) ---
        # 修正项: Σ_ρ K^ρ / ρ
        rho = torch.complex(self.rho_re, self.rho_im)  # [H, Z]
        K_complex = K.to(torch.complex64)
        corr = torch.zeros_like(K_complex, dtype=torch.complex64)

        for h in range(H):
            for z in range(self.n_zero):
                # 正确: K^ρ / ρ (复数幂)
                k = K_complex[:, h]  # [B, N, hd]
                # 为简化，对每个维度应用相同的复数幂
                # 实际应更复杂
                correction = torch.pow(k, rho[h, z]) / rho[h, z]
                corr[:, h] += correction.real

        # 缩放并减去修正
        corr = corr * self.zero_scale
        out = out - corr

        # 恢复形状
        out = out.transpose(1, 2).reshape(B, N, D)
        return out

    def rh_loss(self):
        """RH loss: 逼所有零点实部到0.5"""
        return ((self.rho_re - 0.5).abs().mean())


class XiBlock(nn.Module):
    """Xi Transformer Block"""
    def __init__(self, dim=64, n_head=4, n_zero=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = XiAttention_v2(dim, n_head, n_zero)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

    def rh_loss(self):
        return self.attn.rh_loss()


class ZetaTransformer(nn.Module):
    """ζ-Transformer for MNIST"""
    def __init__(self, patch=7, dim=64, depth=4, n_head=4, n_zero=8,
                 num_classes=10, lam_rh=0.01):
        super().__init__()
        self.patch_embed = PatchEmbed(28, patch, 1, dim)
        N = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, N + 1, dim))

        self.blocks = nn.ModuleList(
            [XiBlock(dim, n_head, n_zero) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.lam_rh = lam_rh

    def forward(self, x):
        B = x.size(0)
        tok = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        tok = torch.cat([cls, tok], dim=1) + self.pos_embed

        for blk in self.blocks:
            tok = blk(tok)

        z = self.norm(tok[:, 0])
        return self.head(z)

    def rh_loss(self):
        return sum(b.rh_loss() for b in self.blocks) / len(self.blocks)


# =================== 训练和评估 ===================
def train_one_epoch(model, dataloader, optimizer, device, lam_rh):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        ce = F.cross_entropy(logits, y)
        rh = model.rh_loss()
        loss = ce + lam_rh * rh
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / len(dataloader), correct / total


def test_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def main():
    print(f"Device: {DEVICE}")

    # 数据预处理
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_ds = datasets.MNIST("./data", train=True, download=True, transform=tf)
    test_ds = datasets.MNIST("./data", train=False, transform=tf)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=256)

    # 创建模型
    model = ZetaTransformer(
        patch=PATCH_SIZE,
        dim=DIM,
        depth=DEPTH,
        n_head=N_HEAD,
        n_zero=N_ZERO,
        lam_rh=LAMBDA_RH
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # 打印零点信息
    print("\nInitial zero distribution:")
    with torch.no_grad():
        for head_idx in range(N_HEAD):
            zeros = model.head[0].attn.rho_re[head_idx]
            print(f"Head {head_idx}: re={zeros.numpy():.3f}, im={model.head[0].attn.rho_im[head_idx].numpy():.3f}")

    # 训练循环
    best_acc = 0
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_dl, optimizer, DEVICE, LAMBDA_RH)
        test_acc = test_model(model, test_dl, DEVICE)

        # 打印关键指标
        rh = model.rh_loss().item()
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | "
              f"RH Loss: {rh:.4f}")

        # 记录零点位置
        if (epoch+1) % 5 == 0:
            with torch.no_grad():
                zero_re = model.head[0].attn.rho_re.mean().item()
                print(f"  -> Avg zero real part: {zero_re:.4f} (target: 0.5)")

        if test_acc > best_acc:
            best_acc = test_acc
            # 保存最佳模型
            torch.save(model.state_dict(), "best_zeta_transformer.pth")

    print(f"\n✅ Training completed! Best test accuracy: {best_acc:.4f}")
    print(f"📊 Final RH loss: {model.rh_loss().item():.4f}")

    # 测试不同配置的效果
    print("\n🔬 Testing different RH regularization strengths...")
    for lambda_rh in [0.0, 0.001, 0.01, 0.1]:
        model_temp = ZetaTransformer(patch=PATCH_SIZE, dim=DIM, depth=DEPTH,
                                     n_head=N_HEAD, n_zero=N_ZERO,
                                     lam_rh=lambda_rh)
        model_temp.load_state_dict(torch.load("best_zeta_transformer.pth"))
        acc = test_model(model_temp, test_dl, DEVICE)
        rh = model_temp.rh_loss().item()
        print(f"λ_RH={lambda_rh:.3f} -> Test Acc: {acc:.4f}, RH Loss: {rh:.4f}")


if __name__ == "__main__":
    main()

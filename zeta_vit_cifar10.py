"""
ζ-ViT for CIFAR-10

Patch=4, ξ-attention, RH loss

CIFAR-10: patch=4 → 64 token + CLS

PyTorch, CPU 可跑，GPU 加速
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


class PatchEmbed(nn.Module):
    """Patch Embedding for CIFAR-10"""
    def __init__(self, img_size=32, patch=4, in_ch=3, dim=192):
        super().__init__()
        assert img_size % patch == 0
        self.n_patch = (img_size // patch) * 12  # Fixed from *2
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        x = self.proj(x)  # [B, D, H, W]
        return x.flatten(2).transpose(1, 2)  # [B, N, D]


class XiAttention(nn.Module):
    """ξ-Attention for ζ-ViT"""
    def __init__(self, dim=192, n_head=3, n_zero=8,
                 im_min=14.0, im_max=48.0):
        super().__init__()
        self.h = n_head
        self.dh = dim // n_head
        self.n_zero = n_zero

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # learnable zeros: rho = re + i im
        self.rho_re = nn.Parameter(torch.randn(n_head, n_zero) * 0.05 + 0.5)
        self.rho_im = nn.Parameter(
            torch.linspace(im_min, im_max, n_zero)
            .unsqueeze(0).repeat(n_head, 1)
        )

        self.temp = nn.Parameter(torch.tensor(1.0))
        self.zero_scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, x):
        B, N, D = x.shape
        H, dh = self.h, self.dh

        qkv = self.qkv(x).view(B, N, 3, H, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, N, dh]

        # arithmetic distance matrix: n_{ij} = 1 + |i-j|
        idx = torch.arange(N, device=x.device)
        n_ij = (idx[:, None] - idx[None, :]).abs().float() + 1.0
        n_ij = n_ij.pow(0.5)  # critical line: n^{1/2}

        # chi = tanh(QK^T / sqrt(dh))
        chi = torch.tanh((q @ k.transpose(-1, -2)) / math.sqrt(dh))

        # local Euler factor: Re log(1/(1 - chi * n^{-1/2}))
        denom = 1.0 - chi / n_ij.unsqueeze(0).unsqueeze(0)
        denom = denom.clamp(min=1e-3, max=1 - 1e-3)
        logits = self.temp * torch.log(1.0 / denom)
        A = torch.softmax(logits, dim=-1)

        out = A @ v  # [B, H, N, dh]

        # explicit-formula correction: out - sum_rho k^rho / rho
        rho = torch.complex(self.rho_re, self.rho_im)
        kc = k.to(torch.complex64)
        corr = torch.zeros_like(kc, dtype=torch.complex64)
        for h in range(H):
            for z in range(self.n_zero):
                corr[:, h] += kc[:, h] * rho[h, z] / rho[h, z]
        out = out - corr.real * self.zero_scale

        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj(out)

    def rh_loss(self):
        return ((self.rho_re - 0.5) * 2).mean()


class XiBlock(nn.Module):
    """Transformer block with ξ-attention"""
    def __init__(self, dim=192, n_head=3, n_zero=8, mlp_ratio=4.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = XiAttention(dim, n_head, n_zero)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x

    def rh_loss(self):
        return self.attn.rh_loss()


class ZetaViT(nn.Module):
    """ζ-ViT for CIFAR-10"""
    def __init__(self, patch=4, dim=192, depth=8, n_head=3,
                 n_zero=8, num_classes=10, lam_rh=0.02):
        super().__init__()
        self.patch = PatchEmbed(32, patch, 3, dim)
        N = self.patch.n_patch
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, N + 1, dim))
        self.blocks = nn.ModuleList(
            [XiBlock(dim, n_head, n_zero) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.lam_rh = lam_rh
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x):
        B = x.size(0)
        tok = self.patch(x)
        cls = self.cls.expand(B, -1, -1)
        tok = torch.cat([cls, tok], dim=1) + self.pos

        for blk in self.blocks:
            tok = blk(tok)

        return self.head(self.norm(tok[:, 0]))

    def rh_loss(self):
        return sum(b.rh_loss() for b in self.blocks) / len(self.blocks)


def get_loaders(batch=128):
    """CIFAR-10 data loaders with augmentation"""
    train_tf = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    train = torchvision.datasets.CIFAR10(
        "./data", train=True, download=True, transform=train_tf)
    test = torchvision.datasets.CIFAR10(
        "./data", train=False, transform=test_tf)
    return (DataLoader(train, batch_size=batch, shuffle=True),
            DataLoader(test, batch_size=256))


def main():
    """Train ζ-ViT on CIFAR-10"""
    train_dl, test_dl = get_loaders()
    print(f"Device: {DEVICE}")
    print(f"Data: {len(train_dl.dataset)} train, {len(test_dl.dataset)} test")

    model = ZetaViT(patch=4, dim=192, depth=8,
                    n_head=3, n_zero=8, lam_rh=0.02).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=5e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)

    print("Training ζ-ViT on CIFAR-10...")
    for epoch in range(100):
        model.train()
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y) + model.lam_rh * model.rh_loss()
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        print(f"ep {epoch:03d} | cifar10_acc={correct/total:.4f} | rh={model.rh_loss().item():.4f}")


if __name__ == "__main__":
    main()

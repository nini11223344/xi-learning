"""
Softmax-free ζ-ViT with Selberg-style spectral attention

Selberg-trace attention (no softmax)

CIFAR-10, PyTorch

Attention = spectral kernel sum
Not softmax, not probabilities
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
CIFAR_STD = (0.2470, 0.2435, 0.2615)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch=4, in_ch=3, dim=192):
        super().__init__()
        self.n_patch = (img_size // patch) * 2  # Fixed: was *2
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        x = self.proj(x)  # [B, D, H, W]
        return x.flatten(2).transpose(1, 2)  # [B, N, D]


class SelbergAttention(nn.Module):
    """Selberg attention (no softmax) with spectral basis"""
    def __init__(self, dim=192, n_head=3, n_eig=8,
                 mu=0.0, im_min=14.0, im_max=48.0):
        super().__init__()
        self.h = n_head
        self.dh = dim // n_head
        self.K = n_eig
        self.mu = mu

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # spectral parameters: t_k (eigenvalues on critical line)
        self.t_k = nn.Parameter(
            torch.linspace(im_min, im_max, n_eig)
            .unsqueeze(0).repeat(n_head, 1)
        )
        # allow small deviation from critical line
        self.re_k = nn.Parameter(torch.zeros(n_head, n_eig) + 0.5)

        self.gain = nn.Parameter(torch.ones(1))
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        B, N, D = x.shape
        H, dh = self.h, self.dh

        qkv = self.qkv(x).view(B, N, 3, H, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, N, dh]

        # arithmetic distance
        idx = torch.arange(N, device=x.device)
        d_ij = torch.log1p((idx[:, None] - idx[None, :]).abs().float())  # log(1+|i-j|)

        # spectral kernel: sum_k e^{i t_k d} / (re_k + i t_k + mu)
        rho = torch.complex(self.re_k, self.t_k)  # [H, K]
        denom = rho + self.mu

        # [H, K, N, N]
        phase = torch.exp(1j * self.t_k[None, :, None, None] * d_ij[None, None, :, :])
        coeff = 1.0 / denom[:, :, None, None]
        kernel = (phase * coeff).real  # take real part

        # modulate by QK similarity (still linear, no softmax)
        sim = (q @ k.transpose(-1, -2)) / math.sqrt(dh)  # [B, H, N, N]
        A = self.gain * kernel.mean(dim=1) * torch.tanh(sim + 1.0)

        out = A @ v  # NOT normalized to probabilities
        out = out * self.scale

        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj(out)

    def rh_loss(self):
        return ((self.re_k - 0.5) * 2).mean()


class SelbergBlock(nn.Module):
    def __init__(self, dim=192, n_head=3, n_eig=8, mlp_ratio=4.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = SelbergAttention(dim, n_head, n_eig)
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


class SelbergViT(nn.Module):
    def __init__(self, patch=4, dim=192, depth=8,
                 n_head=3, n_eig=8, num_classes=10, lam_rh=0.02):
        super().__init__()
        self.patch = PatchEmbed(32, patch, 3, dim)
        N = self.patch.n_patch
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, N + 1, dim))
        self.blocks = nn.ModuleList(
            [SelbergBlock(dim, n_head, n_eig) for _ in range(depth)]
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
    tr = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    te = T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
    train = torchvision.datasets.CIFAR10("./data", True, download=True, transform=tr)
    test = torchvision.datasets.CIFAR10("./data", False, transform=te)
    return DataLoader(train, batch_size=batch, shuffle=True), DataLoader(test, batch_size=256)


def main():
    train_dl, test_dl = get_loaders()
    print(f"Device: {DEVICE}")
    print(f"Data: {len(train_dl.dataset)} train, {len(test_dl.dataset)} test")

    model = SelbergViT(patch=4, dim=192, depth=8,
                       n_head=3, n_eig=8, lam_rh=0.02).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=5e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)

    print("Training Selberg-ViT on CIFAR-10...")
    for epoch in range(100):
        model.train()
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y) + model.lam_rh * model.rh_loss()
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
        print(f"ep {epoch:03d} | cifar10={correct/total:.4f} | rh={model.rh_loss().item():.4f}")


if __name__ == "__main__":
    main()

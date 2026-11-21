"""
Ξ-Net: Whole network = ONE integral operator

Forward = spectral expansion (no layers, no softmax, no attention)
Loss = cross-entropy + trace-formula error + RH loss

CIFAR-10, PyTorch

整个模型 = 一个 ξ 算子 (谱展开)
训练 = 调整本征值，使算子「长得像 ζ」

Not a traditional neural network. It's a spectral operator.
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


class PatchToFunction(nn.Module):
    """
    Convert image patches to function values.
    Each patch becomes a point in an arithmetic space.
    """
    def __init__(self, img_size=32, patch=4, in_ch=3, dim=128):
        super().__init__()
        self.n = (img_size // patch) * 2  # number of patches
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        self.dim = dim

    def forward(self, x):
        x = self.proj(x)  # [B, D, H, W]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        # 不做 layer_norm, raw patches
        return x


class XiOperator(nn.Module):
    """
    The ξ operator.
    The whole network is one operator!

    Spectral kernel: K(p,q) = sum_k phi_k(p) phi_k(q)^H / (rho_k + mu)

    Forward: y = Sigma_k <f, phi_k> / (rho_k + mu) * phi_k
    """
    def __init__(self, N=64, dim=128, K=12, T=40.0, mu=0.5):
        super().__init__()
        self.N = N
        self.dim = dim
        self.K = K
        self.T = T
        self.mu = mu

        # Learnable basis: phi_k(p) = exp(i t_k log(1+p)) * alpha_k
        # alpha_k is a learnable vector of dim dim
        self.alpha = nn.Parameter(torch.randn(K, dim) * 0.01)

        # Eigenvalues / zeros: rho_k = re_k + i t_k
        self.re_k = nn.Parameter(torch.full((K,), 0.5))
        # t_k initialized with GUE-like spacing (Riemann zeros)
        self.t_k = nn.Parameter(
            torch.linspace(14.0, 14.0 + 3.0 * K, K)
        )

        self.readout = nn.Linear(dim, 10)  # 10 classes for CIFAR-10

    def basis(self, p):
        """
        Generate basis functions phi_k(p) for positions p.
        p: [N] (positions)
        Returns: [K, N, dim]
        """
        # t_k: [K, 1, 1] to broadcast
        t_k = self.t_k[:, None, None]
        # log(1+p) for each position
        log_p = torch.log1p(p)
        # Oscillation part: [K, N]
        # For each k, exp(i t_k * log_p)
        # t_k: [K,1], log_p: [1,N] -> [K,N]
        osc = torch.exp(1j * t_k * log_p)
        # Multiply with learnable alpha: [K, 1, dim]
        phi = osc.unsqueeze(-1) * self.alpha[None, :, :]
        return phi  # [K, N, dim]

    def forward(self, f):
        """
        Forward pass: spectral expansion
        f: [B, N, dim]
        """
        # positions: [N]
        p = torch.arange(self.N, device=f.device)
        phi = self.basis(p)  # [K, N, dim]

        # 1. Spectral coefficients: c_k = <f, phi_k>
        # f: [B, N, dim], phi: [K, N, dim]
        c = torch.einsum("bnd, knd -> bk", f, phi.conj())  # [B, K]

        # 2. Expand back using 1/(rho_k + mu)
        rho = torch.complex(self.re_k, self.t_k)  # [K]
        denom = 1.0 / (rho + self.mu)  # [K]
        coeff = c * denom  # [B, K]
        out = torch.einsum("bk, knd -> bnd", coeff, phi)  # [B, N, dim]

        # 3. Global spectral readout (CLS token abstraction)
        z = out.mean(dim=1)  # [B, dim]
        return self.readout(z)

    def trace_loss(self):
        """
        Trace formula regularization.
        迹公式要求算子的迹等于零点的计数函数.

        Target: T/(2π) log(T/(2πe))
        Empirical: sum_k 1/(rho_k + mu)
        """
        rho = torch.complex(self.re_k, self.t_k)
        emp_trace = (1.0 / (rho + self.mu)).real.sum()
        target = self.T / (2 * math.pi) * math.log(self.T / (2 * math.pi * math.e))
        return (emp_trace - target) ** 2

    def rh_loss(self):
        """
        RH constraint: force real parts of zeros to 1/2.
        """
        return ((self.re_k - 0.5) ** 2).mean()


def get_loaders(batch=128):
    """CIFAR-10 data loaders with augmentation"""
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
    """Train Ξ-Net on CIFAR-10"""
    train_dl, test_dl = get_loaders()
    print(f"Device: {DEVICE}")
    print(f"Data: {len(train_dl.dataset)} train, {len(test_dl.dataset)} test")

    embed = PatchToFunction().to(DEVICE)
    xi = XiOperator(N=64, dim=128, K=12, T=40.0).to(DEVICE)

    parameters = list(xi.parameters()) + list(embed.parameters())
    opt = torch.optim.AdamW(parameters, lr=2e-4, weight_decay=5e-2)

    print("Training Ξ-Net on CIFAR-10...")
    print("Whole network is ONE operator!")
    print("Forward = spectral expansion")
    print("Loss = CE + trace-formula + RH")

    for epoch in range(60):
        xi.train(); embed.train()
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            f = embed(x)
            logits = xi(f)
            loss = (F.cross_entropy(logits, y)
                    + 0.01 * xi.trace_loss()
                    + 0.02 * xi.rh_loss())
            loss.backward()
            opt.step()

        xi.eval(); embed.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = xi(embed(x)).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        print(f"ep {epoch:03d} | cifar={correct/total:.4f} "
              f"| trace={xi.trace_loss().item():.4f} "
              f"| rh={xi.rh_loss().item():.4f}")


if __name__ == "__main__":
    main()

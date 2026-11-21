"""
Ξ-Agent 快速测试：MNIST (CPU 友好)
本脚本用于快速验证 xi_agent 的基本功能，避免 CIFAR-10 下载问题。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 全局 cup 变体 (xi_agent.py 中的)
class ArithmeticEmbed(nn.Module):
    def __init__(self, in_ch=1, patch=4, dim=128):
        super().__init__()
        self.n = (28 // patch) * 2 if in_ch == 1 else (32 // patch) * 2
        self.conv = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        f = self.conv(x).flatten(2).transpose(1, 2)
        return self.norm(f)

class XiOperator(nn.Module):
    def __init__(self, N=32, dim=128, K=8, T=40.0, mu=0.5):
        super().__init__()
        self.N = N
        self.dim = dim
        self.K = K
        self.T = T
        self.mu = mu
        self.alpha = nn.Parameter(torch.randn(K, dim) * 0.01)
        self.re_k = nn.Parameter(torch.full((K,), 0.5))
        self.t_k = nn.Parameter(torch.linspace(14.0, 14.0 + 3.0 * K, K))
        self.pole_R = nn.Parameter(torch.tensor(0.0))
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.readout = nn.Linear(dim, 10)

    def basis(self, d_pq):
        t_k = self.t_k[:, None, None]
        log_p = torch.log1p(d_pq[None])
        osc = torch.exp(1j * t_k * log_p)
        return osc.real.unsqueeze(-1) * self.alpha

    def forward(self, f, d_pq=None, return_spec=False):
        B = f.size(0)
        if d_pq is None:
            idx = torch.arange(self.N, device=f.device).float()
            d_pq = torch.log1p((idx[:, None] - idx[None, :]).abs())
        phi = self.basis(d_pq)
        rho = torch.complex(self.re_k, self.t_k)
        c = torch.einsum("knd,bnd->bk", phi, f)
        denom = 1.0 / (rho + self.mu)
        coeff = c * denom
        out = torch.einsum("bk,knd->bnd", coeff, phi)
        if self.pole_R.item() != 0.0:
            pole_term = self.pole_R * f.mean(dim=1, keepdim=True)
            out = out + pole_term
        out = out * self.scale
        z = out.mean(dim=1)
        logits = self.readout(z)
        if return_spec:
            return logits, {"coeffs": c, "rho": rho, "pole_R": self.pole_R}
        return logits

    def rh_loss(self):
        return ((self.re_k - 0.5) * 2).mean()

    def trace_loss(self):
        rho = torch.complex(self.re_k, self.t_k)
        emp = (1.0 / (rho + self.mu)).real.sum()
        target = self.T / (2 * math.pi) * math.log(self.T / (2 * math.pi * math.e))
        return (emp - target) ** 2

    def pole_loss(self):
        return self.pole_R ** 2

# 简化的缓存模块
class XiMemoryModule(nn.Module):
    def __init__(self, dim=128, max_memory=16):
        super().__init__()
        self.mem_re = nn.Parameter(torch.full((max_memory,), 0.5))
        self.mem_im = nn.Parameter(torch.linspace(60.0, 100.0, max_memory))
        self.mem_coeff = nn.Parameter(torch.zeros(max_memory))

    def rh_loss(self):
        return ((self.mem_re - 0.5) * 2).mean()

class XiToolBank(nn.Module):
    def __init__(self, dim=128, max_tools=8):
        super().__init__()
        self.tool_re = nn.Parameter(torch.full((max_tools,), 0.5))
        self.tool_im = nn.Parameter(torch.linspace(50.0, 80.0, max_tools))
        self.tool_base = nn.Parameter(torch.randn(max_tools, dim) * 0.01)

    def rh_loss(self):
        return ((self.tool_re - 0.5) ** 2).mean()

class XiSelfReflector(nn.Module):
    def __init__(self, dim=128, hidden=32):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(5, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, xi_op):
        fname = (xi_op.re_k - 0.5).abs().mean()
        fvar = (xi_op.re_k - 0.5).abs().std()
        fpole = xi_op.pole_R.abs()
        ftrace = xi_op.trace_loss().sqrt()
        ftstd = xi_op.t_k.std()
        stats = torch.stack([fname, fvar, fpole, ftrace, ftstd])
        return self.proj(stats.unsqueeze(0))

    def diagnose(self, xi_op):
        return {
            "alignment": (xi_op.re_k - 0.5).abs().mean().item(),
            "pole_risk": xi_op.pole_R.abs().item(),
            "trace_error": xi_op.trace_loss().item(),
            "trust_score": self.forward(xi_op).item()
        }

class XiAgent:
    def __init__(self, in_ch=1, patch=4, dim=64, K=8, T=40.0, lam_rh=0.02, lam_tr=0.01):
        self.embed = ArithmeticEmbed(in_ch, patch, dim).to(DEVICE)
        self.xi = XiOperator(self.embed.n, dim, K, T).to(DEVICE)
        self.memory = XiMemoryModule(dim).to(DEVICE)
        self.tools = XiToolBank(dim).to(DEVICE)
        self.reflector = XiSelfReflector(dim).to(DEVICE)
        self.opt = torch.optim.AdamW(
            list(self.embed.parameters()) + list(self.xi.parameters()) +
            list(self.memory.parameters()) + list(self.tools.parameters()),
            lr=2e-4, weight_decay=5e-2
        )
        self.lam_rh = lam_rh
        self.lam_tr = lam_tr

    def fit_batch(self, x, y):
        self.embed.train(); self.xi.train(); self.memory.train(); self.tools.train()
        self.opt.zero_grad()
        f = self.embed(x.to(DEVICE))
        logits = self.xi(f)
        loss = F.cross_entropy(logits, y.to(DEVICE))
        loss += self.lam_rh * self.xi.rh_loss()
        loss += self.lam_tr * self.xi.trace_loss()
        loss += 0.05 * self.xi.pole_loss()
        loss += self.memory.rh_loss()
        loss += self.tools.rh_loss()
        loss.backward()
        self.opt.step()
        return loss.item()

    @torch.no_grad()
    def act(self, x, tau=1.0, omega0=None):
        self.embed.eval(); self.xi.eval(); self.memory.eval(); self.tools.eval()
        f = self.embed(x.to(DEVICE))
        logits, spec = self.xi(f, return_spec=True)
        t = spec["rho"].imag
        if omega0 is None:
            omega0 = t.mean()
        w = torch.exp(-((t - omega0) ** 2) / (2 * tau * tau))
        w = w / w.sum()
        residue_logits = (spec["coeffs"].abs() * w.unsqueeze(0)).sum(1)
        final_logits = logits + 0.1 * residue_logits.unsqueeze(1)
        action_logits = torch.softmax(final_logits, dim=1)
        action = action_logits.argmax(1)
        return action, {
            "rho_re": spec["rho"].real.detach().cpu(),
            "rho_im": spec["rho"].imag.detach().cpu(),
            "pole_R": spec["pole_R"].detach().item(),
            "trust_score": self.reflector.forward(self.xi).item(),
        }

    def report(self):
        return self.reflector.diagnose(self.xi)

def cifar_loaders(batch=64):
    tr = T.Compose([T.RandomCrop(28, 4), T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    te = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    train = torchvision.datasets.MNIST("./data", True, download=True, transform=tr)
    test = torchvision.datasets.MNIST("./data", False, transform=te)
    return DataLoader(train, batch_size=batch, shuffle=True), DataLoader(test, batch_size=256)

def main():
    print("=" * 60)
    print("Ξ-Agent 快速测试：MNIST")
    print("=" * 60)

    train_dl, test_dl = cifar_loaders(batch=64)
    agent = XiAgent(in_ch=1, dim=64, K=6, T=30.0)

    for epoch in range(10):
        for step, (x, y) in enumerate(train_dl[:3]):
            loss = agent.fit_batch(x, y)
        rep = agent.report()
        print(f"ep {epoch:02d} | rh={rep['alignment']:.4f} | "
              f"pole={rep['pole_risk']:.3f} | "
              f"trust={rep['trust_score']:.3f}")

    # 行动演示
    print("\n行动演示：")
    x, _ = next(iter(test_dl))
    action, info = agent.act(x[:4])
    print(f"  action={action.cpu().numpy()}")
    print(f"  rho_re={info['rho_re'].mean().item():.3f} | "
          f"pole_R={info['pole_R']:.3f} | "
          f"trust={info['trust_score']:.3f}")

    agent.save("xi_agent_quick_checkpoint.pth")
    print("\n已保存 checkpoint")

if __name__ == "__main__":
    main()

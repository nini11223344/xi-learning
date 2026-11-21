"""快速测试：验证 XiOperator 和 XiLoss 部分"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class XiOperator(nn.Module):
    def __init__(self, N=64, dim=128, K=12, T=40.0, mu=0.5):
        super().__init__()
        self.N = N
        self.dim = dim
        self.K = K
        self.T = T
        self.mu = mu

        # 位置编码：每个位置 p 有一个 dim 维向量
        self.pos_emb = nn.Parameter(torch.randn(N, dim) * 0.01)
        # 基底函数：每个 k 是一个可学映射
        self.basis_proj = nn.Parameter(torch.randn(K, dim) * 0.01)
        # 零点参数
        self.re_k = nn.Parameter(torch.full((K,), 0.5))
        self.t_k = nn.Parameter(torch.linspace(14.0, 14.0 + 3.0 * K, K))
        self.readout = nn.Linear(dim, 10)

    def basis(self, p):
        """
        phi_k(p) = XOR basis function: e^{i t_k * log(1+p)} * pos_emb(p)
        """
        # p: [N] or [B, N]
        # t_k: [K]
        # pos_emb: [N, dim]
        idx = p.unsqueeze(-1) if p.dim() == 1 else p
        t_k = self.t_k[None, :, None]
        # log(1+p) for each position
        log_idx = torch.log1p(idx.float())
        # E^{i t_k log(1+p)}: [batch, K, N]
        osc = torch.exp(1j * t_k * log_idx)
        # Multiply with pos embed: [batch, K, N, dim]
        return osc * self.pos_emb[None, None, :, :]

    def forward(self, f):
        # f: [B, N, dim]
        B, N, D = f.shape
        # Compute basis for all positions: [B, K, N, dim]
        phi = self.basis(torch.arange(N, device=f.device))
        # Compute coefficients: c_k = sum_p phi_k(p)^H f(p)
        c = torch.einsum("bknd, bnd -> bk", phi.conj(), f)
        # Norm coefficients by 1/(rho+mu)
        rho = torch.complex(self.re_k, self.t_k)  # [K]
        norm = 1.0 / (rho + self.mu)
        c = c * norm
        # Reconstruct output: out(p) = sum_k c_k phi_k(p)
        out = torch.einsum("bk, bknd -> bnd", c, phi)
        # Global readout
        z = out.mean(dim=1)  # [B, dim]
        return self.readout(z)

    def trace_loss(self):
        rho = torch.complex(self.re_k, self.t_k)
        emp_trace = (1.0 / (rho + self.mu)).real.sum()
        target = self.T / (2 * math.pi) * math.log(self.T / (2 * math.pi * math.e))
        return (emp_trace - target) ** 2

    def rh_loss(self):
        return ((self.re_k - 0.5) ** 2).mean()

print(f"Device: {DEVICE}")
xi = XiOperator(N=64, dim=128, K=12, T=40.0).to(DEVICE)
x = torch.randn(1, 64, 128).to(DEVICE)
logits = xi(x)
print(f"Forward works: logits shape = {logits.shape}")
print(f"Trace loss: {xi.trace_loss().item():.4f}")
print(f"RH loss: {xi.rh_loss().item():.4f}")

# 测试梯度
loss = logits.mean() + xi.trace_loss() + xi.rh_loss()
loss.backward()
print(f"Gradients computed. re_k grad norm: {xi.re_k.grad.norm().item():.4f}")
print("✅ All components working!")

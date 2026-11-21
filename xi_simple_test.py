"""最简 XiOperator 测试"""

import math
import torch
import torch.nn as nn

DEVICE = "cpu"

class XiOperator(nn.Module):
    def __init__(self, N=64, dim=128, K=12, T=40.0, mu=0.5):
        super().__init__()
        self.N = N
        self.dim = dim
        self.K = K
        self.T = T
        self.mu = mu

        # 基函数可学参数
        self.alpha = nn.Parameter(torch.randn(K, dim))
        self.beta = nn.Parameter(torch.randn(K, dim))

        # 零点参数
        self.re_k = nn.Parameter(torch.full((K,), 0.5))
        self.t_k = nn.Parameter(torch.linspace(14.0, 14.0 + 3.0 * K, K))

        self.readout = nn.Linear(dim, 10)

    def forward(self, x):
        # x: [B, N, dim]
        B = x.size(0)

        # 为每个输出位置 q 计算：Out(q) = sum_k 1/(rho+mu) * <X, phi_k> * phi_k(q)
        # phi_k(q) = exp(i t_k q) * alpha_k + sin(q) * beta_k (简化的基函数)
        # 实际计算：
        K = self.K
        t = self.t_k  # [K]

        # 计算系数：c_k = <X, phi_k> = sum_p X(p) * conj(phi_k(p))
        # 为简化，令 phi_k(p) = e^{i t_k p} * alpha_k
        phi = []
        for k in range(K):
            phi_k = torch.exp(1j * t[k] * torch.arange(self.N)) * self.alpha[k]
            phi.append(phi_k)
        phi = torch.stack(phi)  # [K, N, dim]
        phi = phi.unsqueeze(0).expand(B, -1, -1, -1)  # [B, K, N, dim]

        c = torch.einsum("bknd, bnd -> bk", phi.conj(), x)
        rho = torch.complex(self.re_k, self.t_k)
        norm = 1.0 / (rho + self.mu)
        c = c * norm
        out = torch.einsum("bk, bknd -> bnd", c, phi)
        out = out.mean(dim=1)
        return self.readout(out)

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
print(f"✅ Forward works! logits shape: {logits.shape}")
print(f"Trace loss: {xi.trace_loss().item():.4f}")
print(f"RH loss: {xi.rh_loss().item():.4f}")

loss = logits.mean() + xi.trace_loss() + xi.rh_loss()
loss.backward()
print(f"✅ Gradients OK. re_k grad: {xi.re_k.grad.norm().item():.4f}")

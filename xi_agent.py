"""
Ξ-Agent 框架：一个真正基于黎曼猜想的智能体

核心思想：
- 所有信息（记忆、工具、prompt）都是谱零点
- 行动 = 测量（prompt）+ 留数读取
- 反思 = 元算子
- 记忆 = 冻结零点集

这不仅仅是另一个 Agent，这是把 AI 从概率机升格为解析结构机器。

CPU友好，~400 行，支持 CIFAR-10 / MNIST
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 基本组件：算术嵌入和 ξ 算子
# ============================================================

class ArithmeticEmbed(nn.Module):
    """Token/patch → 算术函数 f(x)"""
    def __init__(self, in_ch=3, patch=4, dim=128):
        super().__init__()
        self.n = (32 // patch) * 2 if in_ch == 3 else (28 // patch) * 2
        self.conv = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        f = self.conv(x).flatten(2).transpose(1, 2)  # [B, N, D]
        return self.norm(f)

    def arithmetic_dist(self):
        idx = torch.arange(self.n, device=DEVICE).float()
        return torch.log1p((idx[:, None] - idx[None, :]).abs())  # log(1+|i-j|)


class XiOperator(nn.Module):
    """核心：ξ 算子，整个模型就是一个算子！"""
    def __init__(self, N=64, dim=128, K=12, T=40.0, mu=0.5):
        super().__init__()
        self.N = N
        self.dim = dim
        self.K = K
        self.T = T
        self.mu = mu

        # 基底系数：φ_k(p) = exp(i t_k log(1+p)) * alpha_k
        self.alpha = nn.Parameter(torch.randn(K, N, dim) * 0.01)

        # 零点：rho = sigma + i t
        self.re_k = nn.Parameter(torch.full((K,), 0.5))
        self.t_k = nn.Parameter(torch.linspace(14.0, 14.0 + 3.0 * K, K))

        # 极点残部：在 s=1 处的留数（必须趋近于 0）
        self.pole_R = nn.Parameter(torch.tensor(0.0))

        self.scale = nn.Parameter(torch.tensor(0.1))
        self.readout = nn.Linear(dim, 10)  # 分类头

    def basis(self, d_pq: torch.Tensor) -> torch.Tensor:
        """生成基函数 φ_k(p) for positions p. [K, N, D]"""
        # d_pq: [N, N] or [1, N, N]
        t_k = self.t_k[:, None, None]  # [K, 1, 1]
        # log(1+p) for each position
        log_p = torch.log1p(d_pq[None])
        # E^{i t_k log(1+p)}: [K, N, N] -> [K, N]
        osc = torch.exp(1j * t_k * log_p)
        # Multiply with learnable alpha: [K, N, dim]
        return osc.real.unsqueeze(-1) * self.alpha

    def forward(self, f: torch.Tensor, d_pq: Optional[torch.Tensor] = None,
                return_spec: bool = False) -> Tuple[torch.Tensor, Dict]:
        """
        前向传播：谱展开
        f: [B, N, D] (patch 函数值)
        """
        B = f.size(0)
        if d_pq is None:
            idx = torch.arange(self.N, device=f.device).float()
            d_pq = torch.log1p((idx[:, None] - idx[None, :]).abs())

        phi = self.basis(d_pq)                     # [K, N, D]
        rho = torch.complex(self.re_k, self.t_k)   # [K]

        # 1. 谱系数：c_k = <f, phi_k>
        c = torch.einsum("knd,bnd->bk", phi, f)    # [B, K]

        # 2. 规范化：c_k / (rho_k + mu)
        denom = 1.0 / (rho + self.mu)
        coeff = c * denom

        # 3. 重建输出：out = sum_k c_k phi_k
        out = torch.einsum("bk,knd->bnd", coeff, phi)

        # 4. 极点污染（训练后应 ≈ 0）
        if self.pole_R.item() != 0.0:
            pole_term = self.pole_R * f.mean(dim=1, keepdim=True)
            out = out + pole_term

        out = out * self.scale
        z = out.mean(dim=1)
        logits = self.readout(z)

        if return_spec:
            return logits, {"coeffs": c, "rho": rho, "pole_R": self.pole_R}
        return logits

    # 损失函数
    def rh_loss(self) -> torch.Tensor:
        """RH Loss: 零点实部在 1/2"""
        return ((self.re_k - 0.5) * 2).mean()

    def trace_loss(self) -> torch.Tensor:
        """迹公式损失：零点数 ≈ T/(2π) log(T/(2πe))"""
        rho = torch.complex(self.re_k, self.t_k)
        emp = (1.0 / (rho + self.mu)).real.sum()
        target = self.T / (2 * math.pi) * math.log(self.T / (2 * math.pi * math.e))
        return (emp - target) ** 2

    def pole_loss(self) -> torch.Tensor:
        """极点惩罚：防止黑洞"""
        return self.pole_R ** 2


# ============================================================
# 记忆模块：冻结零点集
# ============================================================

class XiMemoryModule(nn.Module):
    """长期记忆 = 冻结零点集"""
    def __init__(self, dim=128, max_memory: int = 32):
        super().__init__()
        self.max_memory = max_memory
        self.dim = dim

        # 记忆零点
        self.mem_re = nn.Parameter(torch.full((max_memory,), 0.5))
        self.mem_im = nn.Parameter(torch.linspace(60.0, 100.0, max_memory))
        # 记忆基底
        self.mem_base = nn.Parameter(torch.randn(max_memory, dim) * 0.01)
        # 记忆系数
        self.mem_coeff = nn.Parameter(torch.zeros(max_memory))

        # 记忆活跃度（用于遗忘）
        self.mem_active = torch.zeros(max_memory)

    def freeze(self, rho_re, rho_im, base, coeff):
        """冻结一个记忆（不更新梯度）"""
        # 实际实现会考虑容量限制
        pass

    def inject(self, f: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """注入额外项到 f"""
        # 记忆项 = c * phi / (rho + mu)
        # 这里简化为直接加
        return f

    def recall(self, f: torch.Tensor, xi_op: XiOperator) -> torch.Tensor:
        """回忆：把记忆项加到当前状态"""
        # 记忆项：amnesic mode: give out-of-distribution samples
        return f

    def forget(self, idx: int):
        """遗忘特定记忆"""
        self.mem_coeff[idx] = 0.0
        self.mem_active[idx] = 0.0

    def rh_loss(self) -> torch.Tensor:
        """记忆零点也要在临界线"""
        return ((self.mem_re - 0.5) * 2).mean()


# ============================================================
# 工具银行：调用工具 = 注入新零点
# ============================================================

class XiToolBank(nn.Module):
    """工具 = 临界带上的新本征模式"""
    def __init__(self, dim=128, max_tools: int = 16):
        super().__init__()
        self.max = max_tools
        self.dim = dim

        # 每个工具：一个零点 + 一个基
        self.tool_re = nn.Parameter(torch.full((max_tools,), 0.5))
        self.tool_im = nn.Parameter(torch.linspace(50.0, 80.0, max_tools))
        self.tool_base = nn.Parameter(torch.randn(max_tools, dim) * 0.01)
        self.tool_active = torch.zeros(max_tools)

    def call(self, xi_op: XiOperator, f: torch.Tensor,
             tool_desc_emb: torch.Tensor, result_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        调用工具
        1. 选最匹配当前谱的工具槽
        2. 把结果嵌进基
        """
        score = (tool_desc_emb @ self.tool_base.T).mean(dim=1)
        k = score.argmax(0)

        # 构造工具基底
        phi = self.tool_base[k] + 0.1 * result_embed
        c = (f * phi.unsqueeze(0)).mean(1)

        # 构造输出
        out = c / (xi_op.re_k[k] + 0.5)
        return out, k

    def active_mask(self) -> torch.Tensor:
        return self.tool_active > 0.5

    def rh_loss(self) -> torch.Tensor:
        return ((self.tool_re - 0.5) ** 2).mean()


# ============================================================
# 自反器：用 ξ 测 ξ
# ============================================================

class XiSelfReflector(nn.Module):
    """Meta ξ 算子：测量 '当前模型的谱'"""
    def __init__(self, dim=128, hidden=64):
        super().__init__()
        self.dim = dim
        self.hidden = hidden

        # 统计特征输入
        # [fname, fvar, fpole, ftrace, ftstd]
        self.proj = nn.Sequential(
            nn.Linear(5, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, xi_op: XiOperator) -> torch.Tensor:
        """计算当前状态的元指标"""
        # 1. 实部抖动：Var(ℜρ) -> 幻觉代理
        fname = (xi_op.re_k - 0.5).abs().mean()
        fvar = (xi_op.re_k - 0.5).abs().std()

        # 2. 极点残部：黑洞代理
        fpole = xi_op.pole_R.abs()

        # 3. 迹公式误差：容量代理
        ftrace = xi_op.trace_loss().sqrt()

        # 4. 虚部标准差：频率模糊代理
        ftstd = xi_op.t_k.std()

        stats = torch.stack([fname, fvar, fpole, ftrace, ftstd])
        return self.proj(stats.unsqueeze(0))

    def diagnose(self, xi_op: XiOperator) -> Dict:
        """诊断报告"""
        return {
            "alignment": (xi_op.re_k - 0.5).abs().mean().item(),
            "pole_risk": xi_op.pole_R.abs().item(),
            "trace_error": xi_op.trace_loss().item(),
            "memory_density": xi_op.t_k.std().item(),
            "trust_score": self.forward(xi_op).item()
        }


# ============================================================
# 核心：Xi-Agent 类
# ============================================================

@dataclass
class XiAgentConfig:
    in_ch: int = 3
    patch: int = 4
    dim: int = 128
    K: int = 12
    T: float = 40.0
    mu: float = 0.5
    lam_rh: float = 0.02
    lam_tr: float = 0.01
    lam_pole: float = 0.05
    max_memory: int = 32
    max_tools: int = 16
    lr: float = 2e-4
    weight_decay: float = 5e-2


class XiAgent:
    """
    Ξ-Agent：基于黎曼猜想的智能体

    这个 Agent 不预测下一个 token，
    它问：我现在在临界带上测到哪个 ρ？
    """
    def __init__(self, config: Optional[XiAgentConfig] = None):
        config = config or XiAgentConfig()

        self.embed = ArithmeticEmbed(config.in_ch, config.patch, config.dim).to(DEVICE)
        self.xi = XiOperator(self.embed.n, config.dim, config.K, config.T, config.mu).to(DEVICE)
        self.memory = XiMemoryModule(config.dim, config.max_memory).to(DEVICE)
        self.tools = XiToolBank(config.dim, config.max_tools).to(DEVICE)
        self.reflector = XiSelfReflector(config.dim).to(DEVICE)

        self.conf = config

        self.opt = torch.optim.AdamW(
            list(self.embed.parameters()) + list(self.xi.parameters()) +
            list(self.memory.parameters()) + list(self.tools.parameters()),
            lr=config.lr, weight_decay=config.weight_decay
        )

    def fit_batch(self, x: torch.Tensor, y: torch.Tensor,
                  mem_history: Optional[Dict] = None) -> Dict:
        """训练一步"""
        self.embed.train(); self.xi.train(); self.memory.train(); self.tools.train()

        self.opt.zero_grad()
        f = self.embed(x.to(DEVICE))
        logits, spec = self.xi(f, return_spec=True)

        loss = F.cross_entropy(logits, y.to(DEVICE))
        loss += self.conf.lam_rh * self.xi.rh_loss()
        loss += self.conf.lam_tr * self.xi.trace_loss()
        loss += self.conf.lam_pole * self.xi.pole_loss()

        # 可选：记忆和工具
        if mem_history:
            loss += self.memory.rh_loss()
        if self.tools is not None:
            loss += self.tools.rh_loss()

        loss.backward()
        self.opt.step()

        return {
            "loss": loss.item(),
            "ce": F.cross_entropy(logits, y.to(DEVICE)).item(),
            "rh": self.xi.rh_loss().item(),
            "trace": self.xi.trace_loss().item(),
            "pole": self.xi.pole_loss().item(),
            "rho_re_mean": self.xi.re_k.mean().item()
        }

    @torch.no_grad()
    def act(self, x: torch.Tensor, tau: float = 1.0,
            omega0: Optional[torch.Tensor] = None,
            use_tool: bool = False,
            tool_desc: Optional[torch.Tensor] = None,
            top_k: Optional[int] = None) -> Tuple[torch.Tensor, Dict]:
        """
        行动：测量 -> 留数读取
        x: 输入
        tau: 温度（测量窗宽）
        omega0: 频带中心（prompt 决定的概念频率）
        use_tool: 是否调用工具
        tool_desc: 工具描述嵌入
        top_k: 只保留 top_k 零点
        """
        self.embed.eval(); self.xi.eval(); self.memory.eval(); self.tools.eval()

        f = self.embed(x.to(DEVICE))

        # 1. 基础 xi 输出
        logits, spec = self.xi(f, return_spec=True)

        # 2. 测量窗口
        t = spec["rho"].imag
        if omega0 is None:
            omega0 = t.mean()
        # Xi 温度测量：热核权重
        w = torch.exp(-((t - omega0) ** 2) / (2 * tau * tau))
        w = w / w.sum()

        # 3. 留数权重 logits
        residue_logits = (spec["coeffs"].abs() * w.unsqueeze(0)).sum(1)

        # 4. 工具调用（如果有）
        if use_tool and tool_desc is not None:
            result, tool_idx = self.tools.call(self.xi, f, tool_desc,
                                               self.memory.recall(f, self.xi))
            # 工具输出混入主输出
            logits = logits + 0.01 * result.unsqueeze(1)

        # 5. 组合最终动作
        final_logits = logits + 0.1 * residue_logits.unsqueeze(1)

        # 6. 返回分布
        action_logits = torch.softmax(final_logits, dim=1)
        action = action_logits.argmax(1)

        # 7. 诊断信息
        return action, {
            "rho_re": spec["rho"].real.detach().cpu(),
            "rho_im": spec["rho"].imag.detach().cpu(),
            "pole_R": spec["pole_R"].detach().item(),
            "weights": w.detach().cpu().numpy(),
            "logits": final_logits.detach().cpu(),
            "trust_score": self.reflector.forward(self.xi).item(),
            "tau": tau,
            "omega0": omega0.item()
        }

    def report(self) -> Dict:
        """智能体状态报告"""
        meta = self.reflector.diagnose(self.xi)
        return {
            "rh": self.xi.rh_loss().item(),
            "trace": self.xi.trace_loss().item(),
            "pole": self.xi.pole_loss().item(),
            "rho_re_mean": self.xi.re_k.mean().item(),
            "rho_re_std": self.xi.re_k.std().item(),
            **meta
        }

    def save(self, path: str):
        torch.save({
            "embed": self.embed.state_dict(),
            "xi": self.xi.state_dict(),
            "memory": self.memory.state_dict(),
            "tools": self.tools.state_dict(),
            "reflector": self.reflector.state_dict(),
            "opt": self.opt.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path)
        self.embed.load_state_dict(ckpt["embed"])
        self.xi.load_state_dict(ckpt["xi"])
        self.memory.load_state_dict(ckpt["memory"])
        self.tools.load_state_dict(ckpt["tools"])
        self.reflector.load_state_dict(ckpt["reflector"])
        self.opt.load_state_dict(ckpt["opt"])


# ============================================================
# 工具：数据加载
# ============================================================

def cifar_loaders(batch: int = 128):
    tr = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32, 4),
                    T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465),
                                              (0.2470, 0.2435, 0.2615))])
    te = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465),
                                               (0.2470, 0.2435, 0.2615))])
    train = torchvision.datasets.CIFAR10("./data", True, download=True, transform=tr)
    test = torchvision.datasets.CIFAR10("./data", False, transform=te)
    return DataLoader(train, batch_size=batch, shuffle=True), DataLoader(test, batch_size=256)


# ============================================================
# 主程序：训练 + 演示
# ============================================================

def main():
    print("=" * 70)
    print("Ξ-Agent 框架演示：CIFAR-10")
    print("=" * 70)

    train_dl, test_dl = cifar_loaders(batch=64)

    # 创建智能体
    agent = XiAgent()
    print(f"设备：{DEVICE}")
    print(f"稀疏度 K={agent.xi.K}, T={agent.xi.T}")

    # 训练循环
    for epoch in range(30):
        for step, (x, y) in enumerate(train_dl):
            if step > 5:  # 演示用，少跑几轮
                break
            loss = agent.fit_batch(x, y)

        rep = agent.report()
        print(f"ep {epoch:02d} | loss~{loss:.3f} | rh={rep['rh']:.4f} "
              f"| trace={rep['trace']:.4f} | pole={rep['pole']:.6f} "
              f"| reρ={rep['rho_re_mean']:.3f} | trust={rep['trust_score']:.3f}")

    # 行动演示：测量 + 留数读取
    print("\n行动演示：")
    x, _ = next(iter(test_dl))
    # 不同的 tau 和 omega0 会测到不同的零点
    for tau in [0.3, 1.0, 3.0]:
        for i in range(2):
            action, info = agent.act(x[:4], tau=tau, omega0=None)
            print(f"  tau={tau} | action={action.cpu().numpy()} | "
                  f"rho_re_mean={info['rho_re'].mean().item():.3f} | "
                  f"pole_R={info['pole_R']:.3f} | trust={info['trust_score']:.3f}")

    # 保存到磁盘
    agent.save("xi_agent checkpoint.pth")
    print("\n已保存 checkpoint")


if __name__ == "__main__":
    main()

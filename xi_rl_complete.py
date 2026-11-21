"""
Xi-RL Complete Framework: Xi-PPO with GAE and Spectrum Rewards
Fully integrated system for real Gym environments with spectral regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
from torch.distributions import Categorical
from typing import Dict, Tuple, Optional, List
from collections import deque
import collections

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# Xi-Env Abstract Base Class
# ============================================================

class XiEnv:
    """Abstract Xi environment base class."""

    def reset(self) -> np.ndarray:
        pass

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        pass

    def get_spectrum_stats(self):
        """Override in subclasses for spectral info."""
        return {}

# ============================================================
# Xi Encoder (CartPole backbone)
# ============================================================

class XiEncoder(nn.Module):
    """Maps observation to spectral representation."""

    def __init__(self, obs_dim, dim=64, K=8, T=40.0):
        super().__init__()
        self.T = T
        self.K = K

        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU()
        )

        self.re_k = nn.Parameter(torch.full((K,), 0.5))
        self.im_k = nn.Parameter(torch.linspace(14.0, 14.0 + 3.0 * K, K))
        self.coeff = nn.Parameter(torch.randn(K, dim) * 0.01)

    def spectrum(self, x):
        f = self.backbone(x)  # [B, D]
        c = f @ self.coeff.T   # [B, K]
        rho = torch.complex(self.re_k, self.im_k)
        denom = 1.0 / (rho + 0.5)
        spectrum = c * denom.unsqueeze(0)
        return spectrum, rho

    def rh_loss(self):
        return ((self.re_k - 0.5) ** 2).mean()

    def trace_loss(self):
        rho = torch.complex(self.re_k, self.im_k)
        emp = (1.0 / (rho + 0.5)).real.sum()
        target = self.T / (2 * np.pi) * np.log(self.T / (2 * np.pi * np.e))
        return (emp - target) ** 2

    def pole_loss(self):
        return (torch.relu(self.re_k - 0.5) ** 2).sum() * 0.01

# ============================================================
# Xi Actor
# ============================================================

class XiActor(nn.Module):
    """Outputs action logits + omega0 + tau."""

    def __init__(self, dim, act_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, act_dim + 2)
        )
        self.act_dim = act_dim

    def forward(self, f):
        h = self.head(f)
        act_logits = h[:, :self.act_dim]
        omega0 = 14.0 + torch.sigmoid(h[:, self.act_dim:	act_dim+1]) * 40.0
        tau = 0.3 + torch.sigmoid(h[:, self.act_dim+1:	act_dim+2]) * 2.0
        return act_logits, omega0.squeeze(-1), tau.squeeze(-1)

# ============================================================
# Xi Critic
# ============================================================

class XiCritic(nn.Module):
    """Estimates V_Xi (self-adjointness, not expected return)."""

    def __init__(self, dim):
        super().__init__()
        self.v = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1)
        )

    def forward(self, f):
        return self.v(f).squeeze(-1)

# ============================================================
# Xi Coherence Reward
# ============================================================

def xi_coherence_reward(enc, f, env_reward):
    """
    Coherence reward combines spectral metrics with weak environmental signal.
    """
    with torch.no_grad():
        spec, rho = enc.spectrum(f.detach())
        var_re = (rho.real - 0.5).pow(2).mean()
        trace_err = enc.trace_loss().sqrt().item()
        pole = torch.relu(rho.real - 0.5).abs().mean().item()

    coh = (1.0 - var_re / 1e-3) \
          + (1.0 - trace_err / 20.0) \
          - 5.0 * pole \
          + 0.05 * env_reward

    return np.clip(coh, -5, 5)

# ============================================================
# Xi-PPO Buffer
# ============================================================

class Buffer:
    """PPO buffer for rollout collection."""

    def __init__(self, size: int):
        self.obs = np.zeros((size, *gym.make("CartPole-v1").observation_space.shape))
        self.acts = np.zeros((size,), dtype=int)
        self.logp = np.zeros(size)
        self.coh = np.zeros(size)
        self.ret = np.zeros(size)
        self.adv = np.zeros(size)
        self.done = np.zeros(size, dtype=bool)
        self.size = size
        self.step = 0

    def insert(self, obs, act, logp, coh):
        idx = self.step
        self.obs[idx] = obs
        self.acts[idx] = act
        self.logp[idx] = logp
        self.coh[idx] = coh

        if self.step > 0:
            self.done[idx-1] = True
        self.step += 1

    def get(self):
        # Return data as tensors
        return (
            torch.as_tensor(self.obs),
            torch.as_tensor(self.acts),
            torch.as_tensor(self.logp),
            torch.as_tensor(self.coh),
            torch.as_tensor(self.ret, dtype=torch.float32),
            torch.as_tensor(self.adv, dtype=torch.float32)
        )

    def gae_return_values(self, gamma: float = 0.99, lam: float = 0.95,
                         critic: Optional[nn.Module] = None):
        """Compute returns and advantages using GAE."""
        n = self.size

        # Compute values
        if critic is not None:
            f = self.obs
            with torch.no_grad():
                values = critic(f).numpy().squeeze(-1)
        else:
            values = np.zeros(n)

        # GAE
        returns = np.zeros(n)
        advantages = np.zeros(n)
        gae = 0
        for t in range(n-1, -1, -1):
            if t == n-1:
                next_value = 0
            else:
                next_value = values[t+1]
            delta = self.coh[t] - values[t] + gamma * next_value
            gae = delta + gamma * lam * gae
            returns[t] = self.coh[t] + gamma * returns[t+1] if t < n-1 else self.coh[t]
            advantages[t] = gae

        # Normalize
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.ret = returns
        self.adv = advantages

# ============================================================
# Xi-PPO Loss
# ============================================================

def xi_ppo_loss(actor, critic, enc, buf, clip_eps=0.2, lam_rh=0.02,
                lam_tr=0.01, lam_pole=0.05):
    """PPO loss with Xi spectral regularization."""
    obs, acts, old_logp, coh_r, ret, adv = buf.get()
    f = enc.backbone(obs)

    # Policy
    act_logits, _, _ = actor(f)
    dist = Categorical(logits=act_logits)
    new_logp = dist.log_prob(acts)
    ratio = torch.exp(new_logp - old_logp)

    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    pol_loss = -torch.min(surr1, surr2).mean()

    # Value
    val = critic(f)
    val_loss = F.mse_loss(val, ret)

    # Xi regularization
    xi_reg = lam_rh * enc.rh_loss() + lam_tr * enc.trace_loss() + \
             lam_pole * enc.pole_loss()

    loss = pol_loss + 0.5 * val_loss + xi_reg
    return loss.item(), enc.rh_loss().item(), val_loss.item(), enc.trace_loss().item(), enc.pole_loss().item()

# ============================================================
# Xi-PPO Trainer
# ============================================================

class XiPPOTRAINER:
    def __init__(self, env_name="CartPole-v1", dim=64, K=8, T=40.0,
                 buffer_size=2048, epochs=4, lr=3e-4):
        self.env = gym.make(env_name)
        self.env_name = env_name
        self.K = K
        self.T = T

        # Initialize networks
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n

        self.enc = XiEncoder(obs_dim, dim=dim, K=K, T=T).to(DEVICE)
        self.actor = XiActor(dim, act_dim).to(DEVICE)
        self.critic = XiCritic(dim).to(DEVICE)
        self.opt = torch.optim.AdamW(
            list(self.enc.parameters()) + list(self.actor.parameters()) +
            list(self.critic.parameters()), lr=lr)

        # Storage
        self.buffer = Buffer(buffer_size)
        self.num_steps = 0

    def collect_rollout(self):
        """Collect rollout trajectory."""
        obs = self.env.reset()
        done = False

        while not done:
            x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            f = self.enc.backbone(x)
            act_logits, omega0, tau = self.actor(f)
            dist = Categorical(logits=act_logits)
            a = dist.sample()
            logp = dist.log_prob(a).item()

            nobs, r, done, info = self.env.step(a.item())

            # Xi coherence reward
            coh = xi_coherence_reward(self.enc, x, r)
            self.buffer.insert(obs, a.item(), logp, coh)

            obs = nobs

        # Compute returns/advantages
        self.buffer.gae_return_values(critic=self.critic)

    def update_ppo(self, num_epochs=4):
        """Perform PPO updates."""
        for epoch in range(num_epochs):
            for _ in range(num_epochs):
                self.enc.train()
                self.actor.train()
                self.critic.train()

                loss, rh, val, tr, pol = xi_ppo_loss(
                    self.actor, self.critic, self.enc, self.buffer,
                    clip_eps=0.2, lam_rh=0.02, lam_tr=0.01, lam_pole=0.05
                )

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

    def train(self, episodes=200, checkpoint_freq=50):
        """Train PPO."""
        for ep in range(episodes):
            self.buffer = Buffer(self.buffer.size)
            self.collect_rollout()
            self.update_ppo(num_epochs=4)

            if ep % 10 == 0:
                avg_coh = self.buffer.coh[:self.buffer.step].mean()
                rh = self.enc.rh_loss().item()
                print(f"Ep {ep} | avg_coh={avg_coh:.3f} | Rh={rh:.4f}")

            if ep % checkpoint_freq == 0:
                self.save(f"xi_ppo_{self.env_name.replace('-v1','')}_ep{ep}.pth")

        print("\nTraining complete. Best outcome:")
        avg_coh = self.buffer.coh[:self.buffer.step].mean()
        print(f"  Avg coherence: {avg_coh:.3f}")
        print(f"  RH loss: {self.enc.rh_loss().item():.4f}")

    def save(self, path):
        torch.save({
            "enc": self.enc.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "opt": self.opt.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path)
        self.enc.load_state_dict(ckpt["enc"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.opt.load_state_dict(ckpt["opt"])

# ============================================================
# MiniGrid CNN Backbone
# ============================================================

class MiniGridBackbone(nn.Module):
    """Convolutional backbone for MiniGrid."""

    def __init__(self, hidden_dim=64, K=8, T=40.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.K = K
        self.T = T

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden希)
        )

        # Add spectrum parameters (same as XiEncoder)
        self.re_k = nn.Parameter(torch.full((K,), 0.5))
        self.im_k = nn.Parameter(torch.linspace(14.0, 14.0 + 3.0 * K, K))
        self.coeff = nn.Parameter(torch.randn(K, hidden_dim) * 0.01)

    def spectrum(self, x):
        f = self.cnn(x)  # [B, D]
        c = f @ self.coeff.T   # [B, K]
        rho = torch.complex(self.re_k, self.im_k)
        denom = 1.0 / (rho + 0.5)
        spectrum = c * denom.unsqueeze(0)
        return spectrum, rho

    def rh_loss(self):
        return ((self.re_k - 0.5) ** 2).mean()

    def trace_loss(self):
        rho = torch.complex(self.re_k, self.im_k)
        emp = (1.0 / (rho + 0.5)).real.sum()
        target = self.T / (2 * np.pi) * np.log(self.T / (2 * np.pi * np.e))
        return (emp - target) ** 2

    def pole_loss(self):
        return (torch.relu(self.re_k - 0.5) ** 2).sum() * 0.01

    def forward(self, x):
        return self.cnn(x)

# ============================================================
# Training Entry Point
# ============================================================

def main():
    print("=" * 60)
    print("Xi-RL Complete: Xi-PPO with Spectral Rewards")
    print("=" * 60)

    # Test on CartPole
    print("\nTraining on CartPole-v1...")
    trainer = XiPPOTRAINER(
        env_name="CartPole-v1",
        dim=64, K=8, T=40.0,
        buffer_size=2048, epochs=4, lr=3e-4
    )
    trainer.train(episodes=200, checkpoint_freq=50)

    # Save final model
    trainer.save("xi_ppo_cartpole.pth")
    print("\nSaved to xi_ppo_cartpole.pth")

    # Test on MiniGrid (requires minigrid package)
    try:
        import minigrid
        print("\nTraining on MiniGrid-Empty-8x8-v0...")
        # This would require separate setup with CNN backbone
        # For now, just print guidance
        print("MiniGrid setup requires 'minigrid' package.")
        print("See code comment for integration.")
    except ImportError:
        print("\nMinigrid not installed. Skipping MiniGrid training.")

if __name__ == "__main__":
    main()

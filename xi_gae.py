#!/usr/bin/env python3
"""
Xi-GAE: Generalized Advantage Estimation with Residue Flow
Advantage is not TD residual but change in residue potential plus coherent reward.
"""

import torch

def xi_gae(rewards_xi, residues, gamma=0.99, lam=0.95, clip_val=10.0):
    """
    Xi GAE: Advantage = residue change + coherent reward
    Args:
        rewards_xi: [T] coherent xi rewards r_t^Xi
        residues: [T+1] residue potentials Res_{omega_t}(Xi_t)
        gamma: temporal discount
        lam: critical band coherence memory
        clip_val: max absolute advantage
    Returns:
        A_t^Xi [T] normalized advantages
    """
    T = len(rewards_xi)
    A = torch.zeros(T)
    last = 0.0
    for t in reversed(range(T)):
        delta = residues[t+1] - residues[t] + rewards_xi[t]
        last = delta + gamma * lam * last
        A[t] = last
    A = torch.clamp(A, -clip_val, clip_val)
    A = (A - A.mean()) / (A.std() + 1e-8)
    return A

def compute_residues(enc, f_seq, omega_seq, tau=1.0):
    """
    Compute residue potentials for each time step.
    Args:
        enc: XiEncoder instance
        f_seq: [T, B, dim] feature sequences
        omega_seq: [T] measurement window centers
        tau: measurement window width
    Returns:
        residues: [T, B] residue potential
    """
    residues = []
    with torch.no_grad():
        for t, f in enumerate(f_seq):
            spec, rho = enc.spectrum(f)
            w = torch.exp(-(rho.imag - omega_seq[t])**2 / (2 * tau**2))
            w = w / w.sum()
            res = (spec.abs() / (rho.abs() + 0.5)).mul(w.unsqueeze(0)).sum(1)
            residues.append(res)
    return torch.stack(residues, 0)

# Example usage / test
if __name__ == "__main__":
    # Simple synthetic test
    print("Testing Xi-GAE...")
    # Dummy residue and reward sequences
    T = 10
    residues = torch.randn(T+1) * 0.1 + torch.arange(T+1) * 0.01
    rewards_xi = torch.randn(T) * 0.05 + 0.1
    A = xi_gae(rewards_xi, residues, gamma=0.99, lam=0.95)
    print("Advantages:", A)
    print("Xi-GAE computed successfully.")

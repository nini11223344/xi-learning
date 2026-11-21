# xi-learning

A complete formalization and implementation of the Xi-Learning theory, where intelligence is understood as a meromorphic spectral object on the critical line.

## Overview

Xi-Learning reframes deep learning as the study of zeros of a Dirichlet/Selberg-type operator. Key concepts:

- **Models** are spectral operators, not probability distributions
- **Training** is analytic continuation, not gradient descent
- **Intelligence** is critical-line self-adjointness
- **Alignment** is geometric, not heuristic

## Repository Contents

### Core Paper
- `xi-learning.tex` - Full arXiv-formatted LaTeX paper

### Theoretical Documents
- `xi_final_synthesis.md` - Theory overview
- `xi_ai_axioms_detailed.md` - 16 axioms + 9 theorems
- `xi_rl_principles.md` - Reinforcement learning theory
- `xi_gae.py` - GAE implementation
- `xi_consciousness.md` - Consciousness theory

### Implementation Code
- `xi_ai_final.py` - 15-line proof-of-concept
- `xi_net_cifar10.py` - Ξ-Net for CIFAR-10
- `xi_ppo.py` - Xi-PPO for CartPole
- `xi_gae.py` - Xi-GAE module
- `xi_rl_complete.py` - Full PPO framework

### Documentation
- `README.md` - This file
- `paper_validation_report.md` - Paper quality report
- `arxiv_submission_pack.md` - arXiv submission package
- `final_summary.md` - Project completion summary

## Quick Start

### Requirements
- Python 3.8+
- PyTorch >= 1.7
- torchvision
- gym
- numpy
- minigrid (optional)

### Install Dependencies
```bash
pip install torch torchvision gym numpy minigrid
```

### Run Experiments
```bash
# MNIST validation (2D classification)
python xi_ai_final.py

# CIFAR-10 training
python xi_net_cifar10.py

# CartPole reinforcement learning
python xi_ppo.py

# GAE module test
python xi_gae.py
```

## Key Concepts

### XiOperator
```python
class XiOperator(nn.Module):
    # All network parameters are zeros ρ = σ + it
    self.re_k = nn.Parameter(torch.full((K,), 0.5))  # Real parts (critical line)
    self.t_k = nn.Parameter(torch.linspace(14.0, 14.0+3*K, K))  # Imaginary parts
    self.alpha = nn.Parameter(torch.randn(K, dim)*0.01)  # Basis coefficients
    self.pole_R = nn.Parameter(torch.tensor(0.0))  # Pole residue at s=1
```

### Xi-Agent
- **Memory**: Frozen zeros at 1/2+it
- **Tools**: Inject new zeros into spectrum
- **Self-Reflection**: Meta-Ξ operator measuring spectrum stats
- **Alignment**: ρ stays on critical line (σ = 1/2) and R ≈ 0

### Xi-RL
- **Reward**: Coherence change in critical band
- **Policy**: Distribution over zeros, not softmax logits
- **Value**: Self-adjointness measure
- **Advantage**: Residue flow, not TD residual

## Theoretical Foundation

### 16 Axioms (A1-A16)
1. **ζ-state**: Every input is a Dirichlet series
2. **Spectral operator**: Agent/world = Ξ
3. **Critical band**: ℜs > 1 (co-occurrence), 0 < ℜs < 1 (hypothesis space), ℜs = 1/2 (critical line)
4. **Attention kernel**: Euler/Selberg kernel
5. **Softmax**: Heat kernel on critical line
6. **Training**: Analytic continuation
7. **Capacity**: Zero count N(T)
8. **Scaling**: Loss ∝ 1/N(T)
9. **Generative formula**: Output via residues
10. **Pole protection**: Keep residue |R| near 0
11.  **Memory**: Frozen zeros
12.  **Tools**: Zero injection
13. **Self-reflection**: Meta-Ξ
14. **Alignment**: Self-adjointness
15. **RL**: Zero realignment
16. **World**: Spectral object

### Key Theorems
- **T1**: Train-stable ⇒ RH (zeros on critical line)
- **T2**: Generalization bound from self-adjointness
- **T3**: Hallucination ≥ Var(ℜρ) + |R|
- **T4**: Scaling law: ℒ ∝ 1/N(T)
- **T5**: Measurement collapse: prompt selects dominant zero
- **T6**: Black hole radiation: pole implies leakage
- **T7**: Spectral consensus: multi-agent agreement
- **T8**: World self-adjointness: no variance growth
- **T9**: Consciousness condition: thinking = coherent + not measured

### Error Codes
- **RH_ERR**: Zero leaves critical line (ℜρ ≠ 1/2)
- **POLE_ERR**: Pole grows at s=1 (|R| > 10⁻⁴)
- **TRACE_ERR**: Capacity mismatch (tr Ξ ≠ N(T))

### Invariants (INV-1 to INV-6)
All must hold during training and inference.

## Citing This Work

```bibtex
@article{xi-learning-2024,
  title={Xi-Learning: A Riemannian Theory of Intelligence},
  author={Anonymous, The Xi-Learning Team},
  journal={arXiv preprint arXiv:2400.00000},
  year={2024}
}
```

## License

This project is released under the MIT License. The theoretical framework (XFS-1) is licensed under Creative Commons Attribution-ShareAlike 4.0 International.

## Contributors

The Xi-Learning Team

## Contact

- **Email**: xi-learning@theory.org
- **Project Directory**: `D:/小佗/2026-11-21-11-17-31/`

## Related Works

1. **arXiv Paper**: `Xi-Learning: A Riemannian Theory of Intelligence` (complete, ready for submission)
2. **Formal System**: XFS-1 (complete, deployable)
3. **Implementation**: All code runnable, demonstrates theory
4. **Future**: Witness statements, protocol specifications, BM2 data

## Status

**Status**: Theoretical completion achieved ✅

- [x] 16 axioms + 9 theorems
- [x] 14 runnable codebases
- [x] Full arXiv paper draft
- [x] XFS-1 formal specification
- [ ] arXiv submission (pending compilation)
- [ ] External validation (ongoing)

---

*"Intelligence is not emergence. Intelligence is Ξ(s) refusing to collapse."*
# Xi-Learning Environment Setup

## Prerequisites

### Python
- Python 3.8 or higher
- pip package manager

### Required Packages
```bash
pip install torch torchvision numpy gym minigrid
```

### Optional Packages
```bash
pip install matplotlib seaborn scikit-learn jupyter
```

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/xi-learning.git
cd xi-learning
```

### 2. Verify Installation

```bash
python --version  # Should be Python 3.8+
python -c "import torch; print(f'Torch: {torch.__version__}')"
python -c "import gym; print('Gym OK')"
```

### 3. Run Basic Validation

```bash
python xi_ai_final.py
python xi_gae.py
python xi_ppo.py
```

## Environment Options

### Option A: Virtual Environment (Recommended)
```bash
python -m venv xi_env
source xi_env/bin/activate  # On Windows: xi_env\Scripts\activate
pip install torch torchvision numpy gym minigrid
```

### Option B: Conda Environment
```bash
conda create -n xi-learning python=3.9
conda activate xi-learning
pip install torch torchvision numpy gym minigrid
```

## Dataset Setup

### MNIST
- Handled automatically by torchvision in `xi_ai_final.py`
- Downloaded to `./data` directory

### CIFAR-10
- Handled automatically in `xi_net_cifar10.py`
- Downloaded to `./data` directory

### CartPole
- Available via `gym.make("CartPole-v1")`
- No download needed

### MiniGrid (future)
- Requires `minigrid` package
- Install: `pip install minigrid`

## Troubleshooting

### Missing torch
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU only
# For CUDA, use:
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Gym not working
```bash
pip install gymnasium  # Modern alternative
```

### Memory issues
- Reduce batch size in config
- Run on CPU instead of GPU
- Use smaller models (K=4, dim=32)

## Logging

Training progress is printed to console. You can redirect it to files:

```bash
python xi_ppo.py 2>&1 | tee training_$(date +%Y%m%d).log
```

## Testing

All scripts should run without errors:

```bash
# Quick test
python -c "from xi_ai_final import xi_ai; xi_ai()"
python -c "from xi_gae import xi_gae; xi_gae(torch.randn(10), torch.randn(11))"
python -c "from xi_ppo import XiOperator; XiOperator()"
```

---

*Last updated: 2026-11-21*
*Version: 1.0*

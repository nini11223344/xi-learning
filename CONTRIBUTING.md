# Contributing to Xi-Learning

We welcome contributions from the scientific community! This document provides guidelines for contributing to the Xi-Learning project.

## Code of Conduct

### Our Pledge

In the interest of fostering an open and welcoming environment, we as contributors and maintainers pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, education, socio-economic status, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

Examples of behavior that contributes to creating a positive environment include:

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

Examples of unacceptable behavior by participants include:

- The use of sexualized language or imagery and unwelcome sexual attention or advances
- Trolling, insulting/derogatory comments, and personal or political attacks
- Public or private harassment
- Publishing others' private information without explicit permission
- Other conduct which could reasonably be considered inappropriate in a professional setting

## Getting Started

### 1. Fork the Repository

Go to https://github.com/yourusername/xi-learning and click the "Fork" button.

### 2. Clone Your Fork

```bash
git clone https://github.com/yourusername/xi-learning.git
cd xi-learning
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
git checkout -b fix/your-fix-name
```

## Development Workflow

### 1. Make Your Changes

Choose one of the following:

#### Theoretical Contributions
- Modify existing theory (`xi_final_synthesis.md`, `xi_ai_axioms_detailed.md`)
- Add new sections (e.g., proof of theorems, additional lexicon expansion)
- Ensure consistency with XFS-1 formal system

#### Code Contributions
- Add new implementation (`xi_*.py`)
- Improve existing code (`xi_ppo.py`, `xi_gae.py`, etc.)
- Add tests or examples
- Ensure all changes align with Xi-operator principles

#### Documentation
- Update README.md
- Add new documentation files
- Improve code comments
- Add examples or tutorials

### 2. Test Your Changes

- **For code changes**: Run all relevant scripts to ensure they work
  ```bash
  python xi_ai_final.py
  python xi_gae.py
  python xi_ppo.py
  ```
- **For theoretical changes**: Update `paper_validation_report.md` if needed

### 3. Commit Your Changes

Follow conventional commit format:

```bash
# Theoretical additions
git commit -m "feat: Add new axiom about spectral consensus"

# Code changes
git commit -m "fix: correct XiOperator tensor dimensions"

# Documentation
git commit -m "docs: update README with installation instructions"

# Refactoring
git commit -m "refactor: simplify XiAgent implementation"
```

## Pull Request Process

1. **Ensure your changes are tested** - all experiments should run without errors
2. **Update documentation** - if you change functionality, update the relevant docs
3. **Write a clear PR description** - explain what you did and why it matters
4. **Link to any relevant issues** - reference GitHub issues you're addressing

### PR Checklist

- [ ] Changes are well-documented
- [ ] Code changes are tested
- [ ] Theoretical changes maintain consistency with XFS-1
- [ ] All files follow the project's coding style
- [ ] Commit message follows conventional format

## Review Process

1. **Automated checks** - CI (if set up) checks for basic formatting and compilation
2. **Maintainer review** - Theoretical consistency check and code quality review
3. **Community feedback** - 1 week waiting period for community comments
4. **Merge** - After all feedback is addressed and maintainers approve

## Issue Reporting Guides

### For Code Issues
- Provide: exact error message, commands used, environment info, sample code
- Label: `bug`

### For Theoretical Questions
- Provide: specific claim you want to discuss, related axioms theorems, context
- Label: `question`, `discussion`

### For Feature Requests
- Provide: use case, proposed solution, alignment with project goals
- Label: `enhancement`

## Getting Help

- **Documentation issues**: Open an issue with `docs` label
- **Code help**: Join the discussion in issues or PRs
- **Theoretical questions**: Reference the appropriate axiom or theorem

## Credit

All contributors will be listed in the `README.md` and `LICENSE`. We also consider major contributors for co-authorship on future papers.

---

*This CONTRIBUTING.md is adapted from the contributor covenant but customized for Xi-Learning's formal system approach.*

*Last updated: 2026-11-21*

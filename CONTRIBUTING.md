# Contributing to AxiomLM

Thank you for your interest in contributing to **AxiomLM**! We welcome contributions ranging from low-level GPU/SIMD kernel optimizations to theoretical research guides, architecture modernizations, and bug fixes.

---

## 🛠️ Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/EyadNada/GPT-2.0-124M.git
   cd GPT-2.0-124M
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Install optional developer dependencies (for linting & formatting):**
   ```bash
   pip install -e ".[dev]"
   ```

---

## 🧪 Running the Test Suite

Before opening a pull request, ensure all unit, architecture, optimizer, and kernel tests pass:

```bash
# 1. Run core architecture, optimizer, sampling, and KV-cache tests (25 tests)
python tests/test_all.py

# 2. Run custom low-level GPU and ARM NEON SIMD kernel tests (8 tests)
python tests/test_kernels.py
```

---

## 🎨 Code Style & Quality Standards

- **Python:** Follow PEP 8 style guidelines. Code should be clean, explicitly typed where appropriate, and thoroughly commented when introducing complex mathematical or hardware-specific concepts.
- **Hardware Agnostic:** Ensure core models and training logic run seamlessly across Apple Silicon (`mps`), NVIDIA (`cuda`), and fallback `cpu`.
- **Reproducibility:** When adding new loss curves or architecture benchmarks, include seeds, hyperparameters, and device specs.

---

## 📬 Pull Request Process

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/my-optimization
   ```
2. Commit your changes with clear, semantic commit messages (e.g., `feat:`, `fix:`, `docs:`, `perf:`).
3. Ensure the test suites (`test_all.py` and `test_kernels.py`) pass 100%.
4. Open a Pull Request on GitHub using the PR template.

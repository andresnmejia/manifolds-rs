

**First, install OpenBLAS:**
```bash
# Ubuntu/Debian
sudo apt-get install libopenblas-dev

# Fedora/RHEL
sudo dnf install openblas-devel

# macOS
brew install openblas

# Arch Linux
sudo pacman -S openblas
```

**Then add to `Cargo.toml`:**
```toml
[dependencies]
manifolds-rs = { version = "0.1", features = ["openblas-system"] }
```

TODO:

1. Implement Riemannian [Line Search Methods](https://assets.press.princeton.edu/chapters/absil/Absil_Chap4.pdf). Currently using static step size.

2. Symmetric Positive Definite Matrices and Stiefel Manifolds have not been tested adequately for stability.

3. Implement [Affine Connections and Riemannian Newton's Method](https://assets.press.princeton.edu/chapters/absil/Absil_Chap6.pdf)

4. Optimization methods currently require presenting manifold as embedded submanifold of R^N.

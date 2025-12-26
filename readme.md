This is a Rust implementation of riemannian optimization. We wrote this with a few towards information geometry/statistical manifolds and using the information metric as a tool in regularization.

That being said, many of the methods (included the overall structure of the manifold trait) follow [manifold.jl](https://github.com/JuliaManifolds/Manifolds.jl) closely. In particular, their implementation of "manifolds/core."

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

**WARNING:** Only first order descent is working "genuinely."  (second order) methods require an affine connection which is yet to be implemented.
TODO:

1. Implement Riemannian [Line Search Methods](https://assets.press.princeton.edu/chapters/absil/Absil_Chap4.pdf). Currently using static step size.

2. Symmetric Positive Definite Matrices and Stiefel Manifolds have not been tested adequately for stability.

3. Implement [Affine Connections and Riemannian Newton's Method](https://assets.press.princeton.edu/chapters/absil/Absil_Chap6.pdf)

4. Optimization methods currently require presenting manifold as embedded submanifold of R^N.


*cheeky side note:* we could have implemented spheres as a special case of stiefel, but opted not to for both simplicity, but also for speed reasons (for example in choice of retraction, and later on for second order computations.)

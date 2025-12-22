use ndarray::Array2;
use ndarray_linalg::{Cholesky, Eigh, UPLO};
use num_traits::Float;

use crate::core::{EmbeddedManifold, Error, Manifold, Result};

/// Manifold of Symmetric Positive Definite (SPD) matrices
/// with the affine-invariant Riemannian metric
///
/// SPD(n) = {P ∈ R^{n×n} : P = P^T, P ≻ 0}
///
/// This is the natural manifold for:
/// - Covariance matrices
/// - Fisher Information Matrices
/// - Precision matrices
///
/// The affine-invariant metric makes it invariant under congruence transformations:
/// if P is SPD, then so is A^T P A for any invertible A.
pub struct SPD {
    /// Dimension of the matrices (n x n)
    dim: usize,
}

impl SPD {
    pub fn new(dim: usize) -> Self {
        SPD { dim }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Check if matrix is symmetric positive definite
    fn is_spd(&self, p: &Array2<f64>, tolerance: f64) -> bool {
        if p.shape() != [self.dim, self.dim] {
            return false;
        }

        // Check symmetry
        for i in 0..self.dim {
            for j in i + 1..self.dim {
                if (p[[i, j]] - p[[j, i]]).abs() > tolerance {
                    return false;
                }
            }
        }

        // Check positive definiteness via Cholesky
        p.cholesky(UPLO::Lower).is_ok()
    }
}

impl Manifold for SPD {
    type Point = Array2<f64>;
    type Vector = Array2<f64>;
    type Scalar = f64;

    /// Exponential map using matrix exponential (unchecked)
    /// exp_P(X) = P^{1/2} exp(P^{-1/2} X P^{-1/2}) P^{1/2}
    fn exp_unchecked(&self, p: &Self::Point, x: &Self::Vector) -> Self::Point {
        // Ensure X is symmetric (tangent vectors must be symmetric)
        let x_sym: Array2<f64> = 0.5 * (x + &x.t());

        // Eigendecomposition of P: P = Q D Q^T
        let (eigenvalues, eigenvectors) = p.eigh(UPLO::Lower).expect("eigendecomposition failed");

        // P^{1/2} = Q D^{1/2} Q^T
        let sqrt_eigenvalues = eigenvalues.mapv(|lambda| lambda.sqrt());
        let p_sqrt: Array2<f64> = eigenvectors
            .dot(&Array2::from_diag(&sqrt_eigenvalues))
            .dot(&eigenvectors.t());

        // P^{-1/2} = Q D^{-1/2} Q^T
        let inv_sqrt_eigenvalues = eigenvalues.mapv(|lambda| 1.0 / lambda.sqrt());
        let p_inv_sqrt: Array2<f64> = eigenvectors
            .dot(&Array2::from_diag(&inv_sqrt_eigenvalues))
            .dot(&eigenvectors.t());

        // Y = P^{-1/2} X P^{-1/2}
        let y: Array2<f64> = p_inv_sqrt.dot(&x_sym).dot(&p_inv_sqrt);

        // Symmetrize Y to avoid numerical errors
        let y_sym: Array2<f64> = 0.5 * (&y + &y.t());

        // exp(Y) via eigendecomposition
        let (y_eigenvalues, y_eigenvectors) =
            y_sym.eigh(UPLO::Lower).expect("eigendecomposition failed");
        let exp_y_eigenvalues = y_eigenvalues.mapv(|lambda| lambda.exp());
        let exp_y: Array2<f64> = y_eigenvectors
            .dot(&Array2::from_diag(&exp_y_eigenvalues))
            .dot(&y_eigenvectors.t());

        // Result: P^{1/2} exp(Y) P^{1/2}
        let result: Array2<f64> = p_sqrt.dot(&exp_y).dot(&p_sqrt);

        // Ensure symmetry (numerical errors can break it)
        0.5 * (&result + &result.t())
    }

    /// Exponential map with validation
    fn exp(&self, p: &Self::Point, x: &Self::Vector) -> Result<Self::Point> {
        self.validate_point(p)?;
        self.validate_vector(p, x)?;
        Ok(self.exp_unchecked(p, x))
    }

    /// Logarithmic map using matrix logarithm (unchecked)
    /// log_P(Q) = P^{1/2} log(P^{-1/2} Q P^{-1/2}) P^{1/2}
    fn log_unchecked(&self, p: &Self::Point, q: &Self::Point) -> Self::Vector {
        // Eigendecomposition of P
        let (eigenvalues, eigenvectors) = p.eigh(UPLO::Lower).expect("eigendecomposition failed");

        // P^{1/2} and P^{-1/2}
        let sqrt_eigenvalues = eigenvalues.mapv(|lambda| lambda.sqrt());
        let p_sqrt: Array2<f64> = eigenvectors
            .dot(&Array2::from_diag(&sqrt_eigenvalues))
            .dot(&eigenvectors.t());

        let inv_sqrt_eigenvalues = eigenvalues.mapv(|lambda| 1.0 / lambda.sqrt());
        let p_inv_sqrt: Array2<f64> = eigenvectors
            .dot(&Array2::from_diag(&inv_sqrt_eigenvalues))
            .dot(&eigenvectors.t());

        // Y = P^{-1/2} Q P^{-1/2}
        let y: Array2<f64> = p_inv_sqrt.dot(q).dot(&p_inv_sqrt);

        // Symmetrize Y to avoid numerical errors
        let y_sym: Array2<f64> = 0.5 * (&y + &y.t());

        // log(Y) via eigendecomposition
        let (y_eigenvalues, y_eigenvectors) =
            y_sym.eigh(UPLO::Lower).expect("eigendecomposition failed");
        let log_y_eigenvalues = y_eigenvalues.mapv(|lambda| lambda.ln());
        let log_y: Array2<f64> = y_eigenvectors
            .dot(&Array2::from_diag(&log_y_eigenvalues))
            .dot(&y_eigenvectors.t());

        // Result: P^{1/2} log(Y) P^{1/2}
        let result: Array2<f64> = p_sqrt.dot(&log_y).dot(&p_sqrt);

        // Ensure symmetry (numerical errors can break it)
        0.5 * (&result + &result.t())
    }

    /// Logarithmic map with validation
    fn log(&self, p: &Self::Point, q: &Self::Point) -> Result<Self::Vector> {
        self.validate_point(p)?;
        self.validate_point(q)?;
        Ok(self.log_unchecked(p, q))
    }

    /// Affine-invariant metric: g_P(X, Y) = tr(P^{-1} X P^{-1} Y)
    fn metric(&self, p: &Self::Point, x: &Self::Vector, y: &Self::Vector) -> Self::Scalar {
        // Eigendecomposition of P
        let (eigenvalues, eigenvectors) = match p.eigh(UPLO::Lower) {
            Ok(result) => result,
            Err(_) => return f64::nan(),
        };

        // P^{-1} = Q D^{-1} Q^T
        let inv_eigenvalues = eigenvalues.mapv(|v| 1.0 / v);
        let p_inv = &eigenvectors * &Array2::from_diag(&inv_eigenvalues) * &eigenvectors.t();

        // tr(P^{-1} X P^{-1} Y)
        let temp = p_inv.dot(x).dot(&p_inv).dot(y);
        temp.diag().sum()
    }

    fn project(&self, p: &Self::Point) -> Result<Self::Point> {
        // Ensure symmetry and positive definiteness
        let symmetric = 0.5 * (p + &p.t());

        // Eigendecomposition
        let (eigenvalues, eigenvectors) = symmetric.eigh(UPLO::Lower)?;

        // Clamp eigenvalues to be positive
        let min_eigenvalue = 1e-10;
        let clamped_eigenvalues = eigenvalues.mapv(|x| x.max(min_eigenvalue));

        // Reconstruct
        let result = &eigenvectors * &Array2::from_diag(&clamped_eigenvalues) * &eigenvectors.t();

        Ok(result)
    }

    /// Project onto tangent space (symmetric matrices)
    /// T_P SPD(n) = Sym(n) = {X ∈ R^{n×n} : X = X^T}
    fn project_tangent(&self, _p: &Self::Point, x: &Self::Vector) -> Result<Self::Vector> {
        // Symmetrize
        Ok(0.5 * (x + &x.t()))
    }

    fn validate_point(&self, p: &Self::Point) -> Result<()> {
        if !self.is_spd(p, 1e-10) {
            return Err(Error::NotOnManifold(
                "Matrix is not symmetric positive definite".to_string(),
            ));
        }
        Ok(())
    }

    fn validate_vector(&self, _p: &Self::Point, x: &Self::Vector) -> Result<()> {
        // Check if vector is symmetric (tangent space of SPD is symmetric matrices)
        for i in 0..x.nrows() {
            for j in i + 1..x.ncols() {
                if (x[[i, j]] - x[[j, i]]).abs() > 1e-10 {
                    return Err(Error::InvalidParameter(
                        "Tangent vector must be symmetric".to_string(),
                    ));
                }
            }
        }
        Ok(())
    }
}

impl EmbeddedManifold for SPD {
    fn ambient_dim(&self) -> usize {
        self.dim * (self.dim + 1) / 2 // ✅ CORRECT - upper triangle only
    }

    fn dim(&self) -> usize {
        self.dim * (self.dim + 1) / 2
    }

    fn is_on_manifold(&self, p: &Self::Point, tolerance: f64) -> bool {
        self.is_spd(p, tolerance)
    }

    fn to_ambient(&self, p: &Self::Point) -> Result<ndarray::Array1<f64>> {
        let mut v = ndarray::Array1::zeros(self.dim * (self.dim + 1) / 2);
        let mut idx = 0;

        for i in 0..self.dim {
            for j in i..self.dim {
                // Only upper triangle
                v[idx] = p[[i, j]];
                idx += 1;
            }
        }

        Ok(v)
    }

    fn vector_to_ambient(&self, v: &Self::Vector) -> Result<ndarray::Array1<f64>> {
        self.to_ambient(v)
    }

    fn from_ambient(&self, x: &ndarray::Array1<f64>) -> Result<Self::Point> {
        if x.len() != self.dim * (self.dim + 1) / 2 {
            return Err(Error::DimensionMismatch {
                expected: self.dim * (self.dim + 1) / 2,
                got: x.len(),
            });
        }

        let mut p = Array2::zeros((self.dim, self.dim));
        let mut idx = 0;

        for i in 0..self.dim {
            for j in i..self.dim {
                p[[i, j]] = x[idx];
                if i != j {
                    p[[j, i]] = x[idx]; // Fill lower triangle
                }
                idx += 1;
            }
        }

        self.project(&p)
    }

    fn project_to_ambient_tangent(
        &self,
        p: &Self::Point,
        v: &ndarray::Array1<f64>,
    ) -> Result<ndarray::Array1<f64>> {
        let mut v_mat = Array2::zeros((self.dim, self.dim));
        let mut idx = 0;

        for i in 0..self.dim {
            for j in i..self.dim {
                v_mat[[i, j]] = v[idx];
                if i != j {
                    v_mat[[j, i]] = v[idx];
                }
                idx += 1;
            }
        }

        let v_tan = self.project_tangent(p, &v_mat)?;
        self.vector_to_ambient(&v_tan)
    }

    fn exp_from_ambient(&self, p: &Self::Point, v: &ndarray::Array1<f64>) -> Result<Self::Point> {
        let mut v_mat = Array2::zeros((self.dim, self.dim));
        let mut idx = 0;

        for i in 0..self.dim {
            for j in i..self.dim {
                v_mat[[i, j]] = v[idx];
                if i != j {
                    v_mat[[j, i]] = v[idx];
                }
                idx += 1;
            }
        }

        self.exp(p, &v_mat)
    }
}

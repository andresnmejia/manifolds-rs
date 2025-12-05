use ndarray::Array2;
use ndarray_linalg::{Cholesky, Eigh, UPLO};
use num_traits::Float;

use crate::core::{EmbeddedManifold, Error, Manifold, Result};

/// Manifold of Symmetric Positive Definite (SPD) matrices
/// with the affine-invariant Riemannian metric
///
/// This is the natural manifold for:
/// - Covariance matrices
/// - Fisher Information Matrices
/// - Precision matrices
pub struct SPD {
    /// Dimension of the matrices (n x n)
    dim: usize,
}

impl SPD {
    pub fn new(dim: usize) -> Self {
        SPD { dim }
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
        self.dim * self.dim
    }

    fn dim(&self) -> usize {
        // SPD(n) has dimension n(n+1)/2
        self.dim * (self.dim + 1) / 2
    }

    fn is_on_manifold(&self, p: &Self::Point, tolerance: f64) -> bool {
        self.is_spd(p, tolerance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_spd_exp_log_inverse() {
        let spd = SPD::new(2);
        let p = Array2::from_shape_vec((2, 2), vec![4.0, 1.0, 1.0, 3.0]).unwrap();
        let x = Array2::from_shape_vec((2, 2), vec![0.1, 0.05, 0.05, 0.2]).unwrap();

        let q = spd.exp(&p, &x).unwrap();
        let x_recovered = spd.log(&p, &q).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(x[[i, j]], x_recovered[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_spd_metric_symmetry() {
        let spd = SPD::new(2);
        let p = Array2::from_shape_vec((2, 2), vec![2.0, 0.5, 0.5, 3.0]).unwrap();
        let x = Array2::from_shape_vec((2, 2), vec![0.1, 0.0, 0.0, 0.1]).unwrap();
        let y = Array2::from_shape_vec((2, 2), vec![0.2, 0.1, 0.1, 0.3]).unwrap();

        let gxy = spd.metric(&p, &x, &y);
        let gyx = spd.metric(&p, &y, &x);

        assert_relative_eq!(gxy, gyx, epsilon = 1e-10);
    }
}

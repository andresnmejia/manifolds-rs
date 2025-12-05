use ndarray::Array2;
use ndarray_linalg::{Norm, QR, SVD};

use crate::core::{EmbeddedManifold, Error, Manifold, Result};

/// Stiefel manifold St(n, p) of n×p matrices with orthonormal columns
///
/// St(n, p) = {X ∈ ℝⁿˣᵖ : X^T X = I_p}
///
/// The tangent space at X is:
/// T_X St(n, p) = {Z ∈ ℝⁿˣᵖ : X^T Z + Z^T X = 0}
///
/// This is the space of matrices Y with orthonormal columns.
/// Special cases:
/// - St(n, 1) = S^(n-1) (unit sphere)
/// - St(n, n) = O(n) (orthogonal group)
pub struct Stiefel {
    n: usize, // Ambient dimension (rows)
    p: usize, // Number of columns
}

impl Stiefel {
    /// Create Stiefel manifold St(n, p)
    pub fn new(n: usize, p: usize) -> Result<Self> {
        if p > n {
            return Err(Error::InvalidParameter(format!(
                "Stiefel manifold requires p ≤ n, got p={}, n={}",
                p, n
            )));
        }
        Ok(Stiefel { n, p })
    }

    /// Ambient dimension (rows)
    pub fn n(&self) -> usize {
        self.n
    }

    /// Number of columns
    pub fn p(&self) -> usize {
        self.p
    }

    /// Check if matrix has orthonormal columns
    fn has_orthonormal_columns(&self, x: &Array2<f64>, tolerance: f64) -> bool {
        if x.shape() != [self.n, self.p] {
            return false;
        }

        // Check X^T X = I
        let xtx = x.t().dot(x);
        let identity: Array2<f64> = Array2::eye(self.p);

        let diff = &xtx - &identity;
        diff.norm_l2() < tolerance
    }
}

impl Manifold for Stiefel {
    type Point = Array2<f64>;
    type Vector = Array2<f64>;
    type Scalar = f64;

    /// Exponential map using QR-based retraction (unchecked)
    ///
    /// This is actually a retraction (approximation to exp), but commonly used
    /// exp_X(Z) = qf(X + Z) where qf is QR-based orthonormalization
    fn exp_unchecked(&self, x: &Self::Point, z: &Self::Vector) -> Self::Point {
        // Compute X + Z
        let y: Array2<f64> = x + z;

        // QR decomposition - returns Q (n×n) and R (n×p)
        let (q, r) = match y.qr() {
            Ok((q, r)) => (q, r),
            Err(_) => {
                // Fallback: use SVD if QR fails
                return self.project(x).unwrap_or_else(|_| x.clone());
            }
        };

        // Take first p columns of Q
        let q_p = q.slice(ndarray::s![.., 0..self.p]).to_owned();

        // Ensure positive diagonal of R (for uniqueness)
        let mut result = q_p;
        for i in 0..self.p {
            if r[[i, i]] < 0.0 {
                // Flip sign of column i
                for j in 0..self.n {
                    result[[j, i]] = -result[[j, i]];
                }
            }
        }

        result
    }

    /// Logarithmic map (unchecked)
    ///
    /// For the Stiefel manifold, there's no closed-form logarithm in general.
    /// We use the approximation: log_X(Y) ≈ Y - X (only valid for nearby points)
    fn log_unchecked(&self, x: &Self::Point, y: &Self::Point) -> Self::Vector {
        // Simple approximation for nearby points
        let z: Array2<f64> = y - x; // Explicit type annotation

        // Project onto tangent space to ensure Z ∈ T_X St(n,p)
        self.project_tangent(x, &z).unwrap_or(z)
    }

    /// Riemannian metric (canonical metric from embedding)
    /// g_X(Z, W) = tr(Z^T W)
    fn metric(&self, _x: &Self::Point, z: &Self::Vector, w: &Self::Vector) -> Self::Scalar {
        let ztw = z.t().dot(w);
        ztw.diag().sum()
    }

    /// Project matrix onto Stiefel manifold (orthonormalize columns)
    fn project(&self, x: &Self::Point) -> Result<Self::Point> {
        // Use SVD for robust orthonormalization
        // X = U Σ V^T → project to U[:,:p] * V^T
        let (u, _s, vt) = x.svd(true, true)?;

        let u = u.ok_or_else(|| Error::ComputationFailed("SVD failed to compute U".to_string()))?;
        let vt =
            vt.ok_or_else(|| Error::ComputationFailed("SVD failed to compute V^T".to_string()))?;

        // Take first p columns of U
        let u_p = u.slice(ndarray::s![.., 0..self.p]).to_owned();

        // Result: U[:,:p] * V^T
        let result: Array2<f64> = u_p.dot(&vt);
        Ok(result)
    }

    /// Project onto tangent space at X
    ///
    /// T_X St(n,p) = {Z : X^T Z + Z^T X = 0}
    ///
    /// Projection: Z_tan = Z - X * sym(X^T Z)
    /// where sym(A) = (A + A^T)/2
    fn project_tangent(&self, x: &Self::Point, z: &Self::Vector) -> Result<Self::Vector> {
        // Compute X^T Z
        let xtz: Array2<f64> = x.t().dot(z);

        // Symmetrize: sym(X^T Z) = (X^T Z + Z^T X)/2
        let sym_xtz: Array2<f64> = 0.5 * (&xtz + &xtz.t());

        // Project: Z - X * sym(X^T Z)
        let result: Array2<f64> = z - &x.dot(&sym_xtz);

        Ok(result)
    }

    fn validate_point(&self, x: &Self::Point) -> Result<()> {
        if x.shape() != [self.n, self.p] {
            return Err(Error::DimensionMismatch {
                expected: self.n * self.p,
                got: x.shape()[0] * x.shape()[1],
            });
        }

        if !self.has_orthonormal_columns(x, 1e-8) {
            let xtx = x.t().dot(x);
            let identity: Array2<f64> = Array2::eye(self.p);
            let error = (&xtx - &identity).norm_l2();
            return Err(Error::NotOnManifold(format!(
                "Matrix does not have orthonormal columns: ||X^T X - I|| = {}",
                error
            )));
        }

        Ok(())
    }

    fn validate_vector(&self, x: &Self::Point, z: &Self::Vector) -> Result<()> {
        if z.shape() != [self.n, self.p] {
            return Err(Error::DimensionMismatch {
                expected: self.n * self.p,
                got: z.shape()[0] * z.shape()[1],
            });
        }

        // Check tangent space condition: X^T Z + Z^T X = 0
        let xtz = x.t().dot(z);
        let ztx = z.t().dot(x);
        let skew_error = (&xtz + &ztx).norm_l2();

        if skew_error > 1e-6 {
            return Err(Error::InvalidParameter(format!(
                "Vector not in tangent space: ||X^T Z + Z^T X|| = {}",
                skew_error
            )));
        }

        Ok(())
    }
}

impl EmbeddedManifold for Stiefel {
    fn ambient_dim(&self) -> usize {
        self.n * self.p
    }

    fn dim(&self) -> usize {
        // Dimension of St(n, p) = np - p(p+1)/2
        self.n * self.p - self.p * (self.p + 1) / 2
    }

    fn is_on_manifold(&self, x: &Self::Point, tolerance: f64) -> bool {
        self.has_orthonormal_columns(x, tolerance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::arr2;

    #[test]
    fn test_stiefel_creation() {
        let st = Stiefel::new(5, 3).unwrap();
        assert_eq!(st.n(), 5);
        assert_eq!(st.p(), 3);
        assert_eq!(st.dim(), 5 * 3 - 3 * 4 / 2); // 15 - 6 = 9
    }

    #[test]
    fn test_stiefel_invalid_dimensions() {
        let result = Stiefel::new(3, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_stiefel_project() {
        let st = Stiefel::new(3, 2).unwrap();

        // Random matrix (not orthonormal)
        let x = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);

        let projected = st.project(&x).unwrap();

        // Check orthonormality
        let xtx = projected.t().dot(&projected);
        let identity = Array2::eye(2);

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(xtx[[i, j]], identity[[i, j]], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_stiefel_project_tangent() {
        let st = Stiefel::new(3, 2).unwrap();

        // Point on Stiefel manifold (orthonormal columns)
        let x = arr2(&[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]);

        // Random matrix
        let z = arr2(&[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]);

        let z_tangent = st.project_tangent(&x, &z).unwrap();

        // Check tangent space condition: X^T Z + Z^T X = 0
        let xtz = x.t().dot(&z_tangent);
        let ztx = z_tangent.t().dot(&x);
        let sum = &xtz + &ztx;

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(sum[[i, j]], 0.0, epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_stiefel_exp_stays_on_manifold() {
        let st = Stiefel::new(4, 2).unwrap();

        // Start with orthonormal matrix
        let x = arr2(&[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]);

        // Small tangent vector
        let z = arr2(&[[0.0, 0.1], [-0.1, 0.0], [0.2, 0.0], [0.0, 0.2]]);

        let y = st.exp(&x, &z).unwrap();

        // Check Y is on manifold
        assert!(st.validate_point(&y).is_ok());

        // Check Y^T Y = I
        let yty = y.t().dot(&y);
        let identity = Array2::eye(2);

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(yty[[i, j]], identity[[i, j]], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_stiefel_metric() {
        let st = Stiefel::new(3, 2).unwrap();

        let x = arr2(&[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]);

        let z1 = arr2(&[[0.0, 0.1], [-0.1, 0.0], [0.2, 0.3]]);

        let z2 = arr2(&[[0.0, 0.2], [-0.2, 0.0], [0.1, 0.1]]);

        // Metric should be symmetric
        let g12 = st.metric(&x, &z1, &z2);
        let g21 = st.metric(&x, &z2, &z1);

        assert_relative_eq!(g12, g21, epsilon = 1e-10);
    }

    #[test]
    fn test_stiefel_special_case_sphere() {
        // St(n, 1) should behave like S^(n-1)
        let st = Stiefel::new(3, 1).unwrap();

        // A point (column vector with unit norm)
        let x = arr2(&[[1.0], [0.0], [0.0]]);

        assert!(st.validate_point(&x).is_ok());

        let norm_sq = x.t().dot(&x)[[0, 0]];
        assert_relative_eq!(norm_sq, 1.0, epsilon = 1e-10);
    }
}

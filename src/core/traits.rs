use ndarray::{Array1, Array2};
use num_traits::Float;

use crate::core::error::{Error, Result};

/// Core trait for a Riemannian manifold
/// All manifolds are assumed to be smooth
#[must_use]
pub trait Manifold {
    /// Point on the manifold
    type Point;

    /// Tangent vector at a point
    type Vector;

    /// Scalar field (f64 or f32)
    type Scalar: Float;

    /// Exponential map: TpM → M (unchecked, assumes valid input)
    ///
    /// Maps tangent vector x at point p to a point on the manifold.
    /// This is the fast path - it assumes inputs are valid.
    fn exp_unchecked(&self, p: &Self::Point, x: &Self::Vector) -> Self::Point;

    /// Exponential map with validation
    ///
    /// Validates inputs before calling `exp_unchecked`.
    fn exp(&self, p: &Self::Point, x: &Self::Vector) -> Result<Self::Point> {
        self.validate_point(p)?;
        self.validate_vector(p, x)?;
        Ok(self.exp_unchecked(p, x))
    }

    /// Logarithmic map: M → TpM (unchecked, assumes valid input)
    ///
    /// Inverse of exp: finds tangent vector from p to q.
    /// This is the fast path - it assumes inputs are valid.
    fn log_unchecked(&self, p: &Self::Point, q: &Self::Point) -> Self::Vector;

    /// Logarithmic map with validation
    ///
    /// Validates inputs before calling `log_unchecked`.
    fn log(&self, p: &Self::Point, q: &Self::Point) -> Result<Self::Vector> {
        self.validate_point(p)?;
        self.validate_point(q)?;
        Ok(self.log_unchecked(p, q))
    }

    /// Riemannian metric: g_p(x, y) for x, y ∈ TpM
    fn metric(&self, p: &Self::Point, x: &Self::Vector, y: &Self::Vector) -> Self::Scalar;

    /// Riemannian distance between two points
    fn distance(&self, p: &Self::Point, q: &Self::Point) -> Result<Self::Scalar> {
        let x = self.log(p, q)?;
        Ok(self.metric(p, &x, &x).sqrt())
    }

    /// Project a point onto the manifold (for constraints)
    fn project(&self, p: &Self::Point) -> Result<Self::Point>
    where
        Self::Point: Clone, // Only here!
    {
        Ok(p.clone())
    }

    /// Project a vector onto the tangent space at p
    fn project_tangent(&self, p: &Self::Point, x: &Self::Vector) -> Result<Self::Vector>
    where
        Self::Vector: Clone, // Need this constraint
    {
        Ok(x.clone()) // ✅ Works
    }

    /// Validate that a point lies on the manifold
    ///
    /// Default implementation is optimistic (always returns Ok).
    /// Override to add actual validation.
    fn validate_point(&self, _p: &Self::Point) -> Result<()> {
        Ok(())
    }

    /// Validate that a vector is in the tangent space at p
    ///
    /// Default implementation is optimistic (always returns Ok).
    /// Override to add actual validation.
    fn validate_vector(&self, _p: &Self::Point, _x: &Self::Vector) -> Result<()> {
        Ok(())
    }
}

/// Trait for statistical manifolds with Fisher information metric
pub trait StatisticalManifold: Manifold {
    /// Fisher Information Matrix at parameter θ
    /// G_ij(θ) = E[∂log p(x|θ)/∂θ_i · ∂log p(x|θ)/∂θ_j]
    fn fisher_information(&self, theta: &Self::Point) -> Result<Array2<f64>>;

    /// Compute Fisher metric using FIM
    /// Default implementation: g(X,Y) = X^T G(θ) Y
    fn fisher_metric(
        &self,
        theta: &Self::Point,
        x: &Self::Vector,
        y: &Self::Vector,
    ) -> Result<f64> {
        let g = self.fisher_information(theta)?;
        let x_arr = self.as_array1(x)?;
        let y_arr = self.as_array1(y)?;
        Ok(x_arr.dot(&g.dot(&y_arr)))
    }

    fn as_array1(&self, v: &Self::Vector) -> Result<Array1<f64>>;
}

pub trait EmbeddedManifold: Manifold {
    fn ambient_dim(&self) -> usize;

    fn dim(&self) -> usize;

    fn is_on_manifold(&self, p: &Self::Point, tolerance: f64) -> bool;
}

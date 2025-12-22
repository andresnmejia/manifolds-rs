use crate::core::error::{Error, Result};
use ndarray::{Array1, Array2};
use num_traits::Float;

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
        Self::Point: Clone,
    {
        Ok(p.clone())
    }

    /// Project a vector onto the tangent space at p
    fn project_tangent(&self, p: &Self::Point, x: &Self::Vector) -> Result<Self::Vector>
    where
        Self::Vector: Clone,
    {
        Ok(x.clone())
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

/// Trait for statistical families (NO Manifold dependency)
///
/// This trait represents a parametric family of probability distributions
/// without imposing any manifold structure. It only provides the Fisher
/// information matrix.
pub trait StatisticalFamily {
    /// Parameter type (e.g., Array1<f64>)
    type Param;

    /// Fisher Information Matrix at parameter θ
    /// G_ij(θ) = E[∂log p(x|θ)/∂θ_i · ∂log p(x|θ)/∂θ_j]
    fn fisher_information(&self, theta: &Self::Param) -> Result<Array2<f64>>;

    /// Convert parameter to Array1 for metric computation
    fn param_to_array(&self, theta: &Self::Param) -> Result<Array1<f64>>;

    /// Convert Array1 back to parameter type
    fn array_to_param(&self, arr: &Array1<f64>) -> Result<Self::Param>;

    /// Dimension of parameter space
    fn param_dim(&self) -> usize;
}

/// Wrapper that equips a statistical family with Fisher information metric
///
/// This is the composition-based approach: take a `StatisticalFamily` and
/// turn it into a `Manifold` whose metric is the Fisher information metric.
///
/// # Example
/// ```ignore
/// let gaussian = GaussianFamily::new();
/// let manifold = FisherManifold::new(gaussian);
///
/// // Now `manifold` is a proper Riemannian manifold
/// let dist = manifold.distance(&p, &q)?;
/// ```
pub struct FisherManifold<S> {
    /// The underlying statistical family
    pub family: S,
}

impl<S> FisherManifold<S> {
    /// Create a Fisher manifold from a statistical family
    pub fn new(family: S) -> Self {
        Self { family }
    }

    /// Get reference to underlying family
    pub fn family(&self) -> &S {
        &self.family
    }
}

impl<S: StatisticalFamily> Manifold for FisherManifold<S>
where
    S::Param: Clone,
{
    type Point = S::Param;
    type Vector = Array1<f64>;
    type Scalar = f64;

    /// Exponential map using natural gradient descent
    /// exp_p(v) = p + v (in Euclidean coordinates, projected back to manifold)
    fn exp_unchecked(&self, p: &Self::Point, v: &Self::Vector) -> Self::Point {
        let p_arr = self.family.param_to_array(p).expect("Invalid parameter");
        let q_arr = &p_arr + v;
        self.family.array_to_param(&q_arr).expect("Invalid result")
    }

    /// Logarithmic map (inverse of exp)
    /// log_p(q) = q - p (in Euclidean coordinates)
    fn log_unchecked(&self, p: &Self::Point, q: &Self::Point) -> Self::Vector {
        let p_arr = self.family.param_to_array(p).expect("Invalid parameter p");
        let q_arr = self.family.param_to_array(q).expect("Invalid parameter q");
        &q_arr - &p_arr
    }

    /// Fisher information metric: g(v, w) = v^T G(θ) w
    fn metric(&self, p: &Self::Point, v: &Self::Vector, w: &Self::Vector) -> Self::Scalar {
        let g = self
            .family
            .fisher_information(p)
            .expect("Failed to compute Fisher information");
        v.dot(&g.dot(w))
    }
}

/// Embedded manifolds (e.g., sphere in R^3)
pub trait EmbeddedManifold: Manifold {
    /// Dimension of the ambient Euclidean space
    fn ambient_dim(&self) -> usize;

    /// Intrinsic dimension of the manifold
    fn dim(&self) -> usize;

    /// Check if a point lies on the manifold
    fn is_on_manifold(&self, p: &Self::Point, tolerance: f64) -> bool;

    // ===== REQUIRED CONVERSION METHODS =====

    /// Convert a point on the manifold to ambient coordinates
    ///
    /// For Stiefel St(n,p): vectorize the n×p matrix in column-major order → R^(np)
    ///
    /// # Example
    /// ```ignore
    /// let x = Array2::eye(5, 3);  // Point on St(5,3)
    /// let v = manifold.to_ambient(&x)?;  // v ∈ R^15
    /// ```
    fn to_ambient(&self, p: &Self::Point) -> Result<Array1<f64>>;

    /// Convert a tangent vector to ambient coordinates
    ///
    /// For Stiefel: Point and Vector are both Array2<f64>, so this is
    /// identical to to_ambient. Override if Point ≠ Vector for your manifold.
    ///
    /// # Example
    /// ```ignore
    /// let v_tan = Array2::zeros((5, 3));  // Tangent vector at some point
    /// let v_ambient = manifold.vector_to_ambient(&v_tan)?;  // v ∈ R^15
    /// ```
    fn vector_to_ambient(&self, v: &Self::Vector) -> Result<Array1<f64>>;

    /// Convert ambient coordinates back to a point on the manifold
    ///
    /// For Stiefel St(n,p): reshape R^(np) → n×p matrix, then project to manifold
    /// to ensure orthonormality.
    ///
    /// # Example
    /// ```ignore
    /// let v = Array1::from_vec(vec![1.0; 15]);
    /// let x = manifold.from_ambient(&v)?;  // Projects onto St(5,3)
    /// ```
    fn from_ambient(&self, x: &Array1<f64>) -> Result<Self::Point>;

    // ===== PROVIDED METHODS (default implementations) =====

    /// Project an ambient vector onto the tangent space at p
    ///
    /// Input: v ∈ R^d (ambient coordinates)
    /// Output: v_tan ∈ T_p M (still in ambient coordinates)
    ///
    /// This is the composition: vector_to_ambient ∘ project_tangent ∘ from_ambient
    ///
    /// NOTE: Default implementation assumes Self::Point and Self::Vector can be
    /// converted via from_ambient/to_ambient. For manifolds where Point ≠ Vector,
    /// you must override this method.
    fn project_to_ambient_tangent(&self, p: &Self::Point, v: &Array1<f64>) -> Result<Array1<f64>> {
        // This default implementation only works when Point = Vector (like Stiefel)
        // If your manifold has Point ≠ Vector, override this method

        // For Stiefel: from_ambient gives Array2<f64> which IS both Point and Vector
        // But generically we can't prove this to the compiler

        // Simplest approach: require manifolds to implement this
        Err(Error::InvalidParameter(
            "project_to_ambient_tangent not implemented for this manifold. \
             Override this method in your EmbeddedManifold implementation."
                .to_string(),
        ))
    }

    /// Exponential map from ambient tangent vector
    ///
    /// Input: v ∈ T_p M in ambient coordinates (R^d)
    /// Output: q = exp_p(v) on manifold
    ///
    /// This is the composition: exp ∘ from_ambient
    ///
    /// NOTE: Default implementation assumes Self::Point and Self::Vector can be
    /// converted via from_ambient. For manifolds where Point ≠ Vector,
    /// you must override this method.
    fn exp_from_ambient(&self, p: &Self::Point, v: &Array1<f64>) -> Result<Self::Point> {
        // Same issue as above - from_ambient returns Point but exp expects Vector
        // Require explicit implementation
        Err(Error::InvalidParameter(
            "exp_from_ambient not implemented for this manifold. \
             Override this method in your EmbeddedManifold implementation."
                .to_string(),
        ))
    }
}

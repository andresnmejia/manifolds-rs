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

    
    /// Default implementation always returns Ok.
    
    fn validate_point(&self, _p: &Self::Point) -> Result<()> {
        Ok(())
    }

    /// Validate that a vector is in the tangent space at p.
    /// Default implementation always returns Ok.
    fn validate_vector(&self, _p: &Self::Point, _x: &Self::Vector) -> Result<()> {
        Ok(())
    }
}

/// Trait for statistical families (NO Manifold dependency)

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

pub struct FisherManifold<S> {
    pub family: S,
}

impl<S> FisherManifold<S> {
    pub fn new(family: S) -> Self {
        Self { family }
    }

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

    /// Exponential map t
    
    fn exp_unchecked(&self, p: &Self::Point, v: &Self::Vector) -> Self::Point {
        let p_arr = self.family.param_to_array(p).expect("Invalid parameter");
        let q_arr = &p_arr + v;
        self.family.array_to_param(&q_arr).expect("Invalid result")
    }

    
    fn log_unchecked(&self, p: &Self::Point, q: &Self::Point) -> Self::Vector {
        let p_arr = self.family.param_to_array(p).expect("Invalid parameter p");
        let q_arr = self.family.param_to_array(q).expect("Invalid parameter q");
        &q_arr - &p_arr
    }

    /// Fisher information  g(v, w) = v^T G(θ) w
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


    /// Convert a point on the manifold to ambient coordinates
   
    fn to_ambient(&self, p: &Self::Point) -> Result<Array1<f64>>;

    /// Convert a tangent vector to ambient coordinates
    
    fn vector_to_ambient(&self, v: &Self::Vector) -> Result<Array1<f64>>;

    /// Convert ambient coordinates back to a point on the manifold
    
    fn from_ambient(&self, x: &Array1<f64>) -> Result<Self::Point>;


    ///--------DEFAULTS---------
    
    /// This is the composition: vector_to_ambient( project_tangent ( from_ambient (-)))
    ///
    /// NOTE: Default implementation assumes Self::Point and Self::Vector can be
    /// converted via from_ambient/to_ambient.
    fn project_to_ambient_tangent(&self, p: &Self::Point, v: &Array1<f64>) -> Result<Array1<f64>> {
       
        Err(Error::InvalidParameter(
            "project_to_ambient_tangent not implemented for this manifold. \
             Override this method in your EmbeddedManifold implementation."
                .to_string(),
        ))
    }

    /// NOTE: Default implementation assumes Self::Point and Self::Vector can be
    /// converted via from_ambient. 
    fn exp_from_ambient(&self, p: &Self::Point, v: &Array1<f64>) -> Result<Self::Point> {
        
        Err(Error::InvalidParameter(
            "exp_from_ambient not implemented for this manifold. \
             Override this method in your EmbeddedManifold implementation."
                .to_string(),
        ))
    }
}

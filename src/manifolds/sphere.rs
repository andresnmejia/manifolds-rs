use ndarray::Array1;
use ndarray_linalg::Norm;

use crate::core::{EmbeddedManifold, Error, Manifold, Result};

/// Sphere S^n embedded in R^(n+1) with standard metric
///
/// S^n = {x ∈ R^(n+1) : ||x|| = 1}
///
/// For the sphere, Point and Vector are both Array1<f64> in ambient coordinates,
/// making the EmbeddedManifold implementation particularly clean.
pub struct Sphere {
    dim: usize, // Intrinsic dimension (n)
}

impl Sphere {
    pub fn new(dim: usize) -> Self {
        Sphere { dim }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    fn is_on_sphere(&self, p: &Array1<f64>, tolerance: f64) -> bool {
        if p.len() != self.dim + 1 {
            return false;
        }
        (p.norm_l2() - 1.0).abs() < tolerance
    }
}

impl Manifold for Sphere {
    type Point = Array1<f64>;
    type Vector = Array1<f64>;
    type Scalar = f64;

    /// Exponential map on the sphere
    /// exp_p(v) = cos(||v||)p + sin(||v||)(v/||v||)
    fn exp_unchecked(&self, p: &Self::Point, x: &Self::Vector) -> Self::Point {
        let norm_x = x.norm_l2();

        if norm_x < 1e-10 {
            return p.clone();
        }

        let cos_norm = norm_x.cos();
        let sin_norm = norm_x.sin();

        let result = cos_norm * p + (sin_norm / norm_x) * x;
        &result / result.norm_l2() // Ensure numerical stability
    }

    /// Logarithmic map on the sphere
    /// log_p(q) = (dist(p,q) / sin(dist)) * (q - <p,q>p)
    fn log_unchecked(&self, p: &Self::Point, q: &Self::Point) -> Self::Vector {
        let dot_pq = p.dot(q);
        let dot_clamped = dot_pq.max(-1.0).min(1.0);
        let dist = dot_clamped.acos();

        if dist < 1e-10 {
            return Array1::zeros(p.len());
        }

        let tangent = q - dot_pq * p;
        let norm_tangent = tangent.norm_l2();

        if norm_tangent < 1e-10 {
            return Array1::zeros(p.len());
        }

        (dist / norm_tangent) * tangent
    }

    /// Riemannian metric (Euclidean inner product on tangent space)
    fn metric(&self, _p: &Self::Point, x: &Self::Vector, y: &Self::Vector) -> Self::Scalar {
        x.dot(y)
    }

    /// Geodesic distance on sphere
    /// dist(p,q) = arccos(<p,q>)
    fn distance(&self, p: &Self::Point, q: &Self::Point) -> Result<Self::Scalar> {
        let dot_pq = p.dot(q);
        let dot_clamped = dot_pq.max(-1.0).min(1.0);
        Ok(dot_clamped.acos())
    }

    /// Project point onto sphere (normalize)
    fn project(&self, p: &Self::Point) -> Result<Self::Point> {
        let norm = p.norm_l2();
        if norm < 1e-10 {
            return Err(Error::ComputationFailed(
                "Cannot project zero vector onto sphere".to_string(),
            ));
        }
        Ok(p / norm)
    }

    /// Project vector onto tangent space at p
    /// T_p S^n = {v ∈ R^(n+1) : <p,v> = 0}
    /// proj(v) = v - <p,v>p
    fn project_tangent(&self, p: &Self::Point, x: &Self::Vector) -> Result<Self::Vector> {
        let dot_px = p.dot(x);
        Ok(x - dot_px * p)
    }

    fn validate_point(&self, p: &Self::Point) -> Result<()> {
        if p.len() != self.dim + 1 {
            return Err(Error::DimensionMismatch {
                expected: self.dim + 1,
                got: p.len(),
            });
        }

        if !self.is_on_sphere(p, 1e-8) {
            return Err(Error::NotOnManifold(format!(
                "Point not on sphere: ||p|| = {}",
                p.norm_l2()
            )));
        }

        Ok(())
    }

    fn validate_vector(&self, p: &Self::Point, x: &Self::Vector) -> Result<()> {
        if x.len() != self.dim + 1 {
            return Err(Error::DimensionMismatch {
                expected: self.dim + 1,
                got: x.len(),
            });
        }

        let dot_px = p.dot(x);
        if dot_px.abs() > 1e-6 {
            return Err(Error::InvalidParameter(format!(
                "Vector not tangent to sphere: <p,x> = {}",
                dot_px
            )));
        }

        Ok(())
    }
}

impl EmbeddedManifold for Sphere {
    fn ambient_dim(&self) -> usize {
        self.dim + 1 // S^n embedded in R^(n+1)
    }

    fn dim(&self) -> usize {
        self.dim // Intrinsic dimension of S^n
    }

    fn is_on_manifold(&self, p: &Self::Point, tolerance: f64) -> bool {
        self.is_on_sphere(p, tolerance)
    }

    /// Convert point to ambient coordinates
    /// For sphere: Point = Array1<f64> = ambient coordinates already!
    /// This is just the identity map.
    fn to_ambient(&self, p: &Self::Point) -> Result<Array1<f64>> {
        Ok(p.clone())
    }

    /// Convert tangent vector to ambient coordinates
    /// For sphere: Vector = Array1<f64> = ambient coordinates already!
    /// This is just the identity map.
    fn vector_to_ambient(&self, v: &Self::Vector) -> Result<Array1<f64>> {
        Ok(v.clone())
    }

    /// Convert ambient coordinates to point on manifold
    /// For sphere: normalize to unit length (project onto S^n)
    fn from_ambient(&self, x: &Array1<f64>) -> Result<Self::Point> {
        if x.len() != self.dim + 1 {
            return Err(Error::DimensionMismatch {
                expected: self.dim + 1,
                got: x.len(),
            });
        }

        let norm = x.norm_l2();
        if norm < 1e-10 {
            return Err(Error::ComputationFailed(
                "Cannot normalize zero vector to sphere".to_string(),
            ));
        }

        Ok(x / norm)
    }

    /// Project ambient vector onto tangent space
    /// For sphere: v_tan = v - <p,v>p
    ///
    /// Since Point = Vector = Array1<f64>, this is trivial
    fn project_to_ambient_tangent(&self, p: &Self::Point, v: &Array1<f64>) -> Result<Array1<f64>> {
        if v.len() != self.dim + 1 {
            return Err(Error::DimensionMismatch {
                expected: self.dim + 1,
                got: v.len(),
            });
        }

        // For sphere, ambient coordinates = native coordinates
        // So we just call project_tangent directly
        self.project_tangent(p, v)
    }

    /// Exponential map from ambient tangent vector
    /// For sphere: ambient coordinates = native coordinates
    ///
    /// Since Point = Vector = Array1<f64>, this is trivial
    fn exp_from_ambient(&self, p: &Self::Point, v: &Array1<f64>) -> Result<Self::Point> {
        if v.len() != self.dim + 1 {
            return Err(Error::DimensionMismatch {
                expected: self.dim + 1,
                got: v.len(),
            });
        }

        // For sphere, ambient coordinates = native coordinates
        // So we just call exp directly
        self.exp(p, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::arr1;

    #[test]
    fn test_sphere_projection() {
        let sphere = Sphere::new(2); // S^2
        let p = arr1(&[1.0, 2.0, 3.0]);

        let p_proj = sphere.project(&p).unwrap();
        assert_relative_eq!(p_proj.norm_l2(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sphere_exp_log() {
        let sphere = Sphere::new(2); // S^2
        let p = arr1(&[1.0, 0.0, 0.0]);
        let x = arr1(&[0.0, 0.1, 0.2]); // Tangent vector

        let q = sphere.exp(&p, &x).unwrap();
        let x_recovered = sphere.log(&p, &q).unwrap();

        for i in 0..3 {
            assert_relative_eq!(x[i], x_recovered[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_sphere_distance() {
        let sphere = Sphere::new(2);
        let p = arr1(&[1.0, 0.0, 0.0]);
        let q = arr1(&[0.0, 1.0, 0.0]);

        let dist = sphere.distance(&p, &q).unwrap();
        assert_relative_eq!(dist, std::f64::consts::PI / 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sphere_tangent_projection() {
        let sphere = Sphere::new(2);
        let p = arr1(&[1.0, 0.0, 0.0]);
        let v = arr1(&[0.5, 1.0, 2.0]); // Not tangent

        let v_tan = sphere.project_tangent(&p, &v).unwrap();

        // Check orthogonality: <p, v_tan> = 0
        assert_relative_eq!(p.dot(&v_tan), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sphere_embedded_manifold() {
        let sphere = Sphere::new(2);
        let p = arr1(&[1.0, 0.0, 0.0]);

        // Test to_ambient (identity for sphere)
        let p_ambient = sphere.to_ambient(&p).unwrap();
        assert_eq!(p, p_ambient);

        // Test from_ambient (normalization)
        let x = arr1(&[2.0, 0.0, 0.0]);
        let p_from = sphere.from_ambient(&x).unwrap();
        assert_relative_eq!(p_from.norm_l2(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(p_from[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sphere_retraction() {
        let sphere = Sphere::new(2);
        let p = arr1(&[1.0, 0.0, 0.0]);
        let v = arr1(&[0.0, 0.2, 0.3]);

        // Test that retraction (step + project) stays on manifold
        let p_ambient = sphere.to_ambient(&p).unwrap();
        let p_stepped = &p_ambient + &v;
        let p_retracted = sphere.from_ambient(&p_stepped).unwrap();

        assert_relative_eq!(p_retracted.norm_l2(), 1.0, epsilon = 1e-10);
    }
}

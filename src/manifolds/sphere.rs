use ndarray::Array1;
use ndarray_linalg::Norm;

use crate::core::{EmbeddedManifold, Error, Manifold, Result};

/// Sphere space S^n with standard metric
pub struct Sphere {
    dim: usize,
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

    fn exp_unchecked(&self, p: &Self::Point, x: &Self::Vector) -> Self::Point {
        let norm_x = x.norm_l2();

        if norm_x < 1e-10 {
            return p.clone();
        }

        let cos_norm = norm_x.cos();
        let sin_norm = norm_x.sin();

        let result = cos_norm * p + (sin_norm / norm_x) * x;
        &result / result.norm_l2()
    }

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

    fn metric(&self, _p: &Self::Point, x: &Self::Vector, y: &Self::Vector) -> Self::Scalar {
        x.dot(y)
    }

    fn distance(&self, p: &Self::Point, q: &Self::Point) -> Result<Self::Scalar> {
        let dot_pq = p.dot(q);
        let dot_clamped = dot_pq.max(-1.0).min(1.0);
        Ok(dot_clamped.acos())
    }

    fn project(&self, p: &Self::Point) -> Result<Self::Point> {
        let norm = p.norm_l2();
        if norm < 1e-10 {
            return Err(Error::ComputationFailed(
                "Cannot project zero vector onto sphere".to_string(),
            ));
        }
        Ok(p / norm)
    }

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
        self.dim + 1
    }

    fn dim(&self) -> usize {
        // SPD(n) has dimension n(n+1)/2
        self.dim
    }

    fn is_on_manifold(&self, p: &Self::Point, tolerance: f64) -> bool {
        self.is_on_sphere(p, tolerance)
    }
}

/*
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::arr1;

    #[test]
    fn test_euclidean_exp_log() {
        let euclidean = Euclidean::new(3);
        let p = arr1(&[1.0, 2.0, 3.0]);
        let x = arr1(&[0.1, 0.2, 0.3]);

        let q = euclidean.exp(&p, &x).unwrap();
        let x_recovered = euclidean.log(&p, &q).unwrap();

        assert_relative_eq!(x[0], x_recovered[0], epsilon = 1e-10);
        assert_relative_eq!(x[1], x_recovered[1], epsilon = 1e-10);
        assert_relative_eq!(x[2], x_recovered[2], epsilon = 1e-10);
    }

    #[test]
    fn test_euclidean_distance() {
        let euclidean = Euclidean::new(2);
        let p = arr1(&[0.0, 0.0]);
        let q = arr1(&[3.0, 4.0]);

        let dist = euclidean.distance(&p, &q).unwrap();
        assert_relative_eq!(dist, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_euclidean_validation() {
        let euclidean = Euclidean::new(2);
        let p_valid = arr1(&[1.0, 2.0]);
        let p_invalid = arr1(&[1.0, 2.0, 3.0]);

        assert!(euclidean.validate_point(&p_valid).is_ok());
        assert!(euclidean.validate_point(&p_invalid).is_err());
    }

    #[test]
    fn test_euclidean_unchecked() {
        let euclidean = Euclidean::new(3);
        let p = arr1(&[1.0, 2.0, 3.0]);
        let x = arr1(&[0.1, 0.2, 0.3]);

        // Unchecked version (no validation overhead)
        let q = euclidean.exp_unchecked(&p, &x);
        let x_recovered = euclidean.log_unchecked(&p, &q);

        assert_relative_eq!(x[0], x_recovered[0], epsilon = 1e-10);
    }
 */

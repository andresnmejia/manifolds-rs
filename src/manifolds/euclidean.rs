use ndarray::Array1;
use ndarray_linalg::Norm;

use crate::core::{Manifold, Error, Result};

/// Euclidean space R^n with standard metric
pub struct Euclidean {
    dim: usize,
}

impl Euclidean {
    pub fn new(dim: usize) -> Self {
        Euclidean { dim }
    }
    
    pub fn dim(&self) -> usize {
        self.dim
    }
}

impl Manifold for Euclidean {
    type Point = Array1<f64>;
    type Vector = Array1<f64>;
    type Scalar = f64;
    
    /// Exponential map: simple addition in Euclidean space (unchecked)
    fn exp_unchecked(&self, p: &Self::Point, x: &Self::Vector) -> Self::Point {
        p + x
    }
    
    /// Logarithmic map: simple subtraction in Euclidean space (unchecked)
    fn log_unchecked(&self, p: &Self::Point, q: &Self::Point) -> Self::Vector {
        q - p
    }
    
    /// Standard Euclidean metric: g(x, y) = x^T y
    fn metric(&self, _p: &Self::Point, x: &Self::Vector, y: &Self::Vector) -> Self::Scalar {
        x.dot(y)
    }
    
    /// Distance is Euclidean norm
    fn distance(&self, p: &Self::Point, q: &Self::Point) -> Result<Self::Scalar> {
        let diff = q - p;
        Ok(diff.norm_l2())
    }
    
    /// Validate dimension
    fn validate_point(&self, p: &Self::Point) -> Result<()> {
        if p.len() != self.dim {
            return Err(Error::DimensionMismatch {
                expected: self.dim,
                got: p.len(),
            });
        }
        Ok(())
    }
    
    /// Validate vector dimension
    fn validate_vector(&self, _p: &Self::Point, x: &Self::Vector) -> Result<()> {
        if x.len() != self.dim {
            return Err(Error::DimensionMismatch {
                expected: self.dim,
                got: x.len(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;
    use approx::assert_relative_eq;
    
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
}

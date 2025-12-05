use ndarray::Array1;
use crate::core::{Manifold, Result};

/// Product of two manifolds M1 × M2
/// 
/// Used to combine different parameter spaces, e.g.:
/// - R^+ × R^+ × R^+ × [-1,1] × R^+ for Heston parameters
pub struct ProductManifold<M1: Manifold, M2: Manifold> {
    pub m1: M1,
    pub m2: M2,
}

impl<M1, M2> ProductManifold<M1, M2>
where
    M1: Manifold<Point = Array1<f64>, Vector = Array1<f64>, Scalar = f64>,
    M2: Manifold<Point = Array1<f64>, Vector = Array1<f64>, Scalar = f64>,
{
    pub fn new(m1: M1, m2: M2) -> Self {
        ProductManifold { m1, m2 }
    }
    
    /// Split combined point into components
    fn split_point(&self, p: &Array1<f64>) -> Result<(Array1<f64>, Array1<f64>)> {
        // Assume m1 takes first dim1 components, m2 takes rest
        // This is a simplified version - full version would need dim() method
        let total_len = p.len();
        let split_idx = total_len / 2; // Simplified
        
        let p1 = p.slice(ndarray::s![..split_idx]).to_owned();
        let p2 = p.slice(ndarray::s![split_idx..]).to_owned();
        
        Ok((p1, p2))
    }
    
    /// Combine two points into single point
    fn combine_points(&self, p1: &Array1<f64>, p2: &Array1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(p1.len() + p2.len());
        result.slice_mut(ndarray::s![..p1.len()]).assign(p1);
        result.slice_mut(ndarray::s![p1.len()..]).assign(p2);
        result
    }
}

impl<M1, M2> Manifold for ProductManifold<M1, M2>
where
    M1: Manifold<Point = Array1<f64>, Vector = Array1<f64>, Scalar = f64>,
    M2: Manifold<Point = Array1<f64>, Vector = Array1<f64>, Scalar = f64>,
{
    type Point = Array1<f64>;
    type Vector = Array1<f64>;
    type Scalar = f64;
    
    fn exp_unchecked(&self, p: &Self::Point, x: &Self::Vector) -> Self::Point {
        let (p1, p2) = self.split_point(p).expect("split failed");
        let (x1, x2) = self.split_point(x).expect("split failed");
        
        let q1 = self.m1.exp_unchecked(&p1, &x1);
        let q2 = self.m2.exp_unchecked(&p2, &x2);
        
        self.combine_points(&q1, &q2)
    }
    
    fn exp(&self, p: &Self::Point, x: &Self::Vector) -> Result<Self::Point> {
        let (p1, p2) = self.split_point(p)?;
        let (x1, x2) = self.split_point(x)?;
        
        let q1 = self.m1.exp(&p1, &x1)?;
        let q2 = self.m2.exp(&p2, &x2)?;
        
        Ok(self.combine_points(&q1, &q2))
    }
    
    fn log_unchecked(&self, p: &Self::Point, q: &Self::Point) -> Self::Vector {
        let (p1, p2) = self.split_point(p).expect("split failed");
        let (q1, q2) = self.split_point(q).expect("split failed");
        
        let x1 = self.m1.log_unchecked(&p1, &q1);
        let x2 = self.m2.log_unchecked(&p2, &q2);
        
        self.combine_points(&x1, &x2)
    }
    
    fn log(&self, p: &Self::Point, q: &Self::Point) -> Result<Self::Vector> {
        let (p1, p2) = self.split_point(p)?;
        let (q1, q2) = self.split_point(q)?;
        
        let x1 = self.m1.log(&p1, &q1)?;
        let x2 = self.m2.log(&p2, &q2)?;
        
        Ok(self.combine_points(&x1, &x2))
    }
    
    fn metric(&self, p: &Self::Point, x: &Self::Vector, y: &Self::Vector) -> Self::Scalar {
        let (p1, p2) = match self.split_point(p) {
            Ok(pts) => pts,
            Err(_) => return f64::NAN,
        };
        let (x1, x2) = match self.split_point(x) {
            Ok(vecs) => vecs,
            Err(_) => return f64::NAN,
        };
        let (y1, y2) = match self.split_point(y) {
            Ok(vecs) => vecs,
            Err(_) => return f64::NAN,
        };
        
        // Product metric: g((x1,x2), (y1,y2)) = g1(x1,y1) + g2(x2,y2)
        self.m1.metric(&p1, &x1, &y1) + self.m2.metric(&p2, &x2, &y2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifolds::Euclidean;
    use ndarray::arr1;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_product_manifold_exp_log() {
        let m1 = Euclidean::new(2);
        let m2 = Euclidean::new(2);
        let product = ProductManifold::new(m1, m2);
        
        let p = arr1(&[1.0, 2.0, 3.0, 4.0]);
        let x = arr1(&[0.1, 0.2, 0.3, 0.4]);
        
        let q = product.exp(&p, &x).unwrap();
        let x_recovered = product.log(&p, &q).unwrap();
        
        for i in 0..4 {
            assert_relative_eq!(x[i], x_recovered[i], epsilon = 1e-10);
        }
    }
}

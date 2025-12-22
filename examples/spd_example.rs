use manifolds::prelude::*;
use ndarray::Array2;

fn main() {
    println!("=== SPD Manifold Example ===\n");
    
    // Create SPD manifold of 2x2 matrices
    let spd = SPD::new(2);
    
    // Define two SPD matrices
    let p = Array2::from_shape_vec((2, 2), vec![4.0, 1.0, 1.0, 3.0]).unwrap();
    let q = Array2::from_shape_vec((2, 2), vec![5.0, 0.5, 0.5, 4.0]).unwrap();
    
    println!("Base point P:");
    println!("{:?}\n", p);
    
    println!("Target point Q:");
    println!("{:?}\n", q);
    
    // Compute logarithmic map: tangent vector from P to Q
    let x = spd.log(&p, &q).unwrap();
    println!("Log map log_P(Q) (tangent vector):");
    println!("{:?}\n", x);
    
    // Compute exponential map: should recover Q
    let q_recovered = spd.exp(&p, &x).unwrap();
    println!("Exp map exp_P(log_P(Q)) (should equal Q):");
    println!("{:?}\n", q_recovered);
    
    // Compute Riemannian distance
    let distance = spd.distance(&p, &q).unwrap();
    println!("Riemannian distance d(P, Q) = {:.6}", distance);
    
    // Verify exp∘log = identity
    let error = (&q - &q_recovered).mapv(|x| x.abs()).sum();
    println!("Recovery error: {:.2e}", error);
    
    if error < 1e-10 {
        println!("✓ Exponential and logarithmic maps are inverses");
    }
    
    println!("\n=== Fisher Information Matrix Use Case ===\n");
    println!("SPD manifolds are natural for:");
    println!("1. Covariance matrices in statistics");
    println!("2. Fisher Information Matrices in natural gradient");
    println!("3. Precision matrices in Gaussian processes");
    println!("\nThe affine-invariant metric ensures:");
    println!("- Invariance under linear transformations");
    println!("- Proper handling of scale differences");
    println!("- Natural geometry for positive definite constraints");
}

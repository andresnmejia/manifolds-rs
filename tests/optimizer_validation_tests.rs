use approx::assert_relative_eq;
use manifolds::algorithms::optimization::*;
use manifolds::core::{EmbeddedManifold, Error, Manifold, Result};
use manifolds::manifolds::{Euclidean, Sphere, Stiefel};
use ndarray::{arr1, arr2, Array1, Array2};
use ndarray_linalg::Norm;

// =========================================================================
// TEST 1: Quadratic optimization on Euclidean space
// =========================================================================

struct QuadraticObjective {
    target: Array1<f64>,
}

impl ObjectiveFunction<Euclidean> for QuadraticObjective {
    fn eval(&self, _manifold: &Euclidean, p: &Array1<f64>) -> Result<f64> {
        let diff = p - &self.target;
        Ok(0.5 * diff.dot(&diff))
    }

    fn gradient_ambient(&self, _manifold: &Euclidean, p: &Array1<f64>) -> Result<Array1<f64>> {
        Ok(p - &self.target)
    }

    fn hessian_ambient(&self, _manifold: &Euclidean, _p: &Array1<f64>) -> Result<Array2<f64>> {
        // Hessian of (1/2)||x - target||^2 is I
        Ok(Array2::eye(self.target.len()))
    }
}

#[test]
fn test_euclidean_gradient_descent() {
    let manifold = Euclidean::new(3);
    let target = arr1(&[1.0, 2.0, 3.0]);
    let objective = QuadraticObjective {
        target: target.clone(),
    };

    let p0 = arr1(&[10.0, -5.0, 7.0]);

    let optimizer = RiemannianGradientDescent::new()
        .with_step_size(0.1)
        .with_convergence(Convergence {
            max_iterations: 1000,
            grad_tol: 1e-6,
            f_tol: 1e-9,
            step_tol: 1e-9,
        });

    let result = optimizer.minimize(&manifold, &objective, p0).unwrap();

    assert!(result.converged, "Should converge");
    assert!(result.value < 1e-10, "Should reach minimum");

    // Check we reached the target
    for i in 0..3 {
        assert_relative_eq!(result.point[i], target[i], epsilon = 1e-4);
    }
}

#[test]
fn test_euclidean_newton_method() {
    let manifold = Euclidean::new(3);
    let target = arr1(&[1.0, 2.0, 3.0]);
    let objective = QuadraticObjective {
        target: target.clone(),
    };

    let p0 = arr1(&[10.0, -5.0, 7.0]);

    let optimizer = RiemannianGradientDescent::new()
        .with_newton()
        .with_convergence(Convergence {
            max_iterations: 50,
            grad_tol: 1e-10,
            ..Default::default()
        });

    let result = optimizer
        .minimize_newton(&manifold, &objective, p0)
        .unwrap();

    assert!(result.converged, "Newton should converge quickly");
    assert!(
        result.iterations < 5,
        "Newton should converge in ~1-2 iterations for quadratic"
    );

    // Newton should get us very close to the target
    for i in 0..3 {
        assert_relative_eq!(result.point[i], target[i], epsilon = 1e-8);
    }
}

// =========================================================================
// TEST 2: Optimization on Sphere
// =========================================================================

struct SphericalDistance {
    target: Array1<f64>,
}

impl ObjectiveFunction<Sphere> for SphericalDistance {
    fn eval(&self, manifold: &Sphere, p: &Array1<f64>) -> Result<f64> {
        // Distance squared on sphere: d(p, target)^2
        let dist = manifold.distance(p, &self.target)?;
        Ok(0.5 * dist * dist)
    }

    fn gradient_ambient(&self, manifold: &Sphere, p: &Array1<f64>) -> Result<Array1<f64>> {
        // Gradient of (1/2)d(p,q)^2 is the log map
        manifold.log(p, &self.target)
    }
}

#[test]
fn test_sphere_gradient_descent() {
    let manifold = Sphere::new(2); // S^2 (unit sphere in R^3)

    let target = arr1(&[1.0, 0.0, 0.0]);
    let objective = SphericalDistance {
        target: target.clone(),
    };

    let p0 = arr1(&[0.0, 1.0, 0.0]);

    let optimizer = RiemannianGradientDescent::new()
        .with_retraction(RetractionMethod::Projection)
        .with_line_search(LineSearch::Backtracking {
            initial_alpha: 0.5,
            rho: 0.8,
            c1: 1e-4,
        })
        .with_convergence(Convergence {
            max_iterations: 200,
            grad_tol: 1e-5,
            f_tol: 1e-9,
            step_tol: 1e-9,
        })
        .verbose();

    let result = optimizer.minimize(&manifold, &objective, p0).unwrap();

    assert!(result.converged, "Should converge");
    assert!(
        (result.point.norm_l2() - 1.0).abs() < 1e-8,
        "Should stay on sphere"
    );

    // Check we reached near the target
    let final_dist = manifold.distance(&result.point, &target).unwrap();
    assert!(
        final_dist < 0.1,
        "Should be close to target, got distance: {}",
        final_dist
    );
}

#[test]
fn test_sphere_exponential_vs_projection() {
    let manifold = Sphere::new(2);
    let target = arr1(&[1.0, 0.0, 0.0]);
    let objective = SphericalDistance {
        target: target.clone(),
    };
    let p0 = arr1(&[0.0, 0.0, 1.0]);

    // Test with exponential map
    let optimizer_exp = RiemannianGradientDescent::new()
        .with_retraction(RetractionMethod::Exponential)
        .with_step_size(0.3)
        .with_convergence(Convergence {
            max_iterations: 100,
            grad_tol: 1e-6,
            ..Default::default()
        });

    let result_exp = optimizer_exp
        .minimize(&manifold, &objective, p0.clone())
        .unwrap();

    // Test with projection
    let optimizer_proj = RiemannianGradientDescent::new()
        .with_retraction(RetractionMethod::Projection)
        .with_step_size(0.3)
        .with_convergence(Convergence {
            max_iterations: 100,
            grad_tol: 1e-6,
            ..Default::default()
        });

    let result_proj = optimizer_proj
        .minimize(&manifold, &objective, p0.clone())
        .unwrap();

    // Both should converge to similar points
    assert!(result_exp.converged && result_proj.converged);
    assert!(
        result_exp.value - result_proj.value < 1e-4,
        "Exponential and projection should reach similar objectives"
    );
}

// =========================================================================
// TEST 3: Least-squares on Stiefel manifold
// =========================================================================

struct StiefelLeastSquares {
    a: Array2<f64>, // Target matrix
    n: usize,
    p: usize,
}

impl ResidualFunction<Stiefel> for StiefelLeastSquares {
    fn residual(&self, manifold: &Stiefel, x: &Array2<f64>) -> Result<Array1<f64>> {
        // Residual: r(X) = vec(X - A)
        // Vectorize the difference in column-major order
        let diff = x - &self.a;
        manifold.to_ambient(&diff)
    }

    fn jacobian(&self, manifold: &Stiefel, _x: &Array2<f64>) -> Result<Array2<f64>> {
        // For this simple case, Jacobian w.r.t. ambient coordinates is identity
        // In practice, you'd compute dr/dX accounting for the manifold structure
        let d = manifold.ambient_dim();
        Ok(Array2::eye(d))
    }

    fn num_residuals(&self) -> usize {
        self.n * self.p
    }
}

#[test]
fn test_stiefel_levenberg_marquardt() {
    let n = 5;
    let p = 3;
    let manifold = Stiefel::new(n, p).unwrap();

    // Target: random matrix (not on Stiefel)
    let a = Array2::from_shape_vec(
        (n, p),
        vec![
            1.0, 0.2, 0.3, 0.1, 1.0, 0.1, 0.2, 0.1, 1.0, 0.3, 0.2, 0.1, 0.1, 0.3, 0.2,
        ],
    )
    .unwrap();

    let objective = StiefelLeastSquares { a, n, p };

    // Initial point: identity padded
    let mut p0 = Array2::zeros((n, p));
    for i in 0..p {
        p0[[i, i]] = 1.0;
    }

    let optimizer = RiemannianGradientDescent::new()
        .with_levenberg_marquardt()
        .with_lm_params(1e-2, 2.0, 0.5)
        .with_retraction(RetractionMethod::Projection)
        .with_convergence(Convergence {
            max_iterations: 100,
            grad_tol: 1e-4,
            ..Default::default()
        })
        .verbose();

    let result = optimizer.minimize_lm(&manifold, &objective, p0).unwrap();

    assert!(result.converged, "LM should converge");

    // Verify orthonormality
    let xtx = result.point.t().dot(&result.point);
    for i in 0..p {
        for j in 0..p {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_relative_eq!(xtx[[i, j]], expected, epsilon = 1e-6);
        }
    }
}

#[test]
fn test_stiefel_gauss_newton() {
    let n = 4;
    let p = 2;
    let manifold = Stiefel::new(n, p).unwrap();

    let a = Array2::from_shape_vec((n, p), vec![1.0, 0.0, 0.5, 0.5, 0.0, 1.0, 0.5, 0.5]).unwrap();

    let objective = StiefelLeastSquares { a, n, p };

    let mut p0 = Array2::zeros((n, p));
    for i in 0..p {
        p0[[i, i]] = 1.0;
    }

    let optimizer = RiemannianGradientDescent::new()
        .with_gauss_newton()
        .with_retraction(RetractionMethod::Projection)
        .with_convergence(Convergence {
            max_iterations: 50,
            grad_tol: 1e-5,
            ..Default::default()
        });

    let result = optimizer
        .minimize_gauss_newton(&manifold, &objective, p0)
        .unwrap();

    assert!(result.converged, "Gauss-Newton should converge");
    assert!(
        manifold.is_on_manifold(&result.point, 1e-6),
        "Result should be on Stiefel manifold"
    );
}

// =========================================================================
// TEST 4: Test retraction methods
// =========================================================================

#[test]
fn test_retraction_methods_stiefel() {
    let n = 5;
    let p = 2;
    let manifold = Stiefel::new(n, p).unwrap();

    // Start point
    let mut x = Array2::zeros((n, p));
    x[[0, 0]] = 1.0;
    x[[1, 1]] = 1.0;

    // Tangent vector
    let mut v = Array2::zeros((n, p));
    v[[2, 0]] = 0.1;
    v[[3, 1]] = 0.1;
    let v_tan = manifold.project_tangent(&x, &v).unwrap();
    let v_ambient = manifold.vector_to_ambient(&v_tan).unwrap();

    // Test exponential retraction
    let x_exp = manifold.exp_from_ambient(&x, &v_ambient).unwrap();
    assert!(
        manifold.is_on_manifold(&x_exp, 1e-8),
        "Exp should stay on manifold"
    );

    // Test projection retraction
    let x_proj_ambient = manifold.to_ambient(&x).unwrap();
    let x_stepped = &x_proj_ambient + &v_ambient;
    let x_proj = manifold.from_ambient(&x_stepped).unwrap();
    assert!(
        manifold.is_on_manifold(&x_proj, 1e-8),
        "Projection should return to manifold"
    );

    // Both should be close for small steps
    let dist = manifold.distance(&x_exp, &x_proj).unwrap();
    assert!(
        dist < 0.1,
        "Exp and projection should be similar for small steps"
    );
}

// =========================================================================
// TEST 5: Convergence criteria
// =========================================================================

#[test]
fn test_convergence_criteria() {
    let manifold = Euclidean::new(2);
    let target = arr1(&[0.0, 0.0]);
    let objective = QuadraticObjective {
        target: target.clone(),
    };

    // Test gradient tolerance convergence
    let p0 = arr1(&[10.0, 10.0]); // Start far away

    let optimizer = RiemannianGradientDescent::new()
        .with_step_size(0.1)
        .with_convergence(Convergence {
            max_iterations: 1000,
            grad_tol: 1e-4,  // Loose gradient tolerance
            f_tol: 1e-20,    // Very tight (won't trigger)
            step_tol: 1e-20, // Very tight (won't trigger)
        });

    let result = optimizer
        .minimize(&manifold, &objective, p0.clone())
        .unwrap();

    assert!(result.converged, "Should converge");
    assert!(
        result.message.contains("Gradient"),
        "Expected gradient convergence, got: {}",
        result.message
    );
    assert!(result.grad_norm < 1e-4, "Should meet gradient tolerance");

    // Test that max iterations works
    let optimizer_max = RiemannianGradientDescent::new()
        .with_step_size(0.0001) // Tiny step
        .with_convergence(Convergence {
            max_iterations: 5, // Very few iterations
            grad_tol: 1e-10,
            f_tol: 1e-20,
            step_tol: 1e-20,
        });

    let result_max = optimizer_max
        .minimize(&manifold, &objective, p0.clone())
        .unwrap();
    assert!(
        !result_max.converged,
        "Should not converge with max iterations, but converged = {}",
        result_max.value
    );

    assert!(result_max.iterations == 5, "Should hit max iterations");
}

// =========================================================================
// TEST 6: Line search
// =========================================================================

#[test]
fn test_line_search_vs_fixed_step() {
    let manifold = Euclidean::new(3);
    let target = arr1(&[5.0, 5.0, 5.0]);
    let objective = QuadraticObjective {
        target: target.clone(),
    };
    let p0 = arr1(&[0.0, 0.0, 0.0]);

    // With line search
    let optimizer_ls = RiemannianGradientDescent::new()
        .with_line_search(LineSearch::Backtracking {
            initial_alpha: 1.0,
            rho: 0.5,
            c1: 1e-4,
        })
        .with_convergence(Convergence {
            max_iterations: 100,
            grad_tol: 1e-6,
            ..Default::default()
        });

    let result_ls = optimizer_ls
        .minimize(&manifold, &objective, p0.clone())
        .unwrap();

    // Without line search (fixed step) - use larger step
    let optimizer_fixed = RiemannianGradientDescent::new()
        .with_line_search(LineSearch::None)
        .with_step_size(0.5) // Larger step size
        .with_convergence(Convergence {
            max_iterations: 100,
            grad_tol: 1e-6,
            ..Default::default()
        });

    let result_fixed = optimizer_fixed
        .minimize(&manifold, &objective, p0.clone())
        .unwrap();

    // Both should converge
    assert!(result_ls.converged, "Line search should converge");
    assert!(result_fixed.converged, "Fixed step should converge");

    println!("Line search iterations: {}", result_ls.iterations);
    println!("Fixed step iterations: {}", result_fixed.iterations);

    // Line search should be more efficient
    assert!(
        result_ls.iterations <= result_fixed.iterations,
        "Line search should take fewer or equal iterations"
    );
}

// =========================================================================
// TEST 7: Rosenbrock function (classic test)
// =========================================================================

struct RosenbrockObjective;

impl ObjectiveFunction<Euclidean> for RosenbrockObjective {
    fn eval(&self, _manifold: &Euclidean, p: &Array1<f64>) -> Result<f64> {
        // f(x,y) = (1-x)^2 + 100(y-x^2)^2
        let x = p[0];
        let y = p[1];
        Ok((1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2))
    }

    fn gradient_ambient(&self, _manifold: &Euclidean, p: &Array1<f64>) -> Result<Array1<f64>> {
        let x = p[0];
        let y = p[1];
        Ok(arr1(&[
            -2.0 * (1.0 - x) - 400.0 * x * (y - x * x),
            200.0 * (y - x * x),
        ]))
    }
}

#[test]
fn test_rosenbrock() {
    let manifold = Euclidean::new(2);
    let objective = RosenbrockObjective;
    let p0 = arr1(&[0.0, 0.0]);

    let optimizer = RiemannianGradientDescent::new()
        .with_step_size(0.001)
        .with_line_search(LineSearch::Backtracking {
            initial_alpha: 0.01,
            rho: 0.5,
            c1: 1e-4,
        })
        .with_convergence(Convergence {
            max_iterations: 10000,
            grad_tol: 1e-4,
            f_tol: 1e-8,
            step_tol: 1e-8,
        });

    let result = optimizer.minimize(&manifold, &objective, p0).unwrap();

    // Rosenbrock minimum is at (1, 1)
    println!("Rosenbrock result: {:?}", result.point);
    println!("Iterations: {}", result.iterations);

    // Should get reasonably close
    assert!(result.value < 1e-3, "Should reach near minimum");
}

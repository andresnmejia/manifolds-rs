use crate::core::error::{Error, Result};
use crate::core::traits::EmbeddedManifold;
use ndarray::{Array1, Array2};
use ndarray_linalg::{Norm, Solve};

/// Objective function on a Riemannian manifold embedded in Euclidean space
pub trait ObjectiveFunction<M: EmbeddedManifold> {
    /// Evaluate the function at point p
    fn eval(&self, manifold: &M, p: &M::Point) -> Result<f64>;

    /// Compute the Euclidean gradient in ambient coordinates
    /// Returns ∇f ∈ R^d where d = manifold.ambient_dim()
    fn gradient_ambient(&self, manifold: &M, p: &M::Point) -> Result<Array1<f64>>;

    /// Compute the Riemannian gradient (projected to tangent space)
    /// Returns grad f ∈ T_p M represented in ambient coordinates
    fn riemannian_gradient(&self, manifold: &M, p: &M::Point) -> Result<Array1<f64>> {
        let grad_ambient = self.gradient_ambient(manifold, p)?;
        manifold.project_to_ambient_tangent(p, &grad_ambient)
    }

    /// Compute the Euclidean Hessian in ambient coordinates (for Newton's method)
    /// Returns ∇²f ∈ R^{d×d}
    ///
    /// This is optional - only needed for Newton's method
    fn hessian_ambient(&self, manifold: &M, p: &M::Point) -> Result<Array2<f64>> {
        Err(Error::InvalidParameter(
            "Hessian not implemented for this objective".to_string(),
        ))
    }

    /// Compute the Riemannian Hessian (projected appropriately)
    /// This is more complex - see Absil et al. (2008) Section 5.5
    ///
    /// For now, we'll use the Euclidean Hessian projected to tangent space
    fn riemannian_hessian(&self, manifold: &M, p: &M::Point) -> Result<Array2<f64>> {
        let h = self.hessian_ambient(manifold, p)?;

        // Project rows and columns to tangent space
        // This is a simplification - full Riemannian Hessian is more involved
        Ok(h)
    }
}

/// Least-squares objective for Levenberg-Marquardt
/// f(p) = (1/2) ||r(p)||^2 where r: M → R^m is the residual function
pub trait ResidualFunction<M: EmbeddedManifold> {
    /// Evaluate residual vector r(p) ∈ R^m
    fn residual(&self, manifold: &M, p: &M::Point) -> Result<Array1<f64>>;

    /// Compute the Jacobian J(p) ∈ R^{m×d}
    /// where d = manifold.ambient_dim()
    /// J_ij = ∂r_i/∂x_j where x ∈ R^d are ambient coordinates
    fn jacobian(&self, manifold: &M, p: &M::Point) -> Result<Array2<f64>>;

    /// Number of residuals
    fn num_residuals(&self) -> usize;
}

/// Automatically implements ObjectiveFunction for any ResidualFunction
/// f(p) = (1/2) ||r(p)||^2
impl<M, R> ObjectiveFunction<M> for R
where
    M: EmbeddedManifold,
    R: ResidualFunction<M>,
{
    fn eval(&self, manifold: &M, p: &M::Point) -> Result<f64> {
        let r = self.residual(manifold, p)?;
        Ok(0.5 * r.dot(&r))
    }

    fn gradient_ambient(&self, manifold: &M, p: &M::Point) -> Result<Array1<f64>> {
        let r = self.residual(manifold, p)?;
        let j = self.jacobian(manifold, p)?;

        // grad f = J^T r ∈ R^d
        Ok(j.t().dot(&r))
    }
}

/// Method for taking steps on the manifold
#[derive(Debug, Clone, Copy)]
pub enum RetractionMethod {
    /// Use exponential map: p_new = exp_p(v)
    /// Most accurate but can be expensive (requires QR for Stiefel)
    Exponential,

    /// Use projection retraction: p_new = Proj(p + v)
    /// Simpler: step in ambient space, then project back to manifold
    /// For Stiefel: X_new = qf(X + V) where qf is QR-based orthonormalization
    Projection,
}

impl Default for RetractionMethod {
    fn default() -> Self {
        // Projection is typically faster and works well in practice
        RetractionMethod::Projection
    }
}

/// Line search strategy
#[derive(Debug, Clone, Copy)]
pub enum LineSearch {
    /// No line search, use fixed step size
    None,
    /// Backtracking line search with Armijo condition
    /// Parameters: (initial_alpha, rho, c1)
    /// alpha_{k+1} = rho * alpha_k until f(p + alpha*v) ≤ f(p) + c1*alpha*⟨grad f, v⟩
    Backtracking {
        initial_alpha: f64,
        rho: f64,
        c1: f64,
    },
}

impl Default for LineSearch {
    fn default() -> Self {
        LineSearch::Backtracking {
            initial_alpha: 1.0,
            rho: 0.5,
            c1: 1e-4,
        }
    }
}

/// Convergence criteria
#[derive(Debug, Clone)]
pub struct Convergence {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Tolerance on gradient norm: ||grad f|| < grad_tol
    pub grad_tol: f64,
    /// Tolerance on function value change: |f_k - f_{k-1}| < f_tol
    pub f_tol: f64,
    /// Tolerance on step size: ||p_k - p_{k-1}|| < step_tol
    pub step_tol: f64,
}

impl Default for Convergence {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            grad_tol: 1e-6,
            f_tol: 1e-9,
            step_tol: 1e-9,
        }
    }
}

/// Optimization method
#[derive(Debug, Clone, Copy)]
pub enum OptimizationMethod {
    /// Gradient descent
    GradientDescent,
    /// Levenberg-Marquardt (for least-squares problems)
    LevenbergMarquardt,
    /// Newton's method (requires Hessian)
    Newton,
    /// Gauss-Newton (for least-squares, no damping)
    GaussNewton,
}

/// Optimization result
#[derive(Debug, Clone)]
pub struct OptimizationResult<P> {
    /// Final point
    pub point: P,
    /// Final objective value
    pub value: f64,
    /// Final gradient norm
    pub grad_norm: f64,
    /// Number of iterations
    pub iterations: usize,
    /// Convergence status
    pub converged: bool,
    /// Reason for termination
    pub message: String,
}

/// Riemannian gradient descent optimizer for manifolds with Array2<f64> points
pub struct RiemannianGradientDescent {
    /// Step size (learning rate)
    pub step_size: f64,
    /// Line search strategy
    pub line_search: LineSearch,
    /// Retraction method (exponential vs projection)
    pub retraction: RetractionMethod,
    /// Convergence criteria
    pub convergence: Convergence,
    /// Optimization method
    pub method: OptimizationMethod,
    /// Initial damping parameter for LM (λ_0)
    pub lambda_init: f64,
    /// Factor to increase damping when step is rejected (ν)
    pub lambda_up: f64,
    /// Factor to decrease damping when step is accepted
    pub lambda_down: f64,
    /// Maximum lambda before declaring failure
    pub lambda_max: f64,
    /// Verbose output
    pub verbose: bool,
}

impl Default for RiemannianGradientDescent {
    fn default() -> Self {
        Self {
            step_size: 0.01,
            line_search: LineSearch::default(),
            retraction: RetractionMethod::default(),
            convergence: Convergence::default(),
            method: OptimizationMethod::GradientDescent,
            lambda_init: 1e-3,
            lambda_up: 2.0,
            lambda_down: 0.5,
            lambda_max: 1e10,
            verbose: false,
        }
    }
}

impl RiemannianGradientDescent {
    /// Create a new optimizer with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable Levenberg-Marquardt damping
    pub fn with_levenberg_marquardt(mut self) -> Self {
        self.method = OptimizationMethod::LevenbergMarquardt;
        self
    }

    /// Use Newton's method (requires Hessian)
    pub fn with_newton(mut self) -> Self {
        self.method = OptimizationMethod::Newton;
        self
    }

    /// Use Gauss-Newton (for least-squares, no damping)
    pub fn with_gauss_newton(mut self) -> Self {
        self.method = OptimizationMethod::GaussNewton;
        self
    }

    /// Set retraction method
    pub fn with_retraction(mut self, retraction: RetractionMethod) -> Self {
        self.retraction = retraction;
        self
    }

    /// Set step size
    pub fn with_step_size(mut self, step_size: f64) -> Self {
        self.step_size = step_size;
        self
    }

    /// Set line search strategy
    pub fn with_line_search(mut self, line_search: LineSearch) -> Self {
        self.line_search = line_search;
        self
    }

    /// Set convergence criteria
    pub fn with_convergence(mut self, convergence: Convergence) -> Self {
        self.convergence = convergence;
        self
    }

    /// Set LM parameters
    pub fn with_lm_params(mut self, lambda_init: f64, lambda_up: f64, lambda_down: f64) -> Self {
        self.lambda_init = lambda_init;
        self.lambda_up = lambda_up;
        self.lambda_down = lambda_down;
        self
    }

    /// Enable verbose output
    pub fn verbose(mut self) -> Self {
        self.verbose = true;
        self
    }

    /// Take a step on the manifold using configured retraction method
    fn retract<M>(&self, manifold: &M, p: &M::Point, v_ambient: &Array1<f64>) -> Result<M::Point>
    where
        M: EmbeddedManifold<Scalar = f64>,
        M::Point: Clone,
    {
        match self.retraction {
            RetractionMethod::Exponential => {
                // Use exponential map
                manifold.exp_from_ambient(p, v_ambient)
            }
            RetractionMethod::Projection => {
                // Step-and-project: p_new = Proj(p + v)
                // Convert p to ambient
                let p_ambient = manifold.to_ambient(p)?;

                // Step in ambient space
                let p_stepped = &p_ambient + v_ambient;

                // Project back to manifold
                manifold.from_ambient(&p_stepped)
            }
        }
    }

    /// Optimize using standard Riemannian gradient descent
    ///
    /// All computations are done in ambient Euclidean coordinates R^d,
    /// with projection to tangent space as needed.
    pub fn minimize<M, F>(
        &self,
        manifold: &M,
        objective: &F,
        p0: M::Point,
    ) -> Result<OptimizationResult<M::Point>>
    where
        M: EmbeddedManifold<Scalar = f64>,
        F: ObjectiveFunction<M>,
        M::Point: Clone,
    {
        let mut p = p0;
        let mut f_val = objective.eval(manifold, &p)?;

        if self.verbose {
            println!("Starting Riemannian Gradient Descent");
            println!("{:>6} {:>14} {:>14}", "Iter", "f(p)", "||grad f||");
            println!("{}", "-".repeat(38));
        }

        for iter in 0..self.convergence.max_iterations {
            // Compute Riemannian gradient in ambient coordinates
            let grad_ambient = objective.riemannian_gradient(manifold, &p)?;
            let grad_norm = grad_ambient.norm_l2();

            if self.verbose && iter % 10 == 0 {
                println!("{:6} {:14.6e} {:14.6e}", iter, f_val, grad_norm);
            }

            // Check gradient convergence
            if grad_norm < self.convergence.grad_tol {
                if self.verbose {
                    println!("\n✓ Converged: Gradient norm below tolerance");
                }
                return Ok(OptimizationResult {
                    point: p,
                    value: f_val,
                    grad_norm,
                    iterations: iter,
                    converged: true,
                    message: "Gradient norm below tolerance".to_string(),
                });
            }

            // Search direction: negative gradient
            let direction_ambient = &grad_ambient * (-1.0);

            // Perform line search
            let alpha = self.perform_line_search(
                manifold,
                objective,
                &p,
                &direction_ambient,
                &grad_ambient,
                f_val,
            )?;

            // Take step using configured retraction method
            let step_ambient = &direction_ambient * alpha;
            let p_new = self.retract(manifold, &p, &step_ambient)?;

            // Compute new objective value
            let f_new = objective.eval(manifold, &p_new)?;

            // Check function value convergence
            let f_diff = (f_val - f_new).abs();
            if f_diff < self.convergence.f_tol {
                if self.verbose {
                    println!("\n✓ Converged: Function value change below tolerance");
                }
                return Ok(OptimizationResult {
                    point: p_new,
                    value: f_new,
                    grad_norm,
                    iterations: iter + 1,
                    converged: true,
                    message: "Function value change below tolerance".to_string(),
                });
            }

            // Check step size convergence
            let step_dist = manifold.distance(&p, &p_new)?;
            if step_dist < self.convergence.step_tol {
                if self.verbose {
                    println!("\n✓ Converged: Step size below tolerance");
                }
                return Ok(OptimizationResult {
                    point: p_new,
                    value: f_new,
                    grad_norm,
                    iterations: iter + 1,
                    converged: true,
                    message: "Step size below tolerance".to_string(),
                });
            }

            // Update for next iteration
            p = p_new;
            f_val = f_new;
        }

        // Max iterations reached
        let grad_ambient = objective.riemannian_gradient(manifold, &p)?;
        let grad_norm = grad_ambient.norm_l2();

        if self.verbose {
            println!("\n⚠ Maximum iterations reached");
        }

        Ok(OptimizationResult {
            point: p,
            value: f_val,
            grad_norm,
            iterations: self.convergence.max_iterations,
            converged: false,
            message: "Maximum iterations reached".to_string(),
        })
    }

    /// Optimize using Levenberg-Marquardt on a manifold
    ///
    /// This implements the Riemannian Levenberg-Marquardt algorithm in ambient coordinates:
    /// 1. Solve (J^T J + λ I) δ = -J^T r in R^d
    /// 2. Project δ onto tangent space: δ_tan = proj_{T_p M}(δ)
    /// 3. Take step using configured retraction method
    /// 4. Update λ based on gain ratio
    ///
    /// No reshaping needed - everything stays in R^d!
    pub fn minimize_lm<M, R>(
        &self,
        manifold: &M,
        residual_fn: &R,
        p0: M::Point,
    ) -> Result<OptimizationResult<M::Point>>
    where
        M: EmbeddedManifold<Scalar = f64>,
        R: ResidualFunction<M>,
        M::Point: Clone,
    {
        if !matches!(self.method, OptimizationMethod::LevenbergMarquardt) {
            return Err(Error::InvalidParameter(
                "Optimizer not configured for Levenberg-Marquardt. Call with_levenberg_marquardt()"
                    .to_string(),
            ));
        }

        let mut p = p0;
        let mut lambda = self.lambda_init;
        let mut r = residual_fn.residual(manifold, &p)?;
        let mut f_val = 0.5 * r.dot(&r);

        if self.verbose {
            println!("Starting Riemannian Levenberg-Marquardt");
            println!(
                "{:>6} {:>14} {:>14} {:>14}",
                "Iter", "f(p)", "||grad f||", "λ"
            );
            println!("{}", "-".repeat(52));
        }

        for iter in 0..self.convergence.max_iterations {
            // Compute Jacobian J ∈ R^{m×d} in ambient coordinates
            let j = residual_fn.jacobian(manifold, &p)?;

            // Compute J^T J (Gauss-Newton Hessian approximation)
            let jtj = j.t().dot(&j);

            // Compute J^T r (gradient in ambient space)
            let jtr = j.t().dot(&r);
            let grad_norm = jtr.norm_l2();

            if self.verbose && iter % 10 == 0 {
                println!(
                    "{:6} {:14.6e} {:14.6e} {:14.6e}",
                    iter, f_val, grad_norm, lambda
                );
            }

            // Check gradient convergence
            if grad_norm < self.convergence.grad_tol {
                if self.verbose {
                    println!("\n✓ Converged: Gradient norm below tolerance");
                }
                return Ok(OptimizationResult {
                    point: p,
                    value: f_val,
                    grad_norm,
                    iterations: iter,
                    converged: true,
                    message: "Gradient norm below tolerance".to_string(),
                });
            }

            // Add damping: H = J^T J + λ I
            let mut h = jtj.clone();
            for i in 0..h.nrows() {
                h[[i, i]] += lambda;
            }

            // Solve (J^T J + λ I) δ = -J^T r for the step δ ∈ R^d
            let delta_ambient = match h.solve(&(-&jtr)) {
                Ok(d) => d,
                Err(_) => {
                    // Singular matrix, increase damping and try again
                    lambda *= self.lambda_up;
                    if lambda > self.lambda_max {
                        return Ok(OptimizationResult {
                            point: p,
                            value: f_val,
                            grad_norm,
                            iterations: iter,
                            converged: false,
                            message: "Damping parameter too large (singular Hessian)".to_string(),
                        });
                    }
                    continue;
                }
            };

            // Project δ onto tangent space (now in ambient coordinates - clean!)
            let delta_tangent = manifold.project_to_ambient_tangent(&p, &delta_ambient)?;

            // Take step using configured retraction method
            let p_new = self.retract(manifold, &p, &delta_tangent)?;

            // Evaluate at new point
            let r_new = residual_fn.residual(manifold, &p_new)?;
            let f_new = 0.5 * r_new.dot(&r_new);

            // Compute gain ratio ρ = (f - f_new) / (L(0) - L(δ))
            // where L(δ) ≈ f + δ^T g + (1/2) δ^T H δ
            let actual_reduction = f_val - f_new;
            let predicted_reduction =
                -jtr.dot(&delta_ambient) - 0.5 * delta_ambient.dot(&h.dot(&delta_ambient));

            let gain_ratio = if predicted_reduction.abs() < 1e-15 {
                // Predicted reduction too small, accept step
                1.0
            } else {
                actual_reduction / predicted_reduction
            };

            if gain_ratio > 0.0 {
                // Accept step
                p = p_new;
                r = r_new;
                f_val = f_new;

                // Decrease damping (more aggressive)
                lambda *= self.lambda_down;
                lambda = lambda.max(1e-12); // Prevent underflow

                // Check function value convergence
                if actual_reduction.abs() < self.convergence.f_tol {
                    if self.verbose {
                        println!("\n✓ Converged: Function value change below tolerance");
                    }
                    return Ok(OptimizationResult {
                        point: p,
                        value: f_val,
                        grad_norm,
                        iterations: iter + 1,
                        converged: true,
                        message: "Function value change below tolerance".to_string(),
                    });
                }
            } else {
                // Reject step, increase damping
                lambda *= self.lambda_up;
            }

            // Check if damping is too large
            if lambda > self.lambda_max {
                if self.verbose {
                    println!("\n⚠ Damping parameter too large");
                }
                return Ok(OptimizationResult {
                    point: p,
                    value: f_val,
                    grad_norm,
                    iterations: iter + 1,
                    converged: false,
                    message: "Damping parameter too large".to_string(),
                });
            }
        }

        // Max iterations reached
        let j = residual_fn.jacobian(manifold, &p)?;
        let jtr = j.t().dot(&r);
        let grad_norm = jtr.norm_l2();

        if self.verbose {
            println!("\n⚠ Maximum iterations reached");
        }

        Ok(OptimizationResult {
            point: p,
            value: f_val,
            grad_norm,
            iterations: self.convergence.max_iterations,
            converged: false,
            message: "Maximum iterations reached".to_string(),
        })
    }

    /// Optimize using Newton's method on a manifold
    ///
    /// Newton's method uses second-order information (Hessian) for faster convergence:
    /// 1. Solve ∇²f · δ = -∇f for the Newton step
    /// 2. Project δ onto tangent space
    /// 3. Take step using configured retraction method
    ///
    /// Requires ObjectiveFunction to implement hessian_ambient()
    pub fn minimize_newton<M, F>(
        &self,
        manifold: &M,
        objective: &F,
        p0: M::Point,
    ) -> Result<OptimizationResult<M::Point>>
    where
        M: EmbeddedManifold<Scalar = f64>,
        F: ObjectiveFunction<M>,
        M::Point: Clone,
    {
        if !matches!(self.method, OptimizationMethod::Newton) {
            return Err(Error::InvalidParameter(
                "Optimizer not configured for Newton. Call with_newton()".to_string(),
            ));
        }

        let mut p = p0;
        let mut f_val = objective.eval(manifold, &p)?;

        if self.verbose {
            println!("Starting Riemannian Newton's Method");
            println!("{:>6} {:>14} {:>14}", "Iter", "f(p)", "||grad f||");
            println!("{}", "-".repeat(38));
        }

        for iter in 0..self.convergence.max_iterations {
            // Compute gradient and Hessian in ambient coordinates
            let grad_ambient = objective.gradient_ambient(manifold, &p)?;
            let hess_ambient = objective.hessian_ambient(manifold, &p)?;

            // Project gradient to tangent space
            let grad_tangent = manifold.project_to_ambient_tangent(&p, &grad_ambient)?;
            let grad_norm = grad_tangent.norm_l2();

            if self.verbose && iter % 10 == 0 {
                println!("{:6} {:14.6e} {:14.6e}", iter, f_val, grad_norm);
            }

            // Check gradient convergence
            if grad_norm < self.convergence.grad_tol {
                if self.verbose {
                    println!("\n✓ Converged: Gradient norm below tolerance");
                }
                return Ok(OptimizationResult {
                    point: p,
                    value: f_val,
                    grad_norm,
                    iterations: iter,
                    converged: true,
                    message: "Gradient norm below tolerance".to_string(),
                });
            }

            // Solve Newton system: H · δ = -g
            let delta_ambient = match hess_ambient.solve(&(-&grad_ambient)) {
                Ok(d) => d,
                Err(_) => {
                    // Hessian is singular, fall back to gradient descent
                    if self.verbose {
                        println!("  Warning: Singular Hessian, using gradient descent step");
                    }
                    &grad_tangent * (-self.step_size)
                }
            };

            // Project Newton step to tangent space
            let delta_tangent = manifold.project_to_ambient_tangent(&p, &delta_ambient)?;

            // Optionally perform line search
            let alpha = if matches!(self.line_search, LineSearch::None) {
                1.0 // Full Newton step
            } else {
                self.perform_line_search(
                    manifold,
                    objective,
                    &p,
                    &delta_tangent,
                    &grad_tangent,
                    f_val,
                )?
            };

            // Take step
            let step = &delta_tangent * alpha;
            let p_new = self.retract(manifold, &p, &step)?;

            // Compute new objective value
            let f_new = objective.eval(manifold, &p_new)?;

            // Check function value convergence
            let f_diff = (f_val - f_new).abs();
            if f_diff < self.convergence.f_tol {
                if self.verbose {
                    println!("\n✓ Converged: Function value change below tolerance");
                }
                return Ok(OptimizationResult {
                    point: p_new,
                    value: f_new,
                    grad_norm,
                    iterations: iter + 1,
                    converged: true,
                    message: "Function value change below tolerance".to_string(),
                });
            }

            // Check step size convergence
            let step_norm = step.norm_l2();
            if step_norm < self.convergence.step_tol {
                if self.verbose {
                    println!("\n✓ Converged: Step size below tolerance");
                }
                return Ok(OptimizationResult {
                    point: p_new,
                    value: f_new,
                    grad_norm,
                    iterations: iter + 1,
                    converged: true,
                    message: "Step size below tolerance".to_string(),
                });
            }

            // Update for next iteration
            p = p_new;
            f_val = f_new;
        }

        // Max iterations reached
        let grad_ambient = objective.gradient_ambient(manifold, &p)?;
        let grad_tangent = manifold.project_to_ambient_tangent(&p, &grad_ambient)?;
        let grad_norm = grad_tangent.norm_l2();

        if self.verbose {
            println!("\n⚠ Maximum iterations reached");
        }

        Ok(OptimizationResult {
            point: p,
            value: f_val,
            grad_norm,
            iterations: self.convergence.max_iterations,
            converged: false,
            message: "Maximum iterations reached".to_string(),
        })
    }

    /// Optimize using Gauss-Newton on a manifold (for least-squares)
    ///
    /// Gauss-Newton is like Levenberg-Marquardt but without damping:
    /// 1. Solve (J^T J) δ = -J^T r
    /// 2. Project δ onto tangent space
    /// 3. Take step using configured retraction method
    ///
    /// Faster than LM when well-conditioned, but can fail if J^T J is singular
    pub fn minimize_gauss_newton<M, R>(
        &self,
        manifold: &M,
        residual_fn: &R,
        p0: M::Point,
    ) -> Result<OptimizationResult<M::Point>>
    where
        M: EmbeddedManifold<Scalar = f64>,
        R: ResidualFunction<M>,
        M::Point: Clone,
    {
        if !matches!(self.method, OptimizationMethod::GaussNewton) {
            return Err(Error::InvalidParameter(
                "Optimizer not configured for Gauss-Newton. Call with_gauss_newton()".to_string(),
            ));
        }

        let mut p = p0;
        let mut r = residual_fn.residual(manifold, &p)?;
        let mut f_val = 0.5 * r.dot(&r);

        if self.verbose {
            println!("Starting Riemannian Gauss-Newton");
            println!("{:>6} {:>14} {:>14}", "Iter", "f(p)", "||grad f||");
            println!("{}", "-".repeat(38));
        }

        for iter in 0..self.convergence.max_iterations {
            // Compute Jacobian in ambient coordinates
            let j = residual_fn.jacobian(manifold, &p)?;

            // Compute J^T J and J^T r
            let jtj = j.t().dot(&j);
            let jtr = j.t().dot(&r);
            let grad_norm = jtr.norm_l2();

            if self.verbose && iter % 10 == 0 {
                println!("{:6} {:14.6e} {:14.6e}", iter, f_val, grad_norm);
            }

            // Check gradient convergence
            if grad_norm < self.convergence.grad_tol {
                if self.verbose {
                    println!("\n✓ Converged: Gradient norm below tolerance");
                }
                return Ok(OptimizationResult {
                    point: p,
                    value: f_val,
                    grad_norm,
                    iterations: iter,
                    converged: true,
                    message: "Gradient norm below tolerance".to_string(),
                });
            }

            // Solve Gauss-Newton system: (J^T J) δ = -J^T r
            let delta_ambient = match jtj.solve(&(-&jtr)) {
                Ok(d) => d,
                Err(_) => {
                    return Ok(OptimizationResult {
                        point: p,
                        value: f_val,
                        grad_norm,
                        iterations: iter,
                        converged: false,
                        message: "Singular Hessian (J^T J not invertible)".to_string(),
                    });
                }
            };

            // Project onto tangent space
            let delta_tangent = manifold.project_to_ambient_tangent(&p, &delta_ambient)?;

            // Take step
            let p_new = self.retract(manifold, &p, &delta_tangent)?;

            // Evaluate at new point
            let r_new = residual_fn.residual(manifold, &p_new)?;
            let f_new = 0.5 * r_new.dot(&r_new);

            // Check function value convergence
            let f_diff = (f_val - f_new).abs();
            if f_diff < self.convergence.f_tol {
                if self.verbose {
                    println!("\n✓ Converged: Function value change below tolerance");
                }
                return Ok(OptimizationResult {
                    point: p_new,
                    value: f_new,
                    grad_norm,
                    iterations: iter + 1,
                    converged: true,
                    message: "Function value change below tolerance".to_string(),
                });
            }

            // Update for next iteration
            p = p_new;
            r = r_new;
            f_val = f_new;
        }

        // Max iterations reached
        let j = residual_fn.jacobian(manifold, &p)?;
        let jtr = j.t().dot(&r);
        let grad_norm = jtr.norm_l2();

        if self.verbose {
            println!("\n⚠ Maximum iterations reached");
        }

        Ok(OptimizationResult {
            point: p,
            value: f_val,
            grad_norm,
            iterations: self.convergence.max_iterations,
            converged: false,
            message: "Maximum iterations reached".to_string(),
        })
    }

    /// Perform line search to find step size
    fn perform_line_search<M, F>(
        &self,
        manifold: &M,
        objective: &F,
        p: &M::Point,
        direction_ambient: &Array1<f64>,
        grad_ambient: &Array1<f64>,
        f_val: f64,
    ) -> Result<f64>
    where
        M: EmbeddedManifold<Scalar = f64>,
        F: ObjectiveFunction<M>,
        M::Point: Clone,
    {
        match self.line_search {
            LineSearch::None => Ok(self.step_size),
            LineSearch::Backtracking {
                initial_alpha,
                rho,
                c1,
            } => {
                let mut alpha = initial_alpha;

                // Directional derivative: ⟨grad f, direction⟩ (Euclidean inner product)
                let directional_derivative = grad_ambient.dot(direction_ambient);

                // Armijo condition: f(retract(p, alpha*d)) <= f(p) + c1*alpha*⟨grad f, d⟩
                let max_backtracks = 30;
                for _ in 0..max_backtracks {
                    let step_ambient = direction_ambient * alpha;
                    let p_trial = self.retract(manifold, p, &step_ambient)?;
                    let f_trial = objective.eval(manifold, &p_trial)?;

                    let armijo_rhs = f_val + c1 * alpha * directional_derivative;

                    if f_trial <= armijo_rhs {
                        return Ok(alpha);
                    }

                    alpha *= rho;

                    // Prevent alpha from becoming too small
                    if alpha < 1e-16 {
                        break;
                    }
                }

                // If backtracking fails, return current alpha
                Ok(alpha.max(1e-16))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    // Simple test objective for Euclidean space (treated as flat manifold)
    struct QuadraticObjective {
        target: Array2<f64>,
    }

    impl QuadraticObjective {
        fn new(target: Array2<f64>) -> Self {
            Self { target }
        }
    }

    // We'd need a concrete manifold implementation to test this properly
    // For now, these are just structure tests
}

pub mod algorithms;
pub mod core;
pub mod manifolds;

// Flat re-exports for convenience
pub use core::{EmbeddedManifold, Error, Manifold, Result};

// Re-export manifold types
pub use manifolds::{Euclidean, Sphere, Stiefel, SPD};

// Re-export optimization types
pub use algorithms::optimization::{
    Convergence, LineSearch, ObjectiveFunction, OptimizationMethod, OptimizationResult,
    ResidualFunction, RetractionMethod, RiemannianGradientDescent,
};

/// Convenience re-exports for common use cases
pub mod prelude {
    pub use crate::algorithms::optimization::{
        ObjectiveFunction, ResidualFunction, RiemannianGradientDescent,
    };
    pub use crate::core::{EmbeddedManifold, Error, Manifold, Result};
    pub use crate::manifolds::{Euclidean, Sphere, Stiefel, SPD};
}

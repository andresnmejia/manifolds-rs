//! # Manifolds Core - Riemannian Geometry Library
//!
//! A lightweight Rust library for Riemannian manifold computations.
//!
//! ## Features
//!
//! - **Core manifolds**: Euclidean, SPD matrices, product manifolds
//! - **Statistical manifolds**: Fisher Information Metric support
//! - **Clean abstractions**: Trait-based design for generic algorithms
//! - **Performance**: Unchecked variants for hot paths
//! - **Safety**: Checked variants with validation by default
//!
//! ## Design Philosophy
//!
//! This library provides core Riemannian geometry tools:
//! 1. `Manifold` trait (exp, log, metric)
//! 2. `StatisticalManifold` trait (Fisher Information)
//! 3. Standard manifold implementations
//! 4. Both checked (safe) and unchecked (fast) operations
//!
//! All manifolds are assumed smooth (C^∞).
//!
//! ## Quick Start
//!
//! ```rust
//! use manifolds_core::prelude::*;
//! use ndarray::Array2;
//!
//! // Create SPD manifold
//! let spd = SPD::new(2);
//!
//! // Two positive definite matrices
//! let p = Array2::from_shape_vec((2, 2), vec![4.0, 1.0, 1.0, 3.0]).unwrap();
//! let q = Array2::from_shape_vec((2, 2), vec![5.0, 0.5, 0.5, 4.0]).unwrap();
//!
//! // Compute geodesic (safe, with validation)
//! let tangent = spd.log(&p, &q).unwrap();
//! let distance = spd.distance(&p, &q).unwrap();
//!
//! // Or use unchecked for performance (when you know inputs are valid)
//! let tangent_fast = spd.log_unchecked(&p, &q);
//! ```

pub mod core;
pub mod manifolds;
pub mod algorithms;

// Flat re-exports for convenience
pub use core::{EmbeddedManifold, Error, Manifold, ProductManifold, Result};

pub use manifolds::{sphere, Euclidean, Stiefel, SPD};

/// Convenience re-exports for common use cases
pub mod prelude {
    pub use crate::{
        sphere, EmbeddedManifold, Error, Euclidean, Manifold, ProductManifold, Result, SPD,
    };
}

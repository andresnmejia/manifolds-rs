pub mod error;
pub mod traits;
pub mod product;

pub use error::{Error, Result};
pub use traits::{Manifold, StatisticalManifold, EmbeddedManifold};
pub use product::ProductManifold;

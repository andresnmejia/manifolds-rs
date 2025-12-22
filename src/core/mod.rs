pub mod error;
pub mod product;
pub mod traits;

pub use error::{Error, Result};
pub use product::ProductManifold;
pub use traits::{EmbeddedManifold, FisherManifold, Manifold, StatisticalFamily};

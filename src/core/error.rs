use thiserror::Error;

/// Errors that can occur during manifold operations
#[derive(Debug, Error, Clone)]
pub enum Error {
    /// Point is not on the manifold
    #[error("point not on manifold: {0}")]
    NotOnManifold(String),

    /// Dimension mismatch
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// Computation failed (e.g., matrix not positive definite)
    #[error("computation failed: {0}")]
    ComputationFailed(String),

    /// Invalid parameter value
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),

    /// Numerical error from ndarray-linalg
    #[error("linear algebra error: {0}")]
    LinalgError(String),
}

/// Convert ndarray-linalg errors to Error
impl From<ndarray_linalg::error::LinalgError> for Error {
    fn from(err: ndarray_linalg::error::LinalgError) -> Self {
        Error::LinalgError(format!("{:?}", err))
    }
}

/// Result type for manifold operations
pub type Result<T> = std::result::Result<T, Error>;

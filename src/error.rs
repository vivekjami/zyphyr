use thiserror::Error;

#[derive(Error, Debug)]
pub enum ZyphyrError {
    #[error("Invalid vector dimension: expected {expected}, got {got}")]
    InvalidDimension { expected: usize, got: usize },
    #[error("Vector ID not found: {0}")]
    IdNotFound(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Other error: {0}")]
    Other(String),
}
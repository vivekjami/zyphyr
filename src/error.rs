//! Error types for the Zyphyr vector database

use std::fmt;

/// Main error type for Zyphyr operations
#[derive(Debug, Clone, PartialEq)]
pub enum ZyphyrError {
    /// Vector dimension mismatch
    DimensionMismatch {
        expected: usize,
        actual: usize,
    },
    /// Invalid vector ID
    InvalidVectorId(u64),
    /// Vector not found
    VectorNotFound(u64),
    /// Empty vector provided
    EmptyVector,
    /// Memory alignment error
    AlignmentError(String),
    /// IO related errors
    IoError(String),
    /// Invalid distance metric
    InvalidDistanceMetric,
}

impl fmt::Display for ZyphyrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ZyphyrError::DimensionMismatch { expected, actual } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, actual)
            }
            ZyphyrError::InvalidVectorId(id) => write!(f, "Invalid vector ID: {}", id),
            ZyphyrError::VectorNotFound(id) => write!(f, "Vector not found: {}", id),
            ZyphyrError::EmptyVector => write!(f, "Vector cannot be empty"),
            ZyphyrError::AlignmentError(msg) => write!(f, "Memory alignment error: {}", msg),
            ZyphyrError::IoError(msg) => write!(f, "IO error: {}", msg),
            ZyphyrError::InvalidDistanceMetric => write!(f, "Invalid distance metric"),
        }
    }
}

impl std::error::Error for ZyphyrError {}

/// Result type alias for Zyphyr operations
pub type Result<T> = std::result::Result<T, ZyphyrError>;

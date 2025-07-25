//! Zyphyr - High-Performance Vector Database
//! 
//! A Rust-based vector database designed for billion-scale similarity search
//! with sub-100ms latency requirements.

pub mod error;
pub mod vector;
pub mod utils;

// Re-export commonly used types for easier access
pub use vector::{Vector, VectorCollection, DistanceMetric};
pub use error::ZyphyrError;

// Library version and basic info
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_library_loads() {
        // Basic smoke test to ensure the library compiles and loads
        assert_eq!(VERSION, "0.1.0");
    }
}

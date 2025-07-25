//! Distance metrics for vector similarity calculations

use crate::error::{ZyphyrError, Result};

/// Supported distance metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Euclidean (L2) distance - most common for general vectors
    Euclidean,
    /// Cosine similarity - good for normalized semantic vectors  
    Cosine,
    /// Dot product - fast for normalized vectors
    DotProduct,
}

impl DistanceMetric {
    /// Calculate distance between two vectors
    pub fn calculate(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(ZyphyrError::DimensionMismatch {
                expected: a.len(),
                actual: b.len(),
            });
        }

        if a.is_empty() {
            return Err(ZyphyrError::EmptyVector);
        }

        match self {
            DistanceMetric::Euclidean => Ok(euclidean_distance(a, b)),
            DistanceMetric::Cosine => Ok(cosine_distance(a, b)),
            DistanceMetric::DotProduct => Ok(dot_product_distance(a, b)),
        }
    }

    /// Whether this metric requires normalized vectors for optimal performance
    pub fn requires_normalization(&self) -> bool {
        matches!(self, DistanceMetric::Cosine | DistanceMetric::DotProduct)
    }
}

/// Calculate Euclidean (L2) distance between two vectors
/// Formula: sqrt(sum((a[i] - b[i])^2))
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Calculate cosine distance (1 - cosine similarity)
/// Formula: 1 - (dot(a,b) / (norm(a) * norm(b)))
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0; // Maximum distance for zero vectors
    }
    
    1.0 - (dot_product / (norm_a * norm_b))
}

/// Calculate negative dot product (to make it a distance metric)
/// For normalized vectors, this equals cosine distance
fn dot_product_distance(a: &[f32], b: &[f32]) -> f32 {
    -(a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        
        let distance = DistanceMetric::Euclidean.calculate(&a, &b).unwrap();
        let expected = ((3.0_f32).powi(2) * 3.0).sqrt(); // sqrt(9 + 9 + 9) = sqrt(27)
        
        assert!((distance - expected).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_distance() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        
        let distance = DistanceMetric::Cosine.calculate(&a, &b).unwrap();
        assert!((distance - 1.0).abs() < 1e-6); // Orthogonal vectors have cosine distance of 1
    }

    #[test]
    fn test_dimension_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        
        let result = DistanceMetric::Euclidean.calculate(&a, &b);
        assert!(matches!(result, Err(ZyphyrError::DimensionMismatch { .. })));
    }
}

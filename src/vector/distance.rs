use crate::{Vector, ZyphyrError};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistanceMetric {
    Euclidean,
    Cosine,
    DotProduct,
}

impl DistanceMetric {
    pub fn compute(&self, a: &Vector, b: &Vector) -> Result<f32, ZyphyrError> {
        if a.dim() != b.dim() {
            return Err(ZyphyrError::InvalidDimension {
                expected: a.dim(),
                got: b.dim(),
            });
        }
        match self {
            DistanceMetric::Euclidean => Ok(euclidean_distance(a.data(), b.data())),
            DistanceMetric::Cosine => {
                // Calculate cosine similarity directly without modifying original vectors
                let a_data = a.data();
                let b_data = b.data();
                
                let dot = a_data.iter().zip(b_data.iter()).map(|(x, y)| x * y).sum::<f32>();
                let a_mag = a_data.iter().map(|x| x * x).sum::<f32>().sqrt();
                let b_mag = b_data.iter().map(|x| x * x).sum::<f32>().sqrt();
                
                // Check for zero magnitude
                if a_mag == 0.0 || b_mag == 0.0 {
                    Ok(1.0) // Maximum distance for zero vectors
                } else {
                    Ok(1.0 - (dot / (a_mag * b_mag)))
                }
            }
            DistanceMetric::DotProduct => Ok(dot_product(a.data(), b.data())),
        }
    }
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

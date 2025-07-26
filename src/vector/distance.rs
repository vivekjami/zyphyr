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
                let mut a = a.clone();
                let mut b = b.clone();
                a.normalize();
                b.normalize();
                Ok(1.0 - dot_product(a.data(), b.data()))
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

use crate::ZyphyrError;
use std::mem;

#[repr(C, align(16))]
#[derive(Debug, Clone)]
pub struct Vector {
    id: String,            // Unique identifier
    data: Box<[f32]>,     // Aligned vector data
    dim: usize,           // Vector dimension
    is_normalized: bool,  // Flag for cosine similarity optimization
}

impl Vector {
    pub fn new(id: impl Into<String>, data: Vec<f32>) -> Result<Self, ZyphyrError> {
        let dim = data.len();
        if dim == 0 {
            return Err(ZyphyrError::InvalidDimension { expected: 1, got: 0 });
        }
        let data = data.into_boxed_slice();
        Ok(Vector {
            id: id.into(),
            data,
            dim,
            is_normalized: false,
        })
    }

    pub fn from_slice(id: impl Into<String>, data: &[f32]) -> Result<Self, ZyphyrError> {
        let dim = data.len();
        if dim == 0 {
            return Err(ZyphyrError::InvalidDimension { expected: 1, got: 0 });
        }
        let data = data.to_vec().into_boxed_slice();
        Ok(Vector {
            id: id.into(),
            data,
            dim,
            is_normalized: false,
        })
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn normalize(&mut self) {
        if self.is_normalized {
            return;
        }
        let norm = self.data.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            self.data = self.data.iter().map(|x| x / norm).collect::<Vec<f32>>().into_boxed_slice();
            self.is_normalized = true;
        }
    }

    // Ensure memory alignment for SIMD
    pub fn is_aligned(&self) -> bool {
        let ptr = self.data.as_ptr() as usize;
        ptr % 16 == 0
    }
}
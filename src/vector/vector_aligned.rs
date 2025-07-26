use crate::ZyphyrError;
use crate::utils::alignment::{SIMD_ALIGNMENT, is_aligned, pad_dimension, get_simd_width};
use std::mem;
use aligned_vec::AlignedVec;

#[repr(C, align(32))]  // Increased alignment for AVX-512
#[derive(Debug, Clone)]
pub struct Vector {
    id: String,            // Unique identifier
    data: AlignedVec<f32>, // Properly aligned vector data
    dim: usize,            // Original vector dimension
    padded_dim: usize,     // Padded dimension for SIMD operations
    is_normalized: bool,   // Flag for cosine similarity optimization
}

impl Vector {
    pub fn new(id: impl Into<String>, data: Vec<f32>) -> Result<Self, ZyphyrError> {
        let dim = data.len();
        if dim == 0 {
            return Err(ZyphyrError::InvalidDimension { expected: 1, got: 0 });
        }
        
        // Pad to optimize for SIMD operations
        let simd_width = get_simd_width();
        let padded_dim = pad_dimension(dim, simd_width);
        
        // Create a properly aligned vector
        let mut aligned_data = AlignedVec::with_capacity(SIMD_ALIGNMENT, padded_dim);
        aligned_data.extend_from_slice(&data);
        aligned_data.resize(padded_dim, 0.0); // Pad with zeros
        
        Ok(Vector {
            id: id.into(),
            data: aligned_data,
            dim,
            padded_dim,
            is_normalized: false,
        })
    }

    pub fn from_slice(id: impl Into<String>, data: &[f32]) -> Result<Self, ZyphyrError> {
        let dim = data.len();
        if dim == 0 {
            return Err(ZyphyrError::InvalidDimension { expected: 1, got: 0 });
        }
        
        // Pad to optimize for SIMD operations
        let simd_width = get_simd_width();
        let padded_dim = pad_dimension(dim, simd_width);
        
        // Create a properly aligned vector
        let mut aligned_data = AlignedVec::with_capacity(SIMD_ALIGNMENT, padded_dim);
        aligned_data.extend_from_slice(data);
        aligned_data.resize(padded_dim, 0.0); // Pad with zeros
        
        Ok(Vector {
            id: id.into(),
            data: aligned_data,
            dim,
            padded_dim,
            is_normalized: false,
        })
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn data(&self) -> &[f32] {
        // Return only the unpadded portion
        &self.data[..self.dim]
    }
    
    pub fn raw_data(&self) -> &[f32] {
        // Return the full padded data (for internal use)
        &self.data
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
    
    pub fn padded_dim(&self) -> usize {
        self.padded_dim
    }

    pub fn normalize(&mut self) {
        if self.is_normalized {
            return;
        }
        
        // Calculate the magnitude using only the actual dimensions (not padding)
        let magnitude: f32 = self.data[..self.dim]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
            
        // Avoid division by zero
        if magnitude > 0.0 {
            // Normalize only the actual dimensions (not padding)
            for i in 0..self.dim {
                self.data[i] /= magnitude;
            }
        }
        
        self.is_normalized = true;
    }

    // Ensure memory alignment for SIMD
    pub fn is_aligned(&self) -> bool {
        let ptr = self.data.as_ptr() as *const u8;
        is_aligned(ptr, SIMD_ALIGNMENT)
    }
    
    // Add cache-friendly batch methods
    pub fn batch_distance(&self, others: &[&Vector], metric: crate::DistanceMetric) 
        -> Result<Vec<f32>, ZyphyrError> {
        // Implementation for batch distance calculation
        others.iter()
            .map(|other| metric.compute(self, other))
            .collect()
    }

    // Add memory usage tracking
    pub fn memory_usage(&self) -> usize {
        mem::size_of::<Self>() + 
        self.id.capacity() +
        self.padded_dim * mem::size_of::<f32>()
    }
}

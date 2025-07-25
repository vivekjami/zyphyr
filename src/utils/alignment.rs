//! SIMD alignment utilities for optimal performance

use crate::error::{ZyphyrError, Result};
use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;

/// Alignment requirement for SIMD operations (16 bytes for AVX)
pub const SIMD_ALIGNMENT: usize = 16;

/// Check if a pointer is properly aligned for SIMD operations
pub fn is_simd_aligned<T>(ptr: *const T) -> bool {
    (ptr as usize) % SIMD_ALIGNMENT == 0
}

/// Allocate aligned memory for vector data
pub fn alloc_aligned_f32(len: usize) -> Result<NonNull<f32>> {
    if len == 0 {
        return Err(ZyphyrError::EmptyVector);
    }

    let layout = Layout::from_size_align(len * std::mem::size_of::<f32>(), SIMD_ALIGNMENT)
        .map_err(|_| ZyphyrError::AlignmentError("Invalid layout".to_string()))?;

    let ptr = unsafe { alloc(layout) as *mut f32 };
    
    NonNull::new(ptr)
        .ok_or_else(|| ZyphyrError::AlignmentError("Failed to allocate aligned memory".to_string()))
}

/// Deallocate aligned memory
pub unsafe fn dealloc_aligned_f32(ptr: NonNull<f32>, len: usize) {
    let layout = Layout::from_size_align_unchecked(
        len * std::mem::size_of::<f32>(), 
        SIMD_ALIGNMENT
    );
    dealloc(ptr.as_ptr() as *mut u8, layout);
}

/// Pad vector dimensions to SIMD-friendly multiples
pub fn pad_to_simd_width(dim: usize) -> usize {
    let simd_width = SIMD_ALIGNMENT / std::mem::size_of::<f32>(); // 4 f32s per 16 bytes
    ((dim + simd_width - 1) / simd_width) * simd_width
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_alignment_check() {
        let data = vec![1.0f32; 16];
        let ptr = data.as_ptr();
        // Note: Vec doesn't guarantee SIMD alignment, this is just a basic test
        println!("Pointer alignment: {}", ptr as usize % SIMD_ALIGNMENT);
    }

    #[test]
    fn test_dimension_padding() {
        assert_eq!(pad_to_simd_width(3), 4);
        assert_eq!(pad_to_simd_width(4), 4);
        assert_eq!(pad_to_simd_width(5), 8);
        assert_eq!(pad_to_simd_width(16), 16);
    }
}

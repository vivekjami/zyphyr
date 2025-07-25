//! Utility modules for performance optimization

pub mod alignment;

pub use alignment::{SIMD_ALIGNMENT, is_simd_aligned, alloc_aligned_f32, dealloc_aligned_f32, pad_to_simd_width};

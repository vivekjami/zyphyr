use std::alloc::{alloc, dealloc, Layout};
use std::mem;

/// Alignment required for AVX2/AVX-512 operations
pub const SIMD_ALIGNMENT: usize = 32;

/// Check if a pointer is properly aligned for SIMD operations
pub fn is_aligned(ptr: *const u8, align: usize) -> bool {
    (ptr as usize) % align == 0
}

/// Allocate memory with specific alignment for SIMD operations
pub unsafe fn aligned_alloc<T>(len: usize) -> (*mut T, Layout) {
    let size = mem::size_of::<T>() * len;
    let layout = Layout::from_size_align(size, SIMD_ALIGNMENT)
        .expect("Failed to create memory layout");
    unsafe {
        let ptr = alloc(layout) as *mut T;
        (ptr, layout)
    }
}

/// Deallocate memory that was allocated with aligned_alloc
pub unsafe fn aligned_dealloc<T>(ptr: *mut T, layout: Layout) {
    unsafe {
        dealloc(ptr as *mut u8, layout);
    }
}

/// Pad a dimension to the nearest multiple of SIMD width
pub fn pad_dimension(dim: usize, simd_width: usize) -> usize {
    ((dim + simd_width - 1) / simd_width) * simd_width
}

/// Get the optimal SIMD width for the current platform
pub fn get_simd_width() -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        // Check for AVX-512 support
        if std::arch::is_x86_feature_detected!("avx512f") {
            16 // 16 f32 values in 512 bits
        } else if std::arch::is_x86_feature_detected!("avx2") {
            8  // 8 f32 values in 256 bits
        } else if std::arch::is_x86_feature_detected!("sse") {
            4  // 4 f32 values in 128 bits
        } else {
            1  // No SIMD support
        }
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        // ARM NEON support
        if std::arch::is_aarch64_feature_detected!("neon") {
            4  // 4 f32 values in 128 bits
        } else {
            1  // No SIMD support
        }
    }
    
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        1  // Default for other architectures
    }
}
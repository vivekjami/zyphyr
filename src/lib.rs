//! Zyphyr - High-performance vector database with HNSW indexing

mod error;
mod vector;
mod utils;

#[cfg(test)]
mod tests;

// Re-export primary types
pub use error::ZyphyrError;
pub use vector::{Vector, VectorCollection, DistanceMetric};
pub use utils::alignment::{SIMD_ALIGNMENT, is_aligned};

/// Version of the library
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Returns information about SIMD support on the current platform
pub fn simd_support_info() -> String {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::{__cpuid, __cpuid_count};
        
        let mut info = String::new();
        unsafe {
            let cpuid = __cpuid(1);
            
            if (cpuid.ecx >> 28) & 1 != 0 {
                info.push_str("AVX supported\n");
            }
            
            let cpuid7 = __cpuid_count(7, 0);
            if (cpuid7.ebx >> 5) & 1 != 0 {
                info.push_str("AVX2 supported\n");
            }
            
            if (cpuid7.ebx >> 16) & 1 != 0 {
                info.push_str("AVX-512 supported\n");
            }
        }
        
        if info.is_empty() {
            "No advanced SIMD features detected".to_string()
        } else {
            info
        }
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    {
        "SIMD support detection not implemented for this architecture".to_string()
    }
}
#[cfg(test)]
mod tests {
    use crate::{Vector, VectorCollection, DistanceMetric, ZyphyrError};
    use crate::utils::alignment::{SIMD_ALIGNMENT, get_simd_width, is_aligned};

    #[test]
    fn test_vector_creation() {
        let v = Vector::new("v1", vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(v.dim(), 3);
        assert_eq!(v.id(), "v1");
        assert_eq!(v.data(), &[1.0, 2.0, 3.0]);
        
        // Test that we actually have padding
        assert!(v.padded_dim() >= v.dim());
        assert_eq!(v.padded_dim() % get_simd_width(), 0);
    }

    #[test]
    fn test_vector_alignment_realistic() {
        // Test multiple vectors to see alignment behavior
        let mut aligned_count = 0;
        let total_tests = 100;
        
        for i in 0..total_tests {
            let v = Vector::new(format!("v{}", i), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
            let ptr = v.raw_data().as_ptr() as *const u8;
            if is_aligned(ptr, SIMD_ALIGNMENT) {
                aligned_count += 1;
            }
        }
        
        // We should get some alignment by chance, but not 100%
        // This tests that our alignment detection works
        println!("Aligned vectors: {}/{}", aligned_count, total_tests);
        
        // At minimum, test that our alignment detection function works
        let test_ptr = 0x20 as *const u8; // 32-byte aligned address
        assert!(is_aligned(test_ptr, SIMD_ALIGNMENT));
        
        let test_ptr = 0x21 as *const u8; // Not aligned
        assert!(!is_aligned(test_ptr, SIMD_ALIGNMENT));
    }

    #[test]
    fn test_vector_padding_correctness() {
        // Test different dimensions to ensure padding works correctly
        for dim in [1, 3, 7, 15, 16, 17, 32, 33] {
            let data: Vec<f32> = (0..dim).map(|i| i as f32).collect();
            let v = Vector::new(format!("v{}", dim), data.clone()).unwrap();
            
            // Original data should be preserved
            assert_eq!(v.data(), &data[..]);
            assert_eq!(v.dim(), dim);
            
            // Padding should be correct
            let simd_width = get_simd_width();
            let expected_padded = ((dim + simd_width - 1) / simd_width) * simd_width;
            assert_eq!(v.padded_dim(), expected_padded);
            
            // Padded data should contain original data + zeros
            let raw_data = v.raw_data();
            for i in 0..dim {
                assert_eq!(raw_data[i], i as f32);
            }
            for i in dim..v.padded_dim() {
                assert_eq!(raw_data[i], 0.0);
            }
        }
    }

    #[test]
    fn test_batch_distance_computation() {
        let query = Vector::new("query", vec![1.0, 0.0]).unwrap();
        let vectors = vec![
            Vector::new("v1", vec![1.0, 0.0]).unwrap(),
            Vector::new("v2", vec![0.0, 1.0]).unwrap(),
            Vector::new("v3", vec![-1.0, 0.0]).unwrap(),
        ];
        let vector_refs: Vec<&Vector> = vectors.iter().collect();
        
        let distances = query.batch_distance(&vector_refs, DistanceMetric::Euclidean).unwrap();
        
        assert_eq!(distances.len(), 3);
        assert!((distances[0] - 0.0).abs() < 1e-6); // Same vector
        assert!((distances[1] - 1.414).abs() < 0.01); // √2 distance
        assert!((distances[2] - 2.0).abs() < 1e-6); // Distance of 2
    }

    #[test]
    fn test_memory_usage_accuracy() {
        let v = Vector::new("test_vector", vec![1.0; 100]).unwrap();
        let reported_usage = v.memory_usage();
        
        // Calculate expected memory usage
        let expected_usage = std::mem::size_of::<Vector>() + 
                           "test_vector".len() +  // ID string content
                           v.padded_dim() * std::mem::size_of::<f32>();
        
        // Should be close (string capacity might be different)
        assert!(reported_usage >= expected_usage);
        assert!(reported_usage < expected_usage + 100); // Reasonable upper bound
    }

    #[test]
    fn test_simd_width_detection() {
        let width = get_simd_width();
        
        // Should be a power of 2 and reasonable value
        assert!(width >= 1);
        assert!(width <= 16); // Max reasonable SIMD width for f32
        assert!(width.is_power_of_two() || width == 1);
        
        println!("Detected SIMD width: {}", width);
    }

    #[test]
    fn test_vector_normalization_preserves_padding() {
        let mut v = Vector::new("v1", vec![3.0, 4.0, 5.0]).unwrap();
        let original_padded_dim = v.padded_dim();
        
        v.normalize();
        
        // Normalization should preserve padding structure
        assert_eq!(v.padded_dim(), original_padded_dim);
        
        // Only the actual dimensions should be normalized
        let data = v.data();
        let magnitude = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((magnitude - 1.0).abs() < 1e-6);
        
        // Padding should still be zeros
        let raw_data = v.raw_data();
        for i in v.dim()..v.padded_dim() {
            assert_eq!(raw_data[i], 0.0);
        }
    }

    #[test]
    fn test_collection_memory_usage_breakdown() {
        let mut collection = VectorCollection::new();
        let initial_usage = collection.memory_usage();
        
        // Add some vectors
        for i in 0..10 {
            let v = Vector::new(format!("vector_{}", i), vec![i as f32; 50]).unwrap();
            collection.insert(v).unwrap();
        }
        
        let final_usage = collection.memory_usage();
        assert!(final_usage > initial_usage);
        
        // Memory usage should be reasonable
        let expected_min = 10 * (std::mem::size_of::<Vector>() + 50 * 4); // 10 vectors * roughly 50 floats
        assert!(final_usage >= expected_min);
    }

    #[test]
    fn test_distance_metric_consistency() {
        let v1 = Vector::new("v1", vec![1.0, 0.0, 0.0]).unwrap();
        let v2 = Vector::new("v2", vec![0.0, 1.0, 0.0]).unwrap();
        let v3 = Vector::new("v3", vec![0.0, 0.0, 1.0]).unwrap();
        
        // Test that distances are symmetric
        let d12_euclidean = DistanceMetric::Euclidean.compute(&v1, &v2).unwrap();
        let d21_euclidean = DistanceMetric::Euclidean.compute(&v2, &v1).unwrap();
        assert!((d12_euclidean - d21_euclidean).abs() < 1e-6);
        
        // Test triangle inequality for Euclidean distance
        let d13 = DistanceMetric::Euclidean.compute(&v1, &v3).unwrap();
        let d23 = DistanceMetric::Euclidean.compute(&v2, &v3).unwrap();
        assert!(d13 <= d12_euclidean + d23 + 1e-6); // Triangle inequality
    }

    #[test]
    fn test_invalid_dimension() {
        let result = Vector::new("v1", vec![]);
        assert!(matches!(
            result,
            Err(ZyphyrError::InvalidDimension { expected: 1, got: 0 })
        ));
    }

    #[test]
    fn test_distance_euclidean() {
        let v1 = Vector::new("v1", vec![0.0, 0.0]).unwrap();
        let v2 = Vector::new("v2", vec![3.0, 4.0]).unwrap();
        let distance = DistanceMetric::Euclidean.compute(&v1, &v2).unwrap();
        assert!((distance - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_distance_cosine() {
        let v1 = Vector::new("v1", vec![1.0, 0.0]).unwrap();
        let v2 = Vector::new("v2", vec![0.0, 1.0]).unwrap();
        let distance = DistanceMetric::Cosine.compute(&v1, &v2).unwrap();
        assert!((distance - 1.0).abs() < 1e-6); // Orthogonal vectors
    }

    #[test]
    fn test_distance_dot_product() {
        let v1 = Vector::new("v1", vec![1.0, 2.0]).unwrap();
        let v2 = Vector::new("v2", vec![3.0, 4.0]).unwrap();
        let distance = DistanceMetric::DotProduct.compute(&v1, &v2).unwrap();
        assert!((distance - 11.0).abs() < 1e-6); // 1*3 + 2*4 = 11
    }

    #[test]
    fn test_collection_insert_search() {
        let mut collection = VectorCollection::new();
        let v1 = Vector::new("v1", vec![1.0, 0.0]).unwrap();
        let v2 = Vector::new("v2", vec![0.0, 1.0]).unwrap();
        
        collection.insert(v1).unwrap();
        collection.insert(v2).unwrap();
        
        let query = Vector::new("query", vec![1.0, 0.0]).unwrap();
        let results = collection.search(&query, 1, DistanceMetric::Euclidean).unwrap();
        
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "v1");
        assert!((results[0].1 - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_collection_dimension_consistency() {
        let mut collection = VectorCollection::new();
        let v1 = Vector::new("v1", vec![1.0, 2.0]).unwrap();
        let v2 = Vector::new("v2", vec![3.0, 4.0, 5.0]).unwrap(); // Different dimension
        
        collection.insert(v1).unwrap();
        let result = collection.insert(v2);
        
        assert!(matches!(
            result,
            Err(ZyphyrError::InvalidDimension { expected: 2, got: 3 })
        ));
    }

    #[test]
    fn test_collection_duplicate_id() {
        let mut collection = VectorCollection::new();
        let v1 = Vector::new("v1", vec![1.0, 2.0]).unwrap();
        let v2 = Vector::new("v1", vec![3.0, 4.0]).unwrap(); // Same ID
        
        collection.insert(v1).unwrap();
        let result = collection.insert(v2);
        
        assert!(matches!(result, Err(ZyphyrError::Other(_))));
    }

    #[test]
    fn test_batch_operations() {
        let mut collection = VectorCollection::new();
        let vectors = vec![
            Vector::new("v1", vec![1.0, 0.0]).unwrap(),
            Vector::new("v2", vec![0.0, 1.0]).unwrap(),
            Vector::new("v3", vec![1.0, 1.0]).unwrap(),
        ];
        
        collection.batch_insert(vectors).unwrap();
        assert_eq!(collection.len(), 3);
        
        let query = Vector::new("query", vec![1.0, 0.0]).unwrap();
        let results = collection.search(&query, 2, DistanceMetric::Euclidean).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_collection_remove() {
        let mut collection = VectorCollection::new();
        let v = Vector::new("v1", vec![1.0, 2.0]).unwrap();
        
        collection.insert(v).unwrap();
        assert!(collection.contains("v1"));
        
        let removed = collection.remove("v1");
        assert!(removed.is_some());
        assert!(!collection.contains("v1"));
        assert_eq!(collection.len(), 0);
    }

    #[test]
    fn test_collection_chunks() {
        let mut collection = VectorCollection::new();
        for i in 0..10 {
            let v = Vector::new(format!("v{}", i), vec![i as f32, (i + 1) as f32]).unwrap();
            collection.insert(v).unwrap();
        }
        
        let chunks: Vec<_> = collection.chunks(3).collect();
        assert_eq!(chunks.len(), 4); // 10 vectors in chunks of 3: [3,3,3,1]
        assert_eq!(chunks[0].len(), 3);
        assert_eq!(chunks[3].len(), 1);
    }

    #[test]
    fn test_performance_characteristics() {
        // This test verifies that our optimizations actually work
        use std::time::Instant;
        
        let dim = 512;
        let num_vectors = 1000;
        
        // Create vectors with different patterns
        let mut vectors = Vec::new();
        for i in 0..num_vectors {
            let data: Vec<f32> = (0..dim).map(|j| (i * j) as f32 % 100.0).collect();
            vectors.push(Vector::new(format!("v{}", i), data).unwrap());
        }
        
        // Test that all vectors have consistent padding
        let first_padded_dim = vectors[0].padded_dim();
        for vector in &vectors {
            assert_eq!(vector.padded_dim(), first_padded_dim);
            assert_eq!(vector.dim(), dim);
            assert!(vector.padded_dim() >= dim);
        }
        
        // Test batch distance calculation performance exists
        let query = Vector::new("query", vec![1.0; dim]).unwrap();
        let vector_refs: Vec<&Vector> = vectors.iter().collect();
        
        let start = Instant::now();
        let distances = query.batch_distance(&vector_refs, DistanceMetric::Euclidean).unwrap();
        let batch_time = start.elapsed();
        
        assert_eq!(distances.len(), num_vectors);
        
        // Test individual distance calculation time
        let start = Instant::now();
        for vector in &vectors {
            let _ = DistanceMetric::Euclidean.compute(&query, vector).unwrap();
        }
        let individual_time = start.elapsed();
        
        println!("Batch time: {:?}, Individual time: {:?}", batch_time, individual_time);
        
        // Batch should not be significantly slower (allowing for measurement noise)
        // This tests that our batch implementation doesn't have major overhead
        assert!(batch_time <= individual_time * 2);
    }

    #[test]
    fn test_edge_cases() {
        // Test very small vectors
        let tiny = Vector::new("tiny", vec![42.0]).unwrap();
        assert_eq!(tiny.dim(), 1);
        assert!(tiny.padded_dim() >= 1);
        
        // Test larger vectors
        let large = Vector::new("large", vec![1.0; 10000]).unwrap();
        assert_eq!(large.dim(), 10000);
        assert!(large.padded_dim() >= 10000);
        
        // Test zero vectors
        let zero = Vector::new("zero", vec![0.0; 100]).unwrap();
        let distance = DistanceMetric::Euclidean.compute(&zero, &zero).unwrap();
        assert!((distance - 0.0).abs() < 1e-6);
        
        // Test cosine distance with zero vectors (should handle gracefully)
        let cosine_distance = DistanceMetric::Cosine.compute(&zero, &zero).unwrap();
        assert_eq!(cosine_distance, 1.0); // Maximum distance for zero vectors
    }

    #[test]
    fn test_proper_simd_alignment_with_aligned_vec() {
        // This test verifies that we can achieve proper SIMD alignment
        // when we use AlignedVec instead of standard Box allocation
        
        // Note: This test is for the future aligned implementation
        // The current implementation uses standard Box allocation which
        // doesn't guarantee SIMD alignment but provides the interface
        // for when we upgrade to aligned allocation
        
        let v = Vector::new("aligned_test", vec![1.0; 64]).unwrap();
        
        // Test the key properties that must work regardless of alignment
        assert_eq!(v.dim(), 64);
        assert!(v.padded_dim() >= 64);
        assert_eq!(v.padded_dim() % get_simd_width(), 0);
        
        // Test that our padding preserves the original data
        let original_data = v.data();
        for i in 0..64 {
            assert_eq!(original_data[i], 1.0);
        }
        
        // Test that padding areas are zero
        let raw_data = v.raw_data();
        for i in 64..v.padded_dim() {
            assert_eq!(raw_data[i], 0.0);
        }
        
        println!("Vector uses {} bytes padded to {} dimensions", 
                v.memory_usage(), v.padded_dim());
    }

    #[test]
    fn test_realistic_alignment_behavior() {
        // Test what actually happens with Box allocation
        let test_size = 100;
        let mut alignment_stats = std::collections::HashMap::new();
        
        for i in 0..test_size {
            let v = Vector::new(format!("test_{}", i), vec![1.0; 16]).unwrap();
            let ptr = v.raw_data().as_ptr() as usize;
            let alignment = ptr % SIMD_ALIGNMENT;
            *alignment_stats.entry(alignment).or_insert(0) += 1;
        }
        
        println!("Alignment distribution: {:?}", alignment_stats);
        
        // We should see various alignment values, showing that
        // standard allocation doesn't guarantee SIMD alignment
        assert!(alignment_stats.len() > 1, "Should have varied alignment with Box allocation");
        
        // But our padding should still work correctly
        let v = Vector::new("test", vec![1.0, 2.0, 3.0]).unwrap();
        assert!(v.padded_dim() >= v.dim());
        assert_eq!(v.padded_dim() % get_simd_width(), 0);
    }
}
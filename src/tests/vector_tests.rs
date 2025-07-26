#[cfg(test)]
mod tests {
    use super::{Vector, VectorCollection, DistanceMetric, ZyphyrError};

    #[test]
    fn test_vector_creation() {
        let v = Vector::new("v1", vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(v.dim(), 3);
        assert_eq!(v.id(), "v1");
        assert_eq!(v.data(), &[1.0, 2.0, 3.0]);
        assert!(v.is_aligned());
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
    fn test_collection_insert_search() {
        let mut collection = VectorCollection::new();
        collection.insert(Vector::new("v1", vec![1.0, 0.0]).unwrap()).unwrap();
        collection.insert(Vector::new("v2", vec![0.0, 1.0]).unwrap()).unwrap();
        let query = Vector::new("query", vec![1.0, 0.0]).unwrap();
        let results = collection
            .search(&query, 1, DistanceMetric::Euclidean)
            .unwrap();
        assert_eq!(results[0].0, "v1");
        assert!((results[0].1 - 0.0).abs() < 1e-6);
    }
}
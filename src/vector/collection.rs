use crate::{Vector, ZyphyrError, DistanceMetric};
use std::collections::HashMap;

pub struct VectorCollection {
    vectors: Vec<Vector>,
    id_to_index: HashMap<String, usize>,
}

impl VectorCollection {
    pub fn new() -> Self {
        VectorCollection {
            vectors: Vec::new(),
            id_to_index: HashMap::new(),
        }
    }

    pub fn insert(&mut self, vector: Vector) -> Result<(), ZyphyrError> {
        if self.id_to_index.contains_key(vector.id()) {
            return Err(ZyphyrError::Other(format!("Duplicate ID: {}", vector.id())));
        }
        let index = self.vectors.len();
        self.id_to_index.insert(vector.id().to_string(), index);
        self.vectors.push(vector);
        Ok(())
    }

    pub fn get(&self, id: &str) -> Option<&Vector> {
        self.id_to_index.get(id).map(|&index| &self.vectors[index])
    }

    pub fn search(
        &self,
        query: &Vector,
        k: usize,
        metric: DistanceMetric,
    ) -> Result<Vec<(String, f32)>, ZyphyrError> {
        let mut results: Vec<(String, f32)> = self
            .vectors
            .iter()
            .map(|v| {
                let distance = metric.compute(query, v)?;
                Ok((v.id().to_string(), distance))
            })
            .collect::<Result<Vec<_>, ZyphyrError>>()?;
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results.into_iter().take(k).collect())
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }
}
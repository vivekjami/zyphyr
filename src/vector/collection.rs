use crate::{Vector, ZyphyrError, DistanceMetric};
use std::collections::HashMap;
use std::mem;

pub struct VectorCollection {
    vectors: Vec<Vector>,
    id_to_index: HashMap<String, usize>,
    dimensions: Option<usize>,  // Track consistent dimensions if applicable
}

impl VectorCollection {
    pub fn new() -> Self {
        VectorCollection {
            vectors: Vec::new(),
            id_to_index: HashMap::new(),
            dimensions: None,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        VectorCollection {
            vectors: Vec::with_capacity(capacity),
            id_to_index: HashMap::with_capacity(capacity),
            dimensions: None,
        }
    }

    pub fn insert(&mut self, vector: Vector) -> Result<(), ZyphyrError> {
        // Check for consistent dimensions
        if let Some(dims) = self.dimensions {
            if vector.dim() != dims {
                return Err(ZyphyrError::InvalidDimension { 
                    expected: dims, 
                    got: vector.dim() 
                });
            }
        } else if !self.is_empty() {
            self.dimensions = Some(vector.dim());
        } else {
            self.dimensions = Some(vector.dim());
        }

        if self.id_to_index.contains_key(vector.id()) {
            return Err(ZyphyrError::Other(format!("Duplicate ID: {}", vector.id())));
        }
        
        let index = self.vectors.len();
        self.id_to_index.insert(vector.id().to_string(), index);
        self.vectors.push(vector);
        Ok(())
    }

    // Add batch insertion for efficiency
    pub fn batch_insert(&mut self, vectors: Vec<Vector>) -> Result<(), ZyphyrError> {
        // Pre-allocate capacity
        self.vectors.reserve(vectors.len());
        self.id_to_index.reserve(vectors.len());
        
        for vector in vectors {
            self.insert(vector)?;
        }
        Ok(())
    }

    // Add chunk-based iteration for parallel processing
    pub fn chunks(&self, chunk_size: usize) -> impl Iterator<Item = &[Vector]> {
        self.vectors.chunks(chunk_size)
    }

    // Add memory usage reporting
    pub fn memory_usage(&self) -> usize {
        let vectors_memory: usize = self.vectors.iter()
            .map(|v| v.memory_usage())
            .sum();
            
        let hashmap_memory = self.id_to_index.len() * 
            (mem::size_of::<String>() + mem::size_of::<usize>());
            
        vectors_memory + hashmap_memory + mem::size_of::<Self>()
    }

    pub fn get(&self, id: &str) -> Option<&Vector> {
        self.id_to_index.get(id).map(|&index| &self.vectors[index])
    }

    pub fn get_mut(&mut self, id: &str) -> Option<&mut Vector> {
        let index = *self.id_to_index.get(id)?;
        Some(&mut self.vectors[index])
    }

    pub fn contains(&self, id: &str) -> bool {
        self.id_to_index.contains_key(id)
    }

    pub fn remove(&mut self, id: &str) -> Option<Vector> {
        let index = *self.id_to_index.get(id)?;
        
        // Remove from mapping
        self.id_to_index.remove(id);
        
        // This is inefficient for large collections as it shifts elements
        // Can be optimized by swapping with the last element and updating index
        if index < self.vectors.len() - 1 {
            // If not the last element, swap with last and update index
            let last_index = self.vectors.len() - 1;
            self.vectors.swap(index, last_index);
            
            // Update the mapping for the swapped element
            let swapped_id = self.vectors[index].id().to_string();
            self.id_to_index.insert(swapped_id, index);
        }
        
        // Remove and return
        Some(self.vectors.pop()?)
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
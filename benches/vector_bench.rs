use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use std::hint::black_box;
use zyphyr::{Vector, VectorCollection, DistanceMetric};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

fn generate_random_vector(id: &str, dim: usize, rng: &mut StdRng) -> Vector {
    let data: Vec<f32> = (0..dim)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    Vector::new(id, data).unwrap()
}

fn bench_distance_calculation(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);
    
    let mut group = c.benchmark_group("distance_metrics");
    for dim in [128, 512, 1024].iter() {
        let v1 = generate_random_vector("v1", *dim, &mut rng);
        let v2 = generate_random_vector("v2", *dim, &mut rng);
        
        group.bench_with_input(BenchmarkId::new("euclidean", dim), dim, |b, _| {
            b.iter(|| {
                black_box(
                    DistanceMetric::Euclidean.compute(&v1, &v2).unwrap()
                )
            });
        });
        
        group.bench_with_input(BenchmarkId::new("cosine", dim), dim, |b, _| {
            b.iter(|| {
                black_box(
                    DistanceMetric::Cosine.compute(&v1, &v2).unwrap()
                )
            });
        });
        
        group.bench_with_input(BenchmarkId::new("dot_product", dim), dim, |b, _| {
            b.iter(|| {
                black_box(
                    DistanceMetric::DotProduct.compute(&v1, &v2).unwrap()
                )
            });
        });
    }
    group.finish();
}

fn bench_vector_operations(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);
    let mut group = c.benchmark_group("vector_operations");
    
    for dim in [128, 512, 1024].iter() {
        // Benchmark vector creation
        group.bench_with_input(BenchmarkId::new("creation", dim), dim, |b, &dim| {
            b.iter(|| {
                black_box(
                    generate_random_vector("bench", dim, &mut rng)
                )
            });
        });
        
        // Benchmark normalization
        let v = generate_random_vector("norm_bench", *dim, &mut rng);
        group.bench_with_input(BenchmarkId::new("normalize", dim), dim, |b, _| {
            b.iter(|| {
                let mut v_clone = v.clone();
                black_box(v_clone.normalize())
            });
        });
        
        // Benchmark memory usage calculation
        let v = generate_random_vector("mem_bench", *dim, &mut rng);
        group.bench_with_input(BenchmarkId::new("memory_usage", dim), dim, |b, _| {
            b.iter(|| {
                black_box(v.memory_usage())
            });
        });
    }
    group.finish();
}

fn bench_collection_operations(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);
    let dim = 128;
    
    let mut group = c.benchmark_group("collection_operations");
    
    // Benchmark insertion
    group.bench_function("insert_1000_vectors", |b| {
        b.iter_batched(
            || VectorCollection::new(),
            |mut collection| {
                for i in 0..1000 {
                    let v = generate_random_vector(&format!("v{}", i), dim, &mut rng);
                    collection.insert(v).unwrap();
                }
                black_box(collection)
            },
            criterion::BatchSize::SmallInput,
        );
    });
    
    // Benchmark batch insertion
    group.bench_function("batch_insert_1000_vectors", |b| {
        b.iter_batched(
            || {
                let vectors: Vec<Vector> = (0..1000)
                    .map(|i| generate_random_vector(&format!("v{}", i), dim, &mut rng))
                    .collect();
                (VectorCollection::new(), vectors)
            },
            |(mut collection, vectors)| {
                collection.batch_insert(vectors).unwrap();
                black_box(collection)
            },
            criterion::BatchSize::SmallInput,
        );
    });
    
    // Create a collection with vectors for search benchmarks
    let mut collection = VectorCollection::new();
    for i in 0..1000 {
        let v = generate_random_vector(&format!("v{}", i), dim, &mut rng);
        collection.insert(v).unwrap();
    }
    
    // Benchmark search operation
    let query = generate_random_vector("query", dim, &mut rng);
    group.bench_function("search_1000_vectors", |b| {
        b.iter(|| {
            black_box(
                collection.search(&query, 10, DistanceMetric::Euclidean).unwrap()
            )
        });
    });
    
    // Benchmark different search sizes
    for k in [1, 5, 10, 50].iter() {
        group.bench_with_input(BenchmarkId::new("search_k", k), k, |b, &k| {
            b.iter(|| {
                black_box(
                    collection.search(&query, k, DistanceMetric::Euclidean).unwrap()
                )
            });
        });
    }
    
    // Benchmark memory usage calculation
    group.bench_function("memory_usage_1000_vectors", |b| {
        b.iter(|| {
            black_box(collection.memory_usage())
        });
    });
    
    group.finish();
}

fn bench_parallel_operations(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);
    let dim = 128;
    
    let mut group = c.benchmark_group("parallel_operations");
    
    // Create test data
    let query = generate_random_vector("query", dim, &mut rng);
    let vectors: Vec<Vector> = (0..1000)
        .map(|i| generate_random_vector(&format!("v{}", i), dim, &mut rng))
        .collect();
    let vector_refs: Vec<&Vector> = vectors.iter().collect();
    
    // Benchmark batch distance calculation
    group.bench_function("batch_distance_1000_vectors", |b| {
        b.iter(|| {
            black_box(
                query.batch_distance(&vector_refs, DistanceMetric::Euclidean).unwrap()
            )
        });
    });
    
    // Benchmark chunked iteration
    let mut collection = VectorCollection::new();
    for vector in vectors {
        collection.insert(vector).unwrap();
    }
    
    group.bench_function("chunks_iteration", |b| {
        b.iter(|| {
            let chunks: Vec<_> = collection.chunks(100).collect();
            black_box(chunks)
        });
    });
    
    group.finish();
}

criterion_group!(
    benches, 
    bench_distance_calculation, 
    bench_vector_operations,
    bench_collection_operations,
    bench_parallel_operations
);
criterion_main!(benches);
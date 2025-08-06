use rustrag::{
    core::{
        EmbeddingCache, CacheConfig, CachedEmbeddingService, EmbeddingGenerator,
    },
    utils::{Error, Result},
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokio::time::sleep;

/// Mock embedding generator for testing
#[derive(Clone)]
struct MockEmbeddingGenerator {
    latency_ms: u64,
    dimension: usize,
}

impl MockEmbeddingGenerator {
    fn new(latency_ms: u64, dimension: usize) -> Self {
        Self { latency_ms, dimension }
    }

    fn generate_deterministic_embedding(&self, text: &str) -> Vec<f32> {
        // Create deterministic embeddings based on text content
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let seed = hasher.finish();
        
        (0..self.dimension)
            .map(|i| {
                let value = ((seed.wrapping_add(i as u64) % 1000) as f32 - 500.0) / 500.0;
                value
            })
            .collect()
    }
}

impl EmbeddingGenerator for MockEmbeddingGenerator {
    async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>> {
        // Simulate API latency
        if self.latency_ms > 0 {
            sleep(Duration::from_millis(self.latency_ms)).await;
        }

        Ok(self.generate_deterministic_embedding(text))
    }

    async fn generate_embeddings(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        // Simulate batch processing with latency
        if self.latency_ms > 0 {
            sleep(Duration::from_millis(self.latency_ms * texts.len() as u64 / 2)).await;
        }

        Ok(texts.iter()
            .map(|text| self.generate_deterministic_embedding(text))
            .collect())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("⚡ RustRAG Embedding Cache & Optimization Demo");
    println!("============================================\n");

    // Demo 1: Basic caching operations
    demo_basic_caching().await?;

    // Demo 2: Performance comparison (cached vs uncached)
    demo_performance_comparison().await?;

    // Demo 3: Cache configuration and optimization
    demo_cache_configuration().await?;

    // Demo 4: Persistent cache functionality
    demo_persistent_cache().await?;

    // Demo 5: Cache statistics and monitoring
    demo_cache_statistics().await?;

    // Demo 6: Advanced cache features
    demo_advanced_features().await?;

    // Demo 7: Batch processing optimization
    demo_batch_optimization().await?;

    Ok(())
}

async fn demo_basic_caching() -> Result<()> {
    println!("📦 Demo 1: Basic Caching Operations");
    println!("----------------------------------");

    // Create cache with basic configuration
    let cache = EmbeddingCache::new()?;
    
    let test_texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Rust is a systems programming language",
    ];

    println!("🔸 Testing basic cache operations:");
    
    for (i, text) in test_texts.iter().enumerate() {
        println!("\n  Text {}: \"{}\"", i + 1, text);
        
        // Test cache miss (first time)
        let start = Instant::now();
        let result = cache.get(text, "mock-model", "v1.0").await;
        let miss_time = start.elapsed();
        
        println!("    Cache miss: {:?} ({}μs)", result.is_some(), miss_time.as_micros());
        
        // Add to cache
        let embedding = vec![0.1 * (i as f32 + 1.0); 384];
        cache.put(text, "mock-model", "v1.0", embedding.clone()).await?;
        
        // Test cache hit (second time)
        let start = Instant::now();
        let result = cache.get(text, "mock-model", "v1.0").await;
        let hit_time = start.elapsed();
        
        println!("    Cache hit: {:?} ({}μs)", result.is_some(), hit_time.as_micros());
        
        if let Some(cached_embedding) = result {
            println!("    Embedding dimension: {}", cached_embedding.len());
            println!("    First few values: {:?}", &cached_embedding[..3]);
        }
    }

    // Test cache removal
    cache.remove(&test_texts[0], "mock-model", "v1.0").await?;
    let result = cache.get(&test_texts[0], "mock-model", "v1.0").await;
    println!("\n  After removal - Cache miss: {}", result.is_none());

    Ok(())
}

async fn demo_performance_comparison() -> Result<()> {
    println!("\n\n🚀 Demo 2: Performance Comparison (Cached vs Uncached)");
    println!("-------------------------------------------------------");

    // Create mock embedding generator with artificial latency
    let mock_generator = MockEmbeddingGenerator::new(100, 384); // 100ms latency
    let cache = EmbeddingCache::new()?;
    let cached_service = CachedEmbeddingService::new(
        mock_generator.clone(),
        cache,
        "mock-model".to_string(),
        "v1.0".to_string(),
    );

    let test_texts = [
        "Natural language processing with transformers",
        "Deep learning architectures and neural networks",
        "Vector databases and semantic search",
        "Retrieval augmented generation systems",
        "Large language models and embeddings",
    ];

    println!("🔸 First run (uncached) - Expected ~100ms per embedding:");
    let start_uncached = Instant::now();
    
    for (i, text) in test_texts.iter().enumerate() {
        let start = Instant::now();
        let _embedding = cached_service.generate_embedding_cached(text).await?;
        let elapsed = start.elapsed();
        println!("  Text {}: {}ms", i + 1, elapsed.as_millis());
    }
    
    let total_uncached = start_uncached.elapsed();
    println!("  Total uncached time: {}ms", total_uncached.as_millis());

    println!("\n🔸 Second run (cached) - Expected <1ms per embedding:");
    let start_cached = Instant::now();
    
    for (i, text) in test_texts.iter().enumerate() {
        let start = Instant::now();
        let _embedding = cached_service.generate_embedding_cached(text).await?;
        let elapsed = start.elapsed();
        println!("  Text {}: {}μs", i + 1, elapsed.as_micros());
    }
    
    let total_cached = start_cached.elapsed();
    println!("  Total cached time: {}ms", total_cached.as_millis());

    let speedup = total_uncached.as_millis() as f64 / total_cached.as_millis() as f64;
    println!("  🎯 Speedup: {:.1}x faster with caching!", speedup);

    Ok(())
}

async fn demo_cache_configuration() -> Result<()> {
    println!("\n\n⚙️ Demo 3: Cache Configuration and Optimization");
    println!("-----------------------------------------------");

    // Test different cache configurations
    let configs = [
        ("Small Cache", CacheConfig {
            max_memory_entries: 100,
            max_memory_bytes: 1024 * 1024, // 1MB
            ttl_seconds: 60,
            enable_persistent: false,
            ..Default::default()
        }),
        ("Large Cache", CacheConfig {
            max_memory_entries: 10000,
            max_memory_bytes: 100 * 1024 * 1024, // 100MB
            ttl_seconds: 24 * 60 * 60, // 24 hours
            enable_persistent: false,
            ..Default::default()
        }),
        ("Fast TTL Cache", CacheConfig {
            max_memory_entries: 1000,
            max_memory_bytes: 10 * 1024 * 1024, // 10MB
            ttl_seconds: 2, // 2 seconds
            enable_persistent: false,
            cleanup_interval_seconds: 1, // Cleanup every second
            ..Default::default()
        }),
    ];

    for (name, config) in configs {
        println!("\n🔸 Testing {}", name);
        println!("  Max entries: {}", config.max_memory_entries);
        println!("  Max memory: {} bytes", config.max_memory_bytes);
        println!("  TTL: {} seconds", config.ttl_seconds);

        let cache = EmbeddingCache::with_config(config.clone())?;
        
        // Add some test data
        for i in 0..5 {
            let text = format!("Test embedding number {}", i);
            let embedding = vec![i as f32; 384];
            cache.put(&text, "model", "v1", embedding).await?;
        }

        let stats = cache.get_stats();
        println!("  Memory entries: {}", stats.memory_entries);
        println!("  Memory usage: {} bytes", stats.memory_bytes);

        // Test TTL if it's the fast TTL cache
        if config.ttl_seconds == 2 {
            println!("  Waiting for TTL expiration...");
            sleep(Duration::from_secs(3)).await;
            cache.optimize().await?;
            
            let stats_after = cache.get_stats();
            println!("  Entries after TTL: {}", stats_after.memory_entries);
        }
    }

    Ok(())
}

async fn demo_persistent_cache() -> Result<()> {
    println!("\n\n💾 Demo 4: Persistent Cache Functionality");
    println!("-----------------------------------------");

    // Create temporary directory for cache
    let temp_dir = TempDir::new().map_err(|e| Error::io(format!("Failed to create temp dir: {}", e)))?;
    let cache_dir = temp_dir.path().join("embedding_cache");

    let config = CacheConfig {
        enable_persistent: true,
        cache_dir: cache_dir.clone(),
        enable_compression: true,
        max_memory_entries: 2, // Small memory cache to force persistence
        ..Default::default()
    };

    println!("🔸 Cache directory: {}", cache_dir.display());

    // First cache instance
    {
        let cache1 = EmbeddingCache::with_config(config.clone())?;
        
        let test_data = [
            ("persistent_test_1", vec![0.1, 0.2, 0.3, 0.4]),
            ("persistent_test_2", vec![0.5, 0.6, 0.7, 0.8]),
            ("persistent_test_3", vec![0.9, 1.0, 1.1, 1.2]),
        ];

        for (text, embedding) in &test_data {
            cache1.put(text, "persistent-model", "v1.0", embedding.clone()).await?;
            println!("  Stored: {} -> {} dimensions", text, embedding.len());
        }

        let stats = cache1.get_stats();
        println!("  Memory entries: {}", stats.memory_entries);
        
        // Check files were created
        if cache_dir.exists() {
            let entries = std::fs::read_dir(&cache_dir)
                .map_err(|e| Error::io(format!("Failed to read cache dir: {}", e)))?;
            let file_count = entries.count();
            println!("  Persistent files created: {}", file_count);
        }
    } // cache1 goes out of scope

    println!("\n🔸 Creating new cache instance (simulating restart):");
    
    // Second cache instance (simulating application restart)
    {
        let cache2 = EmbeddingCache::with_config(config)?;
        
        // Try to retrieve from persistent cache
        let result1 = cache2.get("persistent_test_1", "persistent-model", "v1.0").await;
        let result2 = cache2.get("persistent_test_2", "persistent-model", "v1.0").await;
        let result3 = cache2.get("persistent_test_3", "persistent-model", "v1.0").await;

        println!("  Retrieved persistent_test_1: {}", result1.is_some());
        println!("  Retrieved persistent_test_2: {}", result2.is_some());
        println!("  Retrieved persistent_test_3: {}", result3.is_some());

        if let Some(embedding) = result1 {
            println!("  First embedding values: {:?}", embedding);
        }

        let stats = cache2.get_stats();
        println!("  Cache hits after restart: {}", stats.hit_count);
    }

    Ok(())
}

async fn demo_cache_statistics() -> Result<()> {
    println!("\n\n📊 Demo 5: Cache Statistics and Monitoring");
    println!("------------------------------------------");

    let cache = EmbeddingCache::new()?;
    
    // Generate some cache activity
    let texts = vec![
        "Statistics example text one",
        "Statistics example text two",
        "Statistics example text three",
        "Statistics example text four",
        "Statistics example text five",
    ];

    println!("🔸 Generating cache activity...");
    
    // Add items to cache
    for (i, text) in texts.iter().enumerate() {
        let embedding = vec![i as f32; 384];
        cache.put(text, "stats-model", "v1.0", embedding).await?;
    }

    // Generate hits and misses
    for text in &texts {
        cache.get(text, "stats-model", "v1.0").await; // Hit
    }
    
    for i in 0..3 {
        cache.get(&format!("non-existent-text-{}", i), "stats-model", "v1.0").await; // Miss
    }

    // Get and display statistics
    let stats = cache.get_stats();
    
    println!("\n📈 Cache Statistics:");
    println!("  Memory entries: {}", stats.memory_entries);
    println!("  Memory usage: {} bytes ({:.2} KB)", stats.memory_bytes, stats.memory_bytes as f64 / 1024.0);
    println!("  Hit count: {}", stats.hit_count);
    println!("  Miss count: {}", stats.miss_count);
    println!("  Hit rate: {:.1}%", stats.hit_rate() * 100.0);
    println!("  Eviction count: {}", stats.eviction_count);
    println!("  Error count: {}", stats.error_count);
    println!("  Average access time: {:.2}ms", stats.average_access_time_ms);
    
    if let Some(last_cleanup) = stats.last_cleanup {
        println!("  Last cleanup: {} seconds ago", 
                 last_cleanup.elapsed().as_secs());
    }

    Ok(())
}

async fn demo_advanced_features() -> Result<()> {
    println!("\n\n🔬 Demo 6: Advanced Cache Features");
    println!("----------------------------------");

    let config = CacheConfig {
        max_memory_entries: 3, // Small cache to test eviction
        ..Default::default()
    };

    let cache = EmbeddingCache::with_config(config)?;

    println!("🔸 Testing LRU eviction (max 3 entries):");
    
    // Add items that will trigger eviction
    let items = [
        ("first", vec![1.0; 384]),
        ("second", vec![2.0; 384]),  
        ("third", vec![3.0; 384]),
        ("fourth", vec![4.0; 384]), // This should evict "first"
        ("fifth", vec![5.0; 384]),  // This should evict "second"
    ];

    for (name, embedding) in &items {
        cache.put(name, "eviction-test", "v1", embedding.clone()).await?;
        let stats = cache.get_stats();
        println!("  Added '{}' - Cache entries: {}, Evictions: {}", 
                 name, stats.memory_entries, stats.eviction_count);
    }

    println!("\n🔸 Testing cache access patterns:");
    
    // Test which items are still in cache
    for (name, _) in &items {
        let result = cache.get(name, "eviction-test", "v1").await;
        println!("  '{}' in cache: {}", name, result.is_some());
    }

    println!("\n🔸 Testing model versioning:");
    
    // Test different model versions
    let text = "Version test text";
    cache.put(text, "model", "v1.0", vec![1.0; 384]).await?;
    cache.put(text, "model", "v2.0", vec![2.0; 384]).await?;

    let v1_result = cache.get(text, "model", "v1.0").await;
    let v2_result = cache.get(text, "model", "v2.0").await;

    println!("  Same text, v1.0 cached: {}", v1_result.is_some());
    println!("  Same text, v2.0 cached: {}", v2_result.is_some());
    
    if let (Some(v1), Some(v2)) = (v1_result, v2_result) {
        println!("  v1.0 first value: {}", v1[0]);
        println!("  v2.0 first value: {}", v2[0]);
    }

    Ok(())
}

async fn demo_batch_optimization() -> Result<()> {
    println!("\n\n⚡ Demo 7: Batch Processing Optimization");
    println!("--------------------------------------");

    let mock_generator = MockEmbeddingGenerator::new(50, 384); // 50ms per embedding
    let cache = EmbeddingCache::new()?;
    let cached_service = CachedEmbeddingService::new(
        mock_generator,
        cache,
        "batch-model".to_string(),
        "v1.0".to_string(),
    );

    let batch_texts = vec![
        "Batch processing text number one".to_string(),
        "Batch processing text number two".to_string(),
        "Batch processing text number three".to_string(),
        "Batch processing text number four".to_string(),
        "Batch processing text number five".to_string(),
    ];

    println!("🔸 First batch run (all uncached):");
    let start = Instant::now();
    let results1 = cached_service.generate_embeddings_cached(&batch_texts).await?;
    let duration1 = start.elapsed();
    
    println!("  Generated {} embeddings in {}ms", results1.len(), duration1.as_millis());
    println!("  Average per embedding: {:.1}ms", duration1.as_millis() as f64 / results1.len() as f64);

    // Add a few more texts (some cached, some not)
    let mixed_batch = vec![
        batch_texts[0].clone(), // Cached
        batch_texts[1].clone(), // Cached
        "New uncached text one".to_string(), // New
        batch_texts[2].clone(), // Cached
        "New uncached text two".to_string(), // New
    ];

    println!("\n🔸 Mixed batch run (60% cached, 40% new):");
    let start = Instant::now();
    let results2 = cached_service.generate_embeddings_cached(&mixed_batch).await?;
    let duration2 = start.elapsed();
    
    println!("  Generated {} embeddings in {}ms", results2.len(), duration2.as_millis());
    println!("  Average per embedding: {:.1}ms", duration2.as_millis() as f64 / results2.len() as f64);
    
    let efficiency = (1.0 - (duration2.as_millis() as f64 / duration1.as_millis() as f64)) * 100.0;
    println!("  Efficiency improvement: {:.1}%", efficiency);

    println!("\n🔸 Final cache statistics:");
    let final_stats = cached_service.get_cache_stats();
    println!("  Total cache entries: {}", final_stats.memory_entries);
    println!("  Total hits: {}", final_stats.hit_count);
    println!("  Total misses: {}", final_stats.miss_count);
    println!("  Final hit rate: {:.1}%", final_stats.hit_rate() * 100.0);

    Ok(())
}

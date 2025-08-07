use rustrag::performance::{SimpleCacheService, SimpleCacheConfig};
use tokio;

#[tokio::main]
async fn main() {
    println!("Testing Simple Cache Implementation...");
    
    // Create a simple cache configuration
    let config = SimpleCacheConfig {
        enabled: true,
        default_ttl: 60, // 60 seconds
        max_entries: 100,
    };
    
    // Create the cache service
    let cache = SimpleCacheService::new(config);
    
    // Test basic set/get operations
    println!("Testing basic set/get operations...");
    
    // Set a value
    cache.set("test_key", &"Hello, World!".to_string(), None).await.unwrap();
    
    // Get the value
    let retrieved: Option<String> = cache.get("test_key").await.unwrap();
    match retrieved {
        Some(value) => println!("Retrieved value: {}", value),
        None => println!("Value not found!"),
    }
    
    // Test cache miss
    let missing: Option<String> = cache.get("nonexistent_key").await.unwrap();
    match missing {
        Some(value) => println!("Unexpected value: {}", value),
        None => println!("Cache miss works correctly: key not found"),
    }
    
    // Test cache statistics
    let stats = cache.stats().await;
    println!("Cache statistics:");
    println!("  - Hits: {}", stats.hits);
    println!("  - Misses: {}", stats.misses);
    println!("  - Entries: {}", stats.entries);
    println!("  - Hit rate: {:.2}%", stats.hit_rate() * 100.0);
    
    // Test cache deletion
    cache.delete("test_key").await.unwrap();
    let deleted: Option<String> = cache.get("test_key").await.unwrap();
    match deleted {
        Some(value) => println!("Delete failed: value still exists: {}", value),
        None => println!("Delete successful: key removed"),
    }
    
    // Test with different data types
    println!("\nTesting with different data types...");
    
    // Integer
    cache.set("number", &42i32, None).await.unwrap();
    let number: Option<i32> = cache.get("number").await.unwrap();
    println!("Integer: {:?}", number);
    
    // Vector of floats
    let vec_data = vec![1.1, 2.2, 3.3];
    cache.set("vector", &vec_data, None).await.unwrap();
    let retrieved_vec: Option<Vec<f32>> = cache.get("vector").await.unwrap();
    println!("Vector: {:?}", retrieved_vec);
    
    println!("\nCache implementation test completed successfully!");
}

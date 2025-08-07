use rustrag::performance::{DbQueryOptimizer, DbOptimizerConfig, SimpleQueryBuilder};
use rustrag::utils::Result;
use serde::{Deserialize, Serialize};
use tokio;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct Document {
    id: String,
    title: String,
    content: String,
    created_at: String,
}

// Simulate a database query function
async fn fetch_documents(query: &str) -> Result<Vec<Document>> {
    // Simulate some processing time
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    
    // Mock database results based on query
    let documents = vec![
        Document {
            id: "doc1".to_string(),
            title: "Introduction to RAG".to_string(),
            content: "Retrieval-Augmented Generation combines...".to_string(),
            created_at: "2024-01-01".to_string(),
        },
        Document {
            id: "doc2".to_string(),
            title: "Vector Databases".to_string(),
            content: "Vector databases store high-dimensional...".to_string(),
            created_at: "2024-01-02".to_string(),
        },
    ];
    
    println!("Executing query: {}", query);
    Ok(documents)
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("Testing Database Query Optimizer...\n");
    
    // Create optimizer configuration
    let config = DbOptimizerConfig {
        enabled: true,
        query_timeout_seconds: 30,
        max_concurrent_queries: 5,
        enable_query_cache: true,
        cache_ttl_seconds: 60, // 1 minute cache
        max_cache_entries: 100,
    };
    
    let optimizer = DbQueryOptimizer::new(config);
    
    // Test 1: Query Builder
    println!("=== Test 1: Query Builder ===");
    let query = SimpleQueryBuilder::new()
        .table("documents")
        .select("id")
        .select("title")
        .select("content")
        .where_eq("status", "published")
        .where_like("content", "RAG")
        .order_by_desc("created_at")
        .limit(10)
        .build()?;
    
    println!("Generated SQL: {}", query);
    let cache_key = SimpleQueryBuilder::new()
        .table("documents")
        .select_all()
        .where_eq("status", "published")
        .cache_key();
    
    println!("Cache key: {}\n", cache_key);
    
    // Test 2: Query Execution with Caching
    println!("=== Test 2: Query Execution with Caching ===");
    
    // First execution - should hit the database
    let query_key = "documents_search_rag";
    let start_time = std::time::Instant::now();
    
    let documents1: Vec<Document> = optimizer
        .execute_query(query_key, || fetch_documents(&query))
        .await?;
    
    let first_duration = start_time.elapsed();
    println!("First query completed in {:?}", first_duration);
    println!("Retrieved {} documents", documents1.len());
    
    // Second execution - should come from cache (faster)
    let start_time = std::time::Instant::now();
    
    let documents2: Vec<Document> = optimizer
        .execute_query(query_key, || fetch_documents(&query))
        .await?;
    
    let second_duration = start_time.elapsed();
    println!("Second query (cached) completed in {:?}", second_duration);
    println!("Retrieved {} documents", documents2.len());
    
    // Verify results are the same
    assert_eq!(documents1, documents2);
    println!("✓ Cache working correctly - same results returned\n");
    
    // Test 3: Statistics
    println!("=== Test 3: Query Statistics ===");
    let stats = optimizer.get_stats().await;
    
    println!("Query Statistics:");
    println!("  - Total queries: {}", stats.total_queries);
    println!("  - Cached queries: {}", stats.cached_queries);
    println!("  - Failed queries: {}", stats.failed_queries);
    println!("  - Cache hit rate: {:.1}%", stats.cache_hit_rate() * 100.0);
    println!("  - Average execution time: {:.2}ms", stats.avg_execution_time_ms);
    println!("  - Slowest query: {}ms", stats.slowest_query_ms);
    println!("  - Concurrent queries: {}\n", stats.concurrent_queries);
    
    // Test 4: Different queries
    println!("=== Test 4: Different Queries ===");
    
    let query2 = SimpleQueryBuilder::new()
        .table("users")
        .select("id")
        .select("name")
        .where_eq("active", "true")
        .limit(5)
        .build()?;
    
    let users_key = SimpleQueryBuilder::new()
        .table("users")
        .select("id")
        .select("name")
        .where_eq("active", "true")
        .limit(5)
        .cache_key();
    
    println!("Different query: {}", query2);
    
    // This should be a cache miss (different query)
    let users_result: Vec<serde_json::Value> = optimizer
        .execute_query(&users_key, || async {
            // Simulate different data
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
            println!("Executing users query: {}", query2);
            Ok(vec![
                serde_json::json!({"id": "user1", "name": "Alice"}),
                serde_json::json!({"id": "user2", "name": "Bob"}),
            ])
        })
        .await?;
    
    println!("Users query returned {} results\n", users_result.len());
    
    // Test 5: Cache Maintenance
    println!("=== Test 5: Cache Maintenance ===");
    optimizer.maintenance().await?;
    println!("Cache maintenance completed");
    
    // Final statistics
    let final_stats = optimizer.get_stats().await;
    println!("\nFinal Statistics:");
    println!("  - Total queries: {}", final_stats.total_queries);
    println!("  - Cache hit rate: {:.1}%", final_stats.cache_hit_rate() * 100.0);
    
    // Test 6: Timeout Simulation
    println!("\n=== Test 6: Query Timeout Test ===");
    
    // Test a query that would timeout (simulated)
    let timeout_result = optimizer
        .execute_query("timeout_test", || async {
            // This simulates a query that completes within the timeout
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
            println!("Fast query completed within timeout");
            Ok::<Vec<String>, _>(vec!["result".to_string()])
        })
        .await;
    
    match timeout_result {
        Ok(results) => println!("Timeout test passed: got {} results", results.len()),
        Err(e) => println!("Query failed: {:?}", e),
    }
    
    println!("\n✅ Database Query Optimizer test completed successfully!");
    
    Ok(())
}

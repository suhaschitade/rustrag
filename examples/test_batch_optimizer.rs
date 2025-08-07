use rustrag::performance::{BatchOptimizerConfig, create_text_batch_optimizer, create_embedding_batch_optimizer};
use std::time::{Duration, Instant};
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("🚀 Testing Batch Optimizer for Memory Management and Request Batching");
    println!("================================================================");

    // Test 1: Individual vs Batched Processing
    println!("\n1️⃣  Individual vs Batched Processing Test");
    println!("----------------------------------------");
    
    test_individual_processing().await?;
    test_batched_processing().await?;
    
    // Test 2: Memory Management
    println!("\n2️⃣  Memory Management Test");
    println!("-------------------------");
    
    test_memory_management().await?;
    
    // Test 3: Embedding Batch Processing
    println!("\n3️⃣  Embedding Batch Processing Test");
    println!("----------------------------------");
    
    test_embedding_batching().await?;
    
    // Test 4: Performance Comparison
    println!("\n4️⃣  Performance Comparison");
    println!("-------------------------");
    
    test_performance_comparison().await?;

    println!("\n✅ All batch optimizer tests completed successfully!");
    Ok(())
}

async fn test_individual_processing() -> Result<(), Box<dyn std::error::Error>> {
    let config = BatchOptimizerConfig {
        enabled: false,
        ..Default::default()
    };
    
    let optimizer = create_text_batch_optimizer(config);
    
    let start = Instant::now();
    let result = optimizer.process_request("hello world".to_string()).await?;
    let duration = start.elapsed();
    
    println!("Individual processing:");
    println!("  ⏱️  Duration: {:?}", duration);
    println!("  📝 Result: {}", result);
    
    let stats = optimizer.get_stats();
    println!("  📊 Stats: individual={}, batched={}, efficiency={:.1}%", 
             stats.individual_requests, stats.batched_requests, stats.batch_efficiency);
    
    Ok(())
}

async fn test_batched_processing() -> Result<(), Box<dyn std::error::Error>> {
    let config = BatchOptimizerConfig {
        enabled: true,
        max_batch_size: 3,
        batch_timeout_ms: 50,
        ..Default::default()
    };
    
    let optimizer = create_text_batch_optimizer(config);
    
    let start = Instant::now();
    
    // Send multiple requests simultaneously
    let tasks = vec![
        tokio::spawn({
            let opt = optimizer.clone();
            async move { opt.process_request("hello".to_string()).await }
        }),
        tokio::spawn({
            let opt = optimizer.clone();
            async move { opt.process_request("world".to_string()).await }
        }),
        tokio::spawn({
            let opt = optimizer.clone();
            async move { opt.process_request("from".to_string()).await }
        }),
        tokio::spawn({
            let opt = optimizer.clone();
            async move { opt.process_request("rust".to_string()).await }
        }),
    ];
    
    let mut results = Vec::new();
    for task in tasks {
        if let Ok(result) = task.await {
            results.push(result?);
        }
    }
    
    let duration = start.elapsed();
    
    println!("Batch processing:");
    println!("  ⏱️  Duration: {:?}", duration);
    println!("  📝 Results:");
    for (i, result) in results.iter().enumerate() {
        println!("     {}: {}", i + 1, result);
    }
    
    // Allow stats to update
    sleep(Duration::from_millis(10)).await;
    
    let stats = optimizer.get_stats();
    println!("  📊 Stats: total={}, individual={}, batched={}, batches={}, efficiency={:.1}%", 
             stats.total_requests, stats.individual_requests, stats.batched_requests,
             stats.total_batches, stats.batch_efficiency);
    println!("  📈 Average batch size: {:.1}", stats.average_batch_size);
    
    Ok(())
}

async fn test_memory_management() -> Result<(), Box<dyn std::error::Error>> {
    let config = BatchOptimizerConfig {
        enabled: true,
        enable_memory_monitoring: true,
        max_memory_mb: 5,
        cleanup_interval_seconds: 2,
        ..Default::default()
    };
    
    let optimizer = create_text_batch_optimizer(config);
    
    println!("Processing requests to trigger memory management:");
    
    let start = Instant::now();
    
    // Process many requests to trigger memory usage and cleanup
    for i in 0..15 {
        let request = format!("request_number_{}_with_some_extra_text_to_use_memory", i);
        let _result = optimizer.process_request(request).await?;
        
        if i % 5 == 0 {
            let stats = optimizer.get_stats();
            println!("  After {} requests: memory={}MB, cleanups={}", 
                     i + 1, stats.current_memory_usage_mb, stats.memory_cleanups);
        }
        
        // Small delay between requests
        sleep(Duration::from_millis(10)).await;
    }
    
    let duration = start.elapsed();
    
    let final_stats = optimizer.get_stats();
    println!("Memory management results:");
    println!("  ⏱️  Duration: {:?}", duration);
    println!("  🧠 Current memory: {}MB", final_stats.current_memory_usage_mb);
    println!("  📈 Peak memory: {}MB", final_stats.peak_memory_usage_mb);
    println!("  🧹 Cleanups performed: {}", final_stats.memory_cleanups);
    println!("  📊 Total requests: {}", final_stats.total_requests);
    
    Ok(())
}

async fn test_embedding_batching() -> Result<(), Box<dyn std::error::Error>> {
    let config = BatchOptimizerConfig {
        enabled: true,
        max_batch_size: 4,
        batch_timeout_ms: 75,
        ..Default::default()
    };
    
    let optimizer = create_embedding_batch_optimizer(config);
    
    println!("Testing embedding batch processing:");
    
    let texts = vec![
        "artificial intelligence",
        "machine learning",
        "neural networks",
        "deep learning",
        "natural language processing"
    ];
    
    let start = Instant::now();
    
    let mut tasks = Vec::new();
    for text in texts {
        let task = tokio::spawn({
            let opt = optimizer.clone();
            let text = text.to_string();
            async move {
                let embedding = opt.process_request(text.clone()).await?;
                Ok::<(String, Vec<f32>), String>((text, embedding))
            }
        });
        tasks.push(task);
    }
    
    let mut embeddings = Vec::new();
    for task in tasks {
        if let Ok(result) = task.await {
            embeddings.push(result?);
        }
    }
    
    let duration = start.elapsed();
    
    println!("  ⏱️  Duration: {:?}", duration);
    println!("  📝 Generated embeddings:");
    for (text, embedding) in embeddings {
        println!("     '{}' -> [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}]", 
                 text, embedding[0], embedding[1], embedding[2], embedding[3], embedding[4]);
    }
    
    // Allow stats to update
    sleep(Duration::from_millis(10)).await;
    
    let stats = optimizer.get_stats();
    println!("  📊 Embedding stats: efficiency={:.1}%, avg_batch_size={:.1}", 
             stats.batch_efficiency, stats.average_batch_size);
    
    Ok(())
}

async fn test_performance_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("Comparing individual vs batch processing performance:");
    
    let num_requests = 20;
    
    // Test individual processing
    let config_individual = BatchOptimizerConfig {
        enabled: false,
        ..Default::default()
    };
    let optimizer_individual = create_text_batch_optimizer(config_individual);
    
    let start = Instant::now();
    for i in 0..num_requests {
        let _result = optimizer_individual.process_request(format!("request_{}", i)).await?;
    }
    let individual_duration = start.elapsed();
    let individual_stats = optimizer_individual.get_stats();
    
    // Test batch processing
    let config_batch = BatchOptimizerConfig {
        enabled: true,
        max_batch_size: 5,
        batch_timeout_ms: 25,
        ..Default::default()
    };
    let optimizer_batch = create_text_batch_optimizer(config_batch);
    
    let start = Instant::now();
    let mut tasks = Vec::new();
    for i in 0..num_requests {
        let task = tokio::spawn({
            let opt = optimizer_batch.clone();
            async move { opt.process_request(format!("request_{}", i)).await }
        });
        tasks.push(task);
        
        // Small stagger to encourage batching
        if i % 3 == 0 {
            sleep(Duration::from_millis(1)).await;
        }
    }
    
    for task in tasks {
        let _result = task.await??;
    }
    let batch_duration = start.elapsed();
    
    // Allow stats to update
    sleep(Duration::from_millis(10)).await;
    let batch_stats = optimizer_batch.get_stats();
    
    println!("Performance comparison results:");
    println!("  Individual processing:");
    println!("    ⏱️  Duration: {:?}", individual_duration);
    println!("    📊 Requests: {}", individual_stats.total_requests);
    println!("    📈 Efficiency: {:.1}%", individual_stats.batch_efficiency);
    
    println!("  Batch processing:");
    println!("    ⏱️  Duration: {:?}", batch_duration);
    println!("    📊 Total requests: {}", batch_stats.total_requests);
    println!("    🔄 Batched requests: {}", batch_stats.batched_requests);
    println!("    📦 Number of batches: {}", batch_stats.total_batches);
    println!("    📈 Batch efficiency: {:.1}%", batch_stats.batch_efficiency);
    println!("    📊 Average batch size: {:.1}", batch_stats.average_batch_size);
    
    let speedup = individual_duration.as_nanos() as f64 / batch_duration.as_nanos() as f64;
    if speedup > 1.0 {
        println!("  🚀 Batch processing was {:.2}x faster!", speedup);
    } else {
        println!("  ⚖️  Individual processing was {:.2}x faster", 1.0 / speedup);
    }
    
    Ok(())
}

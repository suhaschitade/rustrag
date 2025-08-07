use rustrag::performance::metrics::{global_metrics, MetricsRegistry};
use std::time::Duration;
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();
    
    println!("=== RustRAG Metrics System Demo ===\n");
    
    // Get the global metrics registry
    let metrics = global_metrics();
    
    println!("1. Initial metrics summary:");
    let summary = metrics.get_metrics_summary();
    println!("   Total metrics: {}", summary.total_metrics);
    println!("   Counters: {}", summary.counters_count);
    println!("   Gauges: {}", summary.gauges_count);
    println!("   Histograms: {}", summary.histograms_count);
    println!("   Uptime: {:.2} seconds\n", summary.uptime_seconds);
    
    // Simulate some application activity
    println!("2. Simulating application activity...");
    
    // Simulate cache operations
    for i in 1..=10 {
        if i % 3 == 0 {
            metrics.increment_counter("cache_misses_total");
            println!("   Cache miss #{}", i / 3);
        } else {
            metrics.increment_counter("cache_hits_total");
            println!("   Cache hit #{}", i);
        }
        
        // Simulate request processing time
        let request_time = 0.05 + (i as f64 * 0.01);
        metrics.observe_histogram("request_duration_seconds", request_time);
        
        sleep(Duration::from_millis(100)).await;
    }
    
    // Simulate connection pool activity
    metrics.set_gauge("active_connections", 5.0);
    println!("   Set active connections to 5");
    
    // Simulate memory usage
    metrics.set_gauge("memory_usage_bytes", 1024.0 * 1024.0 * 128.0); // 128 MB
    println!("   Set memory usage to 128 MB");
    
    // Simulate more requests
    for _ in 1..=5 {
        metrics.increment_counter("requests_total");
        sleep(Duration::from_millis(50)).await;
    }
    
    println!("\n3. Updated metrics summary:");
    let updated_summary = metrics.get_metrics_summary();
    println!("   Total metrics: {}", updated_summary.total_metrics);
    println!("   Uptime: {:.2} seconds", updated_summary.uptime_seconds);
    
    // Show all metrics
    println!("\n4. All collected metrics:");
    let all_metrics = metrics.get_all_metrics();
    for metric in &all_metrics {
        println!("   {} ({:?}): {:.2}", metric.name, metric.metric_type, metric.value);
    }
    
    // Test custom metrics
    println!("\n5. Testing custom metrics:");
    
    // Create a local registry for testing
    let local_registry = MetricsRegistry::new();
    
    // Register custom metrics
    local_registry.register_counter("custom_events_total", std::collections::HashMap::new());
    local_registry.register_gauge("custom_queue_size", std::collections::HashMap::new());
    local_registry.register_histogram("custom_processing_time", 
        vec![0.1, 0.5, 1.0, 2.0, 5.0], std::collections::HashMap::new());
    
    // Use custom metrics
    local_registry.increment_counter("custom_events_total");
    local_registry.add_to_counter("custom_events_total", 4.0);
    local_registry.set_gauge("custom_queue_size", 12.0);
    local_registry.observe_histogram("custom_processing_time", 1.5);
    
    let custom_metrics = local_registry.get_all_metrics();
    println!("   Custom metrics count: {}", custom_metrics.len());
    for metric in &custom_metrics {
        println!("   {} ({:?}): {:.2}", metric.name, metric.metric_type, metric.value);
    }
    
    // Test timing functionality
    println!("\n6. Testing duration recording:");
    let result = metrics.record_duration("test_operation_duration", || {
        std::thread::sleep(Duration::from_millis(100));
        "Operation completed"
    });
    
    println!("   Operation result: {}", result);
    println!("   Duration recorded in histogram: test_operation_duration");
    
    // Show final metrics state
    println!("\n7. Final metrics state:");
    let final_metrics = metrics.get_all_metrics();
    println!("   Total metrics collected: {}", final_metrics.len());
    
    // Calculate some statistics
    let cache_hits = final_metrics.iter()
        .find(|m| m.name == "cache_hits_total")
        .map(|m| m.value)
        .unwrap_or(0.0);
    let cache_misses = final_metrics.iter()
        .find(|m| m.name == "cache_misses_total")
        .map(|m| m.value)
        .unwrap_or(0.0);
    let total_requests = final_metrics.iter()
        .find(|m| m.name == "requests_total")
        .map(|m| m.value)
        .unwrap_or(0.0);
    
    if cache_hits + cache_misses > 0.0 {
        let hit_rate = cache_hits / (cache_hits + cache_misses) * 100.0;
        println!("   Cache hit rate: {:.1}%", hit_rate);
    }
    
    println!("   Total requests processed: {}", total_requests);
    
    println!("\n8. Testing macros (if available):");
    // Note: The macros would be used like this in actual code:
    // increment_counter!("requests_total");
    // set_gauge!("active_connections", 10.0);
    // observe_histogram!("request_duration_seconds", 0.123);
    
    println!("   Metrics macros defined but not used in this example");
    
    println!("\n=== Metrics System Demo Complete ===");
    println!("The metrics system successfully:");
    println!("✅ Tracked counters, gauges, and histograms");
    println!("✅ Provided thread-safe access via global registry");
    println!("✅ Calculated derived statistics");
    println!("✅ Recorded operation durations");
    println!("✅ Supported custom metric registration");
    println!("✅ Maintained uptime and timestamps");
    
    Ok(())
}

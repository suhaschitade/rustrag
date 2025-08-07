use rustrag::performance::{
    ConnectionPool, ConnectionPoolConfig, ConnectionFactory,
    create_mock_connection_pool,
};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::time::sleep;

/// Database connection simulator for testing
#[derive(Debug, Clone)]
pub struct DatabaseConnection {
    pub id: String,
    pub database_name: String,
    pub healthy: bool,
    pub created_at: Instant,
    pub query_count: u64,
}

impl DatabaseConnection {
    pub fn new(id: String, database_name: String) -> Self {
        Self {
            id,
            database_name,
            healthy: true,
            created_at: Instant::now(),
            query_count: 0,
        }
    }

    pub async fn execute_query(&mut self, query: &str) -> Result<String, String> {
        if !self.healthy {
            return Err("Connection is unhealthy".to_string());
        }

        // Simulate query execution time
        sleep(Duration::from_millis(50)).await;
        
        self.query_count += 1;
        
        // Simulate occasional failures
        if query.contains("FAIL") {
            self.healthy = false;
            return Err("Query execution failed".to_string());
        }

        Ok(format!("Query '{}' executed successfully on {} (count: {})", 
                   query, self.database_name, self.query_count))
    }

    pub fn make_unhealthy(&mut self) {
        self.healthy = false;
    }
}

/// Database connection factory
pub struct DatabaseConnectionFactory {
    pub database_name: String,
    pub should_fail: bool,
    pub connection_counter: Mutex<u32>,
}

impl DatabaseConnectionFactory {
    pub fn new(database_name: String) -> Self {
        Self {
            database_name,
            should_fail: false,
            connection_counter: Mutex::new(0),
        }
    }

    pub fn set_should_fail(&mut self, should_fail: bool) {
        self.should_fail = should_fail;
    }
}

impl ConnectionFactory<DatabaseConnection> for DatabaseConnectionFactory {
    fn create_connection(&self) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<DatabaseConnection, String>> + Send + '_>> {
        Box::pin(async move {
            if self.should_fail {
                return Err("Database connection failure".to_string());
            }

            // Simulate connection establishment time
            sleep(Duration::from_millis(100)).await;

            let id = {
                let mut counter = self.connection_counter.lock().unwrap();
                *counter += 1;
                format!("db_conn_{}", *counter)
            };
            
            Ok(DatabaseConnection::new(id, self.database_name.clone()))
        })
    }

    fn test_connection(&self, connection: &DatabaseConnection) -> std::pin::Pin<Box<dyn std::future::Future<Output = bool> + Send + '_>> {
        let healthy = connection.healthy;
        Box::pin(async move {
            // Simulate health check query
            sleep(Duration::from_millis(10)).await;
            healthy
        })
    }

    fn close_connection(&self, _connection: DatabaseConnection) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), String>> + Send + '_>> {
        Box::pin(async move {
            // Simulate connection cleanup time
            sleep(Duration::from_millis(20)).await;
            Ok(())
        })
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("🚀 Testing Connection Pool for Resource Management");
    println!("==================================================");

    // Test 1: Basic Connection Pool Operations
    println!("\n1️⃣  Basic Connection Pool Operations");
    println!("-----------------------------------");
    
    test_basic_connection_pool().await?;
    
    // Test 2: Connection Pool Under Load
    println!("\n2️⃣  Connection Pool Under Load");
    println!("-----------------------------");
    
    test_connection_pool_load().await?;
    
    // Test 3: Health Monitoring and Recovery
    println!("\n3️⃣  Health Monitoring and Recovery");
    println!("---------------------------------");
    
    test_health_monitoring().await?;
    
    // Test 4: Database Query Simulation
    println!("\n4️⃣  Database Query Simulation");
    println!("----------------------------");
    
    test_database_queries().await?;
    
    // Test 5: Performance Comparison
    println!("\n5️⃣  Performance Comparison");
    println!("-------------------------");
    
    test_performance_comparison().await?;

    println!("\n✅ All connection pool tests completed successfully!");
    Ok(())
}

async fn test_basic_connection_pool() -> Result<(), Box<dyn std::error::Error>> {
    let config = ConnectionPoolConfig {
        enabled: true,
        min_connections: 3,
        max_connections: 8,
        connection_timeout_ms: 2000,
        ..Default::default()
    };
    
    let pool = create_mock_connection_pool(config);
    pool.initialize().await?;
    
    println!("Basic pool operations:");
    
    // Get initial stats
    let initial_stats = pool.get_stats();
    println!("  📊 Initial connections: {}", initial_stats.total_connections);
    println!("  💤 Idle connections: {}", initial_stats.idle_connections);
    
    // Get multiple connections
    let start = Instant::now();
    let conn1 = pool.get_connection().await?;
    let conn2 = pool.get_connection().await?;
    let conn3 = pool.get_connection().await?;
    let duration = start.elapsed();
    
    println!("  ⏱️  Got 3 connections in: {:?}", duration);
    println!("  🔗 Connection IDs: {}, {}, {}", 
             conn1.connection_id, conn2.connection_id, conn3.connection_id);
    
    // Check active stats
    let active_stats = pool.get_stats();
    println!("  🔥 Active connections: {}", active_stats.active_connections);
    println!("  📈 Pool efficiency: {:.1}%", active_stats.pool_efficiency);
    
    // Return connections
    pool.return_connection(conn1).await;
    pool.return_connection(conn2).await;
    pool.return_connection(conn3).await;
    
    let final_stats = pool.get_stats();
    println!("  📊 Final stats: total={}, idle={}, efficiency={:.1}%",
             final_stats.total_connections, final_stats.idle_connections, 
             final_stats.pool_efficiency);
    
    Ok(())
}

async fn test_connection_pool_load() -> Result<(), Box<dyn std::error::Error>> {
    let config = ConnectionPoolConfig {
        enabled: true,
        min_connections: 2,
        max_connections: 6,
        connection_timeout_ms: 1000,
        ..Default::default()
    };
    
    let pool = Arc::new(create_mock_connection_pool(config));
    pool.initialize().await?;
    
    println!("Load testing with concurrent requests:");
    
    let start = Instant::now();
    
    // Spawn multiple concurrent connection requests
    let mut handles = Vec::new();
    for i in 0..12 {
        let pool_clone = pool.clone();
        let handle = tokio::spawn(async move {
            match pool_clone.get_connection().await {
                Ok(conn) => {
                    // Simulate work
                    sleep(Duration::from_millis(200)).await;
                    pool_clone.return_connection(conn).await;
                    Ok(format!("Request {} completed", i))
                }
                Err(e) => Err(format!("Request {} failed: {}", i, e))
            }
        });
        handles.push(handle);
    }
    
    // Wait for all requests to complete
    let mut successful = 0;
    let mut failed = 0;
    
    for handle in handles {
        match handle.await? {
            Ok(_) => successful += 1,
            Err(e) => {
                failed += 1;
                println!("  ⚠️  {}", e);
            }
        }
    }
    
    let duration = start.elapsed();
    
    println!("  ⏱️  Load test duration: {:?}", duration);
    println!("  ✅ Successful requests: {}", successful);
    println!("  ❌ Failed requests: {}", failed);
    
    let stats = pool.get_stats();
    println!("  📊 Load test stats:");
    println!("    🔗 Connection requests: {}", stats.connection_requests);
    println!("    ✅ Successful connections: {}", stats.successful_connections);
    println!("    ❌ Failed connections: {}", stats.failed_connections);
    println!("    ⏱️  Average connection time: {:.1}ms", stats.average_connection_time_ms);
    println!("    📈 Pool efficiency: {:.1}%", stats.pool_efficiency);
    
    Ok(())
}

async fn test_health_monitoring() -> Result<(), Box<dyn std::error::Error>> {
    let config = ConnectionPoolConfig {
        enabled: true,
        min_connections: 2,
        max_connections: 5,
        health_check_interval_seconds: 2,
        ..Default::default()
    };
    
    let pool = create_mock_connection_pool(config);
    pool.initialize().await?;
    
    println!("Health monitoring test:");
    
    // Get initial stats
    let initial_stats = pool.get_stats();
    println!("  📊 Initial healthy connections: {}", initial_stats.total_connections);
    
    // Wait for health checks to run
    println!("  ⏳ Waiting for health checks to run...");
    sleep(Duration::from_secs(3)).await;
    
    let health_stats = pool.get_stats();
    println!("  ✅ Health checks passed: {}", health_stats.health_checks_passed);
    println!("  ❌ Health checks failed: {}", health_stats.health_checks_failed);
    println!("  🔗 Total connections after health check: {}", health_stats.total_connections);
    
    Ok(())
}

async fn test_database_queries() -> Result<(), Box<dyn std::error::Error>> {
    let config = ConnectionPoolConfig {
        enabled: true,
        min_connections: 2,
        max_connections: 4,
        connection_timeout_ms: 2000,
        ..Default::default()
    };
    
    let factory = Arc::new(DatabaseConnectionFactory::new("test_db".to_string()));
    let pool = ConnectionPool::new(config, factory);
    pool.initialize().await?;
    
    println!("Database query simulation:");
    
    // Execute several queries
    let queries = vec![
        "SELECT * FROM users WHERE active = true",
        "SELECT COUNT(*) FROM documents",
        "SELECT * FROM embeddings LIMIT 100",
        "INSERT INTO logs (message) VALUES ('test')",
        "UPDATE users SET last_seen = NOW()",
    ];
    
    let start = Instant::now();
    
    for (i, query) in queries.iter().enumerate() {
        let mut conn = pool.get_connection().await?;
        
        match conn.connection.execute_query(query).await {
            Ok(result) => {
                println!("  ✅ Query {}: {}", i + 1, result);
            }
            Err(e) => {
                println!("  ❌ Query {} failed: {}", i + 1, e);
            }
        }
        
        pool.return_connection(conn).await;
        
        // Small delay between queries
        sleep(Duration::from_millis(100)).await;
    }
    
    let duration = start.elapsed();
    
    println!("  ⏱️  Total query time: {:?}", duration);
    
    let stats = pool.get_stats();
    println!("  📊 Database connection stats:");
    println!("    🔗 Total connections: {}", stats.total_connections);
    println!("    ✅ Successful connections: {}", stats.successful_connections);
    println!("    📈 Pool efficiency: {:.1}%", stats.pool_efficiency);
    
    pool.shutdown().await?;
    
    Ok(())
}

async fn test_performance_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("Comparing pooled vs non-pooled performance:");
    
    let num_operations = 10;
    
    // Test without pooling
    let config_no_pool = ConnectionPoolConfig {
        enabled: false,
        ..Default::default()
    };
    
    let factory = Arc::new(DatabaseConnectionFactory::new("perf_test_db".to_string()));
    let pool_disabled = ConnectionPool::new(config_no_pool, factory.clone());
    
    let start = Instant::now();
    for i in 0..num_operations {
        let mut conn = pool_disabled.get_connection().await?;
        let _ = conn.connection.execute_query(&format!("SELECT {}", i)).await;
        pool_disabled.return_connection(conn).await;
    }
    let no_pool_duration = start.elapsed();
    
    // Test with pooling
    let config_with_pool = ConnectionPoolConfig {
        enabled: true,
        min_connections: 3,
        max_connections: 6,
        ..Default::default()
    };
    
    let pool_enabled = ConnectionPool::new(config_with_pool, factory);
    pool_enabled.initialize().await?;
    
    let start = Instant::now();
    for i in 0..num_operations {
        let mut conn = pool_enabled.get_connection().await?;
        let _ = conn.connection.execute_query(&format!("SELECT {}", i)).await;
        pool_enabled.return_connection(conn).await;
    }
    let pool_duration = start.elapsed();
    
    println!("Performance comparison results:");
    println!("  Without pooling:");
    println!("    ⏱️  Duration: {:?}", no_pool_duration);
    println!("    📊 Operations: {}", num_operations);
    
    println!("  With pooling:");
    println!("    ⏱️  Duration: {:?}", pool_duration);
    println!("    📊 Operations: {}", num_operations);
    
    let speedup = no_pool_duration.as_nanos() as f64 / pool_duration.as_nanos() as f64;
    if speedup > 1.0 {
        println!("  🚀 Pooling was {:.2}x faster!", speedup);
    } else {
        println!("  ⚖️  Non-pooling was {:.2}x faster", 1.0 / speedup);
    }
    
    let pool_stats = pool_enabled.get_stats();
    println!("  📈 Pool efficiency: {:.1}%", pool_stats.pool_efficiency);
    println!("  ⏱️  Average connection time: {:.1}ms", pool_stats.average_connection_time_ms);
    
    pool_enabled.shutdown().await?;
    
    Ok(())
}

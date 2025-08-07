use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use tokio::time::{timeout, sleep};

/// Configuration for connection pooling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionPoolConfig {
    /// Enable connection pooling
    pub enabled: bool,
    /// Minimum number of connections to maintain
    pub min_connections: usize,
    /// Maximum number of connections allowed
    pub max_connections: usize,
    /// Connection timeout in milliseconds
    pub connection_timeout_ms: u64,
    /// Idle timeout before closing connection (seconds)
    pub idle_timeout_seconds: u64,
    /// Health check interval in seconds
    pub health_check_interval_seconds: u64,
    /// Maximum retry attempts for failed connections
    pub max_retry_attempts: u32,
    /// Retry delay in milliseconds
    pub retry_delay_ms: u64,
}

impl Default for ConnectionPoolConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_connections: 2,
            max_connections: 10,
            connection_timeout_ms: 5000,
            idle_timeout_seconds: 300, // 5 minutes
            health_check_interval_seconds: 60, // 1 minute
            max_retry_attempts: 3,
            retry_delay_ms: 1000,
        }
    }
}

/// Connection pool statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionPoolStats {
    pub total_connections: usize,
    pub active_connections: usize,
    pub idle_connections: usize,
    pub failed_connections: u64,
    pub successful_connections: u64,
    pub connection_requests: u64,
    pub connection_timeouts: u64,
    pub retry_attempts: u64,
    pub health_checks_passed: u64,
    pub health_checks_failed: u64,
    pub average_connection_time_ms: f64,
    pub pool_efficiency: f64, // successful requests / total requests
}

impl ConnectionPoolStats {
    pub fn new() -> Self {
        Self {
            total_connections: 0,
            active_connections: 0,
            idle_connections: 0,
            failed_connections: 0,
            successful_connections: 0,
            connection_requests: 0,
            connection_timeouts: 0,
            retry_attempts: 0,
            health_checks_passed: 0,
            health_checks_failed: 0,
            average_connection_time_ms: 0.0,
            pool_efficiency: 0.0,
        }
    }

    pub fn update_efficiency(&mut self) {
        if self.connection_requests > 0 {
            self.pool_efficiency = (self.successful_connections as f64) / (self.connection_requests as f64) * 100.0;
        }
    }
}

/// Status of a pooled connection
#[derive(Debug, Clone, PartialEq)]
pub enum ConnectionStatus {
    Available,
    InUse,
    Failed,
    HealthCheck,
}

/// A pooled connection wrapper
#[derive(Debug)]
pub struct PooledConnection<T> {
    pub connection: T,
    pub created_at: Instant,
    pub last_used: Instant,
    pub use_count: u64,
    pub status: ConnectionStatus,
    pub connection_id: String,
}

impl<T> PooledConnection<T> {
    pub fn new(connection: T, connection_id: String) -> Self {
        let now = Instant::now();
        Self {
            connection,
            created_at: now,
            last_used: now,
            use_count: 0,
            status: ConnectionStatus::Available,
            connection_id,
        }
    }

    pub fn mark_used(&mut self) {
        self.last_used = Instant::now();
        self.use_count += 1;
        self.status = ConnectionStatus::InUse;
    }

    pub fn mark_available(&mut self) {
        self.status = ConnectionStatus::Available;
    }

    pub fn mark_failed(&mut self) {
        self.status = ConnectionStatus::Failed;
    }

    pub fn is_idle(&self, idle_timeout: Duration) -> bool {
        self.status == ConnectionStatus::Available && 
        self.last_used.elapsed() > idle_timeout
    }

    pub fn is_available(&self) -> bool {
        self.status == ConnectionStatus::Available
    }
}

/// Connection factory trait for creating new connections
pub trait ConnectionFactory<T>: Send + Sync {
    /// Create a new connection
    fn create_connection(&self) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<T, String>> + Send + '_>>;
    
    /// Test if a connection is healthy
    fn test_connection(&self, connection: &T) -> std::pin::Pin<Box<dyn std::future::Future<Output = bool> + Send + '_>>;
    
    /// Close a connection properly
    fn close_connection(&self, connection: T) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), String>> + Send + '_>>;
}

/// Simple connection pool manager
pub struct ConnectionPool<T>
where
    T: Send + Sync + 'static,
{
    config: ConnectionPoolConfig,
    connections: Arc<Mutex<VecDeque<PooledConnection<T>>>>,
    stats: Arc<RwLock<ConnectionPoolStats>>,
    factory: Arc<dyn ConnectionFactory<T>>,
    connection_counter: Arc<Mutex<u64>>,
}

impl<T> ConnectionPool<T>
where
    T: Send + Sync + 'static,
{
    /// Create a new connection pool
    pub fn new(
        config: ConnectionPoolConfig,
        factory: Arc<dyn ConnectionFactory<T>>,
    ) -> Self {
        Self {
            config,
            connections: Arc::new(Mutex::new(VecDeque::new())),
            stats: Arc::new(RwLock::new(ConnectionPoolStats::new())),
            factory,
            connection_counter: Arc::new(Mutex::new(0)),
        }
    }

    /// Initialize the connection pool with minimum connections
    pub async fn initialize(&self) -> Result<(), String> {
        if !self.config.enabled {
            return Ok(());
        }

        for _ in 0..self.config.min_connections {
            match self.create_new_connection().await {
                Ok(conn) => {
                    let mut connections = self.connections.lock().unwrap();
                    connections.push_back(conn);
                }
                Err(e) => {
                    tracing::warn!("Failed to create initial connection: {}", e);
                }
            }
        }

        // Start background maintenance task (disabled for simplicity)
        // self.start_maintenance_task().await;

        Ok(())
    }

    /// Get a connection from the pool
    pub async fn get_connection(&self) -> Result<PooledConnection<T>, String> {
        if !self.config.enabled {
            return self.create_new_connection().await;
        }

        let start = Instant::now();
        
        // Update request stats
        {
            let mut stats = self.stats.write().unwrap();
            stats.connection_requests += 1;
        }

        // Try to get an available connection
        if let Some(mut conn) = self.get_available_connection() {
            conn.mark_used();
            
            // Update stats
            {
                let mut stats = self.stats.write().unwrap();
                stats.successful_connections += 1;
                stats.active_connections += 1;
                stats.idle_connections = stats.idle_connections.saturating_sub(1);
                
                let connection_time = start.elapsed().as_millis() as f64;
                stats.average_connection_time_ms = 
                    (stats.average_connection_time_ms * (stats.successful_connections - 1) as f64 + connection_time) 
                    / stats.successful_connections as f64;
                
                stats.update_efficiency();
            }
            
            return Ok(conn);
        }

        // Try to create a new connection if under limit
        if self.can_create_new_connection() {
            match timeout(
                Duration::from_millis(self.config.connection_timeout_ms),
                self.create_new_connection_with_retry()
            ).await {
                Ok(Ok(mut conn)) => {
                    conn.mark_used();
                    
                    // Update stats
                    {
                        let mut stats = self.stats.write().unwrap();
                        stats.successful_connections += 1;
                        stats.active_connections += 1;
                        
                        let connection_time = start.elapsed().as_millis() as f64;
                        stats.average_connection_time_ms = 
                            (stats.average_connection_time_ms * (stats.successful_connections - 1) as f64 + connection_time) 
                            / stats.successful_connections as f64;
                        
                        stats.update_efficiency();
                    }
                    
                    return Ok(conn);
                }
                Ok(Err(e)) => {
                    // Update failure stats
                    {
                        let mut stats = self.stats.write().unwrap();
                        stats.failed_connections += 1;
                        stats.update_efficiency();
                    }
                    return Err(e);
                }
                Err(_) => {
                    // Timeout
                    {
                        let mut stats = self.stats.write().unwrap();
                        stats.connection_timeouts += 1;
                        stats.failed_connections += 1;
                        stats.update_efficiency();
                    }
                    return Err("Connection timeout".to_string());
                }
            }
        }

        // Pool is full, wait and retry
        Err("Connection pool exhausted".to_string())
    }

    /// Return a connection to the pool
    pub async fn return_connection(&self, mut connection: PooledConnection<T>) {
        if !self.config.enabled {
            let _ = self.factory.close_connection(connection.connection).await;
            return;
        }

        connection.mark_available();
        
        // Test connection health before returning to pool
        if self.factory.test_connection(&connection.connection).await {
            let mut connections = self.connections.lock().unwrap();
            connections.push_back(connection);
            
            // Update stats
            {
                let mut stats = self.stats.write().unwrap();
                stats.active_connections = stats.active_connections.saturating_sub(1);
                stats.idle_connections += 1;
            }
        } else {
            // Connection is unhealthy, close it
            let _ = self.factory.close_connection(connection.connection).await;
            
            {
                let mut stats = self.stats.write().unwrap();
                stats.active_connections = stats.active_connections.saturating_sub(1);
                stats.total_connections = stats.total_connections.saturating_sub(1);
                stats.failed_connections += 1;
            }
        }
    }

    /// Get pool statistics
    pub fn get_stats(&self) -> ConnectionPoolStats {
        let stats = self.stats.read().unwrap();
        let mut current_stats = stats.clone();
        
        // Update current connection counts
        let connections = self.connections.lock().unwrap();
        current_stats.total_connections = connections.len();
        current_stats.idle_connections = connections.iter()
            .filter(|c| c.is_available())
            .count();
        
        current_stats
    }

    /// Get pool configuration
    pub fn get_config(&self) -> &ConnectionPoolConfig {
        &self.config
    }

    /// Shutdown the pool and close all connections
    pub async fn shutdown(&self) -> Result<(), String> {
        let mut connections = self.connections.lock().unwrap();
        let mut errors = Vec::new();
        
        while let Some(conn) = connections.pop_front() {
            if let Err(e) = self.factory.close_connection(conn.connection).await {
                errors.push(e);
            }
        }
        
        if errors.is_empty() {
            Ok(())
        } else {
            Err(format!("Errors during shutdown: {:?}", errors))
        }
    }

    // Private helper methods
    
    fn get_available_connection(&self) -> Option<PooledConnection<T>> {
        let mut connections = self.connections.lock().unwrap();
        
        // Find first available connection
        for i in 0..connections.len() {
            if connections[i].is_available() {
                return connections.remove(i);
            }
        }
        
        None
    }

    fn can_create_new_connection(&self) -> bool {
        let connections = self.connections.lock().unwrap();
        let stats = self.stats.read().unwrap();
        (connections.len() + stats.active_connections) < self.config.max_connections
    }

    async fn create_new_connection(&self) -> Result<PooledConnection<T>, String> {
        let connection = self.factory.create_connection().await?;
        
        let conn_id = {
            let mut counter = self.connection_counter.lock().unwrap();
            *counter += 1;
            format!("conn_{}", *counter)
        };
        
        let pooled_conn = PooledConnection::new(connection, conn_id);
        
        // Update stats
        {
            let mut stats = self.stats.write().unwrap();
            stats.total_connections += 1;
        }
        
        Ok(pooled_conn)
    }

    async fn create_new_connection_with_retry(&self) -> Result<PooledConnection<T>, String> {
        let mut last_error = String::new();
        
        for attempt in 0..self.config.max_retry_attempts {
            match self.create_new_connection().await {
                Ok(conn) => return Ok(conn),
                Err(e) => {
                    last_error = e;
                    
                    {
                        let mut stats = self.stats.write().unwrap();
                        stats.retry_attempts += 1;
                    }
                    
                    if attempt < self.config.max_retry_attempts - 1 {
                        sleep(Duration::from_millis(self.config.retry_delay_ms)).await;
                    }
                }
            }
        }
        
        Err(format!("Failed after {} attempts: {}", self.config.max_retry_attempts, last_error))
    }

    // Maintenance tasks are disabled for simplicity
    // Can be added later as needed
}

/// Mock connection for testing
#[derive(Debug, Clone)]
pub struct MockConnection {
    pub id: String,
    pub healthy: bool,
    pub created_at: Instant,
}

impl MockConnection {
    pub fn new(id: String) -> Self {
        Self {
            id,
            healthy: true,
            created_at: Instant::now(),
        }
    }

    pub fn make_unhealthy(&mut self) {
        self.healthy = false;
    }
}

/// Mock connection factory for testing
pub struct MockConnectionFactory {
    pub should_fail: bool,
    pub connection_counter: Mutex<u32>,
}

impl MockConnectionFactory {
    pub fn new() -> Self {
        Self {
            should_fail: false,
            connection_counter: Mutex::new(0),
        }
    }

    pub fn set_should_fail(&mut self, should_fail: bool) {
        self.should_fail = should_fail;
    }
}

impl ConnectionFactory<MockConnection> for MockConnectionFactory {
    fn create_connection(&self) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<MockConnection, String>> + Send + '_>> {
        Box::pin(async move {
            if self.should_fail {
                return Err("Mock connection failure".to_string());
            }

            let id = {
                let mut counter = self.connection_counter.lock().unwrap();
                *counter += 1;
                format!("mock_conn_{}", *counter)
            };
            
            // Simulate connection creation time
            sleep(Duration::from_millis(10)).await;
            
            Ok(MockConnection::new(id))
        })
    }

    fn test_connection(&self, connection: &MockConnection) -> std::pin::Pin<Box<dyn std::future::Future<Output = bool> + Send + '_>> {
        let healthy = connection.healthy;
        Box::pin(async move { healthy })
    }

    fn close_connection(&self, _connection: MockConnection) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), String>> + Send + '_>> {
        Box::pin(async move {
            // Simulate connection cleanup time
            sleep(Duration::from_millis(5)).await;
            Ok(())
        })
    }
}

/// Helper function to create a simple mock connection pool
pub fn create_mock_connection_pool(config: ConnectionPoolConfig) -> ConnectionPool<MockConnection> {
    let factory = Arc::new(MockConnectionFactory::new());
    ConnectionPool::new(config, factory)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_connection_pool_basic() {
        let config = ConnectionPoolConfig {
            enabled: true,
            min_connections: 2,
            max_connections: 5,
            ..Default::default()
        };
        
        let pool = create_mock_connection_pool(config);
        pool.initialize().await.unwrap();
        
        let stats = pool.get_stats();
        assert!(stats.total_connections >= 2);
        
        // Get a connection
        let conn = pool.get_connection().await.unwrap();
        assert_eq!(conn.status, ConnectionStatus::InUse);
        
        // Return the connection
        pool.return_connection(conn).await;
        
        let final_stats = pool.get_stats();
        assert!(final_stats.successful_connections > 0);
        assert!(final_stats.pool_efficiency > 0.0);
    }

    #[tokio::test]
    async fn test_connection_pool_disabled() {
        let config = ConnectionPoolConfig {
            enabled: false,
            ..Default::default()
        };
        
        let pool = create_mock_connection_pool(config);
        pool.initialize().await.unwrap();
        
        let conn = pool.get_connection().await.unwrap();
        assert_eq!(conn.status, ConnectionStatus::InUse);
        
        pool.return_connection(conn).await;
    }

    #[tokio::test]
    async fn test_connection_pool_retry() {
        let config = ConnectionPoolConfig {
            enabled: true,
            max_retry_attempts: 2,
            retry_delay_ms: 10,
            ..Default::default()
        };
        
        let mut factory = MockConnectionFactory::new();
        factory.set_should_fail(true);
        let pool = ConnectionPool::new(config, Arc::new(factory));
        
        let result = pool.get_connection().await;
        assert!(result.is_err());
        
        let stats = pool.get_stats();
        assert!(stats.retry_attempts > 0);
        assert!(stats.failed_connections > 0);
    }
}

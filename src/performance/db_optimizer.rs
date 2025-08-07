use crate::utils::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Simple database query optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DbOptimizerConfig {
    /// Whether query optimization is enabled
    pub enabled: bool,
    /// Query timeout in seconds
    pub query_timeout_seconds: u64,
    /// Maximum number of concurrent queries
    pub max_concurrent_queries: usize,
    /// Enable query result caching
    pub enable_query_cache: bool,
    /// Query cache TTL in seconds
    pub cache_ttl_seconds: u64,
    /// Maximum cache size (number of entries)
    pub max_cache_entries: usize,
}

impl Default for DbOptimizerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            query_timeout_seconds: 30,
            max_concurrent_queries: 10,
            enable_query_cache: true,
            cache_ttl_seconds: 300, // 5 minutes
            max_cache_entries: 1000,
        }
    }
}

/// Query execution statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueryStats {
    /// Total queries executed
    pub total_queries: u64,
    /// Queries served from cache
    pub cached_queries: u64,
    /// Failed queries
    pub failed_queries: u64,
    /// Average query execution time in milliseconds
    pub avg_execution_time_ms: f64,
    /// Slowest query time in milliseconds
    pub slowest_query_ms: u64,
    /// Current concurrent queries
    pub concurrent_queries: usize,
}

impl QueryStats {
    pub fn cache_hit_rate(&self) -> f64 {
        if self.total_queries == 0 {
            0.0
        } else {
            self.cached_queries as f64 / self.total_queries as f64
        }
    }
}

/// Simple query result cache entry
#[derive(Debug, Clone)]
struct QueryCacheEntry {
    result: String, // JSON serialized result
    created_at: Instant,
    ttl: Duration,
    access_count: u32,
}

impl QueryCacheEntry {
    fn new(result: String, ttl: Duration) -> Self {
        Self {
            result,
            created_at: Instant::now(),
            ttl,
            access_count: 1,
        }
    }

    fn is_expired(&self) -> bool {
        self.created_at.elapsed() > self.ttl
    }

    fn access(&mut self) {
        self.access_count += 1;
    }
}

/// Simple database query optimizer
pub struct DbQueryOptimizer {
    config: DbOptimizerConfig,
    stats: Arc<RwLock<QueryStats>>,
    query_cache: Arc<RwLock<HashMap<String, QueryCacheEntry>>>,
    execution_times: Arc<RwLock<Vec<u64>>>, // For calculating averages
}

impl DbQueryOptimizer {
    /// Create a new database query optimizer
    pub fn new(config: DbOptimizerConfig) -> Self {
        Self {
            config,
            stats: Arc::new(RwLock::new(QueryStats::default())),
            query_cache: Arc::new(RwLock::new(HashMap::new())),
            execution_times: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Execute a query with optimization
    pub async fn execute_query<T, F, Fut>(&self, query_key: &str, query_fn: F) -> Result<T>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
        T: Serialize + for<'de> Deserialize<'de>,
    {
        if !self.config.enabled {
            return query_fn().await;
        }

        // Check cache first if enabled
        if self.config.enable_query_cache {
            if let Some(cached_result) = self.get_from_cache(query_key).await? {
                self.record_cache_hit().await;
                debug!("Query served from cache: {}", query_key);
                return Ok(cached_result);
            }
        }

        // Check concurrent query limit
        self.check_concurrent_limit().await?;

        // Execute query with timing
        let start_time = Instant::now();
        self.increment_concurrent_queries().await;

        let result = match tokio::time::timeout(
            Duration::from_secs(self.config.query_timeout_seconds),
            query_fn(),
        )
        .await
        {
            Ok(result) => result,
            Err(_) => {
                self.decrement_concurrent_queries().await;
                self.record_failed_query().await;
                return Err(Error::internal("Query timeout"));
            }
        };

        self.decrement_concurrent_queries().await;
        let execution_time = start_time.elapsed();

        match result {
            Ok(data) => {
                // Record successful execution
                self.record_successful_query(execution_time).await;

                // Cache result if enabled
                if self.config.enable_query_cache {
                    self.cache_result(query_key, &data).await?;
                }

                debug!(
                    "Query executed successfully: {} ({}ms)",
                    query_key,
                    execution_time.as_millis()
                );

                Ok(data)
            }
            Err(e) => {
                self.record_failed_query().await;
                warn!("Query failed: {} - {:?}", query_key, e);
                Err(e)
            }
        }
    }

    /// Get query statistics
    pub async fn get_stats(&self) -> QueryStats {
        let mut stats = self.stats.read().await.clone();
        
        // Update cache stats
        let cache = self.query_cache.read().await;
        debug!("Current cache size: {} entries", cache.len());
        
        stats
    }

    /// Clear query cache
    pub async fn clear_cache(&self) -> Result<()> {
        let mut cache = self.query_cache.write().await;
        let cleared_count = cache.len();
        cache.clear();
        info!("Cleared {} entries from query cache", cleared_count);
        Ok(())
    }

    /// Perform cache maintenance (remove expired entries)
    pub async fn maintenance(&self) -> Result<()> {
        if !self.config.enable_query_cache {
            return Ok(());
        }

        let mut cache = self.query_cache.write().await;
        let initial_size = cache.len();
        
        // Remove expired entries
        cache.retain(|_, entry| !entry.is_expired());
        
        // If still over limit, remove least accessed entries
        if cache.len() > self.config.max_cache_entries {
            let mut entries: Vec<(String, u32)> = cache
                .iter()
                .map(|(key, entry)| (key.clone(), entry.access_count))
                .collect();
            
            // Sort by access count (ascending - least accessed first)
            entries.sort_by_key(|(_, count)| *count);
            
            let to_remove = cache.len() - self.config.max_cache_entries;
            for (key, _) in entries.iter().take(to_remove) {
                cache.remove(key);
            }
        }

        let final_size = cache.len();
        if initial_size != final_size {
            debug!(
                "Cache maintenance: {} -> {} entries",
                initial_size, final_size
            );
        }

        Ok(())
    }

    // Private helper methods

    async fn get_from_cache<T>(&self, query_key: &str) -> Result<Option<T>>
    where
        T: for<'de> Deserialize<'de>,
    {
        let mut cache = self.query_cache.write().await;
        
        if let Some(entry) = cache.get_mut(query_key) {
            if entry.is_expired() {
                cache.remove(query_key);
                return Ok(None);
            }

            entry.access();
            let result: T = serde_json::from_str(&entry.result)
                .map_err(|e| Error::internal(format!("Cache deserialization error: {}", e)))?;
            
            return Ok(Some(result));
        }

        Ok(None)
    }

    async fn cache_result<T>(&self, query_key: &str, data: &T) -> Result<()>
    where
        T: Serialize,
    {
        let serialized = serde_json::to_string(data)
            .map_err(|e| Error::internal(format!("Cache serialization error: {}", e)))?;

        let mut cache = self.query_cache.write().await;
        
        // Simple eviction if cache is full
        if cache.len() >= self.config.max_cache_entries {
            // Remove one random entry (simple approach)
            if let Some(key) = cache.keys().next().cloned() {
                cache.remove(&key);
            }
        }

        let ttl = Duration::from_secs(self.config.cache_ttl_seconds);
        cache.insert(query_key.to_string(), QueryCacheEntry::new(serialized, ttl));
        
        Ok(())
    }

    async fn check_concurrent_limit(&self) -> Result<()> {
        let stats = self.stats.read().await;
        if stats.concurrent_queries >= self.config.max_concurrent_queries {
            return Err(Error::internal("Maximum concurrent queries exceeded"));
        }
        Ok(())
    }

    async fn increment_concurrent_queries(&self) {
        let mut stats = self.stats.write().await;
        stats.concurrent_queries += 1;
    }

    async fn decrement_concurrent_queries(&self) {
        let mut stats = self.stats.write().await;
        stats.concurrent_queries = stats.concurrent_queries.saturating_sub(1);
    }

    async fn record_successful_query(&self, execution_time: Duration) {
        let execution_ms = execution_time.as_millis() as u64;
        
        {
            let mut stats = self.stats.write().await;
            stats.total_queries += 1;
            
            if execution_ms > stats.slowest_query_ms {
                stats.slowest_query_ms = execution_ms;
            }
        }

        // Update execution times for average calculation
        {
            let mut times = self.execution_times.write().await;
            times.push(execution_ms);
            
            // Keep only last 1000 execution times for memory efficiency
            if times.len() > 1000 {
                times.remove(0);
            }
        }

        // Recalculate average
        self.update_average_execution_time().await;
    }

    async fn record_failed_query(&self) {
        let mut stats = self.stats.write().await;
        stats.total_queries += 1;
        stats.failed_queries += 1;
    }

    async fn record_cache_hit(&self) {
        let mut stats = self.stats.write().await;
        stats.total_queries += 1;
        stats.cached_queries += 1;
    }

    async fn update_average_execution_time(&self) {
        let times = self.execution_times.read().await;
        if !times.is_empty() {
            let avg = times.iter().sum::<u64>() as f64 / times.len() as f64;
            let mut stats = self.stats.write().await;
            stats.avg_execution_time_ms = avg;
        }
    }
}

/// Simple query builder for common database operations
#[derive(Debug, Clone)]
pub struct SimpleQueryBuilder {
    table: Option<String>,
    select_fields: Vec<String>,
    where_conditions: Vec<String>,
    order_by: Option<String>,
    limit: Option<usize>,
}

impl SimpleQueryBuilder {
    pub fn new() -> Self {
        Self {
            table: None,
            select_fields: Vec::new(),
            where_conditions: Vec::new(),
            order_by: None,
            limit: None,
        }
    }

    pub fn table(mut self, table: &str) -> Self {
        self.table = Some(table.to_string());
        self
    }

    pub fn select(mut self, field: &str) -> Self {
        self.select_fields.push(field.to_string());
        self
    }

    pub fn select_all(mut self) -> Self {
        self.select_fields.push("*".to_string());
        self
    }

    pub fn where_eq(mut self, field: &str, value: &str) -> Self {
        self.where_conditions.push(format!("{} = '{}'", field, value));
        self
    }

    pub fn where_like(mut self, field: &str, pattern: &str) -> Self {
        self.where_conditions.push(format!("{} LIKE '%{}%'", field, pattern));
        self
    }

    pub fn order_by_desc(mut self, field: &str) -> Self {
        self.order_by = Some(format!("{} DESC", field));
        self
    }

    pub fn order_by_asc(mut self, field: &str) -> Self {
        self.order_by = Some(format!("{} ASC", field));
        self
    }

    pub fn limit(mut self, count: usize) -> Self {
        self.limit = Some(count);
        self
    }

    pub fn build(&self) -> Result<String> {
        let table = self.table.as_ref()
            .ok_or_else(|| Error::internal("Table name is required"))?;

        let fields = if self.select_fields.is_empty() {
            "*".to_string()
        } else {
            self.select_fields.join(", ")
        };

        let mut query = format!("SELECT {} FROM {}", fields, table);

        if !self.where_conditions.is_empty() {
            query.push_str(&format!(" WHERE {}", self.where_conditions.join(" AND ")));
        }

        if let Some(ref order) = self.order_by {
            query.push_str(&format!(" ORDER BY {}", order));
        }

        if let Some(limit) = self.limit {
            query.push_str(&format!(" LIMIT {}", limit));
        }

        Ok(query)
    }

    /// Generate a cache key for this query
    pub fn cache_key(&self) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let query = self.build().unwrap_or_default();
        let mut hasher = DefaultHasher::new();
        query.hash(&mut hasher);
        format!("query_{:x}", hasher.finish())
    }
}

impl Default for SimpleQueryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_db_optimizer_basic() {
        let config = DbOptimizerConfig::default();
        let optimizer = DbQueryOptimizer::new(config);

        // Test successful query
        let result: i32 = optimizer
            .execute_query("test_query", || async { Ok(42) })
            .await
            .unwrap();

        assert_eq!(result, 42);

        let stats = optimizer.get_stats().await;
        assert_eq!(stats.total_queries, 1);
        assert_eq!(stats.failed_queries, 0);
    }

    #[tokio::test]
    async fn test_query_cache() {
        let mut config = DbOptimizerConfig::default();
        config.enable_query_cache = true;
        let optimizer = DbQueryOptimizer::new(config);

        // First query - should execute
        let result1: String = optimizer
            .execute_query("cached_query", || async { Ok("test_data".to_string()) })
            .await
            .unwrap();

        // Second query - should come from cache
        let result2: String = optimizer
            .execute_query("cached_query", || async { Ok("different_data".to_string()) })
            .await
            .unwrap();

        assert_eq!(result1, "test_data");
        assert_eq!(result2, "test_data"); // Should be from cache

        let stats = optimizer.get_stats().await;
        assert_eq!(stats.total_queries, 2);
        assert_eq!(stats.cached_queries, 1);
        assert_eq!(stats.cache_hit_rate(), 0.5);
    }

    #[test]
    fn test_query_builder() {
        let query = SimpleQueryBuilder::new()
            .table("documents")
            .select("id")
            .select("title")
            .where_eq("status", "active")
            .where_like("content", "search_term")
            .order_by_desc("created_at")
            .limit(10)
            .build()
            .unwrap();

        let expected = "SELECT id, title FROM documents WHERE status = 'active' AND content LIKE '%search_term%' ORDER BY created_at DESC LIMIT 10";
        assert_eq!(query, expected);
    }

    #[test]
    fn test_query_builder_cache_key() {
        let builder = SimpleQueryBuilder::new()
            .table("test")
            .select_all();
        
        let key1 = builder.clone().cache_key();
        let key2 = builder.cache_key();
        assert_eq!(key1, key2);

        let key3 = SimpleQueryBuilder::new()
            .table("test")
            .select("id")
            .cache_key();
        assert_ne!(key1, key3);
    }
}

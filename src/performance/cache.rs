use crate::utils::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing::{debug, error, info, warn};
use tokio::sync::RwLock;
use uuid::Uuid;

#[cfg(feature = "redis")]
use redis::{aio::ConnectionManager, AsyncCommands, Client as RedisClient, RedisError};

#[cfg(feature = "redis")]
use deadpool_redis::{Config as RedisConfig, Manager, Pool as RedisPool, Runtime};

/// Cache configuration settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Whether caching is enabled
    pub enabled: bool,
    /// Redis connection URL
    pub redis_url: Option<String>,
    /// Default TTL for cache entries (in seconds)
    pub default_ttl: u64,
    /// Maximum memory cache size (in MB)
    pub max_memory_size_mb: usize,
    /// Memory cache TTL (in seconds)
    pub memory_ttl: u64,
    /// Cache key prefix
    pub key_prefix: String,
    /// Cache compression settings
    pub compression: CacheCompressionConfig,
    /// Cache eviction policy
    pub eviction_policy: CacheEvictionPolicy,
    /// Cache statistics collection
    pub collect_stats: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            redis_url: Some("redis://localhost:6379".to_string()),
            default_ttl: 3600, // 1 hour
            max_memory_size_mb: 100,
            memory_ttl: 300, // 5 minutes
            key_prefix: "rustrag".to_string(),
            compression: CacheCompressionConfig::default(),
            eviction_policy: CacheEvictionPolicy::LRU,
            collect_stats: true,
        }
    }
}

/// Cache compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheCompressionConfig {
    /// Whether to compress cache entries
    pub enabled: bool,
    /// Compression threshold in bytes (only compress if larger)
    pub threshold_bytes: usize,
    /// Compression level (0-9)
    pub level: u32,
}

impl Default for CacheCompressionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold_bytes: 1024, // 1KB
            level: 6, // Balanced compression
        }
    }
}

/// Cache eviction policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheEvictionPolicy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// Time to Live based
    TTL,
    /// First In, First Out
    FIFO,
}

/// Cache entry metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheEntry<T> {
    /// The cached value
    value: T,
    /// When the entry was created
    created_at: u64,
    /// When the entry expires (timestamp)
    expires_at: u64,
    /// Access count for LFU eviction
    access_count: u32,
    /// Last access timestamp for LRU eviction
    last_accessed: u64,
    /// Entry size in bytes
    size_bytes: usize,
}

impl<T> CacheEntry<T> {
    fn new(value: T, ttl: u64, size_bytes: usize) -> Self {
        let now = current_timestamp();
        Self {
            value,
            created_at: now,
            expires_at: now + ttl,
            access_count: 1,
            last_accessed: now,
            size_bytes,
        }
    }

    fn is_expired(&self) -> bool {
        current_timestamp() > self.expires_at
    }

    fn access(&mut self) {
        self.access_count += 1;
        self.last_accessed = current_timestamp();
    }
}

/// Cache statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheStats {
    /// Total cache hits
    pub hits: u64,
    /// Total cache misses
    pub misses: u64,
    /// Total cache gets (hits + misses)
    pub gets: u64,
    /// Total cache sets
    pub sets: u64,
    /// Total cache deletes
    pub deletes: u64,
    /// Total cache evictions
    pub evictions: u64,
    /// Memory cache size in bytes
    pub memory_size_bytes: usize,
    /// Memory cache entry count
    pub memory_entries: usize,
    /// Redis cache entry count (if available)
    pub redis_entries: Option<usize>,
    /// Average access time in microseconds
    pub avg_access_time_us: f64,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        if self.gets == 0 {
            0.0
        } else {
            self.hits as f64 / self.gets as f64
        }
    }

    pub fn miss_rate(&self) -> f64 {
        1.0 - self.hit_rate()
    }
}

/// Cache layer types
#[derive(Debug, Clone, Copy)]
pub enum CacheLayer {
    /// In-memory L1 cache
    Memory,
    /// Redis L2 cache
    Redis,
    /// Both layers
    Both,
}

/// Multi-level cache service
pub struct CacheService {
    config: CacheConfig,
    /// In-memory L1 cache
    memory_cache: Arc<RwLock<HashMap<String, CacheEntry<Vec<u8>>>>>,
    /// Redis L2 cache connection
    #[cfg(feature = "redis")]
    redis_pool: Option<RedisPool>,
    /// Cache statistics
    stats: Arc<RwLock<CacheStats>>,
}

impl CacheService {
    /// Create a new cache service
    pub async fn new(config: CacheConfig) -> Result<Self> {
        let mut service = Self {
            config: config.clone(),
            memory_cache: Arc::new(RwLock::new(HashMap::new())),
            #[cfg(feature = "redis")]
            redis_pool: None,
            stats: Arc::new(RwLock::new(CacheStats::default())),
        };

        // Initialize Redis connection if configured
        #[cfg(feature = "redis")]
        if let Some(redis_url) = &config.redis_url {
            service.redis_pool = Some(service.create_redis_pool(redis_url).await?);
            info!("Redis cache initialized: {}", redis_url);
        }

        if !config.enabled {
            warn!("Cache service is disabled");
        }

        Ok(service)
    }

    #[cfg(feature = "redis")]
    async fn create_redis_pool(&self, redis_url: &str) -> Result<RedisPool> {
        let cfg = RedisConfig::from_url(redis_url);
        let manager = Manager::new(cfg.connection.unwrap())
            .map_err(|e| Error::cache(format!("Failed to create Redis manager: {}", e)))?;
        
        let pool = RedisPool::builder(manager)
            .max_size(16)
            .build()
            .map_err(|e| Error::cache(format!("Failed to create Redis pool: {}", e)))?;

        Ok(pool)
    }

    /// Get a value from cache
    pub async fn get<T>(&self, key: &str) -> Result<Option<T>>
    where
        T: for<'de> Deserialize<'de> + serde::Serialize,
    {
        if !self.config.enabled {
            return Ok(None);
        }

        let start_time = std::time::Instant::now();
        let full_key = format!("{}:{}", self.config.key_prefix, key);

        // Try L1 memory cache first
        if let Some(value) = self.get_from_memory(&full_key).await? {
            self.record_hit(start_time).await;
            return Ok(Some(value));
        }

        // Try L2 Redis cache
        #[cfg(feature = "redis")]
        if let Some(value) = self.get_from_redis(&full_key).await? {
            // Store in memory cache for faster future access
            if let Ok(serialized) = bincode::serialize(&value) {
                self.set_in_memory(&full_key, serialized, self.config.memory_ttl).await?;
            }
            self.record_hit(start_time).await;
            return Ok(Some(value));
        }

        self.record_miss(start_time).await;
        Ok(None)
    }

    /// Set a value in cache
    pub async fn set<T>(&self, key: &str, value: &T, ttl: Option<u64>) -> Result<()>
    where
        T: Serialize,
    {
        if !self.config.enabled {
            return Ok(());
        }

        let full_key = format!("{}:{}", self.config.key_prefix, key);
        let ttl = ttl.unwrap_or(self.config.default_ttl);
        
        let serialized = bincode::serialize(value)
            .map_err(|e| Error::cache(format!("Failed to serialize cache value: {}", e)))?;

        // Optionally compress large entries
        let data = if self.config.compression.enabled && serialized.len() > self.config.compression.threshold_bytes {
            self.compress_data(&serialized)?
        } else {
            serialized
        };

        // Store in both layers
        self.set_in_memory(&full_key, data.clone(), std::cmp::min(ttl, self.config.memory_ttl)).await?;
        
        #[cfg(feature = "redis")]
        self.set_in_redis(&full_key, &data, ttl).await?;

        self.record_set().await;
        debug!("Cache set: key={}, size={} bytes", full_key, data.len());
        Ok(())
    }

    /// Delete a value from cache
    pub async fn delete(&self, key: &str) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let full_key = format!("{}:{}", self.config.key_prefix, key);

        // Remove from both layers
        self.delete_from_memory(&full_key).await?;
        
        #[cfg(feature = "redis")]
        self.delete_from_redis(&full_key).await?;

        self.record_delete().await;
        debug!("Cache delete: key={}", full_key);
        Ok(())
    }

    /// Clear entire cache
    pub async fn clear(&self) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Clear memory cache
        {
            let mut memory = self.memory_cache.write().await;
            memory.clear();
        }

        // Clear Redis cache (only keys with our prefix)
        #[cfg(feature = "redis")]
        if let Some(pool) = &self.redis_pool {
            let mut conn = pool.get().await
                .map_err(|e| Error::cache(format!("Failed to get Redis connection: {}", e)))?;
            
            let pattern = format!("{}:*", self.config.key_prefix);
            let keys: Vec<String> = conn.keys(&pattern).await
                .map_err(|e| Error::cache(format!("Failed to get Redis keys: {}", e)))?;
            
            if !keys.is_empty() {
                let _: () = conn.del(&keys).await
                    .map_err(|e| Error::cache(format!("Failed to delete Redis keys: {}", e)))?;
            }
        }

        info!("Cache cleared");
        Ok(())
    }

    /// Get cache statistics
    pub async fn stats(&self) -> CacheStats {
        let mut stats = self.stats.read().await.clone();
        
        // Update memory stats
        let memory = self.memory_cache.read().await;
        stats.memory_entries = memory.len();
        stats.memory_size_bytes = memory.values()
            .map(|entry| entry.size_bytes)
            .sum();

        // Update Redis stats if available
        #[cfg(feature = "redis")]
        if let Some(pool) = &self.redis_pool {
            if let Ok(mut conn) = pool.get().await {
                let pattern = format!("{}:*", self.config.key_prefix);
                if let Ok(keys) = conn.keys::<&str, Vec<String>>(&pattern).await {
                    stats.redis_entries = Some(keys.len());
                }
            }
        }

        stats
    }

    /// Perform cache maintenance (cleanup expired entries)
    pub async fn maintenance(&self) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let mut evicted = 0;
        let now = current_timestamp();

        // Clean expired entries from memory cache
        {
            let mut memory = self.memory_cache.write().await;
            let mut to_remove = Vec::new();

            for (key, entry) in memory.iter() {
                if entry.is_expired() {
                    to_remove.push(key.clone());
                }
            }

            for key in to_remove {
                memory.remove(&key);
                evicted += 1;
            }

            // Check if we need to evict more entries due to memory constraints
            let total_size_mb = memory.values()
                .map(|entry| entry.size_bytes)
                .sum::<usize>() / (1024 * 1024);

            if total_size_mb > self.config.max_memory_size_mb {
                let excess_entries = memory.len() - (self.config.max_memory_size_mb * 1024 * 1024) / 
                    (memory.values().map(|e| e.size_bytes).sum::<usize>() / memory.len().max(1));
                
                let mut entries_to_remove: Vec<(String, CacheEntry<Vec<u8>>)> = memory.iter()
                    .map(|(key, entry)| (key.clone(), entry.clone()))
                    .collect();
                    
                match self.config.eviction_policy {
                    CacheEvictionPolicy::LRU => {
                        entries_to_remove.sort_by_key(|(_, entry)| entry.last_accessed);
                    }
                    CacheEvictionPolicy::LFU => {
                        entries_to_remove.sort_by_key(|(_, entry)| entry.access_count);
                    }
                    CacheEvictionPolicy::TTL => {
                        entries_to_remove.sort_by_key(|(_, entry)| entry.expires_at);
                    }
                    CacheEvictionPolicy::FIFO => {
                        entries_to_remove.sort_by_key(|(_, entry)| entry.created_at);
                    }
                }

                for (key, _) in entries_to_remove.iter().take(excess_entries) {
                    memory.remove(key);
                    evicted += 1;
                }
            }
        }

        if evicted > 0 {
            self.record_evictions(evicted).await;
            debug!("Cache maintenance: evicted {} entries", evicted);
        }

        Ok(())
    }

    // Internal helper methods

    async fn get_from_memory<T>(&self, key: &str) -> Result<Option<T>>
    where
        T: for<'de> Deserialize<'de>,
    {
        let mut memory = self.memory_cache.write().await;
        
        if let Some(entry) = memory.get_mut(key) {
            if entry.is_expired() {
                memory.remove(key);
                return Ok(None);
            }
            
            entry.access();
            let data = if self.config.compression.enabled {
                self.decompress_data(&entry.value)?
            } else {
                entry.value.clone()
            };
            
            let value = bincode::deserialize(&data)
                .map_err(|e| Error::cache(format!("Failed to deserialize cache value: {}", e)))?;
            
            return Ok(Some(value));
        }

        Ok(None)
    }

    async fn set_in_memory(&self, key: &str, data: Vec<u8>, ttl: u64) -> Result<()> {
        let mut memory = self.memory_cache.write().await;
        let size = data.len();
        let entry = CacheEntry::new(data, ttl, size);
        memory.insert(key.to_string(), entry);
        Ok(())
    }

    async fn delete_from_memory(&self, key: &str) -> Result<()> {
        let mut memory = self.memory_cache.write().await;
        memory.remove(key);
        Ok(())
    }

    #[cfg(feature = "redis")]
    async fn get_from_redis<T>(&self, key: &str) -> Result<Option<T>>
    where
        T: for<'de> Deserialize<'de>,
    {
        if let Some(pool) = &self.redis_pool {
            let mut conn = pool.get().await
                .map_err(|e| Error::cache(format!("Failed to get Redis connection: {}", e)))?;
            
            let data: Option<Vec<u8>> = conn.get(key).await
                .map_err(|e| Error::cache(format!("Failed to get from Redis: {}", e)))?;
            
            if let Some(data) = data {
                let decompressed = if self.config.compression.enabled {
                    self.decompress_data(&data)?
                } else {
                    data
                };
                
                let value = bincode::deserialize(&decompressed)
                    .map_err(|e| Error::cache(format!("Failed to deserialize Redis value: {}", e)))?;
                
                return Ok(Some(value));
            }
        }
        
        Ok(None)
    }

    #[cfg(feature = "redis")]
    async fn set_in_redis(&self, key: &str, data: &[u8], ttl: u64) -> Result<()> {
        if let Some(pool) = &self.redis_pool {
            let mut conn = pool.get().await
                .map_err(|e| Error::cache(format!("Failed to get Redis connection: {}", e)))?;
            
            let _: () = conn.set_ex(key, data, ttl).await
                .map_err(|e| Error::cache(format!("Failed to set in Redis: {}", e)))?;
        }
        
        Ok(())
    }

    #[cfg(feature = "redis")]
    async fn delete_from_redis(&self, key: &str) -> Result<()> {
        if let Some(pool) = &self.redis_pool {
            let mut conn = pool.get().await
                .map_err(|e| Error::cache(format!("Failed to get Redis connection: {}", e)))?;
            
            let _: () = conn.del(key).await
                .map_err(|e| Error::cache(format!("Failed to delete from Redis: {}", e)))?;
        }
        
        Ok(())
    }

    fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        use std::io::prelude::*;
        use flate2::Compression;
        use flate2::write::GzEncoder;

        let mut encoder = GzEncoder::new(Vec::new(), Compression::new(self.config.compression.level));
        encoder.write_all(data)
            .map_err(|e| Error::cache(format!("Failed to compress data: {}", e)))?;
        encoder.finish()
            .map_err(|e| Error::cache(format!("Failed to finish compression: {}", e)))
    }

    fn decompress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        use std::io::prelude::*;
        use flate2::read::GzDecoder;

        let mut decoder = GzDecoder::new(data);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)
            .map_err(|e| Error::cache(format!("Failed to decompress data: {}", e)))?;
        Ok(decompressed)
    }

    // Statistics recording methods

    async fn record_hit(&self, start_time: std::time::Instant) {
        let mut stats = self.stats.write().await;
        stats.hits += 1;
        stats.gets += 1;
        self.update_avg_time(&mut stats, start_time);
    }

    async fn record_miss(&self, start_time: std::time::Instant) {
        let mut stats = self.stats.write().await;
        stats.misses += 1;
        stats.gets += 1;
        self.update_avg_time(&mut stats, start_time);
    }

    async fn record_set(&self) {
        let mut stats = self.stats.write().await;
        stats.sets += 1;
    }

    async fn record_delete(&self) {
        let mut stats = self.stats.write().await;
        stats.deletes += 1;
    }

    async fn record_evictions(&self, count: u32) {
        let mut stats = self.stats.write().await;
        stats.evictions += count as u64;
    }

    fn update_avg_time(&self, stats: &mut CacheStats, start_time: std::time::Instant) {
        let duration_us = start_time.elapsed().as_micros() as f64;
        stats.avg_access_time_us = if stats.gets == 1 {
            duration_us
        } else {
            (stats.avg_access_time_us * (stats.gets - 1) as f64 + duration_us) / stats.gets as f64
        };
    }
}

/// Cached embedding service that uses the cache layer
pub struct CachedEmbeddingService<T> {
    inner: T,
    cache: Arc<CacheService>,
    ttl: u64,
}

impl<T> CachedEmbeddingService<T> {
    pub fn new(inner: T, cache: Arc<CacheService>, ttl: u64) -> Self {
        Self {
            inner,
            cache,
            ttl,
        }
    }

    pub async fn get_embedding(&self, text: &str) -> Result<Option<Vec<f32>>>
    where
        T: EmbeddingProvider,
    {
        let cache_key = format!("embedding:{}", sha256::digest(text));
        
        // Try cache first
        if let Some(embedding) = self.cache.get::<Vec<f32>>(&cache_key).await? {
            return Ok(Some(embedding));
        }

        // Generate embedding
        let embedding = self.inner.generate_embedding(text).await?;
        
        // Cache the result
        self.cache.set(&cache_key, &embedding, Some(self.ttl)).await?;
        
        Ok(Some(embedding))
    }
}

/// Trait for embedding providers to work with caching
#[async_trait::async_trait]
pub trait EmbeddingProvider {
    async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>>;
}

/// Get current timestamp in seconds
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Cache-specific error extension
impl Error {
    pub fn cache(message: String) -> Self {
        Error::internal(message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_memory_cache() {
        let config = CacheConfig {
            enabled: true,
            redis_url: None, // Disable Redis for this test
            ..Default::default()
        };

        let cache = CacheService::new(config).await.unwrap();

        // Test set/get
        cache.set("test_key", &"test_value".to_string(), None).await.unwrap();
        let value: Option<String> = cache.get("test_key").await.unwrap();
        assert_eq!(value, Some("test_value".to_string()));

        // Test miss
        let missing: Option<String> = cache.get("missing_key").await.unwrap();
        assert_eq!(missing, None);

        // Test delete
        cache.delete("test_key").await.unwrap();
        let deleted: Option<String> = cache.get("test_key").await.unwrap();
        assert_eq!(deleted, None);
    }

    #[tokio::test]
    async fn test_cache_expiration() {
        let config = CacheConfig {
            enabled: true,
            redis_url: None,
            default_ttl: 1, // 1 second
            memory_ttl: 1,
            ..Default::default()
        };

        let cache = CacheService::new(config).await.unwrap();

        cache.set("expire_key", &"expire_value".to_string(), Some(1)).await.unwrap();
        
        // Should be available immediately
        let value: Option<String> = cache.get("expire_key").await.unwrap();
        assert_eq!(value, Some("expire_value".to_string()));

        // Wait for expiration
        tokio::time::sleep(Duration::from_secs(2)).await;
        
        // Should be expired now
        let expired: Option<String> = cache.get("expire_key").await.unwrap();
        assert_eq!(expired, None);
    }

    #[tokio::test]
    async fn test_cache_stats() {
        let config = CacheConfig {
            enabled: true,
            redis_url: None,
            collect_stats: true,
            ..Default::default()
        };

        let cache = CacheService::new(config).await.unwrap();

        // Perform some operations
        cache.set("key1", &"value1".to_string(), None).await.unwrap();
        let _: Option<String> = cache.get("key1").await.unwrap(); // hit
        let _: Option<String> = cache.get("key2").await.unwrap(); // miss

        let stats = cache.stats().await;
        assert_eq!(stats.sets, 1);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.gets, 2);
        assert_eq!(stats.hit_rate(), 0.5);
    }

    #[test]
    fn test_cache_entry() {
        let entry = CacheEntry::new("test".to_string(), 60, 4);
        assert!(!entry.is_expired());
        assert_eq!(entry.access_count, 1);
        
        let mut entry_expired = CacheEntry::new("test".to_string(), 0, 4);
        // Simulate time passing
        std::thread::sleep(Duration::from_millis(10));
        assert!(entry_expired.is_expired());
    }
}

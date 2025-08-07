use crate::utils::{Error, Result};
use crate::performance::metrics::global_metrics;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Simple cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleCacheConfig {
    /// Whether caching is enabled
    pub enabled: bool,
    /// Default TTL for cache entries (in seconds)
    pub default_ttl: u64,
    /// Maximum number of entries in cache
    pub max_entries: usize,
}

impl Default for SimpleCacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_ttl: 3600, // 1 hour
            max_entries: 1000,
        }
    }
}

/// Cache entry with TTL
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheEntry<T> {
    value: T,
    expires_at: u64,
    created_at: u64,
    access_count: u32,
}

impl<T> CacheEntry<T> {
    fn new(value: T, ttl: u64) -> Self {
        let now = current_timestamp();
        Self {
            value,
            expires_at: now + ttl,
            created_at: now,
            access_count: 0,
        }
    }

    fn is_expired(&self) -> bool {
        current_timestamp() > self.expires_at
    }

    fn access(&mut self) {
        self.access_count += 1;
    }
}

/// Cache statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SimpleCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub entries: usize,
}

impl SimpleCacheStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// Simple in-memory cache service
pub struct SimpleCacheService {
    config: SimpleCacheConfig,
    storage: Arc<RwLock<HashMap<String, CacheEntry<Vec<u8>>>>>,
    stats: Arc<RwLock<SimpleCacheStats>>,
}

impl SimpleCacheService {
    /// Create a new simple cache service
    pub fn new(config: SimpleCacheConfig) -> Self {
        Self {
            config,
            storage: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(SimpleCacheStats::default())),
        }
    }

    /// Get a value from cache
    pub async fn get<T>(&self, key: &str) -> Result<Option<T>>
    where
        T: for<'de> Deserialize<'de>,
    {
        if !self.config.enabled {
            return Ok(None);
        }

        let mut storage = self.storage.write().await;
        
        if let Some(entry) = storage.get_mut(key) {
            if entry.is_expired() {
                storage.remove(key);
                self.record_miss().await;
                return Ok(None);
            }

            entry.access();
            
            // Deserialize the value
            let value = bincode::deserialize(&entry.value)
                .map_err(|e| Error::internal(format!("Cache deserialization error: {}", e)))?;
            
            self.record_hit().await;
            debug!("Cache hit: key={}", key);
            return Ok(Some(value));
        }

        self.record_miss().await;
        debug!("Cache miss: key={}", key);
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

        let serialized = bincode::serialize(value)
            .map_err(|e| Error::internal(format!("Cache serialization error: {}", e)))?;

        let ttl = ttl.unwrap_or(self.config.default_ttl);
        let entry = CacheEntry::new(serialized, ttl);

        let mut storage = self.storage.write().await;

        // Simple LRU eviction if cache is full
        if storage.len() >= self.config.max_entries && !storage.contains_key(key) {
            self.evict_oldest(&mut storage).await;
        }

        storage.insert(key.to_string(), entry);
        
        debug!("Cache set: key={}, ttl={}", key, ttl);
        Ok(())
    }

    /// Delete a value from cache
    pub async fn delete(&self, key: &str) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let mut storage = self.storage.write().await;
        storage.remove(key);
        
        debug!("Cache delete: key={}", key);
        Ok(())
    }

    /// Clear entire cache
    pub async fn clear(&self) -> Result<()> {
        let mut storage = self.storage.write().await;
        storage.clear();
        
        let mut stats = self.stats.write().await;
        *stats = SimpleCacheStats::default();
        
        info!("Cache cleared");
        Ok(())
    }

    /// Get cache statistics
    pub async fn stats(&self) -> SimpleCacheStats {
        let mut stats = self.stats.read().await.clone();
        
        let storage = self.storage.read().await;
        stats.entries = storage.len();
        
        stats
    }

    /// Perform cache maintenance (cleanup expired entries)
    pub async fn maintenance(&self) -> Result<u32> {
        if !self.config.enabled {
            return Ok(0);
        }

        let mut storage = self.storage.write().await;
        let mut removed = 0;
        
        // Remove expired entries
        let expired_keys: Vec<String> = storage
            .iter()
            .filter(|(_, entry)| entry.is_expired())
            .map(|(key, _)| key.clone())
            .collect();

        for key in expired_keys {
            storage.remove(&key);
            removed += 1;
        }

        if removed > 0 {
            debug!("Cache maintenance: removed {} expired entries", removed);
        }

        Ok(removed)
    }

    /// Evict the oldest entry (simple LRU)
    async fn evict_oldest(&self, storage: &mut HashMap<String, CacheEntry<Vec<u8>>>) {
        if let Some(oldest_key) = storage
            .iter()
            .min_by_key(|(_, entry)| entry.created_at)
            .map(|(key, _)| key.clone())
        {
            storage.remove(&oldest_key);
            debug!("Cache evicted oldest entry: key={}", oldest_key);
        }
    }

    async fn record_hit(&self) {
        let mut stats = self.stats.write().await;
        stats.hits += 1;
        
        // Update global metrics
        global_metrics().increment_counter("cache_hits_total");
    }

    async fn record_miss(&self) {
        let mut stats = self.stats.write().await;
        stats.misses += 1;
        
        // Update global metrics
        global_metrics().increment_counter("cache_misses_total");
    }
}

/// Cached embedding service using simple cache
pub struct SimpleCachedEmbeddingService<T> {
    inner: T,
    cache: Arc<SimpleCacheService>,
    ttl: u64,
}

impl<T> SimpleCachedEmbeddingService<T> {
    pub fn new(inner: T, cache: Arc<SimpleCacheService>, ttl: u64) -> Self {
        Self {
            inner,
            cache,
            ttl,
        }
    }

    pub async fn get_embedding(&self, text: &str) -> Result<Vec<f32>>
    where
        T: SimpleEmbeddingProvider,
    {
        let cache_key = format!("embedding:{}", sha256::digest(text));
        
        // Try cache first
        if let Some(embedding) = self.cache.get::<Vec<f32>>(&cache_key).await? {
            return Ok(embedding);
        }

        // Generate embedding
        let embedding = self.inner.generate_embedding(text).await?;
        
        // Cache the result
        self.cache.set(&cache_key, &embedding, Some(self.ttl)).await?;
        
        Ok(embedding)
    }
}

/// Simple embedding provider trait
#[async_trait::async_trait]
pub trait SimpleEmbeddingProvider {
    async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>>;
}

/// Get current timestamp in seconds
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_simple_cache_basic_operations() {
        let config = SimpleCacheConfig::default();
        let cache = SimpleCacheService::new(config);

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
        let config = SimpleCacheConfig {
            enabled: true,
            default_ttl: 1, // 1 second
            max_entries: 100,
        };
        let cache = SimpleCacheService::new(config);

        cache.set("expire_key", &"expire_value".to_string(), Some(1)).await.unwrap();
        
        // Should be available immediately
        let value: Option<String> = cache.get("expire_key").await.unwrap();
        assert_eq!(value, Some("expire_value".to_string()));

        // Wait for expiration
        sleep(Duration::from_secs(2)).await;
        
        // Should be expired now
        let expired: Option<String> = cache.get("expire_key").await.unwrap();
        assert_eq!(expired, None);
    }

    #[tokio::test]
    async fn test_cache_stats() {
        let config = SimpleCacheConfig::default();
        let cache = SimpleCacheService::new(config);

        // Perform some operations
        cache.set("key1", &"value1".to_string(), None).await.unwrap();
        let _: Option<String> = cache.get("key1").await.unwrap(); // hit
        let _: Option<String> = cache.get("key2").await.unwrap(); // miss

        let stats = cache.stats().await;
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.hit_rate(), 0.5);
    }

    #[tokio::test]
    async fn test_cache_maintenance() {
        let config = SimpleCacheConfig {
            enabled: true,
            default_ttl: 1, // 1 second
            max_entries: 100,
        };
        let cache = SimpleCacheService::new(config);

        // Add some entries that will expire
        cache.set("key1", &"value1".to_string(), Some(1)).await.unwrap();
        cache.set("key2", &"value2".to_string(), Some(1)).await.unwrap();
        
        // Wait for expiration
        sleep(Duration::from_secs(2)).await;
        
        // Run maintenance
        let removed = cache.maintenance().await.unwrap();
        assert_eq!(removed, 2);
        
        let stats = cache.stats().await;
        assert_eq!(stats.entries, 0);
    }

    #[tokio::test] 
    async fn test_cache_eviction() {
        let config = SimpleCacheConfig {
            enabled: true,
            default_ttl: 3600,
            max_entries: 2, // Small limit to test eviction
        };
        let cache = SimpleCacheService::new(config);

        // Fill cache beyond limit
        cache.set("key1", &"value1".to_string(), None).await.unwrap();
        cache.set("key2", &"value2".to_string(), None).await.unwrap();
        cache.set("key3", &"value3".to_string(), None).await.unwrap(); // Should evict oldest

        let stats = cache.stats().await;
        assert_eq!(stats.entries, 2); // Should be at max limit

        // The oldest entry should be evicted
        let value1: Option<String> = cache.get("key1").await.unwrap();
        assert_eq!(value1, None); // Should be evicted
        
        let value3: Option<String> = cache.get("key3").await.unwrap();
        assert_eq!(value3, Some("value3".to_string())); // Should exist
    }
}

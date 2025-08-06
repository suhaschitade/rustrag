use crate::utils::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::fs;

/// Configuration for embedding cache
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of embeddings to cache in memory
    pub max_memory_entries: usize,
    /// Maximum memory usage in bytes (approximate)
    pub max_memory_bytes: usize,
    /// Time-to-live for cached embeddings
    pub ttl_seconds: u64,
    /// Whether to enable persistent cache
    pub enable_persistent: bool,
    /// Directory for persistent cache files
    pub cache_dir: PathBuf,
    /// Whether to enable compression for persistent cache
    pub enable_compression: bool,
    /// Interval for background cache cleanup
    pub cleanup_interval_seconds: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_memory_entries: 10000,
            max_memory_bytes: 100 * 1024 * 1024, // 100MB
            ttl_seconds: 24 * 60 * 60, // 24 hours
            enable_persistent: true,
            cache_dir: PathBuf::from("cache/embeddings"),
            enable_compression: true,
            cleanup_interval_seconds: 300, // 5 minutes
        }
    }
}

/// Cache entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    pub embedding: Vec<f32>,
    pub created_at: u64,
    pub last_accessed: u64,
    pub access_count: u64,
    pub content_hash: u64,
    pub model_version: String,
}

impl CacheEntry {
    pub fn new(embedding: Vec<f32>, content_hash: u64, model_version: String) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            embedding,
            created_at: now,
            last_accessed: now,
            access_count: 1,
            content_hash,
            model_version,
        }
    }

    pub fn is_expired(&self, ttl_seconds: u64) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        now - self.created_at > ttl_seconds
    }

    pub fn touch(&mut self) {
        self.last_accessed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.access_count += 1;
    }

    pub fn estimated_size_bytes(&self) -> usize {
        // Rough estimation: 4 bytes per f32 + metadata overhead
        self.embedding.len() * 4 + 64
    }
}

/// Cache key for embeddings
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    pub content_hash: u64,
    pub model_name: String,
    pub model_version: String,
}

impl CacheKey {
    pub fn new(content: &str, model_name: &str, model_version: &str) -> Self {
        Self {
            content_hash: Self::hash_content(content),
            model_name: model_name.to_string(),
            model_version: model_version.to_string(),
        }
    }

    fn hash_content(content: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        hasher.finish()
    }
}

/// Cache statistics for monitoring
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub memory_entries: usize,
    pub memory_bytes: usize,
    pub persistent_entries: usize,
    pub hit_count: u64,
    pub miss_count: u64,
    pub eviction_count: u64,
    pub error_count: u64,
    pub last_cleanup: Option<Instant>,
    pub total_access_time_ms: u64,
    pub average_access_time_ms: f64,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hit_count + self.miss_count;
        if total == 0 {
            0.0
        } else {
            self.hit_count as f64 / total as f64
        }
    }

    pub fn update_access_time(&mut self, duration: Duration) {
        let ms = duration.as_millis() as u64;
        self.total_access_time_ms += ms;
        let total_accesses = self.hit_count + self.miss_count;
        if total_accesses > 0 {
            self.average_access_time_ms = self.total_access_time_ms as f64 / total_accesses as f64;
        }
    }
}

/// LRU (Least Recently Used) cache implementation
struct LruCache {
    entries: HashMap<CacheKey, CacheEntry>,
    access_order: VecDeque<CacheKey>,
    max_entries: usize,
    max_bytes: usize,
    current_bytes: usize,
}

impl LruCache {
    fn new(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            entries: HashMap::new(),
            access_order: VecDeque::new(),
            max_entries,
            max_bytes,
            current_bytes: 0,
        }
    }

    fn get(&mut self, key: &CacheKey) -> Option<&CacheEntry> {
        if self.entries.contains_key(key) {
            // Move to front (most recently used)
            self.move_to_front(key);
            
            // Now get the entry and touch it
            if let Some(entry) = self.entries.get_mut(key) {
                entry.touch();
            }
            
            // Return immutable reference
            self.entries.get(key)
        } else {
            None
        }
    }

    fn insert(&mut self, key: CacheKey, mut entry: CacheEntry) -> Option<Vec<CacheKey>> {
        let entry_size = entry.estimated_size_bytes();
        let mut evicted = Vec::new();

        // Remove existing entry if present
        if let Some(old_entry) = self.entries.remove(&key) {
            self.current_bytes -= old_entry.estimated_size_bytes();
            self.remove_from_access_order(&key);
        }

        // Evict entries if necessary
        while (self.entries.len() >= self.max_entries || 
               self.current_bytes + entry_size > self.max_bytes) && 
              !self.access_order.is_empty() {
            
            if let Some(evict_key) = self.access_order.pop_back() {
                if let Some(evict_entry) = self.entries.remove(&evict_key) {
                    self.current_bytes -= evict_entry.estimated_size_bytes();
                    evicted.push(evict_key);
                }
            }
        }

        // Insert new entry
        entry.touch();
        self.entries.insert(key.clone(), entry);
        self.access_order.push_front(key);
        self.current_bytes += entry_size;

        if evicted.is_empty() { None } else { Some(evicted) }
    }

    fn remove(&mut self, key: &CacheKey) -> Option<CacheEntry> {
        if let Some(entry) = self.entries.remove(key) {
            self.current_bytes -= entry.estimated_size_bytes();
            self.remove_from_access_order(key);
            Some(entry)
        } else {
            None
        }
    }

    fn move_to_front(&mut self, key: &CacheKey) {
        self.remove_from_access_order(key);
        self.access_order.push_front(key.clone());
    }

    fn remove_from_access_order(&mut self, key: &CacheKey) {
        if let Some(pos) = self.access_order.iter().position(|k| k == key) {
            self.access_order.remove(pos);
        }
    }

    fn cleanup_expired(&mut self, ttl_seconds: u64) -> Vec<CacheKey> {
        let mut expired = Vec::new();
        let keys_to_check: Vec<_> = self.entries.keys().cloned().collect();
        
        for key in keys_to_check {
            if let Some(entry) = self.entries.get(&key) {
                if entry.is_expired(ttl_seconds) {
                    expired.push(key);
                }
            }
        }

        for key in &expired {
            self.remove(key);
        }

        expired
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn current_memory_usage(&self) -> usize {
        self.current_bytes
    }
}

/// High-performance embedding cache with multiple storage layers
pub struct EmbeddingCache {
    config: CacheConfig,
    memory_cache: Arc<RwLock<LruCache>>,
    stats: Arc<RwLock<CacheStats>>,
    cleanup_handle: Option<tokio::task::JoinHandle<()>>,
}

impl EmbeddingCache {
    /// Create a new embedding cache with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(CacheConfig::default())
    }

    /// Create a new embedding cache with custom configuration
    pub fn with_config(config: CacheConfig) -> Result<Self> {
        let memory_cache = Arc::new(RwLock::new(LruCache::new(
            config.max_memory_entries,
            config.max_memory_bytes,
        )));

        let stats = Arc::new(RwLock::new(CacheStats::default()));

        // Ensure cache directory exists
        if config.enable_persistent {
            std::fs::create_dir_all(&config.cache_dir)
                .map_err(|e| Error::internal(format!("Failed to create cache directory: {}", e)))?;
        }

        let mut cache = Self {
            config,
            memory_cache,
            stats,
            cleanup_handle: None,
        };

        // Start background cleanup task
        cache.start_cleanup_task();

        Ok(cache)
    }

    /// Get embedding from cache
    pub async fn get(&self, content: &str, model_name: &str, model_version: &str) -> Option<Vec<f32>> {
        let start_time = Instant::now();
        let key = CacheKey::new(content, model_name, model_version);

        // Try memory cache first
        if let Ok(mut cache) = self.memory_cache.write() {
            if let Some(entry) = cache.get(&key) {
                // Update statistics
                if let Ok(mut stats) = self.stats.write() {
                    stats.hit_count += 1;
                    stats.update_access_time(start_time.elapsed());
                }
                return Some(entry.embedding.clone());
            }
        }

        // Try persistent cache if enabled
        if self.config.enable_persistent {
            if let Ok(embedding) = self.get_from_persistent(&key).await {
                // Cache hit in persistent storage - load into memory
                let entry = CacheEntry::new(
                    embedding.clone(),
                    key.content_hash,
                    model_version.to_string(),
                );

                if let Ok(mut cache) = self.memory_cache.write() {
                    cache.insert(key, entry);
                }

                // Update statistics
                if let Ok(mut stats) = self.stats.write() {
                    stats.hit_count += 1;
                    stats.update_access_time(start_time.elapsed());
                }

                return Some(embedding);
            }
        }

        // Cache miss
        if let Ok(mut stats) = self.stats.write() {
            stats.miss_count += 1;
            stats.update_access_time(start_time.elapsed());
        }

        None
    }

    /// Put embedding into cache
    pub async fn put(
        &self,
        content: &str,
        model_name: &str,
        model_version: &str,
        embedding: Vec<f32>,
    ) -> Result<()> {
        let key = CacheKey::new(content, model_name, model_version);
        let entry = CacheEntry::new(embedding.clone(), key.content_hash, model_version.to_string());

        // Insert into memory cache
        let evicted = if let Ok(mut cache) = self.memory_cache.write() {
            cache.insert(key.clone(), entry.clone())
        } else {
            return Err(Error::internal("Failed to acquire cache lock"));
        };

        // Update eviction statistics
        if let Some(evicted_keys) = evicted {
            if let Ok(mut stats) = self.stats.write() {
                stats.eviction_count += evicted_keys.len() as u64;
            }
        }

        // Store in persistent cache if enabled
        if self.config.enable_persistent {
            if let Err(e) = self.put_to_persistent(&key, &entry).await {
                tracing::warn!("Failed to store embedding in persistent cache: {}", e);
                if let Ok(mut stats) = self.stats.write() {
                    stats.error_count += 1;
                }
            }
        }

        Ok(())
    }

    /// Remove embedding from cache
    pub async fn remove(&self, content: &str, model_name: &str, model_version: &str) -> Result<()> {
        let key = CacheKey::new(content, model_name, model_version);

        // Remove from memory cache
        if let Ok(mut cache) = self.memory_cache.write() {
            cache.remove(&key);
        }

        // Remove from persistent cache if enabled
        if self.config.enable_persistent {
            if let Err(e) = self.remove_from_persistent(&key).await {
                tracing::warn!("Failed to remove embedding from persistent cache: {}", e);
            }
        }

        Ok(())
    }

    /// Clear all cached embeddings
    pub async fn clear(&self) -> Result<()> {
        // Clear memory cache
        if let Ok(mut cache) = self.memory_cache.write() {
            *cache = LruCache::new(self.config.max_memory_entries, self.config.max_memory_bytes);
        }

        // Clear persistent cache if enabled
        if self.config.enable_persistent {
            if let Err(e) = fs::remove_dir_all(&self.config.cache_dir).await {
                tracing::warn!("Failed to clear persistent cache: {}", e);
            }
            fs::create_dir_all(&self.config.cache_dir).await
                .map_err(|e| Error::internal(format!("Failed to recreate cache directory: {}", e)))?;
        }

        // Reset statistics
        if let Ok(mut stats) = self.stats.write() {
            *stats = CacheStats::default();
        }

        Ok(())
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> CacheStats {
        let mut stats = if let Ok(stats_guard) = self.stats.read() {
            stats_guard.clone()
        } else {
            tracing::warn!("Failed to acquire stats lock");
            CacheStats::default()
        };

        // Update current memory statistics
        if let Ok(cache) = self.memory_cache.read() {
            stats.memory_entries = cache.len();
            stats.memory_bytes = cache.current_memory_usage();
        }

        stats
    }

    /// Optimize cache performance by cleaning up expired entries
    pub async fn optimize(&self) -> Result<()> {
        let cleanup_count = if let Ok(mut cache) = self.memory_cache.write() {
            let expired = cache.cleanup_expired(self.config.ttl_seconds);
            expired.len()
        } else {
            0
        };

        if cleanup_count > 0 {
            tracing::info!("Cleaned up {} expired cache entries", cleanup_count);
        }

        // Update statistics
        if let Ok(mut stats) = self.stats.write() {
            stats.last_cleanup = Some(Instant::now());
        }

        Ok(())
    }

    /// Start background cleanup task
    fn start_cleanup_task(&mut self) {
        let memory_cache = Arc::clone(&self.memory_cache);
        let stats = Arc::clone(&self.stats);
        let ttl_seconds = self.config.ttl_seconds;
        let cleanup_interval = Duration::from_secs(self.config.cleanup_interval_seconds);

        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(cleanup_interval);
            
            loop {
                interval.tick().await;
                
                // Cleanup expired entries
                let cleanup_count = if let Ok(mut cache) = memory_cache.write() {
                    let expired = cache.cleanup_expired(ttl_seconds);
                    expired.len()
                } else {
                    0
                };

                if cleanup_count > 0 {
                    tracing::debug!("Background cleanup removed {} expired entries", cleanup_count);
                }

                // Update statistics
                if let Ok(mut stats_guard) = stats.write() {
                    stats_guard.last_cleanup = Some(Instant::now());
                }
            }
        });

        self.cleanup_handle = Some(handle);
    }

    /// Get cache entry from persistent storage
    async fn get_from_persistent(&self, key: &CacheKey) -> Result<Vec<f32>> {
        let file_path = self.persistent_file_path(key);
        
        if !file_path.exists() {
            return Err(Error::not_found("Cache entry not found in persistent storage"));
        }

        let data = fs::read(&file_path).await
            .map_err(|e| Error::internal(format!("Failed to read cache file: {}", e)))?;

        let entry: CacheEntry = if self.config.enable_compression {
            // Simple compression using bincode (in production, consider zstd or lz4)
            bincode::deserialize(&data)
                .map_err(|e| Error::internal(format!("Failed to deserialize cache entry: {}", e)))?
        } else {
            serde_json::from_slice(&data)
                .map_err(|e| Error::internal(format!("Failed to deserialize cache entry: {}", e)))?
        };

        // Check if entry is expired
        if entry.is_expired(self.config.ttl_seconds) {
            // Remove expired file
            if let Err(e) = fs::remove_file(&file_path).await {
                tracing::warn!("Failed to remove expired cache file: {}", e);
            }
            return Err(Error::not_found("Cache entry expired"));
        }

        Ok(entry.embedding)
    }

    /// Store cache entry to persistent storage
    async fn put_to_persistent(&self, key: &CacheKey, entry: &CacheEntry) -> Result<()> {
        let file_path = self.persistent_file_path(key);
        
        if let Some(parent) = file_path.parent() {
            fs::create_dir_all(parent).await
                .map_err(|e| Error::internal(format!("Failed to create cache subdirectory: {}", e)))?;
        }

        let data = if self.config.enable_compression {
            bincode::serialize(entry)
                .map_err(|e| Error::internal(format!("Failed to serialize cache entry: {}", e)))?
        } else {
            serde_json::to_vec(entry)
                .map_err(|e| Error::internal(format!("Failed to serialize cache entry: {}", e)))?
        };

        fs::write(&file_path, data).await
            .map_err(|e| Error::internal(format!("Failed to write cache file: {}", e)))?;

        Ok(())
    }

    /// Remove cache entry from persistent storage
    async fn remove_from_persistent(&self, key: &CacheKey) -> Result<()> {
        let file_path = self.persistent_file_path(key);
        
        if file_path.exists() {
            fs::remove_file(&file_path).await
                .map_err(|e| Error::internal(format!("Failed to remove cache file: {}", e)))?;
        }

        Ok(())
    }

    /// Generate file path for persistent cache entry
    fn persistent_file_path(&self, key: &CacheKey) -> PathBuf {
        let filename = format!(
            "{}_{}_{}_{}.cache",
            key.content_hash,
            key.model_name.replace('/', "_"),
            key.model_version.replace('/', "_"),
            if self.config.enable_compression { "bin" } else { "json" }
        );
        
        self.config.cache_dir.join(filename)
    }
}

impl Drop for EmbeddingCache {
    fn drop(&mut self) {
        if let Some(handle) = self.cleanup_handle.take() {
            handle.abort();
        }
    }
}

/// Cache-aware embedding service wrapper
pub struct CachedEmbeddingService<T> {
    inner: T,
    cache: EmbeddingCache,
    model_name: String,
    model_version: String,
}

impl<T> CachedEmbeddingService<T> {
    pub fn new(inner: T, cache: EmbeddingCache, model_name: String, model_version: String) -> Self {
        Self {
            inner,
            cache,
            model_name,
            model_version,
        }
    }

    pub async fn generate_embedding_cached(&self, text: &str) -> Result<Vec<f32>>
    where
        T: EmbeddingGenerator,
    {
        // Try cache first
        if let Some(embedding) = self.cache.get(text, &self.model_name, &self.model_version).await {
            return Ok(embedding);
        }

        // Generate embedding
        let embedding = self.inner.generate_embedding(text).await?;

        // Cache the result
        self.cache.put(text, &self.model_name, &self.model_version, embedding.clone()).await?;

        Ok(embedding)
    }

    pub async fn generate_embeddings_cached(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>
    where
        T: EmbeddingGenerator,
    {
        let mut results = Vec::with_capacity(texts.len());
        let mut uncached_texts = Vec::new();
        let mut uncached_indices = Vec::new();

        // Check cache for each text
        for (i, text) in texts.iter().enumerate() {
            if let Some(embedding) = self.cache.get(text, &self.model_name, &self.model_version).await {
                results.push(Some(embedding));
            } else {
                results.push(None);
                uncached_texts.push(text.clone());
                uncached_indices.push(i);
            }
        }

        // Generate embeddings for uncached texts
        if !uncached_texts.is_empty() {
            let new_embeddings = self.inner.generate_embeddings(&uncached_texts).await?;
            
            // Cache and store results
            for (idx, embedding) in uncached_indices.into_iter().zip(new_embeddings.into_iter()) {
                self.cache.put(&texts[idx], &self.model_name, &self.model_version, embedding.clone()).await?;
                results[idx] = Some(embedding);
            }
        }

        // Convert to final result
        Ok(results.into_iter().map(|opt| opt.unwrap()).collect())
    }

    pub fn get_cache_stats(&self) -> CacheStats {
        self.cache.get_stats()
    }

    pub async fn clear_cache(&self) -> Result<()> {
        self.cache.clear().await
    }
}

/// Trait for embedding generation (to be implemented by actual embedding services)
pub trait EmbeddingGenerator {
    async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>>;
    async fn generate_embeddings(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cache_basic_operations() {
        let cache = EmbeddingCache::new().unwrap();
        
        let embedding = vec![0.1, 0.2, 0.3];
        
        // Test put and get
        cache.put("test content", "model", "v1", embedding.clone()).await.unwrap();
        let retrieved = cache.get("test content", "model", "v1").await;
        
        assert_eq!(retrieved, Some(embedding));
        
        // Test cache miss
        let miss = cache.get("other content", "model", "v1").await;
        assert_eq!(miss, None);
    }

    #[tokio::test]
    async fn test_cache_expiration() {
        let config = CacheConfig {
            ttl_seconds: 1, // 1 second TTL
            ..Default::default()
        };
        
        let cache = EmbeddingCache::with_config(config).unwrap();
        let embedding = vec![0.1, 0.2, 0.3];
        
        cache.put("test content", "model", "v1", embedding.clone()).await.unwrap();
        
        // Should be available immediately
        let retrieved = cache.get("test content", "model", "v1").await;
        assert_eq!(retrieved, Some(embedding));
        
        // Wait for expiration
        tokio::time::sleep(Duration::from_secs(2)).await;
        
        // Should be expired now
        cache.optimize().await.unwrap();
        let expired = cache.get("test content", "model", "v1").await;
        assert_eq!(expired, None);
    }

    #[tokio::test]
    async fn test_cache_statistics() {
        let cache = EmbeddingCache::new().unwrap();
        let embedding = vec![0.1, 0.2, 0.3];
        
        cache.put("test1", "model", "v1", embedding.clone()).await.unwrap();
        cache.put("test2", "model", "v1", embedding.clone()).await.unwrap();
        
        // Generate hits and misses
        cache.get("test1", "model", "v1").await;
        cache.get("test2", "model", "v1").await;
        cache.get("test3", "model", "v1").await; // miss
        
        let stats = cache.get_stats();
        assert_eq!(stats.hit_count, 2);
        assert_eq!(stats.miss_count, 1);
        assert!(stats.hit_rate() > 0.5);
    }
}

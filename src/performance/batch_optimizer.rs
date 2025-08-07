use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use tokio::time::timeout;

/// Configuration for batch processing optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchOptimizerConfig {
    /// Enable batch optimization
    pub enabled: bool,
    /// Maximum batch size for processing
    pub max_batch_size: usize,
    /// Maximum time to wait for batch to fill (milliseconds)
    pub batch_timeout_ms: u64,
    /// Maximum memory usage before forcing cleanup (MB)
    pub max_memory_mb: usize,
    /// Enable memory monitoring
    pub enable_memory_monitoring: bool,
    /// Cleanup interval in seconds
    pub cleanup_interval_seconds: u64,
}

impl Default for BatchOptimizerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_batch_size: 50,
            batch_timeout_ms: 100,
            max_memory_mb: 256,
            enable_memory_monitoring: true,
            cleanup_interval_seconds: 30,
        }
    }
}

/// Statistics for batch processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchStats {
    pub total_requests: u64,
    pub batched_requests: u64,
    pub individual_requests: u64,
    pub total_batches: u64,
    pub average_batch_size: f64,
    pub memory_cleanups: u64,
    pub current_memory_usage_mb: usize,
    pub peak_memory_usage_mb: usize,
    pub batch_efficiency: f64, // batched vs individual
}

impl BatchStats {
    pub fn new() -> Self {
        Self {
            total_requests: 0,
            batched_requests: 0,
            individual_requests: 0,
            total_batches: 0,
            average_batch_size: 0.0,
            memory_cleanups: 0,
            current_memory_usage_mb: 0,
            peak_memory_usage_mb: 0,
            batch_efficiency: 0.0,
        }
    }

    pub fn update_batch_efficiency(&mut self) {
        if self.total_requests > 0 {
            self.batch_efficiency = (self.batched_requests as f64) / (self.total_requests as f64) * 100.0;
        }
    }

    pub fn update_average_batch_size(&mut self) {
        if self.total_batches > 0 {
            self.average_batch_size = (self.batched_requests as f64) / (self.total_batches as f64);
        }
    }
}

/// A pending request in the batch queue
#[derive(Debug)]
struct PendingRequest<T, R> {
    pub request: T,
    pub response_sender: tokio::sync::oneshot::Sender<Result<R, String>>,
    pub timestamp: Instant,
}

/// Simple memory tracker for monitoring usage
#[derive(Debug)]
struct MemoryTracker {
    pub allocated_mb: usize,
    pub peak_mb: usize,
    pub last_cleanup: Instant,
}

impl MemoryTracker {
    pub fn new() -> Self {
        Self {
            allocated_mb: 0,
            peak_mb: 0,
            last_cleanup: Instant::now(),
        }
    }

    pub fn allocate(&mut self, mb: usize) {
        self.allocated_mb += mb;
        if self.allocated_mb > self.peak_mb {
            self.peak_mb = self.allocated_mb;
        }
    }

    pub fn deallocate(&mut self, mb: usize) {
        if self.allocated_mb >= mb {
            self.allocated_mb -= mb;
        } else {
            self.allocated_mb = 0;
        }
    }

    pub fn needs_cleanup(&self, max_mb: usize, cleanup_interval: Duration) -> bool {
        self.allocated_mb > max_mb || self.last_cleanup.elapsed() > cleanup_interval
    }

    pub fn cleanup(&mut self) {
        // Simulate garbage collection/cleanup
        self.allocated_mb = (self.allocated_mb as f64 * 0.8) as usize; // 20% reduction
        self.last_cleanup = Instant::now();
    }
}

/// Main batch optimizer for managing request batching and memory
#[derive(Clone)]
pub struct BatchOptimizer<T, R>
where
    T: Clone + Send + Sync + 'static,
    R: Clone + Send + Sync + 'static,
{
    config: BatchOptimizerConfig,
    stats: Arc<RwLock<BatchStats>>,
    memory_tracker: Arc<RwLock<MemoryTracker>>,
    pending_requests: Arc<RwLock<VecDeque<PendingRequest<T, R>>>>,
    batch_processor: Arc<dyn Fn(Vec<T>) -> Result<Vec<R>, String> + Send + Sync>,
}

impl<T, R> BatchOptimizer<T, R>
where
    T: Clone + Send + Sync + 'static,
    R: Clone + Send + Sync + 'static,
{
    /// Create a new batch optimizer
    pub fn new<F>(config: BatchOptimizerConfig, batch_processor: F) -> Self
    where
        F: Fn(Vec<T>) -> Result<Vec<R>, String> + Send + Sync + 'static,
    {
        Self {
            config,
            stats: Arc::new(RwLock::new(BatchStats::new())),
            memory_tracker: Arc::new(RwLock::new(MemoryTracker::new())),
            pending_requests: Arc::new(RwLock::new(VecDeque::new())),
            batch_processor: Arc::new(batch_processor),
        }
    }

    /// Process a single request, potentially batching it with others
    pub async fn process_request(&self, request: T) -> Result<R, String> {
        if !self.config.enabled {
            // Process immediately if batching is disabled
            return self.process_individual(request).await;
        }

        // Check memory usage first
        self.maybe_cleanup_memory().await;

        let (tx, rx) = tokio::sync::oneshot::channel();
        
        // Add to pending queue
        {
            let mut pending = self.pending_requests.write().unwrap();
            pending.push_back(PendingRequest {
                request: request.clone(),
                response_sender: tx,
                timestamp: Instant::now(),
            });

            // Update memory tracking (simulate request memory usage)
            if self.config.enable_memory_monitoring {
                let mut tracker = self.memory_tracker.write().unwrap();
                tracker.allocate(1); // 1MB per request approximation
            }
        }

        // Try to process batch if conditions are met
        self.maybe_process_batch().await;

        // Wait for response or timeout
        let timeout_duration = Duration::from_millis(self.config.batch_timeout_ms * 2);
        match timeout(timeout_duration, rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err("Response channel closed".to_string()),
            Err(_) => {
                // Timeout - process individually
                self.process_individual(request).await
            }
        }
    }

    /// Process an individual request without batching
    async fn process_individual(&self, request: T) -> Result<R, String> {
        let batch_results = (self.batch_processor)(vec![request])?;
        
        // Update stats
        {
            let mut stats = self.stats.write().unwrap();
            stats.total_requests += 1;
            stats.individual_requests += 1;
            stats.update_batch_efficiency();
        }

        batch_results.into_iter().next()
            .ok_or_else(|| "No result from batch processor".to_string())
    }

    /// Check if batch should be processed based on size or timeout
    async fn maybe_process_batch(&self) {
        let should_process = {
            let pending = self.pending_requests.read().unwrap();
            let batch_size = pending.len();
            let oldest_timestamp = pending.front().map(|r| r.timestamp);
            
            batch_size >= self.config.max_batch_size ||
            (batch_size > 0 && 
             oldest_timestamp.map_or(false, |ts| 
                ts.elapsed() >= Duration::from_millis(self.config.batch_timeout_ms)))
        };

        if should_process {
            self.process_current_batch().await;
        }
    }

    /// Process all pending requests as a batch
    async fn process_current_batch(&self) {
        let batch_requests = {
            let mut pending = self.pending_requests.write().unwrap();
            let batch_size = std::cmp::min(pending.len(), self.config.max_batch_size);
            
            if batch_size == 0 {
                return;
            }

            let mut batch = Vec::with_capacity(batch_size);
            for _ in 0..batch_size {
                if let Some(req) = pending.pop_front() {
                    batch.push(req);
                }
            }
            batch
        };

        if batch_requests.is_empty() {
            return;
        }

        // Extract requests for processing
        let requests: Vec<T> = batch_requests.iter()
            .map(|pr| pr.request.clone())
            .collect();

        // Process the batch
        let batch_size = requests.len();
        let results = (self.batch_processor)(requests);

        // Send responses back
        match results {
            Ok(responses) => {
                for (req, response) in batch_requests.into_iter().zip(responses.into_iter()) {
                    let _ = req.response_sender.send(Ok(response));
                }

                // Update stats
                {
                    let mut stats = self.stats.write().unwrap();
                    stats.total_requests += batch_size as u64;
                    stats.batched_requests += batch_size as u64;
                    stats.total_batches += 1;
                    stats.update_average_batch_size();
                    stats.update_batch_efficiency();
                }
            }
            Err(error) => {
                for req in batch_requests {
                    let _ = req.response_sender.send(Err(error.clone()));
                }

                // Update stats for failed batch
                {
                    let mut stats = self.stats.write().unwrap();
                    stats.total_requests += batch_size as u64;
                    stats.individual_requests += batch_size as u64;
                }
            }
        }

        // Update memory tracking
        if self.config.enable_memory_monitoring {
            let mut tracker = self.memory_tracker.write().unwrap();
            tracker.deallocate(batch_size); // Release memory for processed requests
        }
    }

    /// Check memory usage and cleanup if needed
    async fn maybe_cleanup_memory(&self) {
        if !self.config.enable_memory_monitoring {
            return;
        }

        let needs_cleanup = {
            let tracker = self.memory_tracker.read().unwrap();
            tracker.needs_cleanup(
                self.config.max_memory_mb,
                Duration::from_secs(self.config.cleanup_interval_seconds)
            )
        };

        if needs_cleanup {
            self.cleanup_memory().await;
        }
    }

    /// Perform memory cleanup
    async fn cleanup_memory(&self) {
        {
            let mut tracker = self.memory_tracker.write().unwrap();
            tracker.cleanup();
        }

        // Update stats
        {
            let mut stats = self.stats.write().unwrap();
            stats.memory_cleanups += 1;
            
            let tracker = self.memory_tracker.read().unwrap();
            stats.current_memory_usage_mb = tracker.allocated_mb;
            stats.peak_memory_usage_mb = tracker.peak_mb;
        }
    }

    /// Get current statistics
    pub fn get_stats(&self) -> BatchStats {
        let mut stats = self.stats.read().unwrap().clone();
        
        if self.config.enable_memory_monitoring {
            let tracker = self.memory_tracker.read().unwrap();
            stats.current_memory_usage_mb = tracker.allocated_mb;
            stats.peak_memory_usage_mb = tracker.peak_mb;
        }
        
        stats
    }

    /// Get current configuration
    pub fn get_config(&self) -> &BatchOptimizerConfig {
        &self.config
    }

    /// Force process any pending requests
    pub async fn flush_pending(&self) {
        self.process_current_batch().await;
    }
}

/// Helper function to create a simple text processing batch optimizer
pub fn create_text_batch_optimizer(
    config: BatchOptimizerConfig
) -> BatchOptimizer<String, String> {
    BatchOptimizer::new(config, |batch: Vec<String>| {
        // Simple example: convert to uppercase and add batch info
        let results: Vec<String> = batch.into_iter()
            .enumerate()
            .map(|(i, text)| format!("BATCH[{}]: {}", i, text.to_uppercase()))
            .collect();
        Ok(results)
    })
}

/// Helper function to create an embedding batch optimizer
pub fn create_embedding_batch_optimizer(
    config: BatchOptimizerConfig
) -> BatchOptimizer<String, Vec<f32>> {
    BatchOptimizer::new(config, |batch: Vec<String>| {
        // Simple example: generate mock embeddings
        let results: Vec<Vec<f32>> = batch.into_iter()
            .map(|text| {
                // Mock embedding generation based on text length and hash
                let len = text.len() as f32;
                let hash = text.chars().map(|c| c as u32).sum::<u32>() as f32;
                vec![len / 100.0, hash / 10000.0, 0.5, -0.2, 0.8]
            })
            .collect();
        Ok(results)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::future;

    #[tokio::test]
    async fn test_batch_optimizer_individual() {
        let config = BatchOptimizerConfig {
            enabled: false,
            ..Default::default()
        };
        
        let optimizer = create_text_batch_optimizer(config);
        let result = optimizer.process_request("hello world".to_string()).await.unwrap();
        assert!(result.contains("HELLO WORLD"));
        
        let stats = optimizer.get_stats();
        assert_eq!(stats.individual_requests, 1);
        assert_eq!(stats.batched_requests, 0);
    }

    #[tokio::test]
    async fn test_batch_optimizer_batching() {
        let config = BatchOptimizerConfig {
            enabled: true,
            max_batch_size: 2,
            batch_timeout_ms: 50,
            ..Default::default()
        };
        
        let optimizer = create_text_batch_optimizer(config);
        
        // Send two requests simultaneously
        let handle1 = tokio::spawn({
            let opt = optimizer.clone();
            async move { opt.process_request("hello".to_string()).await }
        });
        
        let handle2 = tokio::spawn({
            let optimizer = optimizer.clone();
            async move { optimizer.process_request("world".to_string()).await }
        });

        let results = futures::try_join!(handle1, handle2);
        assert!(results.is_ok());

        // Allow some time for stats to update
        tokio::time::sleep(Duration::from_millis(10)).await;
        
        let stats = optimizer.get_stats();
        assert_eq!(stats.total_requests, 2);
        println!("Stats: {:?}", stats);
    }

    #[tokio::test]
    async fn test_memory_monitoring() {
        let config = BatchOptimizerConfig {
            enabled: true,
            enable_memory_monitoring: true,
            max_memory_mb: 5,
            ..Default::default()
        };
        
        let optimizer = create_text_batch_optimizer(config);
        
        // Process several requests to trigger memory usage
        for i in 0..10 {
            let _ = optimizer.process_request(format!("request_{}", i)).await;
        }
        
        let stats = optimizer.get_stats();
        assert!(stats.current_memory_usage_mb > 0 || stats.memory_cleanups > 0);
    }
}

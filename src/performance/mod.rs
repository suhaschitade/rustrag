pub mod cache;
pub mod simple_cache;
pub mod db_optimizer;
pub mod batch_optimizer;
pub mod connection_pool;
pub mod metrics;

pub use cache::{
    CacheService, CacheConfig, CacheStats, CacheLayer, CacheEvictionPolicy,
    CacheCompressionConfig, CachedEmbeddingService, EmbeddingProvider,
};

pub use simple_cache::{
    SimpleCacheService, SimpleCacheConfig, SimpleCacheStats,
    SimpleCachedEmbeddingService, SimpleEmbeddingProvider,
};

pub use db_optimizer::{
    DbQueryOptimizer, DbOptimizerConfig, QueryStats, SimpleQueryBuilder,
};

pub use batch_optimizer::{
    BatchOptimizer, BatchOptimizerConfig, BatchStats,
    create_text_batch_optimizer, create_embedding_batch_optimizer,
};

pub use connection_pool::{
    ConnectionPool, ConnectionPoolConfig, ConnectionPoolStats, PooledConnection,
    ConnectionFactory, ConnectionStatus, MockConnection, MockConnectionFactory,
    create_mock_connection_pool,
};

pub use metrics::{
    MetricsRegistry, MetricValue, MetricType, MetricsSummary,
    Counter, Gauge, Histogram, HistogramData, HistogramBucket,
    global_metrics,
};

// Re-export main performance components
pub use cache::CacheService as PerformanceCacheService;
pub use simple_cache::SimpleCacheService as SimplePerformanceCacheService;
pub use batch_optimizer::BatchOptimizer as PerformanceBatchOptimizer;
pub use connection_pool::ConnectionPool as PerformanceConnectionPool;

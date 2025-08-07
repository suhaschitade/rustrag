pub mod content_validator;
pub mod document_formats;
pub mod document_processor;
pub mod embeddings;
pub mod retrieval;
pub mod query_service;
pub mod hybrid_search;
pub mod query_expansion;
// pub mod enhanced_query_processor; // TODO: Implement when QueryProcessor types are available
pub mod relevance_scorer;
pub mod enhanced_retrieval;
pub mod reranking;

// Re-exports for easier access
pub mod similarity_search;
pub mod embedding_cache;
pub mod generation;

pub use content_validator::{ContentValidator, ValidationConfig, ValidationResult};
pub use document_processor::{DocumentProcessor, ChunkingStrategy};
pub use document_formats::DocumentFormatProcessor;
pub use embeddings::{EmbeddingService, EmbeddingProvider, EmbeddingConfig, EmbeddingServiceBuilder};
pub use embedding_cache::{
    EmbeddingCache, CacheConfig, CacheStats, CachedEmbeddingService, EmbeddingGenerator,
};
pub use generation::{GenerationService, GenerationConfig};
pub use query_service::{QueryService, QueryServiceConfig, QueryServiceStatistics};
pub use retrieval::{RetrievalService, RetrievalConfig, RankedChunk, RetrievalStats};
pub use similarity_search::{
    SimilaritySearchEngine, SearchConfig, SearchFilters, SearchResult, SearchStats,
    DistanceMetric,
};
pub use hybrid_search::{
    HybridSearchConfig, HybridSearchScorer, HybridSearchResult, DocumentStats,
    KeywordSearchResult, build_document_stats, calculate_cosine_similarity,
};
pub use query_expansion::{
    QueryExpansionService, QueryExpansionConfig, ExpansionResult, NegationInfo, NegationHandling,
};
// TODO: Implement when QueryProcessor types are available
// pub use enhanced_query_processor::{
//     EnhancedQueryProcessor, EnhancedQueryProcessorConfig, EnhancedProcessedQuery, QueryProcessingStats,
// };
pub use relevance_scorer::{
    RelevanceScorer, RelevanceScorerFactory, RelevanceConfig, RelevanceScore, RelevanceExplanation,
    QueryAnalysis, QueryType as RelevanceQueryType, QueryIntent, QueryComplexity,
    NamedEntity, EntityType, TemporalContext, DocumentQuality,
};
pub use enhanced_retrieval::{
    EnhancedRetrievalService, EnhancedRetrievalServiceBuilder, EnhancedRetrievalConfig,
    EnhancedRankedChunk, EnhancedRetrievalStats, RetrievalStrategy,
};
pub use reranking::{
    RerankingService, RerankingConfig, RerankingStrategy, RerankedResult,
    RerankingScoreComponents, RerankingStats, DiversityConfig, QualityConfig, TemporalConfig,
};

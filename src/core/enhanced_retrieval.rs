use crate::models::{Query, DocumentChunk};
use crate::core::{
    EmbeddingService, RelevanceScorer, RelevanceScorerFactory, QueryAnalysis, RelevanceScore, RelevanceConfig,
    HybridSearchScorer, HybridSearchConfig, build_document_stats,
};
use crate::storage::VectorStore;
use crate::utils::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Configuration for enhanced retrieval with advanced relevance scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedRetrievalConfig {
    /// Default maximum number of chunks to retrieve
    pub default_max_chunks: usize,
    /// Default similarity threshold for initial filtering
    pub default_similarity_threshold: f32,
    /// Relevance scoring configuration
    pub relevance_config: RelevanceConfig,
    /// Whether to enable multi-stage ranking
    pub enable_multi_stage_ranking: bool,
    /// Number of candidates to retrieve in first stage (should be > max_chunks)
    pub first_stage_candidates: usize,
    /// Whether to enable adaptive scoring based on query characteristics
    pub enable_adaptive_scoring: bool,
    /// Whether to enable result explanation generation
    pub enable_explanations: bool,
    /// Minimum relevance score for including results
    pub min_relevance_score: f32,
}

impl Default for EnhancedRetrievalConfig {
    fn default() -> Self {
        Self {
            default_max_chunks: 10,
            default_similarity_threshold: 0.6,
            relevance_config: RelevanceConfig::default(),
            enable_multi_stage_ranking: true,
            first_stage_candidates: 50,
            enable_adaptive_scoring: true,
            enable_explanations: true,
            min_relevance_score: 0.3,
        }
    }
}

/// Enhanced ranked chunk with advanced relevance information
#[derive(Debug, Clone)]
pub struct EnhancedRankedChunk {
    pub chunk: DocumentChunk,
    pub relevance: RelevanceScore,
    pub rank: usize,
    pub retrieval_stage: String,
}

/// Retrieval performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EnhancedRetrievalStats {
    pub total_queries: u64,
    pub total_chunks_retrieved: u64,
    pub average_relevance_score: f32,
    pub average_confidence: f32,
    pub multi_stage_usage: u64,
    pub adaptive_scoring_usage: u64,
    pub average_first_stage_time_ms: f64,
    pub average_scoring_time_ms: f64,
    pub average_total_time_ms: f64,
    pub query_type_distribution: HashMap<String, u64>,
    pub scoring_method_distribution: HashMap<String, u64>,
}

/// Query-specific retrieval strategy
#[derive(Debug, Clone)]
pub enum RetrievalStrategy {
    Semantic,    // Focus on semantic similarity
    Keyword,     // Focus on keyword matching
    Balanced,    // Balanced approach
    Quality,     // Focus on document quality
    Fresh,       // Focus on recent content
    Adaptive,    // Choose based on query analysis
}

/// Enhanced retrieval service with advanced relevance scoring
pub struct EnhancedRetrievalService {
    config: EnhancedRetrievalConfig,
    embedding_service: Arc<EmbeddingService>,
    vector_store: Arc<dyn VectorStore + Send + Sync>,
    relevance_scorer: Arc<RelevanceScorer>,
    hybrid_scorer: Arc<HybridSearchScorer>,
    stats: Arc<tokio::sync::RwLock<EnhancedRetrievalStats>>,
}

impl EnhancedRetrievalService {
    /// Create a new enhanced retrieval service
    pub fn new(
        embedding_service: Arc<EmbeddingService>,
        vector_store: Arc<dyn VectorStore + Send + Sync>,
    ) -> Self {
        let config = EnhancedRetrievalConfig::default();
        let relevance_scorer = Arc::new(RelevanceScorerFactory::create_general());
        
        // Create document stats for hybrid search (empty for now, can be populated)
        let empty_chunks = Vec::new();
        let document_stats = tokio::runtime::Handle::current().block_on(async {
            build_document_stats(&empty_chunks).await
        });
        
        let hybrid_scorer = Arc::new(HybridSearchScorer::new(
            HybridSearchConfig::default(),
            Arc::new(document_stats),
        ));
        
        Self::with_config(config, embedding_service, vector_store, relevance_scorer, hybrid_scorer)
    }

    /// Create with custom configuration and components
    pub fn with_config(
        config: EnhancedRetrievalConfig,
        embedding_service: Arc<EmbeddingService>,
        vector_store: Arc<dyn VectorStore + Send + Sync>,
        relevance_scorer: Arc<RelevanceScorer>,
        hybrid_scorer: Arc<HybridSearchScorer>,
    ) -> Self {
        let stats = Arc::new(tokio::sync::RwLock::new(EnhancedRetrievalStats::default()));
        
        Self {
            config,
            embedding_service,
            vector_store,
            relevance_scorer,
            hybrid_scorer,
            stats,
        }
    }

    /// Retrieve and rank chunks using advanced relevance scoring
    pub async fn retrieve_chunks(&self, query: &Query) -> Result<Vec<EnhancedRankedChunk>> {
        let start_time = std::time::Instant::now();
        
        info!("Enhanced retrieval for query: {}", query.text);
        
        // Stage 1: Query Analysis
        let query_analysis = self.relevance_scorer.analyze_query(query);
        debug!("Query analysis: type={:?}, intent={:?}, complexity={:?}", 
               query_analysis.query_type, query_analysis.intent, query_analysis.complexity);
        
        // Stage 2: Adaptive Strategy Selection
        let strategy = if self.config.enable_adaptive_scoring {
            self.select_strategy(&query_analysis)
        } else {
            RetrievalStrategy::Balanced
        };
        
        debug!("Selected retrieval strategy: {:?}", strategy);
        
        // Stage 3: Initial Candidate Retrieval
        let first_stage_start = std::time::Instant::now();
        let candidates = self.retrieve_candidates(query, &query_analysis, &strategy).await?;
        let first_stage_time = first_stage_start.elapsed();
        
        info!("Retrieved {} candidates in {:.2}ms", 
              candidates.len(), first_stage_time.as_millis());
        
        if candidates.is_empty() {
            warn!("No candidates retrieved for query");
            return Ok(Vec::new());
        }
        
        // Stage 4: Advanced Relevance Scoring
        let scoring_start = std::time::Instant::now();
        let scored_chunks = self.score_candidates(query, &query_analysis, candidates, &strategy).await?;
        let scoring_time = scoring_start.elapsed();
        
        info!("Scored {} candidates in {:.2}ms", 
              scored_chunks.len(), scoring_time.as_millis());
        
        // Stage 5: Final Ranking and Filtering
        let final_results = self.finalize_results(scored_chunks, &query_analysis).await?;
        
        let total_time = start_time.elapsed();
        
        // Update statistics
        self.update_stats(&query_analysis, &final_results, &strategy, 
                         first_stage_time, scoring_time, total_time).await;
        
        info!("Enhanced retrieval completed: {} results in {:.2}ms", 
              final_results.len(), total_time.as_millis());
        
        Ok(final_results)
    }

    /// Select optimal retrieval strategy based on query analysis
    fn select_strategy(&self, query_analysis: &QueryAnalysis) -> RetrievalStrategy {
        match (&query_analysis.query_type, &query_analysis.intent) {
            // Procedural queries benefit from keyword precision
            (crate::core::relevance_scorer::QueryType::Procedural, _) => RetrievalStrategy::Keyword,
            
            // Navigational queries need exact matches
            (crate::core::relevance_scorer::QueryType::Navigational, _) => RetrievalStrategy::Keyword,
            
            // Learning queries benefit from quality documents
            (_, crate::core::QueryIntent::Learning) => RetrievalStrategy::Quality,
            
            // Time-sensitive queries need fresh content
            (_, _) if query_analysis.temporal_context.is_some() => RetrievalStrategy::Fresh,
            
            // Analytical queries benefit from semantic understanding
            (crate::core::relevance_scorer::QueryType::Analytical, _) => RetrievalStrategy::Semantic,
            
            // Default to balanced approach
            _ => RetrievalStrategy::Balanced,
        }
    }

    /// Retrieve initial candidates using vector similarity
    async fn retrieve_candidates(
        &self,
        query: &Query,
        _query_analysis: &QueryAnalysis,
        _strategy: &RetrievalStrategy,
    ) -> Result<Vec<(DocumentChunk, f32)>> {
        // Generate embedding for query
        let query_embedding = self.embedding_service
            .generate_embedding(&query.text)
            .await
            .map_err(|e| Error::embedding(format!("Failed to generate query embedding: {}", e)))?;
        
        // Determine number of candidates to retrieve
        let num_candidates = if self.config.enable_multi_stage_ranking {
            self.config.first_stage_candidates
        } else {
            query.options.max_chunks.unwrap_or(self.config.default_max_chunks)
        };
        
        // Build search filters
        let filters = crate::core::SearchFilters {
            document_ids: query.options.document_ids.clone(),
            ..Default::default()
        };
        
        // Perform vector similarity search
        let retrieved_chunks = self.vector_store
            .similarity_search(&query_embedding, num_candidates, Some(&filters))
            .await
            .map_err(|e| Error::vector_db(format!("Vector search failed: {}", e)))?;
        
        // Convert to DocumentChunk with similarity scores
        let candidates: Vec<(DocumentChunk, f32)> = retrieved_chunks
            .into_iter()
            .map(|retrieved| {
                let chunk = DocumentChunk {
                    id: retrieved.chunk_id,
                    document_id: retrieved.document_id,
                    content: retrieved.content,
                    chunk_index: retrieved.chunk_index,
                    embedding: retrieved.embedding,
                    metadata: retrieved.metadata,
                    created_at: retrieved.created_at,
                };
                (chunk, retrieved.similarity_score)
            })
            .collect();
        
        Ok(candidates)
    }

    /// Score candidates using advanced relevance algorithms
    async fn score_candidates(
        &self,
        query: &Query,
        query_analysis: &QueryAnalysis,
        candidates: Vec<(DocumentChunk, f32)>,
        strategy: &RetrievalStrategy,
    ) -> Result<Vec<(DocumentChunk, RelevanceScore)>> {
        // Select appropriate relevance scorer based on strategy
        let scorer = self.select_scorer(strategy);
        
        // Prepare data for batch scoring
        let query_embedding = self.embedding_service
            .generate_embedding(&query.text)
            .await
            .map_err(|e| Error::embedding(format!("Failed to generate query embedding: {}", e)))?;
        
        // Calculate keyword scores using hybrid search
        let chunks_for_hybrid: Vec<DocumentChunk> = candidates.iter()
            .map(|(chunk, _)| chunk.clone())
            .collect();
        
        let hybrid_results = self.hybrid_scorer
            .search_chunks(query, &chunks_for_hybrid, &query_embedding)
            .await
            .map_err(|e| Error::search(format!("Hybrid search failed: {}", e)))?;
        
        // Create keyword score lookup
        let keyword_scores: HashMap<uuid::Uuid, f32> = hybrid_results
            .into_iter()
            .map(|result| (result.chunk.id, result.keyword_score))
            .collect();
        
        // Score each candidate with advanced relevance scoring
        let mut scored_candidates = Vec::new();
        
        for (chunk, semantic_score) in candidates {
            let keyword_score = keyword_scores.get(&chunk.id).copied().unwrap_or(0.0);
            
            let relevance_score = scorer.calculate_relevance(
                query_analysis,
                &chunk,
                semantic_score,
                keyword_score,
            ).await?;
            
            scored_candidates.push((chunk, relevance_score));
        }
        
        Ok(scored_candidates)
    }

    /// Select appropriate scorer based on strategy
    fn select_scorer(&self, strategy: &RetrievalStrategy) -> Arc<RelevanceScorer> {
        match strategy {
            RetrievalStrategy::Semantic => Arc::new(RelevanceScorerFactory::create_semantic_focused()),
            RetrievalStrategy::Keyword => Arc::new(RelevanceScorerFactory::create_keyword_focused()),
            RetrievalStrategy::Quality => Arc::new(RelevanceScorerFactory::create_quality_focused()),
            RetrievalStrategy::Fresh => Arc::new(RelevanceScorerFactory::create_time_sensitive()),
            RetrievalStrategy::Balanced | RetrievalStrategy::Adaptive => self.relevance_scorer.clone(),
        }
    }

    /// Finalize results with ranking and filtering
    async fn finalize_results(
        &self,
        mut scored_chunks: Vec<(DocumentChunk, RelevanceScore)>,
        _query_analysis: &QueryAnalysis,
    ) -> Result<Vec<EnhancedRankedChunk>> {
        // Filter by minimum relevance score
        scored_chunks.retain(|(_, score)| score.overall_score >= self.config.min_relevance_score);
        
        // Sort by relevance score (already done in relevance scorer, but ensure consistency)
        scored_chunks.sort_by(|a, b| {
            b.1.overall_score
                .partial_cmp(&a.1.overall_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Limit to requested number of chunks
        let max_chunks = self.config.default_max_chunks; // Can be made query-specific
        scored_chunks.truncate(max_chunks);
        
        // Create final enhanced ranked chunks
        let results: Vec<EnhancedRankedChunk> = scored_chunks
            .into_iter()
            .enumerate()
            .map(|(index, (chunk, relevance))| EnhancedRankedChunk {
                chunk,
                relevance,
                rank: index + 1,
                retrieval_stage: if self.config.enable_multi_stage_ranking {
                    "multi-stage".to_string()
                } else {
                    "single-stage".to_string()
                },
            })
            .collect();
        
        Ok(results)
    }

    /// Update performance statistics
    async fn update_stats(
        &self,
        query_analysis: &QueryAnalysis,
        results: &[EnhancedRankedChunk],
        strategy: &RetrievalStrategy,
        first_stage_time: std::time::Duration,
        scoring_time: std::time::Duration,
        total_time: std::time::Duration,
    ) {
        let mut stats = self.stats.write().await;
        stats.total_queries += 1;
        stats.total_chunks_retrieved += results.len() as u64;
        
        // Update average relevance score
        if !results.is_empty() {
            let total_relevance: f32 = results.iter()
                .map(|r| r.relevance.overall_score)
                .sum();
            let avg_relevance = total_relevance / results.len() as f32;
            
            stats.average_relevance_score = self.update_running_average_f32(
                stats.average_relevance_score,
                avg_relevance,
                stats.total_queries,
            );
            
            // Update average confidence
            let total_confidence: f32 = results.iter()
                .map(|r| r.relevance.confidence)
                .sum();
            let avg_confidence = total_confidence / results.len() as f32;
            
            stats.average_confidence = self.update_running_average_f32(
                stats.average_confidence,
                avg_confidence,
                stats.total_queries,
            );
        }
        
        // Update timing statistics
        stats.average_first_stage_time_ms = self.update_running_average_f64(
            stats.average_first_stage_time_ms,
            first_stage_time.as_millis() as f64,
            stats.total_queries,
        );
        
        stats.average_scoring_time_ms = self.update_running_average_f64(
            stats.average_scoring_time_ms,
            scoring_time.as_millis() as f64,
            stats.total_queries,
        );
        
        stats.average_total_time_ms = self.update_running_average_f64(
            stats.average_total_time_ms,
            total_time.as_millis() as f64,
            stats.total_queries,
        );
        
        // Update feature usage statistics
        if self.config.enable_multi_stage_ranking {
            stats.multi_stage_usage += 1;
        }
        
        if self.config.enable_adaptive_scoring {
            stats.adaptive_scoring_usage += 1;
        }
        
        // Update query type distribution
        let query_type_str = format!("{:?}", query_analysis.query_type);
        *stats.query_type_distribution.entry(query_type_str).or_insert(0) += 1;
        
        // Update scoring method distribution
        let strategy_str = format!("{:?}", strategy);
        *stats.scoring_method_distribution.entry(strategy_str).or_insert(0) += 1;
    }

    /// Update running average for f32
    fn update_running_average_f32(&self, current_avg: f32, new_value: f32, count: u64) -> f32 {
        let count_f = count as f32;
        ((current_avg * (count_f - 1.0)) + new_value) / count_f
    }
    
    /// Update running average for f64
    fn update_running_average_f64(&self, current_avg: f64, new_value: f64, count: u64) -> f64 {
        let count_f = count as f64;
        ((current_avg * (count_f - 1.0)) + new_value) / count_f
    }

    /// Get current performance statistics
    pub async fn get_stats(&self) -> EnhancedRetrievalStats {
        self.stats.read().await.clone()
    }

    /// Get current configuration
    pub fn get_config(&self) -> &EnhancedRetrievalConfig {
        &self.config
    }

    /// Explain retrieval results for debugging and transparency
    pub fn explain_results(&self, results: &[EnhancedRankedChunk]) -> Vec<String> {
        if !self.config.enable_explanations {
            return vec!["Explanations disabled".to_string()];
        }
        
        let mut explanations = Vec::new();
        
        for (i, result) in results.iter().enumerate() {
            let explanation = format!(
                "Rank {}: Overall Score: {:.3} (Confidence: {:.3})\n\
                 - Semantic: {:.3}, Keyword: {:.3}, Quality: {:.3}\n\
                 - Freshness: {:.3}, Context: {:.3}, Boost: {:.3}\n\
                 - Main factors: {}\n\
                 - Positive signals: {}\n\
                 - Content preview: {}...",
                i + 1,
                result.relevance.overall_score,
                result.relevance.confidence,
                result.relevance.semantic_score,
                result.relevance.keyword_score,
                result.relevance.quality_score,
                result.relevance.freshness_score,
                result.relevance.context_score,
                result.relevance.boost_multiplier,
                result.relevance.explanation.main_factors.join(", "),
                result.relevance.explanation.positive_signals.join(", "),
                result.chunk.content.chars().take(100).collect::<String>(),
            );
            explanations.push(explanation);
        }
        
        explanations
    }

    /// Benchmark different retrieval strategies
    pub async fn benchmark_strategies(&self, query: &Query) -> Result<HashMap<String, (Vec<EnhancedRankedChunk>, std::time::Duration)>> {
        let strategies = vec![
            RetrievalStrategy::Semantic,
            RetrievalStrategy::Keyword,
            RetrievalStrategy::Balanced,
            RetrievalStrategy::Quality,
            RetrievalStrategy::Fresh,
        ];
        
        let mut results = HashMap::new();
        
        for strategy in strategies {
            let start = std::time::Instant::now();
            
            // Temporarily override config for single strategy test
            let mut temp_config = self.config.clone();
            temp_config.enable_adaptive_scoring = false;
            
            let temp_service = EnhancedRetrievalService::with_config(
                temp_config,
                self.embedding_service.clone(),
                self.vector_store.clone(),
                self.select_scorer(&strategy),
                self.hybrid_scorer.clone(),
            );
            
            let chunks = temp_service.retrieve_chunks(query).await?;
            let duration = start.elapsed();
            
            results.insert(format!("{:?}", strategy), (chunks, duration));
        }
        
        Ok(results)
    }
}

/// Builder for enhanced retrieval service
pub struct EnhancedRetrievalServiceBuilder {
    config: EnhancedRetrievalConfig,
    embedding_service: Option<Arc<EmbeddingService>>,
    vector_store: Option<Arc<dyn VectorStore + Send + Sync>>,
    relevance_config: Option<RelevanceConfig>,
}

impl EnhancedRetrievalServiceBuilder {
    pub fn new() -> Self {
        Self {
            config: EnhancedRetrievalConfig::default(),
            embedding_service: None,
            vector_store: None,
            relevance_config: None,
        }
    }

    pub fn with_config(mut self, config: EnhancedRetrievalConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_embedding_service(mut self, service: Arc<EmbeddingService>) -> Self {
        self.embedding_service = Some(service);
        self
    }

    pub fn with_vector_store(mut self, store: Arc<dyn VectorStore + Send + Sync>) -> Self {
        self.vector_store = Some(store);
        self
    }

    pub fn with_relevance_config(mut self, config: RelevanceConfig) -> Self {
        self.relevance_config = Some(config);
        self
    }

    pub fn enable_multi_stage_ranking(mut self, enabled: bool) -> Self {
        self.config.enable_multi_stage_ranking = enabled;
        self
    }

    pub fn enable_adaptive_scoring(mut self, enabled: bool) -> Self {
        self.config.enable_adaptive_scoring = enabled;
        self
    }

    pub fn with_max_chunks(mut self, max_chunks: usize) -> Self {
        self.config.default_max_chunks = max_chunks;
        self
    }

    pub fn build(mut self) -> Result<EnhancedRetrievalService> {
        let embedding_service = self.embedding_service
            .ok_or_else(|| Error::configuration("Embedding service is required"))?;
        
        let vector_store = self.vector_store
            .ok_or_else(|| Error::configuration("Vector store is required"))?;

        if let Some(relevance_config) = self.relevance_config {
            self.config.relevance_config = relevance_config;
        }

        let relevance_scorer = Arc::new(RelevanceScorer::new(self.config.relevance_config.clone()));
        
        // Create document stats for hybrid search
        let empty_chunks = Vec::new();
        let document_stats = tokio::runtime::Handle::current().block_on(async {
            build_document_stats(&empty_chunks).await
        });
        
        let hybrid_scorer = Arc::new(HybridSearchScorer::new(
            HybridSearchConfig::default(),
            Arc::new(document_stats),
        ));

        Ok(EnhancedRetrievalService::with_config(
            self.config,
            embedding_service,
            vector_store,
            relevance_scorer,
            hybrid_scorer,
        ))
    }
}

impl Default for EnhancedRetrievalServiceBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;
    use chrono::Utc;

    fn create_test_query(text: &str) -> Query {
        Query::new(text.to_string())
    }

    #[tokio::test]
    async fn test_strategy_selection() {
        let config = EnhancedRetrievalConfig::default();
        let relevance_scorer = Arc::new(RelevanceScorerFactory::create_general());
        let empty_chunks = Vec::new();
        let document_stats = build_document_stats(&empty_chunks).await;
        let hybrid_scorer = Arc::new(HybridSearchScorer::new(
            HybridSearchConfig::default(),
            Arc::new(document_stats),
        ));
        
        // Mock services for testing (in real implementation, would use actual services)
        // This test focuses on strategy selection logic
        
        // TODO: This test needs proper mock implementations
        // let service = EnhancedRetrievalService {
        //     config,
        //     embedding_service: Arc::new(crate::core::EmbeddingService::mock()), // Would need mock implementation
        //     vector_store: Arc::new(crate::storage::MockVectorStore::new()), // Would need mock implementation
        //     relevance_scorer,
        //     hybrid_scorer,
        //     stats: Arc::new(tokio::sync::RwLock::new(EnhancedRetrievalStats::default())),
        // };
        
        // TODO: Re-enable when mock implementations are available
        // Test different query types lead to different strategies
        // let procedural_query = create_test_query("How to implement machine learning?");
        // let query_analysis = service.relevance_scorer.analyze_query(&procedural_query);
        // let strategy = service.select_strategy(&query_analysis);
        // 
        // // Procedural queries should prefer keyword strategy
        // assert!(matches!(strategy, RetrievalStrategy::Keyword));
    }

    #[test]
    fn test_enhanced_retrieval_config() {
        let config = EnhancedRetrievalConfig::default();
        
        assert!(config.enable_multi_stage_ranking);
        assert!(config.enable_adaptive_scoring);
        assert!(config.enable_explanations);
        assert_eq!(config.default_max_chunks, 10);
        assert_eq!(config.first_stage_candidates, 50);
    }

    #[test]
    fn test_builder_pattern() {
        let builder = EnhancedRetrievalServiceBuilder::new()
            .enable_multi_stage_ranking(false)
            .enable_adaptive_scoring(true)
            .with_max_chunks(20);
        
        assert!(!builder.config.enable_multi_stage_ranking);
        assert!(builder.config.enable_adaptive_scoring);
        assert_eq!(builder.config.default_max_chunks, 20);
    }
}

use crate::models::{Query, DocumentChunk};
use crate::core::{QueryAnalysis, RelevanceScore};
use crate::utils::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Configuration for result reranking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankingConfig {
    /// Whether reranking is enabled
    pub enabled: bool,
    /// Number of candidates to consider for reranking (should be > final_count)
    pub candidate_count: usize,
    /// Final number of results to return after reranking
    pub final_count: usize,
    /// Reranking strategy to use
    pub strategy: RerankingStrategy,
    /// Weight given to original relevance scores vs reranking scores
    pub original_score_weight: f32,
    /// Whether to enable learning-to-rank features
    pub enable_learning_to_rank: bool,
    /// Whether to enable cross-encoder reranking
    pub enable_cross_encoder: bool,
    /// Diversity parameters for result diversification
    pub diversity_config: DiversityConfig,
    /// Quality boost parameters
    pub quality_config: QualityConfig,
    /// Temporal relevance parameters
    pub temporal_config: TemporalConfig,
}

impl Default for RerankingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            candidate_count: 50,
            final_count: 10,
            strategy: RerankingStrategy::Hybrid,
            original_score_weight: 0.3,
            enable_learning_to_rank: false,
            enable_cross_encoder: false,
            diversity_config: DiversityConfig::default(),
            quality_config: QualityConfig::default(),
            temporal_config: TemporalConfig::default(),
        }
    }
}

/// Reranking strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RerankingStrategy {
    /// Use original relevance scores only
    Original,
    /// Apply diversity-based reranking
    Diversity,
    /// Apply quality-based reranking
    Quality,
    /// Apply temporal relevance reranking
    Temporal,
    /// Apply cross-encoder neural reranking (requires model)
    CrossEncoder,
    /// Apply learning-to-rank approach
    LearningToRank,
    /// Hybrid approach combining multiple signals
    Hybrid,
}

/// Configuration for result diversification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiversityConfig {
    /// Enable diversity-based reranking
    pub enabled: bool,
    /// Weight for diversity score in final ranking
    pub diversity_weight: f32,
    /// Minimum similarity threshold for considering documents similar
    pub similarity_threshold: f32,
    /// Maximum number of results from the same document
    pub max_per_document: usize,
    /// Penalty factor for similar content
    pub similarity_penalty: f32,
}

impl Default for DiversityConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            diversity_weight: 0.2,
            similarity_threshold: 0.85,
            max_per_document: 3,
            similarity_penalty: 0.5,
        }
    }
}

/// Configuration for quality-based reranking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityConfig {
    /// Enable quality-based reranking
    pub enabled: bool,
    /// Weight for quality score in final ranking
    pub quality_weight: f32,
    /// Boost factor for high-quality documents
    pub quality_boost: f32,
    /// Penalty for low-quality documents
    pub quality_penalty: f32,
    /// Minimum content length for quality assessment
    pub min_content_length: usize,
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            quality_weight: 0.15,
            quality_boost: 1.2,
            quality_penalty: 0.8,
            min_content_length: 100,
        }
    }
}

/// Configuration for temporal relevance reranking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalConfig {
    /// Enable temporal relevance reranking
    pub enabled: bool,
    /// Weight for recency score in final ranking
    pub recency_weight: f32,
    /// Decay factor for older content (per day)
    pub recency_decay: f32,
    /// Boost factor for very recent content
    pub recent_boost: f32,
    /// Threshold in days for considering content "recent"
    pub recent_threshold_days: u32,
}

impl Default for TemporalConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            recency_weight: 0.1,
            recency_decay: 0.999, // Very gradual decay
            recent_boost: 1.1,
            recent_threshold_days: 7,
        }
    }
}

/// Reranked result with detailed scoring information
#[derive(Debug, Clone)]
pub struct RerankedResult {
    pub chunk: DocumentChunk,
    pub original_score: f32,
    pub reranked_score: f32,
    pub final_score: f32,
    pub rank: usize,
    pub original_rank: usize,
    pub score_components: RerankingScoreComponents,
    pub explanation: String,
}

/// Detailed components of reranking score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankingScoreComponents {
    pub relevance_score: f32,
    pub diversity_score: f32,
    pub quality_score: f32,
    pub recency_score: f32,
    pub cross_encoder_score: Option<f32>,
    pub learning_to_rank_score: Option<f32>,
}

impl Default for RerankingScoreComponents {
    fn default() -> Self {
        Self {
            relevance_score: 0.0,
            diversity_score: 0.0,
            quality_score: 0.0,
            recency_score: 0.0,
            cross_encoder_score: None,
            learning_to_rank_score: None,
        }
    }
}

/// Statistics for reranking performance
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RerankingStats {
    pub total_rerankings: u64,
    pub average_reranking_time_ms: f64,
    pub rank_changes_distribution: HashMap<i32, u64>, // Rank change -> count
    pub strategy_usage: HashMap<String, u64>,
    pub average_score_improvement: f32,
    pub diversity_improvements: u64,
    pub quality_improvements: u64,
    pub temporal_improvements: u64,
}

/// Result reranking service
pub struct RerankingService {
    config: RerankingConfig,
    stats: Arc<tokio::sync::RwLock<RerankingStats>>,
}

impl RerankingService {
    /// Create a new reranking service
    pub fn new(config: RerankingConfig) -> Self {
        Self {
            config,
            stats: Arc::new(tokio::sync::RwLock::new(RerankingStats::default())),
        }
    }

    /// Rerank search results based on configured strategy
    pub async fn rerank_results(
        &self,
        query: &Query,
        query_analysis: &QueryAnalysis,
        candidates: Vec<(DocumentChunk, RelevanceScore)>,
    ) -> Result<Vec<RerankedResult>> {
        if !self.config.enabled || candidates.is_empty() {
            // Convert to RerankedResult without reranking
            return Ok(candidates.into_iter().enumerate().map(|(idx, (chunk, relevance))| {
                RerankedResult {
                    chunk,
                    original_score: relevance.overall_score,
                    reranked_score: relevance.overall_score,
                    final_score: relevance.overall_score,
                    rank: idx + 1,
                    original_rank: idx + 1,
                    score_components: RerankingScoreComponents {
                        relevance_score: relevance.overall_score,
                        ..Default::default()
                    },
                    explanation: "No reranking applied".to_string(),
                }
            }).collect());
        }

        let start_time = std::time::Instant::now();
        
        info!("Reranking {} candidates using strategy: {:?}", 
              candidates.len(), self.config.strategy);

        // Apply reranking strategy
        let reranked_results = match self.config.strategy {
            RerankingStrategy::Original => self.apply_original_ranking(candidates).await?,
            RerankingStrategy::Diversity => self.apply_diversity_reranking(query, candidates).await?,
            RerankingStrategy::Quality => self.apply_quality_reranking(candidates).await?,
            RerankingStrategy::Temporal => self.apply_temporal_reranking(candidates).await?,
            RerankingStrategy::CrossEncoder => self.apply_cross_encoder_reranking(query, candidates).await?,
            RerankingStrategy::LearningToRank => self.apply_learning_to_rank(query, query_analysis, candidates).await?,
            RerankingStrategy::Hybrid => self.apply_hybrid_reranking(query, query_analysis, candidates).await?,
        };

        let reranking_time = start_time.elapsed();

        // Update statistics
        self.update_stats(&reranked_results, reranking_time).await;

        info!("Reranking completed: {} results in {:.2}ms", 
              reranked_results.len(), reranking_time.as_millis());

        Ok(reranked_results)
    }

    /// Apply original ranking (no reranking)
    async fn apply_original_ranking(
        &self,
        candidates: Vec<(DocumentChunk, RelevanceScore)>,
    ) -> Result<Vec<RerankedResult>> {
        let results = candidates
            .into_iter()
            .enumerate()
            .map(|(idx, (chunk, relevance))| RerankedResult {
                chunk,
                original_score: relevance.overall_score,
                reranked_score: relevance.overall_score,
                final_score: relevance.overall_score,
                rank: idx + 1,
                original_rank: idx + 1,
                score_components: RerankingScoreComponents {
                    relevance_score: relevance.overall_score,
                    ..Default::default()
                },
                explanation: "Original ranking maintained".to_string(),
            })
            .take(self.config.final_count)
            .collect();

        Ok(results)
    }

    /// Apply diversity-based reranking to avoid similar results
    async fn apply_diversity_reranking(
        &self,
        _query: &Query,
        candidates: Vec<(DocumentChunk, RelevanceScore)>,
    ) -> Result<Vec<RerankedResult>> {
        if !self.config.diversity_config.enabled {
            return self.apply_original_ranking(candidates).await;
        }

        debug!("Applying diversity-based reranking");

        let mut results = Vec::new();
        let mut used_documents = HashMap::new();
        let mut selected_content = Vec::new();

        for (original_rank, (chunk, relevance)) in candidates.into_iter().enumerate() {
            // Check document count limit
            let doc_count = used_documents.entry(chunk.document_id).or_insert(0);
            if *doc_count >= self.config.diversity_config.max_per_document {
                debug!("Skipping chunk from document {} (limit reached)", chunk.document_id);
                continue;
            }

            // Calculate diversity score
            let diversity_score = self.calculate_diversity_score(&chunk, &selected_content);
            
            // Calculate final score with diversity
            let relevance_score = relevance.overall_score;
            let weighted_diversity = diversity_score * self.config.diversity_config.diversity_weight;
            let final_score = relevance_score + weighted_diversity;

            let explanation = format!(
                "Relevance: {:.3}, Diversity: {:.3}, Final: {:.3}",
                relevance_score, diversity_score, final_score
            );

            results.push(RerankedResult {
                chunk: chunk.clone(),
                original_score: relevance_score,
                reranked_score: final_score,
                final_score,
                rank: 0, // Will be set after sorting
                original_rank: original_rank + 1,
                score_components: RerankingScoreComponents {
                    relevance_score,
                    diversity_score,
                    ..Default::default()
                },
                explanation,
            });

            *doc_count += 1;
            selected_content.push(chunk);

            if results.len() >= self.config.final_count {
                break;
            }
        }

        // Sort by final score and assign ranks
        results.sort_by(|a, b| b.final_score.partial_cmp(&a.final_score).unwrap_or(std::cmp::Ordering::Equal));
        for (idx, result) in results.iter_mut().enumerate() {
            result.rank = idx + 1;
        }

        Ok(results)
    }

    /// Calculate diversity score for a chunk compared to already selected content
    fn calculate_diversity_score(&self, chunk: &DocumentChunk, selected_content: &[DocumentChunk]) -> f32 {
        if selected_content.is_empty() {
            return 1.0; // Maximum diversity for first item
        }

        let mut min_similarity: f32 = 1.0;
        
        for selected_chunk in selected_content {
            // Simple content-based similarity (can be enhanced with embeddings)
            let similarity = self.calculate_text_similarity(&chunk.content, &selected_chunk.content);
            min_similarity = min_similarity.min(similarity);
        }

        // Diversity score is inverse of similarity
        let diversity = 1.0 - min_similarity;
        
        // Apply penalty if too similar to existing content
        if min_similarity > self.config.diversity_config.similarity_threshold {
            diversity * (1.0 - self.config.diversity_config.similarity_penalty)
        } else {
            diversity
        }
    }

    /// Simple text similarity calculation (Jaccard similarity on words)
    fn calculate_text_similarity(&self, text1: &str, text2: &str) -> f32 {
        use std::collections::HashSet;
        
        let words1: HashSet<&str> = text1.split_whitespace().collect();
        let words2: HashSet<&str> = text2.split_whitespace().collect();
        
        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();
        
        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    /// Apply quality-based reranking
    async fn apply_quality_reranking(
        &self,
        candidates: Vec<(DocumentChunk, RelevanceScore)>,
    ) -> Result<Vec<RerankedResult>> {
        if !self.config.quality_config.enabled {
            return self.apply_original_ranking(candidates).await;
        }

        debug!("Applying quality-based reranking");

        let mut results: Vec<RerankedResult> = candidates
            .into_iter()
            .enumerate()
            .map(|(original_rank, (chunk, relevance))| {
                let quality_score = self.calculate_quality_score(&chunk);
                let relevance_score = relevance.overall_score;
                
                // Apply quality boost or penalty
                let quality_multiplier = if quality_score > 0.7 {
                    self.config.quality_config.quality_boost
                } else if quality_score < 0.3 {
                    self.config.quality_config.quality_penalty
                } else {
                    1.0
                };

                let quality_weighted = quality_score * self.config.quality_config.quality_weight;
                let final_score = (relevance_score * quality_multiplier) + quality_weighted;

                let explanation = format!(
                    "Relevance: {:.3}, Quality: {:.3} (x{:.2}), Final: {:.3}",
                    relevance_score, quality_score, quality_multiplier, final_score
                );

                RerankedResult {
                    chunk,
                    original_score: relevance_score,
                    reranked_score: final_score,
                    final_score,
                    rank: 0, // Will be set after sorting
                    original_rank: original_rank + 1,
                    score_components: RerankingScoreComponents {
                        relevance_score,
                        quality_score,
                        ..Default::default()
                    },
                    explanation,
                }
            })
            .collect();

        // Sort by final score and assign ranks
        results.sort_by(|a, b| b.final_score.partial_cmp(&a.final_score).unwrap_or(std::cmp::Ordering::Equal));
        for (idx, result) in results.iter_mut().enumerate() {
            result.rank = idx + 1;
        }

        // Limit to final count
        results.truncate(self.config.final_count);

        Ok(results)
    }

    /// Calculate quality score for a document chunk
    fn calculate_quality_score(&self, chunk: &DocumentChunk) -> f32 {
        let mut quality_score: f32 = 0.5; // Base score

        // Content length factor
        let content_length = chunk.content.len();
        if content_length >= self.config.quality_config.min_content_length {
            quality_score += 0.2;
        } else {
            quality_score -= 0.2;
        }

        // Check for well-formed content (simple heuristics)
        let sentences = chunk.content.split('.').count();
        let words = chunk.content.split_whitespace().count();
        
        if words > 10 && sentences > 1 {
            quality_score += 0.2;
        }

        // Check for structured content (headers, lists, etc.)
        if chunk.content.contains('\n') || chunk.content.contains("- ") || chunk.content.contains("1. ") {
            quality_score += 0.1;
        }

        // Metadata quality indicators
        let metadata = &chunk.metadata;
        if metadata.get("title").is_some() {
            quality_score += 0.1;
        }
        if metadata.get("author").is_some() {
            quality_score += 0.1;
        }

        quality_score.clamp(0.0, 1.0)
    }

    /// Apply temporal relevance reranking
    async fn apply_temporal_reranking(
        &self,
        candidates: Vec<(DocumentChunk, RelevanceScore)>,
    ) -> Result<Vec<RerankedResult>> {
        if !self.config.temporal_config.enabled {
            return self.apply_original_ranking(candidates).await;
        }

        debug!("Applying temporal relevance reranking");

        let now = chrono::Utc::now();

        let mut results: Vec<RerankedResult> = candidates
            .into_iter()
            .enumerate()
            .map(|(original_rank, (chunk, relevance))| {
                let recency_score = self.calculate_recency_score(&chunk, now);
                let relevance_score = relevance.overall_score;
                
                let recency_weighted = recency_score * self.config.temporal_config.recency_weight;
                let final_score = relevance_score + recency_weighted;

                let explanation = format!(
                    "Relevance: {:.3}, Recency: {:.3}, Final: {:.3}",
                    relevance_score, recency_score, final_score
                );

                RerankedResult {
                    chunk,
                    original_score: relevance_score,
                    reranked_score: final_score,
                    final_score,
                    rank: 0, // Will be set after sorting
                    original_rank: original_rank + 1,
                    score_components: RerankingScoreComponents {
                        relevance_score,
                        recency_score,
                        ..Default::default()
                    },
                    explanation,
                }
            })
            .collect();

        // Sort by final score and assign ranks
        results.sort_by(|a, b| b.final_score.partial_cmp(&a.final_score).unwrap_or(std::cmp::Ordering::Equal));
        for (idx, result) in results.iter_mut().enumerate() {
            result.rank = idx + 1;
        }

        // Limit to final count
        results.truncate(self.config.final_count);

        Ok(results)
    }

    /// Calculate recency score based on document creation time
    fn calculate_recency_score(&self, chunk: &DocumentChunk, now: chrono::DateTime<chrono::Utc>) -> f32 {
        let days_old = (now - chunk.created_at).num_days() as f32;
        
        // Apply recent boost for very recent content
        if days_old <= self.config.temporal_config.recent_threshold_days as f32 {
            return self.config.temporal_config.recent_boost;
        }
        
        // Apply exponential decay for older content
        let decay_factor = self.config.temporal_config.recency_decay.powf(days_old);
        decay_factor.max(0.1) // Minimum score
    }

    /// Apply cross-encoder reranking (placeholder - requires actual model)
    async fn apply_cross_encoder_reranking(
        &self,
        _query: &Query,
        candidates: Vec<(DocumentChunk, RelevanceScore)>,
    ) -> Result<Vec<RerankedResult>> {
        warn!("Cross-encoder reranking not yet implemented, falling back to original ranking");
        self.apply_original_ranking(candidates).await
    }

    /// Apply learning-to-rank approach (placeholder)
    async fn apply_learning_to_rank(
        &self,
        _query: &Query,
        _query_analysis: &QueryAnalysis,
        candidates: Vec<(DocumentChunk, RelevanceScore)>,
    ) -> Result<Vec<RerankedResult>> {
        warn!("Learning-to-rank reranking not yet implemented, falling back to original ranking");
        self.apply_original_ranking(candidates).await
    }

    /// Apply hybrid reranking combining multiple signals
    async fn apply_hybrid_reranking(
        &self,
        query: &Query,
        query_analysis: &QueryAnalysis,
        candidates: Vec<(DocumentChunk, RelevanceScore)>,
    ) -> Result<Vec<RerankedResult>> {
        debug!("Applying hybrid reranking");

        let now = chrono::Utc::now();
        let mut results = Vec::new();
        let mut used_documents = HashMap::new();
        let mut selected_content = Vec::new();

        for (original_rank, (chunk, relevance)) in candidates.into_iter().enumerate() {
            // Apply document limit for diversity
            let doc_count = used_documents.entry(chunk.document_id).or_insert(0);
            if *doc_count >= self.config.diversity_config.max_per_document {
                continue;
            }

            // Calculate all component scores
            let relevance_score = relevance.overall_score;
            let quality_score = self.calculate_quality_score(&chunk);
            let recency_score = self.calculate_recency_score(&chunk, now);
            let diversity_score = self.calculate_diversity_score(&chunk, &selected_content);

            // Combine scores with adaptive weighting based on query analysis
            let weights = self.calculate_adaptive_weights(query_analysis);
            
            let final_score = self.config.original_score_weight * relevance_score +
                            weights.quality * quality_score +
                            weights.recency * recency_score +
                            weights.diversity * diversity_score;

            let explanation = format!(
                "Relevance: {:.3} (w={:.2}), Quality: {:.3} (w={:.2}), Recency: {:.3} (w={:.2}), Diversity: {:.3} (w={:.2}), Final: {:.3}",
                relevance_score, self.config.original_score_weight,
                quality_score, weights.quality,
                recency_score, weights.recency,
                diversity_score, weights.diversity,
                final_score
            );

            results.push(RerankedResult {
                chunk: chunk.clone(),
                original_score: relevance_score,
                reranked_score: final_score,
                final_score,
                rank: 0, // Will be set after sorting
                original_rank: original_rank + 1,
                score_components: RerankingScoreComponents {
                    relevance_score,
                    quality_score,
                    recency_score,
                    diversity_score,
                    ..Default::default()
                },
                explanation,
            });

            *doc_count += 1;
            selected_content.push(chunk);

            if results.len() >= self.config.candidate_count {
                break;
            }
        }

        // Sort by final score and assign ranks
        results.sort_by(|a, b| b.final_score.partial_cmp(&a.final_score).unwrap_or(std::cmp::Ordering::Equal));
        for (idx, result) in results.iter_mut().enumerate() {
            result.rank = idx + 1;
        }

        // Limit to final count
        results.truncate(self.config.final_count);

        Ok(results)
    }

    /// Calculate adaptive weights based on query analysis
    fn calculate_adaptive_weights(&self, query_analysis: &QueryAnalysis) -> AdaptiveWeights {
        let mut weights = AdaptiveWeights::default();

        // Adjust weights based on query characteristics
        match query_analysis.query_type {
            crate::core::relevance_scorer::QueryType::Factual => {
                weights.quality = 0.25;
                weights.diversity = 0.25;
                weights.recency = 0.15;
            },
            crate::core::relevance_scorer::QueryType::Navigational => {
                weights.quality = 0.20;
                weights.diversity = 0.10;
                weights.recency = 0.20;
            },
            crate::core::relevance_scorer::QueryType::Procedural => {
                weights.quality = 0.30;
                weights.diversity = 0.15;
                weights.recency = 0.25;
            },
            crate::core::relevance_scorer::QueryType::Analytical => {
                weights.quality = 0.35;
                weights.diversity = 0.30;
                weights.recency = 0.10;
            },
            crate::core::relevance_scorer::QueryType::Comparative => {
                weights.quality = 0.28;
                weights.diversity = 0.35;
                weights.recency = 0.12;
            },
            crate::core::relevance_scorer::QueryType::Conversational => {
                weights.quality = 0.20;
                weights.diversity = 0.20;
                weights.recency = 0.18;
            },
        }

        // Boost recency for time-sensitive queries
        if query_analysis.temporal_context.is_some() {
            weights.recency += 0.2;
            weights.quality -= 0.1;
            weights.diversity -= 0.1;
        }

        // Ensure weights are valid
        weights.normalize();
        weights
    }

    /// Update reranking statistics
    async fn update_stats(&self, results: &[RerankedResult], reranking_time: std::time::Duration) {
        let mut stats = self.stats.write().await;
        stats.total_rerankings += 1;

        // Update average reranking time
        let time_ms = reranking_time.as_millis() as f64;
        stats.average_reranking_time_ms = 
            (stats.average_reranking_time_ms * (stats.total_rerankings - 1) as f64 + time_ms) / 
            stats.total_rerankings as f64;

        // Track rank changes
        for result in results {
            let rank_change = result.rank as i32 - result.original_rank as i32;
            *stats.rank_changes_distribution.entry(rank_change).or_insert(0) += 1;
        }

        // Track strategy usage
        let strategy_name = format!("{:?}", self.config.strategy);
        *stats.strategy_usage.entry(strategy_name).or_insert(0) += 1;

        // Track improvements (simplified metric)
        let total_score_change: f32 = results.iter()
            .map(|r| r.final_score - r.original_score)
            .sum();
        
        if total_score_change > 0.0 {
            let avg_improvement = total_score_change / results.len() as f32;
            stats.average_score_improvement = 
                (stats.average_score_improvement * (stats.total_rerankings - 1) as f32 + avg_improvement) / 
                stats.total_rerankings as f32;
        }
    }

    /// Get reranking statistics
    pub async fn get_stats(&self) -> RerankingStats {
        self.stats.read().await.clone()
    }

    /// Reset statistics
    pub async fn reset_stats(&self) {
        let mut stats = self.stats.write().await;
        *stats = RerankingStats::default();
    }
}

/// Adaptive weights for hybrid reranking
#[derive(Debug, Clone)]
struct AdaptiveWeights {
    pub quality: f32,
    pub recency: f32,
    pub diversity: f32,
}

impl Default for AdaptiveWeights {
    fn default() -> Self {
        Self {
            quality: 0.2,
            recency: 0.15,
            diversity: 0.2,
        }
    }
}

impl AdaptiveWeights {
    /// Normalize weights to ensure they sum to reasonable bounds
    fn normalize(&mut self) {
        let total = self.quality + self.recency + self.diversity;
        if total > 1.0 {
            self.quality /= total;
            self.recency /= total;
            self.diversity /= total;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::DocumentChunk;
    use uuid::Uuid;

    fn create_test_chunk(id: Uuid, content: &str, created_days_ago: i64) -> DocumentChunk {
        DocumentChunk {
            id,
            document_id: Uuid::new_v4(),
            content: content.to_string(),
            chunk_index: 0,
            embedding: vec![0.1, 0.2, 0.3],
            metadata: serde_json::Value::Object(serde_json::Map::new()),
            created_at: chrono::Utc::now() - chrono::Duration::days(created_days_ago),
        }
    }

    #[tokio::test]
    async fn test_diversity_reranking() {
        let config = RerankingConfig {
            strategy: RerankingStrategy::Diversity,
            final_count: 3,
            ..Default::default()
        };
        
        let service = RerankingService::new(config);
        
        let chunks = vec![
            (create_test_chunk(Uuid::new_v4(), "Machine learning algorithms", 1), 
             RelevanceScore { overall_score: 0.9, confidence: 0.8, ..Default::default() }),
            (create_test_chunk(Uuid::new_v4(), "Deep learning networks", 1), 
             RelevanceScore { overall_score: 0.8, confidence: 0.8, ..Default::default() }),
            (create_test_chunk(Uuid::new_v4(), "Database management systems", 1), 
             RelevanceScore { overall_score: 0.7, confidence: 0.8, ..Default::default() }),
        ];
        
        let query = Query {
            text: "AI algorithms".to_string(),
            ..Default::default()
        };
        
        let query_analysis = QueryAnalysis::default();
        
        let results = service.rerank_results(&query, &query_analysis, chunks).await.unwrap();
        assert_eq!(results.len(), 3);
        assert!(results[0].score_components.diversity_score >= 0.0);
    }

    #[test]
    fn test_text_similarity() {
        let service = RerankingService::new(RerankingConfig::default());
        
        let similarity = service.calculate_text_similarity(
            "machine learning algorithms",
            "deep learning networks"
        );
        
        assert!(similarity > 0.0);
        assert!(similarity < 1.0);
        
        let identical_similarity = service.calculate_text_similarity(
            "same text",
            "same text"
        );
        
        assert_eq!(identical_similarity, 1.0);
    }

    #[test]
    fn test_quality_score_calculation() {
        let service = RerankingService::new(RerankingConfig::default());
        
        let high_quality_chunk = create_test_chunk(
            Uuid::new_v4(),
            "This is a well-structured document with multiple sentences. It contains detailed information about machine learning algorithms. The content is comprehensive and informative.",
            1
        );
        
        let low_quality_chunk = create_test_chunk(
            Uuid::new_v4(),
            "Short text",
            1
        );
        
        let high_quality_score = service.calculate_quality_score(&high_quality_chunk);
        let low_quality_score = service.calculate_quality_score(&low_quality_chunk);
        
        assert!(high_quality_score > low_quality_score);
    }
}

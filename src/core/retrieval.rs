use crate::models::{Query, DocumentChunk};
use crate::core::{EmbeddingService, SimilaritySearchEngine, SearchConfig, SearchFilters};
use crate::storage::VectorStore;
use crate::utils::{Error, Result};
use std::sync::Arc;

/// Configuration for retrieval service
#[derive(Debug, Clone)]
pub struct RetrievalConfig {
    /// Default maximum number of chunks to retrieve
    pub default_max_chunks: usize,
    /// Default similarity threshold
    pub default_similarity_threshold: f32,
    /// Whether to enable hybrid search (vector + keyword)
    pub enable_hybrid_search: bool,
    /// Weight for vector similarity in hybrid search (0.0-1.0)
    pub vector_weight: f32,
    /// Weight for keyword matching in hybrid search (0.0-1.0) 
    pub keyword_weight: f32,
    /// Whether to enable result reranking
    pub enable_reranking: bool,
    /// Alpha parameter for diversity-based reranking
    pub diversity_alpha: f32,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            default_max_chunks: 10,
            default_similarity_threshold: 0.7,
            enable_hybrid_search: true,
            vector_weight: 0.7,
            keyword_weight: 0.3,
            enable_reranking: true,
            diversity_alpha: 0.5,
        }
    }
}

/// Ranked document chunk with relevance metrics
#[derive(Debug, Clone)]
pub struct RankedChunk {
    pub chunk: DocumentChunk,
    pub similarity_score: f32,
    pub keyword_score: f32,
    pub combined_score: f32,
    pub rank: usize,
    pub relevance_explanation: String,
}

/// Retrieval statistics for monitoring
#[derive(Debug, Clone, Default)]
pub struct RetrievalStats {
    pub total_queries: u64,
    pub total_chunks_retrieved: u64,
    pub average_similarity_score: f32,
    pub cache_hit_rate: f32,
    pub average_retrieval_time_ms: f64,
}

/// Retrieval service for finding relevant document chunks
pub struct RetrievalService {
    config: RetrievalConfig,
    embedding_service: Arc<EmbeddingService>,
    vector_store: Arc<dyn VectorStore + Send + Sync>,
    search_engine: SimilaritySearchEngine,
    stats: Arc<std::sync::RwLock<RetrievalStats>>,
}

impl RetrievalService {
    /// Create a new retrieval service
    pub fn new(
        embedding_service: Arc<EmbeddingService>,
        vector_store: Arc<dyn VectorStore + Send + Sync>,
    ) -> Self {
        Self::with_config(
            RetrievalConfig::default(),
            embedding_service,
            vector_store,
        )
    }

    /// Create a new retrieval service with custom configuration
    pub fn with_config(
        config: RetrievalConfig,
        embedding_service: Arc<EmbeddingService>,
        vector_store: Arc<dyn VectorStore + Send + Sync>,
    ) -> Self {
        let search_config = SearchConfig {
            similarity_threshold: config.default_similarity_threshold,
            max_results: config.default_max_chunks,
            ..Default::default()
        };
        let search_engine = SimilaritySearchEngine::new(vector_store.clone());
        let stats = Arc::new(std::sync::RwLock::new(RetrievalStats::default()));

        Self {
            config,
            embedding_service,
            vector_store,
            search_engine,
            stats,
        }
    }

    /// Retrieve and rank relevant chunks for a query
    pub async fn retrieve_chunks(&self, query: &Query) -> Result<Vec<RankedChunk>> {
        let start_time = std::time::Instant::now();
        
        tracing::info!("Retrieving chunks for query: {}", query.text);
        
        // Generate embedding for the query
        let query_embedding = self.embedding_service
            .generate_embedding(&query.text)
            .await
            .map_err(|e| Error::embedding(format!("Failed to generate query embedding: {}", e)))?;

        // Determine search parameters from query options
        let max_chunks = query.options.max_chunks.unwrap_or(self.config.default_max_chunks);
        let _similarity_threshold = query.options.similarity_threshold
            .unwrap_or(self.config.default_similarity_threshold);

        // Build search filters from query options - using available fields
        let filters = SearchFilters {
            document_ids: query.options.document_ids.clone(),
            ..Default::default()
        };

        // Perform vector similarity search using the available interface
        let retrieved_chunks = self.vector_store
            .similarity_search(&query_embedding, max_chunks * 2, Some(&filters))
            .await
            .map_err(|e| Error::vector_db(format!("Vector search failed: {}", e)))?;

        // Convert RetrievedChunk to DocumentChunk objects
        let mut chunks: Vec<DocumentChunk> = retrieved_chunks
            .into_iter()
            .map(|retrieved| DocumentChunk {
                id: retrieved.chunk_id,
                document_id: retrieved.document_id,
                content: retrieved.content,
                chunk_index: retrieved.chunk_index,
                embedding: retrieved.embedding,
                metadata: retrieved.metadata,
                created_at: retrieved.created_at,
            })
            .collect();

        // Apply additional filters
        chunks = self.apply_filters(chunks, query).await?;

        // Perform hybrid search if enabled
        let ranked_chunks = if self.config.enable_hybrid_search {
            self.hybrid_search(query, chunks, &query_embedding).await?
        } else {
            self.vector_only_search(chunks, &query_embedding).await?
        };

        // Rerank results if enabled
        let final_results = if self.config.enable_reranking {
            self.rerank_results(ranked_chunks, query).await?
        } else {
            ranked_chunks
        };

        // Limit to requested number of chunks
        let limited_results: Vec<RankedChunk> = final_results
            .into_iter()
            .take(max_chunks)
            .enumerate()
            .map(|(i, mut chunk)| {
                chunk.rank = i + 1;
                chunk
            })
            .collect();

        // Update statistics
        self.update_stats(&limited_results, start_time.elapsed()).await;

        tracing::info!(
            "Retrieved {} chunks for query in {:.2}ms",
            limited_results.len(),
            start_time.elapsed().as_millis()
        );

        Ok(limited_results)
    }

    /// Calculate similarity between two embeddings
    pub fn calculate_similarity(&self, embedding1: &[f32], embedding2: &[f32]) -> f32 {
        if embedding1.len() != embedding2.len() {
            return 0.0;
        }

        // Cosine similarity calculation
        let dot_product: f32 = embedding1.iter()
            .zip(embedding2.iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm1: f32 = embedding1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = embedding2.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            return 0.0;
        }

        dot_product / (norm1 * norm2)
    }

    /// Apply filters to chunk results
    async fn apply_filters(
        &self,
        chunks: Vec<DocumentChunk>,
        query: &Query,
    ) -> Result<Vec<DocumentChunk>> {
        let mut filtered = chunks;

        // Filter by document IDs if specified
        if let Some(doc_ids) = &query.options.document_ids {
            filtered.retain(|chunk| doc_ids.contains(&chunk.document_id));
        }

        // Filter by tags if specified in query options
        if let Some(tags) = &query.options.filter_tags {
            filtered.retain(|chunk| {
                if let Ok(metadata) = serde_json::from_value::<std::collections::HashMap<String, serde_json::Value>>(chunk.metadata.clone()) {
                    if let Some(chunk_tags) = metadata.get("tags") {
                        if let Ok(chunk_tags) = serde_json::from_value::<Vec<String>>(chunk_tags.clone()) {
                            return tags.iter().any(|tag| chunk_tags.contains(tag));
                        }
                    }
                }
                false
            });
        }

        // Filter by category if specified in query options
        if let Some(category) = &query.options.filter_category {
            filtered.retain(|chunk| {
                if let Ok(metadata) = serde_json::from_value::<std::collections::HashMap<String, serde_json::Value>>(chunk.metadata.clone()) {
                    if let Some(chunk_category) = metadata.get("category") {
                        if let Ok(chunk_category) = serde_json::from_value::<String>(chunk_category.clone()) {
                            return &chunk_category == category;
                        }
                    }
                }
                false
            });
        }

        Ok(filtered)
    }

    /// Perform hybrid search combining vector and keyword matching
    async fn hybrid_search(
        &self,
        query: &Query,
        chunks: Vec<DocumentChunk>,
        query_embedding: &[f32],
    ) -> Result<Vec<RankedChunk>> {
        let mut ranked_chunks = Vec::new();
        let query_terms: Vec<String> = query.text
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();

        for chunk in chunks {
            // Calculate vector similarity
            let similarity_score = if let Some(embedding) = &chunk.embedding {
                self.calculate_similarity(query_embedding, embedding)
            } else {
                0.0
            };

            // Calculate keyword matching score
            let keyword_score = self.calculate_keyword_score(&query_terms, &chunk.content);

            // Combine scores
            let combined_score = (similarity_score * self.config.vector_weight) +
                               (keyword_score * self.config.keyword_weight);

            let explanation = format!(
                "Vector: {:.3} ({:.1}%), Keyword: {:.3} ({:.1}%), Combined: {:.3}",
                similarity_score,
                self.config.vector_weight * 100.0,
                keyword_score,
                self.config.keyword_weight * 100.0,
                combined_score
            );

            ranked_chunks.push(RankedChunk {
                chunk,
                similarity_score,
                keyword_score,
                combined_score,
                rank: 0, // Will be set later
                relevance_explanation: explanation,
            });
        }

        // Sort by combined score
        ranked_chunks.sort_by(|a, b| b.combined_score.partial_cmp(&a.combined_score).unwrap());

        Ok(ranked_chunks)
    }

    /// Perform vector-only search
    async fn vector_only_search(
        &self,
        chunks: Vec<DocumentChunk>,
        query_embedding: &[f32],
    ) -> Result<Vec<RankedChunk>> {
        let mut ranked_chunks = Vec::new();

        for chunk in chunks {
            let similarity_score = if let Some(embedding) = &chunk.embedding {
                self.calculate_similarity(query_embedding, embedding)
            } else {
                0.0
            };

            let explanation = format!("Vector similarity: {:.3}", similarity_score);

            ranked_chunks.push(RankedChunk {
                chunk,
                similarity_score,
                keyword_score: 0.0,
                combined_score: similarity_score,
                rank: 0, // Will be set later
                relevance_explanation: explanation,
            });
        }

        // Sort by similarity score
        ranked_chunks.sort_by(|a, b| b.similarity_score.partial_cmp(&a.similarity_score).unwrap());

        Ok(ranked_chunks)
    }

    /// Calculate keyword matching score using TF-IDF-like approach
    fn calculate_keyword_score(&self, query_terms: &[String], content: &str) -> f32 {
        let content_lower = content.to_lowercase();
        let content_terms: Vec<&str> = content_lower.split_whitespace().collect();
        let content_length = content_terms.len() as f32;

        if content_length == 0.0 || query_terms.is_empty() {
            return 0.0;
        }

        let mut score = 0.0;
        let mut matched_terms = 0;

        for term in query_terms {
            let term_frequency = content_terms.iter().filter(|&&t| t == term).count() as f32;
            if term_frequency > 0.0 {
                // Simple TF score (can be enhanced with IDF)
                let tf = term_frequency / content_length;
                score += tf;
                matched_terms += 1;
            }
        }

        // Normalize by number of query terms and boost for coverage
        if matched_terms > 0 {
            let coverage = matched_terms as f32 / query_terms.len() as f32;
            score = (score / query_terms.len() as f32) * (1.0 + coverage)
        }

        score.min(1.0) // Ensure score doesn't exceed 1.0
    }

    /// Rerank results using diversity and relevance
    async fn rerank_results(
        &self,
        ranked_chunks: Vec<RankedChunk>,
        _query: &Query,
    ) -> Result<Vec<RankedChunk>> {
        // Simple diversity-based reranking using Maximal Marginal Relevance (MMR)
        if ranked_chunks.len() <= 1 {
            return Ok(ranked_chunks);
        }

        let mut reranked = Vec::new();
        let mut remaining = ranked_chunks.clone();

        // Always include the top result
        if let Some(top) = remaining.first() {
            reranked.push(top.clone());
            remaining.remove(0);
        }

        // Iteratively select documents that balance relevance and diversity
        while !remaining.is_empty() && reranked.len() < ranked_chunks.len() {
            let mut best_idx = 0;
            let mut best_mmr = f32::NEG_INFINITY;

            for (i, candidate) in remaining.iter().enumerate() {
                // Calculate maximum similarity to already selected documents
                let mut max_similarity: f32 = 0.0;
                for selected in &reranked {
                    if let (Some(cand_emb), Some(sel_emb)) = (&candidate.chunk.embedding, &selected.chunk.embedding) {
                        let sim = self.calculate_similarity(cand_emb, sel_emb);
                        max_similarity = max_similarity.max(sim);
                    }
                }

                // MMR formula: λ * relevance - (1-λ) * max_similarity
                let mmr = self.config.diversity_alpha * candidate.combined_score - 
                         (1.0 - self.config.diversity_alpha) * max_similarity;

                if mmr > best_mmr {
                    best_mmr = mmr;
                    best_idx = i;
                }
            }

            reranked.push(remaining.remove(best_idx));
        }

        Ok(reranked)
    }

    /// Update retrieval statistics
    async fn update_stats(&self, results: &[RankedChunk], duration: std::time::Duration) {
        if let Ok(mut stats) = self.stats.write() {
            stats.total_queries += 1;
            stats.total_chunks_retrieved += results.len() as u64;
            
            if !results.is_empty() {
                let total_score: f32 = results.iter().map(|r| r.combined_score).sum();
                let new_avg_score = total_score / results.len() as f32;
                
                // Update running average
                let query_count = stats.total_queries as f32;
                stats.average_similarity_score = 
                    ((stats.average_similarity_score * (query_count - 1.0)) + new_avg_score) / query_count;
            }

            // Update average retrieval time
            let duration_ms = duration.as_millis() as f64;
            let query_count = stats.total_queries as f64;
            stats.average_retrieval_time_ms = 
                ((stats.average_retrieval_time_ms * (query_count - 1.0)) + duration_ms) / query_count;
        }
    }

    /// Get retrieval statistics
    pub fn get_stats(&self) -> RetrievalStats {
        self.stats.read().unwrap().clone()
    }

    /// Get retrieval configuration
    pub fn get_config(&self) -> &RetrievalConfig {
        &self.config
    }
}

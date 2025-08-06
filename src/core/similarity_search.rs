use crate::models::{QueryOptions, RetrievedChunk};
use crate::storage::VectorStore;
use crate::utils::{Error, Result};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

/// Distance metrics for similarity search
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistanceMetric {
    /// Cosine distance (1 - cosine_similarity)
    Cosine,
    /// Euclidean (L2) distance
    Euclidean,
    /// Manhattan (L1) distance  
    Manhattan,
    /// Dot product (for normalized vectors)
    DotProduct,
}

impl Default for DistanceMetric {
    fn default() -> Self {
        DistanceMetric::Cosine
    }
}

/// Search configuration options
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// Maximum number of results to return
    pub max_results: usize,
    /// Minimum similarity threshold (0.0 to 1.0)
    pub similarity_threshold: f32,
    /// Distance metric to use
    pub distance_metric: DistanceMetric,
    /// Whether to include chunk metadata in results
    pub include_metadata: bool,
    /// Whether to enable query preprocessing
    pub preprocess_query: bool,
    /// Maximum query length in characters
    pub max_query_length: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            max_results: 10,
            similarity_threshold: 0.0,
            distance_metric: DistanceMetric::Cosine,
            include_metadata: true,
            preprocess_query: true,
            max_query_length: 1000,
        }
    }
}

/// Search filters for refining results
#[derive(Debug, Clone, Default)]
pub struct SearchFilters {
    /// Filter by document IDs
    pub document_ids: Option<Vec<Uuid>>,
    /// Filter by chunk index range
    pub chunk_index_range: Option<(i32, i32)>,
    /// Filter by metadata fields
    pub metadata_filters: HashMap<String, serde_json::Value>,
    /// Filter by content length range
    pub content_length_range: Option<(usize, usize)>,
    /// Filter by creation date range
    pub created_after: Option<chrono::DateTime<chrono::Utc>>,
    pub created_before: Option<chrono::DateTime<chrono::Utc>>,
}

/// Search result with similarity score and ranking information
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub chunk: RetrievedChunk,
    pub similarity_score: f32,
    pub distance: f32,
    pub rank: usize,
    pub explanation: Option<String>,
}

/// Advanced similarity search engine
pub struct SimilaritySearchEngine {
    vector_store: Arc<dyn VectorStore + Send + Sync>,
    config: SearchConfig,
}

impl SimilaritySearchEngine {
    /// Create a new similarity search engine
    pub fn new(vector_store: Arc<dyn VectorStore + Send + Sync>) -> Self {
        Self {
            vector_store,
            config: SearchConfig::default(),
        }
    }

    /// Create a similarity search engine with custom configuration
    pub fn with_config(
        vector_store: Arc<dyn VectorStore + Send + Sync>,
        config: SearchConfig,
    ) -> Self {
        Self {
            vector_store,
            config,
        }
    }

    /// Perform similarity search with query embedding
    pub async fn search(
        &self,
        query_embedding: Vec<f32>,
        filters: Option<SearchFilters>,
        options: Option<QueryOptions>,
    ) -> Result<Vec<SearchResult>> {
        // Validate query embedding
        self.validate_query_embedding(&query_embedding)?;

        // Apply search configuration overrides from options
        let effective_config = self.merge_config_with_options(&self.config, options.as_ref());

        // Perform vector similarity search
        let raw_results = self.vector_store
            .similarity_search(
                &query_embedding,
                effective_config.max_results,
                filters.as_ref(),
            )
            .await?;

        // Process and rank results
        let mut search_results = self.process_raw_results(
            raw_results,
            &query_embedding,
            &effective_config,
        )?;

        // Apply similarity threshold filtering
        search_results.retain(|result| result.similarity_score >= effective_config.similarity_threshold);

        // Sort by similarity score (descending)
        search_results.sort_by(|a, b| b.similarity_score.partial_cmp(&a.similarity_score)
            .unwrap_or(std::cmp::Ordering::Equal));

        // Update ranks after sorting
        for (index, result) in search_results.iter_mut().enumerate() {
            result.rank = index + 1;
        }

        // Limit results to max_results
        search_results.truncate(effective_config.max_results);

        Ok(search_results)
    }

    /// Search with query text (requires embedding generation)
    pub async fn search_with_text(
        &self,
        query_text: &str,
        embedding_generator: impl Fn(&str) -> Result<Vec<f32>>,
        filters: Option<SearchFilters>,
        options: Option<QueryOptions>,
    ) -> Result<Vec<SearchResult>> {
        // Preprocess query if enabled
        let processed_query = if self.config.preprocess_query {
            self.preprocess_query(query_text)?
        } else {
            query_text.to_string()
        };

        // Generate embedding for the query
        let query_embedding = embedding_generator(&processed_query)?;

        // Perform search with embedding
        self.search(query_embedding, filters, options).await
    }

    /// Find similar chunks to a given chunk
    pub async fn find_similar_chunks(
        &self,
        source_chunk_id: Uuid,
        max_results: usize,
        exclude_same_document: bool,
    ) -> Result<Vec<SearchResult>> {
        // Get the source chunk with its embedding
        let source_chunk = self.vector_store
            .get_chunk_by_id(&source_chunk_id)
            .await?
            .ok_or_else(|| Error::not_found(&format!("Chunk not found: {}", source_chunk_id)))?;

        let source_embedding = source_chunk.embedding
            .ok_or_else(|| Error::validation("embedding", "Source chunk has no embedding"))?;

        // Set up filters to exclude the source chunk
        let filters = SearchFilters::default();
        if exclude_same_document {
            // Note: This would need to be implemented in the vector store
            // For now, we'll filter post-search
        }

        // Perform similarity search
        let mut results = self.search(
            source_embedding,
            Some(filters),
            Some(QueryOptions {
                max_chunks: Some(max_results + 10), // Get extra to account for filtering
                ..Default::default()
            }),
        ).await?;

        // Filter out the source chunk and same document if requested
        results.retain(|result| {
            result.chunk.id != source_chunk_id &&
            (!exclude_same_document || result.chunk.document_id != source_chunk.document_id)
        });

        // Limit to requested number of results
        results.truncate(max_results);

        Ok(results)
    }

    /// Batch similarity search for multiple queries
    pub async fn batch_search(
        &self,
        query_embeddings: Vec<Vec<f32>>,
        filters: Option<SearchFilters>,
        options: Option<QueryOptions>,
    ) -> Result<Vec<Vec<SearchResult>>> {
        let mut results = Vec::with_capacity(query_embeddings.len());
        
        for embedding in query_embeddings {
            let search_result = self.search(
                embedding,
                filters.clone(),
                options.clone(),
            ).await?;
            results.push(search_result);
        }

        Ok(results)
    }

    /// Get search statistics and performance metrics
    pub async fn get_search_stats(&self) -> Result<SearchStats> {
        let collection_info = self.vector_store.get_collection_info().await?;
        
        Ok(SearchStats {
            total_vectors: collection_info.vectors_count.unwrap_or(0) as usize,
            vector_dimension: collection_info.vectors_config.as_ref()
                .map(|config| config.size as usize)
                .unwrap_or(0),
            index_status: collection_info.status,
            distance_metric: self.config.distance_metric,
            similarity_threshold: self.config.similarity_threshold,
        })
    }

    // Private helper methods
    
    fn validate_query_embedding(&self, embedding: &[f32]) -> Result<()> {
        if embedding.is_empty() {
            return Err(Error::validation("query_embedding", "Query embedding cannot be empty"));
        }

        // Check for NaN or infinite values
        for (i, &value) in embedding.iter().enumerate() {
            if !value.is_finite() {
                return Err(Error::validation(
                    "embedding_value", 
                    &format!("Invalid embedding value at index {}: {}", i, value)
                ));
            }
        }

        Ok(())
    }

    fn merge_config_with_options(&self, config: &SearchConfig, options: Option<&QueryOptions>) -> SearchConfig {
        let mut effective_config = config.clone();
        
        if let Some(opts) = options {
            if let Some(max_chunks) = opts.max_chunks {
                effective_config.max_results = max_chunks;
            }
            if let Some(threshold) = opts.similarity_threshold {
                effective_config.similarity_threshold = threshold;
            }
            // Note: include_metadata is not in QueryOptions, keeping default
            // if let Some(include_metadata) = opts.include_metadata {
            //     effective_config.include_metadata = include_metadata;
            // }
        }
        
        effective_config
    }

    fn process_raw_results(
        &self,
        raw_results: Vec<RetrievedChunk>,
        query_embedding: &[f32],
        config: &SearchConfig,
    ) -> Result<Vec<SearchResult>> {
        let mut search_results = Vec::with_capacity(raw_results.len());

        for (index, chunk) in raw_results.into_iter().enumerate() {
            let embedding = chunk.embedding.as_ref()
                .ok_or_else(|| Error::validation("chunk_embedding", "Chunk missing embedding"))?;

            // Calculate similarity score based on distance metric
            let (distance, similarity_score) = self.calculate_similarity(
                query_embedding,
                embedding,
                config.distance_metric,
            )?;

            let search_result = SearchResult {
                chunk,
                similarity_score,
                distance,
                rank: index + 1, // Will be updated after sorting
                explanation: if config.include_metadata {
                    Some(format!("Distance: {:.4}, Metric: {:?}", distance, config.distance_metric))
                } else {
                    None
                },
            };

            search_results.push(search_result);
        }

        Ok(search_results)
    }

    fn calculate_similarity(
        &self,
        query_embedding: &[f32],
        chunk_embedding: &[f32],
        metric: DistanceMetric,
    ) -> Result<(f32, f32)> {
        if query_embedding.len() != chunk_embedding.len() {
            return Err(Error::validation(
                "embedding_dimension",
                &format!("Embedding dimension mismatch: query={}, chunk={}", 
                        query_embedding.len(), chunk_embedding.len())
            ));
        }

        let distance = match metric {
            DistanceMetric::Cosine => self.cosine_distance(query_embedding, chunk_embedding),
            DistanceMetric::Euclidean => self.euclidean_distance(query_embedding, chunk_embedding),
            DistanceMetric::Manhattan => self.manhattan_distance(query_embedding, chunk_embedding),
            DistanceMetric::DotProduct => -self.dot_product(query_embedding, chunk_embedding), // Negative for distance
        };

        // Convert distance to similarity score (0.0 to 1.0, higher is better)
        let similarity_score = match metric {
            DistanceMetric::Cosine => 1.0 - distance.max(0.0).min(2.0) / 2.0,
            DistanceMetric::Euclidean => 1.0 / (1.0 + distance),
            DistanceMetric::Manhattan => 1.0 / (1.0 + distance),
            DistanceMetric::DotProduct => (-distance).max(0.0).min(1.0), // Assuming normalized vectors
        };

        Ok((distance, similarity_score))
    }

    fn cosine_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            return 1.0; // Maximum distance for zero vectors
        }
        
        1.0 - (dot_product / (norm_a * norm_b))
    }

    fn euclidean_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    fn manhattan_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .sum()
    }

    fn dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    fn preprocess_query(&self, query: &str) -> Result<String> {
        if query.len() > self.config.max_query_length {
            return Err(Error::validation(
                "query_length",
                &format!("Query too long: {} > {}", query.len(), self.config.max_query_length)
            ));
        }

        // Basic query preprocessing
        let processed = query
            .trim()
            .to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace() || ".,!?-".contains(*c))
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");

        Ok(processed)
    }
}

/// Search statistics and performance metrics
#[derive(Debug, Clone)]
pub struct SearchStats {
    pub total_vectors: usize,
    pub vector_dimension: usize,
    pub index_status: String,
    pub distance_metric: DistanceMetric,
    pub similarity_threshold: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_calculations() {
        let engine = SimilaritySearchEngine::new(Arc::new(crate::storage::MockVectorStore));
        
        let vec_a = vec![1.0, 0.0, 0.0];
        let vec_b = vec![0.0, 1.0, 0.0];
        let vec_c = vec![1.0, 0.0, 0.0]; // Same as vec_a

        // Test cosine distance
        let (dist_ab, sim_ab) = engine.calculate_similarity(&vec_a, &vec_b, DistanceMetric::Cosine).unwrap();
        let (dist_ac, sim_ac) = engine.calculate_similarity(&vec_a, &vec_c, DistanceMetric::Cosine).unwrap();
        
        assert!((dist_ab - 1.0).abs() < 0.001); // 90 degrees = cosine distance of 1.0
        assert!(dist_ac.abs() < 0.001); // Same vectors = cosine distance of 0.0
        assert!(sim_ac > sim_ab); // Same vectors should have higher similarity

        // Test euclidean distance
        let (dist_euclidean, _) = engine.calculate_similarity(&vec_a, &vec_b, DistanceMetric::Euclidean).unwrap();
        assert!((dist_euclidean - (2.0_f32).sqrt()).abs() < 0.001);
    }

    #[test]
    fn test_query_preprocessing() {
        let engine = SimilaritySearchEngine::new(Arc::new(crate::storage::MockVectorStore));
        
        let query = "  Hello, WORLD!!! How are you?  ";
        let processed = engine.preprocess_query(query).unwrap();
        assert_eq!(processed, "hello, world!!! how are you?");

        // Test length limit
        let long_query = "a".repeat(2000);
        assert!(engine.preprocess_query(&long_query).is_err());
    }
}

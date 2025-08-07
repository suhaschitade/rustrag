use std::sync::Arc;
use std::time::Instant;
use tracing::{info, warn, error};
use uuid::Uuid;

use crate::api::{ApiResult, QueryProcessingResponse, RetrievedChunk, Citation, internal_error};
use crate::core::{
    RetrievalService, GenerationService, EmbeddingService,
    generation::GenerationConfig, retrieval::RetrievalConfig,
};
use crate::models::{Query, QueryOptions};
use crate::storage::VectorStore;
use crate::utils::Result;

/// Configuration for the query service
#[derive(Debug, Clone)]
pub struct QueryServiceConfig {
    pub retrieval: RetrievalConfig,
    pub generation: GenerationConfig,
    /// Default model to use for processing
    pub default_model: String,
    /// Whether to enable query logging
    pub enable_query_logging: bool,
    /// Maximum query processing timeout in seconds
    pub max_processing_time_seconds: u64,
}

impl Default for QueryServiceConfig {
    fn default() -> Self {
        Self {
            retrieval: RetrievalConfig::default(),
            generation: GenerationConfig::default(),
            default_model: "gpt-4".to_string(),
            enable_query_logging: true,
            max_processing_time_seconds: 30,
        }
    }
}

/// Main service for processing RAG queries end-to-end
pub struct QueryService {
    config: QueryServiceConfig,
    retrieval_service: Arc<RetrievalService>,
    generation_service: Arc<GenerationService>,
    embedding_service: Arc<EmbeddingService>,
}

impl QueryService {
    /// Create a new query service with dependencies
    pub fn new(
        embedding_service: Arc<EmbeddingService>,
        vector_store: Arc<dyn VectorStore + Send + Sync>,
    ) -> Result<Self> {
        Self::with_config(
            QueryServiceConfig::default(),
            embedding_service,
            vector_store,
        )
    }

    /// Create a new query service with custom configuration
    pub fn with_config(
        config: QueryServiceConfig,
        embedding_service: Arc<EmbeddingService>,
        vector_store: Arc<dyn VectorStore + Send + Sync>,
    ) -> Result<Self> {
        let retrieval_service = Arc::new(RetrievalService::with_config(
            config.retrieval.clone(),
            embedding_service.clone(),
            vector_store,
        ));

        let generation_service = Arc::new(GenerationService::with_config(
            config.generation.clone()
        )?);

        Ok(Self {
            config,
            retrieval_service,
            generation_service,
            embedding_service,
        })
    }

    /// Process a complete RAG query from text input to final response
    pub async fn process_query(
        &self,
        query_text: String,
        options: Option<QueryOptions>,
        model_override: Option<String>,
    ) -> ApiResult<QueryProcessingResponse> {
        let start_time = Instant::now();
        let query_id = Uuid::new_v4();

        info!("Starting RAG query processing (ID: {}): {}", query_id, query_text);

        // Validate query text
        if query_text.trim().is_empty() {
            return Err(internal_error("Query text cannot be empty"));
        }

        if query_text.len() > 10000 {
            return Err(internal_error("Query text exceeds maximum length of 10,000 characters"));
        }

        // Build query object
        let query_options = options.unwrap_or_default();
        let query = Query {
            id: query_id,
            text: query_text.clone(),
            options: query_options,
            created_at: chrono::Utc::now(),
        };

        // Determine which model to use
        let model_used = model_override.unwrap_or_else(|| self.config.default_model.clone());

        // Step 1: Retrieve relevant chunks
        info!("Step 1: Retrieving relevant document chunks for query {}", query_id);
        let retrieved_chunks = match self.retrieval_service.retrieve_chunks(&query).await {
            Ok(chunks) => {
                info!("Retrieved {} chunks for query {}", chunks.len(), query_id);
                chunks
            }
            Err(e) => {
                error!("Failed to retrieve chunks for query {}: {}", query_id, e);
                return Err(internal_error(&format!("Retrieval failed: {}", e)));
            }
        };

        // Step 2: Generate response using LLM
        info!("Step 2: Generating response for query {} with {} chunks", query_id, retrieved_chunks.len());
        let query_response = match self.generation_service.generate_response(&query, retrieved_chunks.clone()).await {
            Ok(response) => {
                info!("Generated response for query {} in {}ms", query_id, start_time.elapsed().as_millis());
                response
            }
            Err(e) => {
                error!("Failed to generate response for query {}: {}", query_id, e);
                return Err(internal_error(&format!("Generation failed: {}", e)));
            }
        };

        // Step 3: Calculate confidence score
        let confidence_score = self.calculate_confidence_score(&retrieved_chunks, &query_response.answer);

        // Step 4: Convert to API response format
        let processing_time = start_time.elapsed().as_millis() as u64;
        
        let api_chunks: Vec<RetrievedChunk> = retrieved_chunks
            .iter()
            .map(|ranked_chunk| RetrievedChunk {
                id: ranked_chunk.chunk.id,
                document_id: ranked_chunk.chunk.document_id,
                content: ranked_chunk.chunk.content.clone(),
                similarity_score: ranked_chunk.combined_score,
                chunk_index: ranked_chunk.chunk.chunk_index as u32,
                metadata: ranked_chunk.chunk.metadata.clone(),
            })
            .collect();

        let api_citations: Vec<Citation> = query_response.citations
            .iter()
            .map(|citation| Citation {
                document_id: citation.document_id,
                document_title: citation.document_title.clone(),
                chunk_id: citation.chunk_id,
                page_number: citation.page_number,
                confidence: retrieved_chunks
                    .iter()
                    .find(|rc| rc.chunk.id == citation.chunk_id)
                    .map(|rc| rc.combined_score)
                    .unwrap_or(0.0),
            })
            .collect();

        let response = QueryProcessingResponse {
            query_id,
            query: query_text,
            answer: query_response.answer,
            confidence_score,
            retrieved_chunks: api_chunks,
            citations: api_citations,
            processing_time_ms: processing_time,
            model_used,
        };

        // Log query if enabled
        if self.config.enable_query_logging {
            self.log_query_completion(&response).await;
        }

        info!("Completed RAG query processing (ID: {}) in {}ms", query_id, processing_time);
        Ok(response)
    }

    /// Process a search-only query (no generation)
    pub async fn search_documents(
        &self,
        query_text: String,
        options: Option<QueryOptions>,
    ) -> ApiResult<Vec<RetrievedChunk>> {
        let start_time = Instant::now();
        let query_id = Uuid::new_v4();

        info!("Starting document search (ID: {}): {}", query_id, query_text);

        // Validate query text
        if query_text.trim().is_empty() {
            return Err(internal_error("Query text cannot be empty"));
        }

        // Build query object
        let query_options = options.unwrap_or_default();
        let query = Query {
            id: query_id,
            text: query_text.clone(),
            options: query_options,
            created_at: chrono::Utc::now(),
        };

        // Retrieve relevant chunks
        let retrieved_chunks = match self.retrieval_service.retrieve_chunks(&query).await {
            Ok(chunks) => chunks,
            Err(e) => {
                error!("Failed to retrieve chunks for search {}: {}", query_id, e);
                return Err(internal_error(&format!("Search failed: {}", e)));
            }
        };

        // Convert to API response format
        let api_chunks: Vec<RetrievedChunk> = retrieved_chunks
            .iter()
            .map(|ranked_chunk| RetrievedChunk {
                id: ranked_chunk.chunk.id,
                document_id: ranked_chunk.chunk.document_id,
                content: ranked_chunk.chunk.content.clone(),
                similarity_score: ranked_chunk.combined_score,
                chunk_index: ranked_chunk.chunk.chunk_index as u32,
                metadata: ranked_chunk.chunk.metadata.clone(),
            })
            .collect();

        let processing_time = start_time.elapsed().as_millis();
        info!("Completed document search (ID: {}) in {}ms, found {} chunks", 
              query_id, processing_time, api_chunks.len());

        Ok(api_chunks)
    }

    /// Calculate confidence score based on retrieval results and answer quality
    fn calculate_confidence_score(
        &self,
        retrieved_chunks: &[crate::core::retrieval::RankedChunk],
        _answer: &str,
    ) -> f32 {
        if retrieved_chunks.is_empty() {
            return 0.0;
        }

        // Calculate weighted confidence based on:
        // 1. Top similarity scores (70% weight)
        // 2. Number of relevant chunks (20% weight)
        // 3. Consistency across chunks (10% weight)

        // Get top 3 similarity scores for primary confidence
        let top_scores: Vec<f32> = retrieved_chunks
            .iter()
            .take(3)
            .map(|chunk| chunk.combined_score)
            .collect();

        let primary_confidence = if !top_scores.is_empty() {
            let _sum: f32 = top_scores.iter().sum();
            let weighted_sum = match top_scores.len() {
                1 => top_scores[0],
                2 => top_scores[0] * 0.7 + top_scores[1] * 0.3,
                _ => top_scores[0] * 0.5 + top_scores[1] * 0.3 + top_scores[2] * 0.2,
            };
            weighted_sum
        } else {
            0.0
        };

        // Chunk quantity factor (more relevant chunks = higher confidence)
        let quantity_factor = match retrieved_chunks.len() {
            0 => 0.0,
            1 => 0.7,
            2..=3 => 0.85,
            4..=6 => 1.0,
            _ => 0.95, // Too many chunks might indicate ambiguous query
        };

        // Consistency factor (how similar are the top chunk scores)
        let consistency_factor = if retrieved_chunks.len() >= 2 {
            let scores: Vec<f32> = retrieved_chunks.iter().take(5).map(|c| c.combined_score).collect();
            let mean: f32 = scores.iter().sum::<f32>() / scores.len() as f32;
            let variance: f32 = scores.iter()
                .map(|score| (score - mean).powi(2))
                .sum::<f32>() / scores.len() as f32;
            let std_dev = variance.sqrt();
            
            // Lower standard deviation = higher consistency = higher confidence
            (1.0 - (std_dev / mean).min(1.0)).max(0.5)
        } else {
            0.8 // Single chunk gets neutral consistency
        };

        // Combine factors
        let final_confidence = (primary_confidence * 0.7) + 
                             (quantity_factor * 0.2) + 
                             (consistency_factor * 0.1);

        final_confidence.min(1.0).max(0.0)
    }

    /// Log query completion for analytics
    async fn log_query_completion(&self, response: &QueryProcessingResponse) {
        info!(
            query_id = %response.query_id,
            processing_time_ms = response.processing_time_ms,
            chunks_retrieved = response.retrieved_chunks.len(),
            citations_count = response.citations.len(),
            confidence_score = response.confidence_score,
            model_used = %response.model_used,
            "Query processing completed"
        );
    }

    /// Get service statistics
    pub fn get_statistics(&self) -> QueryServiceStatistics {
        let retrieval_stats = self.retrieval_service.get_stats();
        
        QueryServiceStatistics {
            total_queries_processed: retrieval_stats.total_queries,
            average_processing_time_ms: retrieval_stats.average_retrieval_time_ms,
            average_chunks_retrieved: if retrieval_stats.total_queries > 0 {
                retrieval_stats.total_chunks_retrieved as f64 / retrieval_stats.total_queries as f64
            } else {
                0.0
            },
            average_confidence_score: retrieval_stats.average_similarity_score,
        }
    }

    /// Get query service configuration
    pub fn get_config(&self) -> &QueryServiceConfig {
        &self.config
    }

    /// Update service configuration
    pub fn update_config(&mut self, config: QueryServiceConfig) -> Result<()> {
        // Note: This would require rebuilding the services with new config
        // For now, we'll just update the config and log a warning
        warn!("Query service configuration updated. Some changes may require service restart.");
        self.config = config;
        Ok(())
    }
}

/// Statistics for query service performance monitoring
#[derive(Debug, Clone)]
pub struct QueryServiceStatistics {
    pub total_queries_processed: u64,
    pub average_processing_time_ms: f64,
    pub average_chunks_retrieved: f64,
    pub average_confidence_score: f32,
}

impl Default for QueryServiceStatistics {
    fn default() -> Self {
        Self {
            total_queries_processed: 0,
            average_processing_time_ms: 0.0,
            average_chunks_retrieved: 0.0,
            average_confidence_score: 0.0,
        }
    }
}

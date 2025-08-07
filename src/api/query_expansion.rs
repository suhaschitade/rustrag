use crate::core::{
    QueryExpansionService, QueryExpansionConfig, ExpansionResult,
};
use crate::core::relevance_scorer::QueryType;
use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::info;

/// Query expansion request
#[derive(Debug, Deserialize, Serialize)]
pub struct QueryExpansionRequest {
    /// The query to expand and refine
    pub query: String,
    /// Configuration options (optional)
    pub options: Option<QueryExpansionOptions>,
}

/// Query expansion options
#[derive(Debug, Deserialize, Serialize)]
pub struct QueryExpansionOptions {
    /// Enable synonym expansion
    pub enable_synonyms: Option<bool>,
    /// Enable semantic expansion
    pub enable_semantic_expansion: Option<bool>,
    /// Enable query refinement
    pub enable_refinement: Option<bool>,
    /// Maximum expanded terms to add
    pub max_expanded_terms: Option<usize>,
    /// Enable negation handling
    pub enable_negation_handling: Option<bool>,
    /// Enable domain-specific expansion
    pub enable_domain_expansion: Option<bool>,
}

/// Query expansion response
#[derive(Debug, Serialize, Deserialize)]
pub struct QueryExpansionResponse {
    /// Original query
    pub original_query: String,
    /// Expanded query with additional terms
    pub expanded_query: String,
    /// Refined query with improved structure
    pub refined_query: String,
    /// Added synonyms
    pub synonyms: Vec<String>,
    /// Added semantic terms
    pub semantic_terms: Vec<String>,
    /// Detected negations
    pub negations: Vec<NegationResponse>,
    /// Alternative formulations
    pub alternatives: Vec<String>,
    /// Term weights for ranking
    pub term_weights: HashMap<String, f32>,
    /// Expansion confidence score
    pub confidence: f32,
}

/// Negation information in response
#[derive(Debug, Serialize, Deserialize)]
pub struct NegationResponse {
    pub negated_term: String,
    pub position: usize,
    pub handling_strategy: String,
}

/// Enhanced query processing request
#[derive(Debug, Deserialize, Serialize)]
pub struct EnhancedProcessingRequest {
    /// The query to process
    pub query: String,
    /// Processing configuration (optional)
    pub config: Option<ProcessingConfig>,
}

/// Processing configuration options
#[derive(Debug, Deserialize, Serialize)]
pub struct ProcessingConfig {
    pub enable_advanced_expansion: Option<bool>,
    pub use_expansion_alternatives: Option<bool>,
    pub apply_term_weights: Option<bool>,
    pub expansion_options: Option<QueryExpansionOptions>,
}

/// Enhanced query processing response
#[derive(Debug, Serialize, Deserialize)]
pub struct EnhancedProcessingResponse {
    /// Original query
    pub original_query: String,
    /// Final optimized query
    pub final_query: String,
    /// Query type detected
    pub query_type: String,
    /// Alternative formulations
    pub alternatives: Vec<String>,
    /// Term weights
    pub term_weights: HashMap<String, f32>,
    /// Overall confidence
    pub overall_confidence: f32,
    /// Processing statistics
    pub stats: ProcessingStatsResponse,
    /// Expansion details (if available)
    pub expansion_details: Option<ExpansionDetailsResponse>,
}

/// Processing statistics response
#[derive(Debug, Serialize, Deserialize)]
pub struct ProcessingStatsResponse {
    pub original_term_count: usize,
    pub processed_term_count: usize,
    pub key_terms_count: usize,
    pub synonyms_added: usize,
    pub semantic_terms_added: usize,
    pub negations_handled: usize,
    pub alternatives_generated: usize,
    pub expansion_confidence: f32,
    pub overall_confidence: f32,
}

/// Expansion details response
#[derive(Debug, Serialize, Deserialize)]
pub struct ExpansionDetailsResponse {
    pub synonyms: Vec<String>,
    pub semantic_terms: Vec<String>,
    pub negations: Vec<NegationResponse>,
    pub confidence: f32,
}

/// Query analysis request for batch processing
#[derive(Debug, Deserialize, Serialize)]
pub struct BatchAnalysisRequest {
    /// List of queries to analyze
    pub queries: Vec<String>,
    /// Maximum number of queries to process (default: 10)
    pub limit: Option<usize>,
}

/// Batch analysis response
#[derive(Debug, Serialize, Deserialize)]
pub struct BatchAnalysisResponse {
    pub results: Vec<QueryAnalysisResult>,
    pub summary: BatchSummary,
}

/// Individual query analysis result
#[derive(Debug, Serialize, Deserialize)]
pub struct QueryAnalysisResult {
    pub query: String,
    pub query_type: String,
    pub confidence: f32,
    pub expansion_quality: f32,
    pub complexity_score: f32,
    pub recommendations: Vec<String>,
}

/// Batch processing summary
#[derive(Debug, Serialize, Deserialize)]
pub struct BatchSummary {
    pub total_queries: usize,
    pub processed_queries: usize,
    pub average_confidence: f32,
    pub query_type_distribution: HashMap<String, usize>,
}

/// Application state for query expansion endpoints
#[derive(Clone)]
pub struct QueryExpansionState {
    pub expansion_service: Arc<QueryExpansionService>,
    // TODO: Re-enable when EnhancedQueryProcessor is implemented
    // pub enhanced_processor: Arc<EnhancedQueryProcessor>,
}

impl QueryExpansionState {
    pub fn new() -> Self {
        Self {
            expansion_service: Arc::new(QueryExpansionService::new()),
            // enhanced_processor: Arc::new(EnhancedQueryProcessor::new()),
        }
    }

    // TODO: Re-enable when types are available
    // pub fn with_config(
    //     expansion_config: QueryExpansionConfig,
    //     processor_config: EnhancedQueryProcessorConfig,
    // ) -> Self {
    //     Self {
    //         expansion_service: Arc::new(QueryExpansionService::with_config(expansion_config)),
    //         enhanced_processor: Arc::new(EnhancedQueryProcessor::with_config(processor_config)),
    //     }
    // }
}

/// Create query expansion router
pub fn create_query_expansion_router() -> Router<QueryExpansionState> {
    Router::new()
        .route("/expand", post(expand_query))
        // TODO: Re-enable when EnhancedQueryProcessor is implemented
        // .route("/process", post(process_query_enhanced))
        // .route("/analyze", post(analyze_queries_batch))
        .route("/config", get(get_expansion_config))
        .route("/health", get(health_check))
}

/// Expand and refine a single query
async fn expand_query(
    State(state): State<QueryExpansionState>,
    Json(request): Json<QueryExpansionRequest>,
) -> Result<Json<QueryExpansionResponse>, StatusCode> {
    info!("Expanding query: {}", request.query);

    // Configure expansion service if options provided
    let expansion_service = if let Some(options) = request.options {
        let mut config = QueryExpansionConfig::default();
        
        if let Some(enable_synonyms) = options.enable_synonyms {
            config.enable_synonyms = enable_synonyms;
        }
        if let Some(enable_semantic) = options.enable_semantic_expansion {
            config.enable_semantic_expansion = enable_semantic;
        }
        if let Some(enable_refinement) = options.enable_refinement {
            config.enable_refinement = enable_refinement;
        }
        if let Some(max_terms) = options.max_expanded_terms {
            config.max_expanded_terms = max_terms;
        }
        if let Some(enable_negation) = options.enable_negation_handling {
            config.enable_negation_handling = enable_negation;
        }
        if let Some(enable_domain) = options.enable_domain_expansion {
            config.enable_domain_expansion = enable_domain;
        }

        Arc::new(QueryExpansionService::with_config(config))
    } else {
        state.expansion_service.clone()
    };

    let expansion_result = expansion_service
        .expand_query(&request.query)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let response = QueryExpansionResponse {
        original_query: expansion_result.original_query,
        expanded_query: expansion_result.expanded_query,
        refined_query: expansion_result.refined_query,
        synonyms: expansion_result.synonyms,
        semantic_terms: expansion_result.semantic_terms,
        negations: expansion_result
            .negations
            .into_iter()
            .map(|neg| NegationResponse {
                negated_term: neg.negated_term,
                position: neg.position,
                handling_strategy: format!("{:?}", neg.handling_strategy),
            })
            .collect(),
        alternatives: expansion_result.alternatives,
        term_weights: expansion_result.term_weights,
        confidence: expansion_result.confidence,
    };

    Ok(Json(response))
}

// TODO: Re-implement when EnhancedQueryProcessor types are available
/*
/// Process query with enhanced capabilities
async fn process_query_enhanced(
    State(state): State<QueryExpansionState>,
    Json(request): Json<EnhancedProcessingRequest>,
) -> Result<Json<EnhancedProcessingResponse>, StatusCode> {
    info!("Processing enhanced query: {}", request.query);

    // Configure processor if options provided
    let processor = if let Some(config) = request.config {
        let mut processor_config = EnhancedQueryProcessorConfig::default();
        
        if let Some(enable_advanced) = config.enable_advanced_expansion {
            processor_config.enable_advanced_expansion = enable_advanced;
        }
        if let Some(use_alternatives) = config.use_expansion_alternatives {
            processor_config.use_expansion_alternatives = use_alternatives;
        }
        if let Some(apply_weights) = config.apply_term_weights {
            processor_config.apply_term_weights = apply_weights;
        }

        // Apply expansion options if provided
        if let Some(expansion_opts) = config.expansion_options {
            if let Some(enable_synonyms) = expansion_opts.enable_synonyms {
                processor_config.expansion_config.enable_synonyms = enable_synonyms;
            }
            if let Some(enable_semantic) = expansion_opts.enable_semantic_expansion {
                processor_config.expansion_config.enable_semantic_expansion = enable_semantic;
            }
            if let Some(max_terms) = expansion_opts.max_expanded_terms {
                processor_config.expansion_config.max_expanded_terms = max_terms;
            }
        }

        Arc::new(EnhancedQueryProcessor::with_config(processor_config))
    } else {
        state.enhanced_processor.clone()
    };

    let processed = processor
        .process_query_enhanced(&request.query)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let stats = processor.get_processing_stats(&processed);

    let response = EnhancedProcessingResponse {
        original_query: processed.basic.original,
        final_query: processed.final_query,
        query_type: format!("{:?}", processed.basic.query_type),
        alternatives: processed.alternatives,
        term_weights: processed.term_weights,
        overall_confidence: processed.overall_confidence,
        stats: ProcessingStatsResponse {
            original_term_count: stats.original_term_count,
            processed_term_count: stats.processed_term_count,
            key_terms_count: stats.key_terms_count,
            synonyms_added: stats.synonyms_added,
            semantic_terms_added: stats.semantic_terms_added,
            negations_handled: stats.negations_handled,
            alternatives_generated: stats.alternatives_generated,
            expansion_confidence: stats.expansion_confidence,
            overall_confidence: stats.overall_confidence,
        },
        expansion_details: processed.expansion.map(|exp| ExpansionDetailsResponse {
            synonyms: exp.synonyms,
            semantic_terms: exp.semantic_terms,
            negations: exp
                .negations
                .into_iter()
                .map(|neg| NegationResponse {
                    negated_term: neg.negated_term,
                    position: neg.position,
                    handling_strategy: format!("{:?}", neg.handling_strategy),
                })
                .collect(),
            confidence: exp.confidence,
        }),
    };

    Ok(Json(response))
}

/// Analyze multiple queries in batch
async fn analyze_queries_batch(
    State(state): State<QueryExpansionState>,
    Json(request): Json<BatchAnalysisRequest>,
) -> Result<Json<BatchAnalysisResponse>, StatusCode> {
    info!("Analyzing {} queries in batch", request.queries.len());

    let limit = request.limit.unwrap_or(10).min(50); // Cap at 50
    let queries_to_process: Vec<_> = request.queries.into_iter().take(limit).collect();
    
    let mut results = Vec::new();
    let mut type_distribution = HashMap::new();
    let mut total_confidence = 0.0;
    let mut processed_count = 0;

    for query in &queries_to_process {
        match state.enhanced_processor.process_query_enhanced(query).await {
            Ok(processed) => {
                let stats = state.enhanced_processor.get_processing_stats(&processed);
                
                // Calculate complexity score
                let complexity_score = calculate_complexity_score(&processed);
                
                // Generate recommendations
                let recommendations = generate_recommendations(&processed);
                
                let query_type_str = format!("{:?}", processed.basic.query_type);
                *type_distribution.entry(query_type_str.clone()).or_insert(0) += 1;
                
                results.push(QueryAnalysisResult {
                    query: query.clone(),
                    query_type: query_type_str,
                    confidence: processed.overall_confidence,
                    expansion_quality: stats.expansion_confidence,
                    complexity_score,
                    recommendations,
                });
                
                total_confidence += processed.overall_confidence;
                processed_count += 1;
            }
            Err(_) => {
                // Skip failed queries but continue processing
                results.push(QueryAnalysisResult {
                    query: query.clone(),
                    query_type: "Error".to_string(),
                    confidence: 0.0,
                    expansion_quality: 0.0,
                    complexity_score: 0.0,
                    recommendations: vec!["Query processing failed".to_string()],
                });
            }
        }
    }

    let average_confidence = if processed_count > 0 {
        total_confidence / processed_count as f32
    } else {
        0.0
    };

    let response = BatchAnalysisResponse {
        results,
        summary: BatchSummary {
            total_queries: queries_to_process.len(),
            processed_queries: processed_count,
            average_confidence,
            query_type_distribution: type_distribution,
        },
    };

    Ok(Json(response))
}

/// Get current expansion configuration
async fn get_expansion_config(
    State(_state): State<QueryExpansionState>,
) -> Json<QueryExpansionConfig> {
    Json(QueryExpansionConfig::default())
}

/// Health check endpoint
async fn health_check() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "query-expansion",
        "timestamp": chrono::Utc::now().to_rfc3339()
    }))
}

/// Calculate complexity score for a query
fn calculate_complexity_score(processed: &EnhancedProcessedQuery) -> f32 {
    let mut score = 0.0;
    
    // Base complexity from term count
    score += processed.basic.tokens.len() as f32 * 0.1;
    
    // Boost for technical queries
    if matches!(processed.basic.query_type, crate::core::QueryType::Technical) {
        score += 0.3;
    }
    
    // Boost for negations
    if let Some(expansion) = &processed.expansion {
        score += expansion.negations.len() as f32 * 0.2;
    }
    
    // Boost for multiple alternatives
    score += processed.alternatives.len() as f32 * 0.1;
    
    score.min(1.0)
}

/// Generate recommendations for query improvement
fn generate_recommendations(processed: &EnhancedProcessedQuery) -> Vec<String> {
    let mut recommendations = Vec::new();
    
    if processed.overall_confidence < 0.5 {
        recommendations.push("Consider rephrasing the query for better clarity".to_string());
    }
    
    if processed.basic.tokens.len() < 3 {
        recommendations.push("Add more specific terms to improve search accuracy".to_string());
    }
    
    if let Some(expansion) = &processed.expansion {
        if expansion.synonyms.is_empty() && expansion.semantic_terms.is_empty() {
            recommendations.push("Query could benefit from synonym expansion".to_string());
        }
        
        if !expansion.negations.is_empty() {
            recommendations.push("Query contains negations - ensure results exclude specified terms".to_string());
        }
    }
    
    if matches!(processed.basic.query_type, crate::core::QueryType::Unknown) {
        recommendations.push("Query intent unclear - consider restructuring as a question or command".to_string());
    }
    
    if recommendations.is_empty() {
        recommendations.push("Query looks good!".to_string());
    }
    
    recommendations
}
*/

/// Get current expansion configuration
async fn get_expansion_config(
    State(_state): State<QueryExpansionState>,
) -> Json<QueryExpansionConfig> {
    Json(QueryExpansionConfig::default())
}

/// Health check endpoint
async fn health_check() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "query-expansion",
        "timestamp": chrono::Utc::now().to_rfc3339()
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;
    use axum_test::TestServer;

    async fn create_test_server() -> TestServer {
        let state = QueryExpansionState::new();
        let app = create_query_expansion_router().with_state(state);
        TestServer::new(app).unwrap()
    }

    #[tokio::test]
    async fn test_expand_query_endpoint() {
        let server = create_test_server().await;
        
        let request = QueryExpansionRequest {
            query: "machine learning algorithms".to_string(),
            options: None,
        };
        
        let response = server
            .post("/expand")
            .json(&request)
            .await;
            
        assert_eq!(response.status_code(), StatusCode::OK);
        
        let expansion: QueryExpansionResponse = response.json();
        assert_eq!(expansion.original_query, "machine learning algorithms");
        assert!(!expansion.expanded_query.is_empty());
        assert!(expansion.confidence > 0.0);
    }

    #[tokio::test]
    #[ignore] // Disabled until EnhancedQueryProcessor is implemented
    async fn test_process_query_endpoint() {
        let server = create_test_server().await;
        
        let request = EnhancedProcessingRequest {
            query: "How to implement REST API?".to_string(),
            config: None,
        };
        
        let response = server
            .post("/process")
            .json(&request)
            .await;
            
        assert_eq!(response.status_code(), StatusCode::OK);
        
        let processing: EnhancedProcessingResponse = response.json();
        assert_eq!(processing.original_query, "How to implement REST API?");
        assert!(!processing.final_query.is_empty());
        assert!(processing.overall_confidence > 0.0);
    }

    #[tokio::test]
    #[ignore] // Disabled until EnhancedQueryProcessor is implemented
    async fn test_batch_analysis_endpoint() {
        let server = create_test_server().await;
        
        let request = BatchAnalysisRequest {
            queries: vec![
                "What is Rust?".to_string(),
                "database error".to_string(),
                "show me documents".to_string(),
            ],
            limit: Some(5),
        };
        
        let response = server
            .post("/analyze")
            .json(&request)
            .await;
            
        assert_eq!(response.status_code(), StatusCode::OK);
        
        let analysis: BatchAnalysisResponse = response.json();
        assert_eq!(analysis.summary.total_queries, 3);
        assert!(!analysis.results.is_empty());
    }
}

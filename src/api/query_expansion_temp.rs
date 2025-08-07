use crate::core::{
    QueryExpansionService, QueryExpansionConfig, ExpansionResult,
};
use axum::{
    extract::State,
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
#[derive(Debug, Deserialize)]
pub struct QueryExpansionRequest {
    /// The query to expand and refine
    pub query: String,
    /// Configuration options (optional)
    pub options: Option<QueryExpansionOptions>,
}

/// Query expansion options
#[derive(Debug, Deserialize)]
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
#[derive(Debug, Serialize)]
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
#[derive(Debug, Serialize)]
pub struct NegationResponse {
    pub negated_term: String,
    pub position: usize,
    pub handling_strategy: String,
}

/// Application state for query expansion endpoints
#[derive(Clone)]
pub struct QueryExpansionState {
    pub expansion_service: Arc<QueryExpansionService>,
}

impl QueryExpansionState {
    pub fn new() -> Self {
        Self {
            expansion_service: Arc::new(QueryExpansionService::new()),
        }
    }

    pub fn with_config(expansion_config: QueryExpansionConfig) -> Self {
        Self {
            expansion_service: Arc::new(QueryExpansionService::with_config(expansion_config)),
        }
    }
}

/// Create query expansion router
pub fn create_query_expansion_router() -> Router<QueryExpansionState> {
    Router::new()
        .route("/expand", post(expand_query))
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
}

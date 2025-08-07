use axum::{
    extract::{Path, Query as AxumQuery},
    response::{Json, Sse, sse::Event},
};
use chrono::{DateTime, Utc};
use futures::stream::{self, Stream};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tracing::{info, error};
use uuid::Uuid;

use crate::api::{
    ApiResponse, ApiResult, QueryProcessingResponse, RetrievedChunk,
    PaginatedResponse, PaginationParams,
    validation_error, internal_error,
};
use crate::core::{
    QueryService,
};
use crate::models::QueryOptions;
use crate::storage::vector_store::InMemoryVectorStore;

// Request/Response types
#[derive(Debug, Serialize, Deserialize)]
pub struct QueryRequest {
    pub query: String,
    pub max_chunks: Option<usize>,
    pub similarity_threshold: Option<f32>,
    pub include_citations: Option<bool>,
    pub document_ids: Option<Vec<Uuid>>, // Filter to specific documents
    pub model: Option<String>, // Specify LLM model to use
    pub temperature: Option<f32>, // LLM temperature
    pub max_tokens: Option<u32>, // Max response tokens
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BatchQueryRequest {
    pub queries: Vec<QueryRequest>,
    pub parallel: Option<bool>, // Process queries in parallel
}

#[derive(Debug, Serialize)]
pub struct QueryHistoryItem {
    pub id: Uuid,
    pub query: String,
    pub answer: String,
    pub processing_time_ms: u64,
    pub chunk_count: u32,
    pub citation_count: u32,
    pub model_used: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Deserialize)]
pub struct QueryHistoryQuery {
    pub user_id: Option<String>,
    pub model: Option<String>,
    pub min_processing_time: Option<u64>,
    pub max_processing_time: Option<u64>,
    #[serde(flatten)]
    pub pagination: PaginationParams,
}

#[derive(Debug, Serialize)]
pub struct StreamingChunk {
    pub chunk_type: String, // "chunk", "final", "error"
    pub content: String,
    pub metadata: Option<HashMap<String, Value>>,
}

// ==== Query Processing Endpoints ====

/// Process a query and return RAG response
pub async fn process_query(
    Json(request): Json<QueryRequest>,
) -> ApiResult<Json<ApiResponse<QueryProcessingResponse>>> {
    let _start_time = Instant::now();
    let query_id = Uuid::new_v4();
    
    info!("Processing query (ID: {}): {}", query_id, request.query);
    
    // Validate request
    if request.query.trim().is_empty() {
        return Err(validation_error("query", "Query cannot be empty"));
    }
    
    if request.query.len() > 10000 {
        return Err(validation_error("query", "Query cannot exceed 10,000 characters"));
    }
    
    // Validate optional parameters
    if let Some(threshold) = request.similarity_threshold {
        if threshold < 0.0 || threshold > 1.0 {
            return Err(validation_error("similarity_threshold", "Similarity threshold must be between 0.0 and 1.0"));
        }
    }
    
    if let Some(max_chunks) = request.max_chunks {
        if max_chunks == 0 || max_chunks > 50 {
            return Err(validation_error("max_chunks", "Max chunks must be between 1 and 50"));
        }
    }
    
    // Create QueryOptions from request parameters
    let query_options = QueryOptions {
        max_chunks: request.max_chunks,
        similarity_threshold: request.similarity_threshold,
        include_citations: request.include_citations.unwrap_or(true),
        document_ids: request.document_ids.clone(),
        filter_tags: None, // Not available in current request struct
        filter_category: None, // Not available in current request struct
        temperature: request.temperature,
        max_tokens: request.max_tokens,
    };

    // Initialize services (in production, these would be dependency-injected)
    let query_service = match create_query_service().await {
        Ok(service) => service,
        Err(e) => {
            error!("Failed to initialize query service: {}", e);
            return Err(internal_error("Service initialization failed"));
        }
    };

    // Process the query using the RAG pipeline
    let response = match query_service.process_query(
        request.query.clone(),
        Some(query_options),
        request.model.clone(),
    ).await {
        Ok(response) => response,
        Err(e) => {
            error!("Query processing failed: {}", e);
            return Err(e);
        }
    };
    
    info!("Query processed successfully (ID: {}) in {}ms", response.query_id, response.processing_time_ms);
    
    Ok(Json(ApiResponse::success(response)))
}

/// Stream query processing with real-time updates
pub async fn stream_query(
    Json(request): Json<QueryRequest>,
) -> ApiResult<Sse<impl Stream<Item = Result<Event, axum::Error>>>> {
    let query_id = Uuid::new_v4();
    info!("Starting streaming query (ID: {}): {}", query_id, request.query);
    
    // Validate request
    if request.query.trim().is_empty() {
        return Err(validation_error("query", "Query cannot be empty"));
    }
    
    // Create a stream of events for real-time updates
    let stream = stream::iter(vec![
        Ok(Event::default().json_data(StreamingChunk {
            chunk_type: "start".to_string(),
            content: format!("Starting query processing for: {}", request.query),
            metadata: {
                let mut map = HashMap::new();
                map.insert("query_id".to_string(), serde_json::Value::String(query_id.to_string()));
                Some(map)
            },
        }).unwrap()),
        
        Ok(Event::default().json_data(StreamingChunk {
            chunk_type: "chunk".to_string(),
            content: "Retrieved relevant documents...".to_string(),
            metadata: {
                let mut map = HashMap::new();
                map.insert("step".to_string(), serde_json::Value::String("retrieval".to_string()));
                Some(map)
            },
        }).unwrap()),
        
        Ok(Event::default().json_data(StreamingChunk {
            chunk_type: "chunk".to_string(),
            content: "Generating response...".to_string(),
            metadata: {
                let mut map = HashMap::new();
                map.insert("step".to_string(), serde_json::Value::String("generation".to_string()));
                Some(map)
            },
        }).unwrap()),
        
        Ok(Event::default().json_data(StreamingChunk {
            chunk_type: "final".to_string(),
            content: "This is a placeholder streaming response. Real streaming will be implemented with actual LLM integration.".to_string(),
            metadata: {
                let mut map = HashMap::new();
                map.insert("query_id".to_string(), serde_json::Value::String(query_id.to_string()));
                map.insert("processing_time_ms".to_string(), serde_json::Value::Number(serde_json::Number::from(150)));
                map.insert("confidence".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(0.85).unwrap()));
                Some(map)
            },
        }).unwrap()),
    ]);
    
    Ok(Sse::new(stream).keep_alive(axum::response::sse::KeepAlive::default()))
}

/// Process multiple queries in batch
pub async fn batch_process_queries(
    Json(request): Json<BatchQueryRequest>,
) -> ApiResult<Json<ApiResponse<Vec<QueryProcessingResponse>>>> {
    info!("Processing batch of {} queries", request.queries.len());
    
    if request.queries.is_empty() {
        return Err(validation_error("queries", "At least one query is required"));
    }
    
    if request.queries.len() > 10 {
        return Err(validation_error("queries", "Cannot process more than 10 queries at once"));
    }
    
    // TODO: Implement actual batch processing
    // For now, return empty results
    let responses = vec![];
    
    Ok(Json(ApiResponse::success_with_message(
        responses,
        format!("{} queries processed", request.queries.len()),
    )))
}

// ==== Query History Management ====

/// Get a specific query result by ID
pub async fn get_query_result(
    Path(query_id): Path<Uuid>,
) -> ApiResult<Json<ApiResponse<QueryProcessingResponse>>> {
    info!("Retrieving query result: {}", query_id);
    
    // TODO: Implement actual query result retrieval from database
    // For now, return a mock result
    let mock_response = QueryProcessingResponse {
        query_id,
        query: "What is the purpose of this document?".to_string(),
        answer: "This is a stored query result.".to_string(),
        confidence_score: 0.88,
        retrieved_chunks: vec![],
        citations: vec![],
        processing_time_ms: 1250,
        model_used: "gpt-4".to_string(),
    };
    
    Ok(Json(ApiResponse::success(mock_response)))
}

/// List query history with pagination and filtering
pub async fn list_query_history(
    AxumQuery(params): AxumQuery<QueryHistoryQuery>,
) -> ApiResult<Json<ApiResponse<PaginatedResponse<QueryHistoryItem>>>> {
    info!("Retrieving query history with pagination");
    
    params.pagination.validate().map_err(|e| validation_error("pagination", &e))?;
    
    // TODO: Implement actual query history retrieval
    let history_items = vec![];
    let total_count = 0;
    
    let paginated_response = PaginatedResponse::new(
        history_items,
        params.pagination.page,
        params.pagination.limit,
        total_count,
    );
    
    Ok(Json(ApiResponse::success(paginated_response)))
}

/// Search documents without generating responses
pub async fn search_documents(
    Json(request): Json<QueryRequest>,
) -> ApiResult<Json<ApiResponse<Vec<RetrievedChunk>>>> {
    let start_time = Instant::now();
    let query_id = Uuid::new_v4();
    
    info!("Starting document search (ID: {}): {}", query_id, request.query);
    
    // Validate request
    if request.query.trim().is_empty() {
        return Err(validation_error("query", "Query cannot be empty"));
    }
    
    // Create QueryOptions from request parameters
    let query_options = QueryOptions {
        max_chunks: request.max_chunks,
        similarity_threshold: request.similarity_threshold,
        include_citations: request.include_citations.unwrap_or(false), // No citations for search-only
        document_ids: request.document_ids.clone(),
        filter_tags: None,
        filter_category: None,
        temperature: None, // Not needed for search-only
        max_tokens: None, // Not needed for search-only
    };

    // Initialize services
    let query_service = match create_query_service().await {
        Ok(service) => service,
        Err(e) => {
            error!("Failed to initialize query service: {}", e);
            return Err(internal_error("Service initialization failed"));
        }
    };

    // Perform document search
    let chunks = match query_service.search_documents(
        request.query.clone(),
        Some(query_options),
    ).await {
        Ok(chunks) => chunks,
        Err(e) => {
            error!("Document search failed: {}", e);
            return Err(e);
        }
    };

    let processing_time = start_time.elapsed().as_millis();
    let chunks_count = chunks.len();
    info!("Document search completed (ID: {}) in {}ms, found {} chunks", 
          query_id, processing_time, chunks_count);
    
    Ok(Json(ApiResponse::success_with_message(
        chunks,
        format!("Found {} relevant document chunks", chunks_count),
    )))
}

// ==== Helper Functions ====

/// Create a query service instance (temporary implementation)
/// In production, this would be provided via dependency injection
async fn create_query_service() -> Result<QueryService, Box<dyn std::error::Error + Send + Sync>> {
    // Initialize embedding service with mock provider for development
    use crate::core::embeddings::EmbeddingServiceBuilder;
    
    let embedding_service = Arc::new(
        EmbeddingServiceBuilder::mock()?
    );
    
    // Initialize in-memory vector store (temporary)
    let vector_store = Arc::new(InMemoryVectorStore::new()) as Arc<dyn crate::storage::VectorStore + Send + Sync>;
    
    // Create query service
    let query_service = QueryService::new(embedding_service, vector_store)?;
    
    Ok(query_service)
}

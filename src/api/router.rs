use axum::{
    middleware,
    routing::{get, post},
    Router,
};
use tower::ServiceBuilder;
use tower_http::{
    cors::{Any, CorsLayer},
    timeout::TimeoutLayer,
    trace::TraceLayer,
};
use std::time::Duration;

use super::middleware::{
    auth_middleware,
    content_type_middleware,
    logging_middleware,
    rate_limit_middleware,
    request_id_middleware,
    security_headers_middleware,
};
use super::{
    health::*,
    documents::*,
    queries::*,
};

/// Build the main API router with all endpoints and middleware
pub fn create_api_router() -> Router {
    // Health endpoints (no authentication required)
    let health_router = Router::new()
        .route("/health", get(health_check))
        .route("/health/detailed", get(detailed_health_check))
        .route("/health/ready", get(readiness_check));

    // Document management endpoints
    let documents_router = Router::new()
        .route("/documents", post(upload_document).get(list_documents))
        .route("/documents/:id", get(get_document).delete(delete_document).patch(update_document))
        .route("/documents/:id/content", get(get_document_content))
        .route("/documents/:id/metadata", get(get_document_metadata).patch(update_document_metadata))
        .route("/documents/:id/chunks", get(list_document_chunks))
        .route("/documents/:id/reprocess", post(reprocess_document))
        .route("/documents/batch", post(batch_upload_documents).delete(batch_delete_documents))
        .route("/documents/search", get(search_documents))
        .layer(middleware::from_fn(auth_middleware));

    // Query processing endpoints
    let queries_router = Router::new()
        .route("/query", post(process_query))
        .route("/query/stream", post(stream_query))
        .route("/query/batch", post(batch_process_queries))
        .route("/queries/:id", get(get_query_result))
        .route("/queries", get(list_query_history))
        .layer(middleware::from_fn(auth_middleware));

    // Admin endpoints (require special permissions)
    let admin_router = Router::new()
        .route("/admin/stats", get(get_system_stats))
        .route("/admin/config", get(get_system_config).patch(update_system_config))
        .route("/admin/maintenance", post(trigger_maintenance))
        .route("/admin/cache/clear", post(clear_cache))
        .layer(middleware::from_fn(auth_middleware)); // TODO: Add admin-only middleware

    // API versioning - v1 routes
    let v1_router = Router::new()
        .merge(health_router)
        .merge(documents_router)
        .merge(queries_router)
        .merge(admin_router);

    // Main router with middleware stack
    Router::new()
        .nest("/api/v1", v1_router)
        .route("/", get(api_info))
        .layer(
            ServiceBuilder::new()
                // Request tracing and ID
                .layer(middleware::from_fn(request_id_middleware))
                .layer(TraceLayer::new_for_http())
                .layer(middleware::from_fn(logging_middleware))
                
                // Security middleware
                .layer(middleware::from_fn(security_headers_middleware))
                .layer(CorsLayer::new()
                    .allow_origin(Any) // TODO: Configure allowed origins properly
                    .allow_methods([
                        axum::http::Method::GET,
                        axum::http::Method::POST,
                        axum::http::Method::PUT,
                        axum::http::Method::PATCH,
                        axum::http::Method::DELETE,
                    ])
                    .allow_headers(Any)
                )
                
                // Rate limiting and timeouts
                .layer(middleware::from_fn(rate_limit_middleware))
                .layer(TimeoutLayer::new(Duration::from_secs(30)))
                
                // Content validation
                .layer(middleware::from_fn(content_type_middleware))
        )
}

/// API information endpoint
async fn api_info() -> axum::Json<serde_json::Value> {
    axum::Json(serde_json::json!({
        "name": "RustRAG API",
        "version": crate::VERSION,
        "description": "Enterprise Contextual AI Assistant Platform",
        "documentation_url": "/api/v1/docs",
        "health_check": "/api/v1/health",
        "endpoints": {
            "documents": "/api/v1/documents",
            "queries": "/api/v1/query",
            "admin": "/api/v1/admin"
        },
        "authentication": "API Key required (X-API-Key header or Authorization: Bearer <token>)"
    }))
}

// Placeholder admin handlers (to be implemented)
async fn get_system_stats() -> axum::Json<serde_json::Value> {
    axum::Json(serde_json::json!({
        "message": "System stats endpoint - to be implemented",
        "implementation": "Activity 8: Performance Optimization"
    }))
}

async fn get_system_config() -> axum::Json<serde_json::Value> {
    axum::Json(serde_json::json!({
        "message": "System config endpoint - to be implemented",
        "implementation": "Activity 1: Configuration Management"
    }))
}

async fn update_system_config() -> axum::Json<serde_json::Value> {
    axum::Json(serde_json::json!({
        "message": "Update system config endpoint - to be implemented",
        "implementation": "Activity 1: Configuration Management"
    }))
}

async fn trigger_maintenance() -> axum::Json<serde_json::Value> {
    axum::Json(serde_json::json!({
        "message": "Maintenance trigger endpoint - to be implemented",
        "implementation": "Activity 8: Performance Optimization"
    }))
}

async fn clear_cache() -> axum::Json<serde_json::Value> {
    axum::Json(serde_json::json!({
        "message": "Cache clear endpoint - to be implemented",
        "implementation": "Activity 8: Performance Optimization"
    }))
}

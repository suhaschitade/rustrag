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
    content_type_middleware,
    logging_middleware,
    request_id_middleware,
    security_headers_middleware,
};
use super::auth::{
    auth_middleware,
    ApiKeyStore,
};
use super::rate_limiter::{
    rate_limit_middleware,
    create_rate_limiter,
};
use super::auth_endpoints::*;
use super::{
    health::*,
    documents::*,
    queries,
    query_expansion::{create_query_expansion_router, QueryExpansionState},
};

/// Build the main API router with all endpoints and middleware
pub fn create_api_router() -> Router {
    // Initialize authentication and rate limiting services
    let api_key_store = ApiKeyStore::new();
    let rate_limiter = create_rate_limiter();

    // Health endpoints (no authentication required)
    let health_router = Router::new()
        .route("/health", get(health_check))
        .route("/health/detailed", get(detailed_health_check))
        .route("/health/ready", get(readiness_check));

    // Authentication management endpoints (require separate handling due to state)
    let auth_router = Router::new()
        .route("/auth/keys", post(create_api_key).get(list_api_keys))
        .route("/auth/keys/:id", get(get_api_key).patch(update_api_key).delete(delete_api_key))
        .route("/auth/keys/:id/revoke", post(revoke_api_key))
        .route("/auth/me", get(get_current_auth_info))
        .route("/auth/stats", get(get_auth_stats))
        // Permission testing endpoints (for development)
        .route("/auth/test/read", get(test_read_permission))
        .route("/auth/test/write", get(test_write_permission))
        .route("/auth/test/delete", get(test_delete_permission))
        .route("/auth/test/admin", get(test_admin_permission))
        .with_state(api_key_store.clone())
        .layer(middleware::from_fn_with_state(api_key_store.clone(), auth_middleware));

    // Rate limiting endpoints (require rate limiter state)
    let rate_limit_router = Router::new()
        .route("/auth/rate-limit", get(get_rate_limit_status))
        .route("/auth/rate-limits", get(get_all_rate_limit_statuses))
        .with_state(rate_limiter.clone())
        .layer(middleware::from_fn_with_state(api_key_store.clone(), auth_middleware));

    // Document management endpoints
    let documents_router = Router::new()
        .route("/documents", post(upload_document).get(list_documents))
        .route("/documents/search", get(search_documents))
        .route("/documents/:id", get(get_document).delete(delete_document).patch(update_document))
        .route("/documents/:id/content", get(get_document_content))
        .route("/documents/:id/metadata", get(get_document_metadata).patch(update_document_metadata))
        .route("/documents/:id/chunks", get(list_document_chunks))
        .route("/documents/:id/reprocess", post(reprocess_document))
        .route("/documents/batch", post(batch_upload_documents).delete(batch_delete_documents))
        .layer(middleware::from_fn_with_state(api_key_store.clone(), auth_middleware));

    // Query processing endpoints
    let queries_router = Router::new()
        .route("/query", post(queries::process_query))
        .route("/query/search", post(queries::search_documents))
        .route("/query/stream", post(queries::stream_query))
        .route("/query/batch", post(queries::batch_process_queries))
        .route("/queries/:id", get(queries::get_query_result))
        .route("/queries", get(queries::list_query_history))
        .layer(middleware::from_fn_with_state(api_key_store.clone(), auth_middleware));

    // Query expansion and refinement endpoints
    let query_expansion_state = QueryExpansionState::new();
    let query_expansion_router = create_query_expansion_router()
        .with_state(query_expansion_state)
        .layer(middleware::from_fn_with_state(api_key_store.clone(), auth_middleware));

    // Admin endpoints (require special permissions)
    let admin_router = Router::new()
        .route("/admin/stats", get(get_system_stats))
        .route("/admin/config", get(get_system_config).patch(update_system_config))
        .route("/admin/maintenance", post(trigger_maintenance))
        .route("/admin/cache/clear", post(clear_cache))
        .layer(middleware::from_fn_with_state(api_key_store.clone(), auth_middleware));

    // API versioning - v1 routes (merge main routers, rate limit router separate for now)
    let v1_router = Router::new()
        .merge(health_router)
        .merge(auth_router)
        .merge(rate_limit_router)
        .merge(documents_router)
        .merge(queries_router)
        .nest("/query-expansion", query_expansion_router)
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
                .layer(middleware::from_fn_with_state(rate_limiter.clone(), rate_limit_middleware))
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
            "query_expansion": "/api/v1/query-expansion",
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

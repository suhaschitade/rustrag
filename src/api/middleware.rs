use axum::{
    extract::Request,
    http::{HeaderMap, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use std::time::Instant;
use tracing::{info, warn};
use uuid::Uuid;

/// Request ID header name
pub const REQUEST_ID_HEADER: &str = "x-request-id";

/// Add request ID to all requests
pub async fn request_id_middleware(mut request: Request, next: Next) -> Response {
    let request_id = Uuid::new_v4().to_string();
    
    // Add request ID to headers for downstream handlers
    request.headers_mut().insert(
        REQUEST_ID_HEADER,
        request_id.parse().unwrap(),
    );

    let mut response = next.run(request).await;
    
    // Add request ID to response headers
    response.headers_mut().insert(
        REQUEST_ID_HEADER,
        request_id.parse().unwrap(),
    );

    response
}

/// Logging middleware to track request/response cycles
pub async fn logging_middleware(request: Request, next: Next) -> Response {
    let start = Instant::now();
    let method = request.method().clone();
    let uri = request.uri().clone();
    let request_id = request
        .headers()
        .get(REQUEST_ID_HEADER)
        .and_then(|h| h.to_str().ok())
        .unwrap_or("unknown")
        .to_string();

    info!(
        request_id = %request_id,
        method = %method,
        uri = %uri,
        "Processing request"
    );

    let response = next.run(request).await;
    let duration = start.elapsed();

    info!(
        request_id = %request_id,
        method = %method,
        uri = %uri,
        status = %response.status(),
        duration_ms = %duration.as_millis(),
        "Request completed"
    );

    response
}

/// Basic API key authentication middleware
pub async fn auth_middleware(request: Request, next: Next) -> Result<Response, impl IntoResponse> {
    // Skip auth for health endpoint
    if request.uri().path() == "/health" {
        return Ok(next.run(request).await);
    }

    let api_key = extract_api_key(request.headers())?;
    
    // TODO: Implement proper API key validation
    // For now, just check if it's not empty
    if api_key.is_empty() {
        warn!("Authentication failed: empty API key");
        return Err((
            StatusCode::UNAUTHORIZED,
            Json(json!({
                "error": "AUTHENTICATION_ERROR",
                "message": "Invalid or missing API key",
                "code": 401
            })),
        ));
    }

    info!("Authentication successful for API key: {}...", &api_key[..4.min(api_key.len())]);
    Ok(next.run(request).await)
}

/// Extract API key from request headers
fn extract_api_key(headers: &HeaderMap) -> Result<String, (StatusCode, Json<serde_json::Value>)> {
    // Check Authorization header (Bearer token)
    if let Some(auth_header) = headers.get("authorization") {
        if let Ok(auth_str) = auth_header.to_str() {
            if let Some(token) = auth_str.strip_prefix("Bearer ") {
                return Ok(token.to_string());
            }
        }
    }

    // Check X-API-Key header
    if let Some(api_key_header) = headers.get("x-api-key") {
        if let Ok(api_key) = api_key_header.to_str() {
            return Ok(api_key.to_string());
        }
    }

    Err((
        StatusCode::UNAUTHORIZED,
        Json(json!({
            "error": "AUTHENTICATION_ERROR",
            "message": "Missing API key. Provide via 'Authorization: Bearer <token>' or 'X-API-Key' header",
            "code": 401
        })),
    ))
}

/// Simple rate limiting middleware (in-memory, not production-ready)
pub async fn rate_limit_middleware(request: Request, next: Next) -> Response {
    // TODO: Implement proper rate limiting with Redis or similar
    // This is a placeholder that always allows requests
    
    let client_ip = request
        .headers()
        .get("x-forwarded-for")
        .or_else(|| request.headers().get("x-real-ip"))
        .and_then(|h| h.to_str().ok())
        .unwrap_or("unknown");

    // For now, just log the rate limit check
    info!("Rate limit check passed for client: {}", client_ip);
    
    next.run(request).await
}

/// Content-Type validation middleware for JSON endpoints
pub async fn content_type_middleware(request: Request, next: Next) -> Result<Response, impl IntoResponse> {
    let method = request.method();
    let content_type = request.headers().get("content-type");
    
    // Only check content-type for POST/PUT/PATCH requests
    if matches!(method, &axum::http::Method::POST | &axum::http::Method::PUT | &axum::http::Method::PATCH) {
        match content_type {
            Some(ct) if ct.to_str().unwrap_or("").starts_with("application/json") => {
                // Valid JSON content type
            }
            Some(ct) if ct.to_str().unwrap_or("").starts_with("multipart/form-data") => {
                // Valid for file uploads
            }
            _ => {
                return Err((
                    StatusCode::UNSUPPORTED_MEDIA_TYPE,
                    Json(json!({
                        "error": "UNSUPPORTED_MEDIA_TYPE",
                        "message": "Content-Type must be 'application/json' or 'multipart/form-data'",
                        "code": 415
                    })),
                ));
            }
        }
    }
    
    Ok(next.run(request).await)
}

/// Security headers middleware
pub async fn security_headers_middleware(request: Request, next: Next) -> Response {
    let mut response = next.run(request).await;
    
    let headers = response.headers_mut();
    
    // Add security headers
    headers.insert("X-Content-Type-Options", "nosniff".parse().unwrap());
    headers.insert("X-Frame-Options", "DENY".parse().unwrap());
    headers.insert("X-XSS-Protection", "1; mode=block".parse().unwrap());
    headers.insert("Referrer-Policy", "strict-origin-when-cross-origin".parse().unwrap());
    headers.insert("Content-Security-Policy", "default-src 'self'".parse().unwrap());
    
    response
}

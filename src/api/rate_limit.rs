use axum::{
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use std::{collections::HashMap, sync::{Arc, RwLock}, time::{Duration, Instant}};
use tracing::{info, warn};

use crate::api::types::ErrorResponse;

/// Simple in-memory rate limiter
#[derive(Clone)]
pub struct RateLimiter {
    // Map of client_id -> (last_request_time, request_count)
    clients: Arc<RwLock<HashMap<String, (Instant, u32)>>>,
    max_requests_per_window: u32,
    window_duration: Duration,
}

impl RateLimiter {
    pub fn new(max_requests_per_window: u32, window_duration: Duration) -> Self {
        Self {
            clients: Arc::new(RwLock::new(HashMap::new())),
            max_requests_per_window,
            window_duration,
        }
    }

    /// Check if a client has exceeded the rate limit
    pub fn check_rate_limit(&self, client_id: &str) -> bool {
        let mut clients = self.clients.write().unwrap();
        let now = Instant::now();
        
        match clients.get_mut(client_id) {
            Some((last_time, count)) => {
                if now.duration_since(*last_time) > self.window_duration {
                    // Reset the window
                    *last_time = now;
                    *count = 1;
                    true
                } else if *count >= self.max_requests_per_window {
                    // Rate limit exceeded
                    false
                } else {
                    // Increment counter
                    *count += 1;
                    true
                }
            },
            None => {
                // First request from this client
                clients.insert(client_id.to_string(), (now, 1));
                true
            }
        }
    }

    /// Clean up old entries (optional maintenance)
    pub fn cleanup_old_entries(&self) {
        let mut clients = self.clients.write().unwrap();
        let now = Instant::now();
        let cleanup_threshold = self.window_duration * 2;
        
        clients.retain(|_, (last_time, _)| {
            now.duration_since(*last_time) < cleanup_threshold
        });
    }
}

/// Extract client identifier from request
fn extract_client_id(request: &Request) -> String {
    // Try to get API key from headers
    if let Some(api_key) = request.headers().get("x-api-key") {
        if let Ok(key_str) = api_key.to_str() {
            if !key_str.is_empty() {
                return format!("api_key:{}", key_str);
            }
        }
    }

    // Try to get bearer token
    if let Some(auth_header) = request.headers().get("authorization") {
        if let Ok(auth_str) = auth_header.to_str() {
            if let Some(token) = auth_str.strip_prefix("Bearer ") {
                if !token.trim().is_empty() {
                    return format!("bearer:{}", token);
                }
            }
        }
    }

    // Fall back to IP address
    request.headers()
        .get("x-forwarded-for")
        .or_else(|| request.headers().get("x-real-ip"))
        .and_then(|h| h.to_str().ok())
        .map(|ip| format!("ip:{}", ip))
        .unwrap_or_else(|| "unknown".to_string())
}

/// Rate limiting middleware
pub async fn rate_limit_middleware(
    State(rate_limiter): State<RateLimiter>,
    request: Request,
    next: Next,
) -> Response {
    let client_id = extract_client_id(&request);
    
    if !rate_limiter.check_rate_limit(&client_id) {
        warn!("Rate limit exceeded for client: {}", client_id);
        
        let error_response = ErrorResponse::new(
            "RATE_LIMIT_EXCEEDED", 
            "Too many requests. Please try again later."
        );
        
        return (StatusCode::TOO_MANY_REQUESTS, axum::Json(error_response)).into_response();
    }
    
    info!("Request allowed for client: {}", client_id);
    next.run(request).await
}

/// Create default rate limiter (100 requests per hour)
pub fn create_default_rate_limiter() -> RateLimiter {
    RateLimiter::new(100, Duration::from_secs(3600))
}

/// Create strict rate limiter (10 requests per minute)
pub fn create_strict_rate_limiter() -> RateLimiter {
    RateLimiter::new(10, Duration::from_secs(60))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;

    #[test]
    fn test_rate_limiter_basic() {
        let rate_limiter = RateLimiter::new(3, Duration::from_secs(10));
        
        // First 3 requests should pass
        assert!(rate_limiter.check_rate_limit("test_client"));
        assert!(rate_limiter.check_rate_limit("test_client"));
        assert!(rate_limiter.check_rate_limit("test_client"));
        
        // 4th request should be rate limited
        assert!(!rate_limiter.check_rate_limit("test_client"));
    }

    #[test]
    fn test_rate_limiter_window_reset() {
        let rate_limiter = RateLimiter::new(2, Duration::from_millis(100));
        
        // Use up the limit
        assert!(rate_limiter.check_rate_limit("test_client"));
        assert!(rate_limiter.check_rate_limit("test_client"));
        assert!(!rate_limiter.check_rate_limit("test_client"));
        
        // Wait for window to reset
        sleep(Duration::from_millis(150));
        
        // Should be allowed again
        assert!(rate_limiter.check_rate_limit("test_client"));
    }

    #[test]
    fn test_rate_limiter_different_clients() {
        let rate_limiter = RateLimiter::new(1, Duration::from_secs(10));
        
        // Different clients should have separate limits
        assert!(rate_limiter.check_rate_limit("client1"));
        assert!(rate_limiter.check_rate_limit("client2"));
        
        // But same client should be limited
        assert!(!rate_limiter.check_rate_limit("client1"));
        assert!(!rate_limiter.check_rate_limit("client2"));
    }
}

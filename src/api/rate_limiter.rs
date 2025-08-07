use axum::{
    extract::{Request, State},
    http::{HeaderMap, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tracing::{debug, info, warn};

use crate::api::{ErrorResponse, AuthContext};

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Maximum requests per window
    pub max_requests: u32,
    /// Window duration in seconds
    pub window_seconds: u64,
    /// Whether to enable rate limiting
    pub enabled: bool,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            max_requests: 1000,
            window_seconds: 3600, // 1 hour
            enabled: true,
        }
    }
}

/// Rate limit bucket for tracking requests
#[derive(Debug, Clone)]
struct RateLimitBucket {
    /// Number of requests made in current window
    request_count: u32,
    /// Window start time (Unix timestamp)
    window_start: u64,
    /// Window duration in seconds
    window_duration: u64,
    /// Maximum requests allowed in window
    max_requests: u32,
}

impl RateLimitBucket {
    fn new(max_requests: u32, window_duration: u64) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        Self {
            request_count: 0,
            window_start: now,
            window_duration,
            max_requests,
        }
    }

    /// Check if a request is allowed and increment counter if so
    fn try_consume(&mut self) -> RateLimitResult {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Check if we need to reset the window
        if now >= self.window_start + self.window_duration {
            // Reset window
            self.window_start = now;
            self.request_count = 0;
            debug!("Rate limit window reset for bucket");
        }

        // Check if request is allowed
        if self.request_count >= self.max_requests {
            let reset_time = self.window_start + self.window_duration;
            return RateLimitResult::Exceeded {
                limit: self.max_requests,
                remaining: 0,
                reset_time,
                retry_after: reset_time.saturating_sub(now),
            };
        }

        // Allow request and increment counter
        self.request_count += 1;
        RateLimitResult::Allowed {
            limit: self.max_requests,
            remaining: self.max_requests.saturating_sub(self.request_count),
            reset_time: self.window_start + self.window_duration,
        }
    }
}

/// Result of rate limit check
#[derive(Debug)]
pub enum RateLimitResult {
    /// Request is allowed
    Allowed {
        limit: u32,
        remaining: u32,
        reset_time: u64,
    },
    /// Rate limit exceeded
    Exceeded {
        limit: u32,
        remaining: u32,
        reset_time: u64,
        retry_after: u64,
    },
}

/// In-memory rate limiter (fallback when Redis is not available)
#[derive(Debug, Clone)]
pub struct InMemoryRateLimiter {
    buckets: Arc<RwLock<HashMap<String, RateLimitBucket>>>,
    default_config: RateLimitConfig,
}

impl InMemoryRateLimiter {
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            buckets: Arc::new(RwLock::new(HashMap::new())),
            default_config: config,
        }
    }

    /// Check rate limit for a client
    pub fn check_rate_limit(&self, client_key: &str, custom_limit: Option<u32>) -> RateLimitResult {
        if !self.default_config.enabled {
            return RateLimitResult::Allowed {
                limit: u32::MAX,
                remaining: u32::MAX,
                reset_time: 0,
            };
        }

        let max_requests = custom_limit.unwrap_or(self.default_config.max_requests);
        
        let mut buckets = self.buckets.write().unwrap();
        let bucket = buckets
            .entry(client_key.to_string())
            .or_insert_with(|| RateLimitBucket::new(max_requests, self.default_config.window_seconds));

        // Update bucket limits if they've changed
        bucket.max_requests = max_requests;
        bucket.window_duration = self.default_config.window_seconds;

        bucket.try_consume()
    }

    /// Clean up expired buckets (should be called periodically)
    pub fn cleanup_expired_buckets(&self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let mut buckets = self.buckets.write().unwrap();
        buckets.retain(|_, bucket| {
            // Keep buckets that are still in their window or recently used
            now < bucket.window_start + bucket.window_duration + 3600 // Keep for 1 extra hour
        });

        debug!("Cleaned up expired rate limit buckets, remaining: {}", buckets.len());
    }

    /// Get current status for a client (for monitoring)
    pub fn get_client_status(&self, client_key: &str) -> Option<(u32, u32, u64)> {
        let buckets = self.buckets.read().unwrap();
        buckets.get(client_key).map(|bucket| {
            let reset_time = bucket.window_start + bucket.window_duration;
            (bucket.request_count, bucket.max_requests, reset_time)
        })
    }

    /// Get all client statuses (admin only)
    pub fn get_all_statuses(&self) -> HashMap<String, (u32, u32, u64)> {
        let buckets = self.buckets.read().unwrap();
        buckets
            .iter()
            .map(|(key, bucket)| {
                let reset_time = bucket.window_start + bucket.window_duration;
                (key.clone(), (bucket.request_count, bucket.max_requests, reset_time))
            })
            .collect()
    }
}

impl Default for InMemoryRateLimiter {
    fn default() -> Self {
        Self::new(RateLimitConfig::default())
    }
}

/// Rate limiter that can use Redis or fall back to in-memory
#[derive(Debug, Clone)]
pub struct RateLimiter {
    in_memory: InMemoryRateLimiter,
    config: RateLimitConfig,
}

impl RateLimiter {
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            in_memory: InMemoryRateLimiter::new(config.clone()),
            config,
        }
    }

    /// Check rate limit for a client
    pub async fn check_rate_limit(&self, client_key: &str, custom_limit: Option<u32>) -> RateLimitResult {
        // For now, use in-memory implementation
        // TODO: Implement Redis backend
        self.in_memory.check_rate_limit(client_key, custom_limit)
    }

    /// Clean up expired data
    pub async fn cleanup(&self) {
        self.in_memory.cleanup_expired_buckets();
    }

    /// Get status for monitoring
    pub async fn get_status(&self, client_key: &str) -> Option<(u32, u32, u64)> {
        self.in_memory.get_client_status(client_key)
    }

    /// Get all statuses (admin only)
    pub async fn get_all_statuses(&self) -> HashMap<String, (u32, u32, u64)> {
        self.in_memory.get_all_statuses()
    }
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new(RateLimitConfig::default())
    }
}

/// Extract client identifier for rate limiting
fn get_client_identifier(headers: &HeaderMap, auth_context: Option<&AuthContext>) -> String {
    // Use API key ID if authenticated
    if let Some(context) = auth_context {
        return format!("api_key:{}", context.api_key_id);
    }

    // Fall back to IP address
    let client_ip = headers
        .get("x-forwarded-for")
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.split(',').next())
        .map(|s| s.trim())
        .or_else(|| {
            headers
                .get("x-real-ip")
                .and_then(|h| h.to_str().ok())
        })
        .unwrap_or("unknown");

    format!("ip:{}", client_ip)
}

/// Rate limiting middleware
pub async fn rate_limit_middleware(
    State(rate_limiter): State<RateLimiter>,
    request: Request,
    next: Next,
) -> Result<Response, impl IntoResponse> {
    // Get client identifier
    let auth_context = request.extensions().get::<AuthContext>();
    let client_id = get_client_identifier(request.headers(), auth_context);

    // Get custom rate limit from auth context
    let custom_limit = auth_context.and_then(|ctx| ctx.rate_limit_per_hour);

    // Check rate limit
    match rate_limiter.check_rate_limit(&client_id, custom_limit).await {
        RateLimitResult::Allowed { limit, remaining, reset_time } => {
            info!(
                client_id = %client_id,
                limit = limit,
                remaining = remaining,
                reset_time = reset_time,
                "Rate limit check passed"
            );

            let mut response = next.run(request).await;
            
            // Add rate limit headers to response
            let headers = response.headers_mut();
            headers.insert("X-RateLimit-Limit", limit.to_string().parse().unwrap());
            headers.insert("X-RateLimit-Remaining", remaining.to_string().parse().unwrap());
            headers.insert("X-RateLimit-Reset", reset_time.to_string().parse().unwrap());

            Ok(response)
        }
        RateLimitResult::Exceeded { limit, remaining, reset_time, retry_after } => {
            warn!(
                client_id = %client_id,
                limit = limit,
                remaining = remaining,
                reset_time = reset_time,
                retry_after = retry_after,
                "Rate limit exceeded"
            );

            let mut error_response = (
                StatusCode::TOO_MANY_REQUESTS,
                Json(ErrorResponse::new(
                    "RATE_LIMIT_EXCEEDED",
                    &format!("Rate limit exceeded. Try again in {} seconds", retry_after),
                    429,
                )),
            ).into_response();

            // Add rate limit headers
            let headers = error_response.headers_mut();
            headers.insert("X-RateLimit-Limit", limit.to_string().parse().unwrap());
            headers.insert("X-RateLimit-Remaining", remaining.to_string().parse().unwrap());
            headers.insert("X-RateLimit-Reset", reset_time.to_string().parse().unwrap());
            headers.insert("Retry-After", retry_after.to_string().parse().unwrap());

            Err(error_response)
        }
    }
}

/// Global rate limiting configuration
#[derive(Debug, Clone)]
pub struct GlobalRateLimitConfig {
    /// Per-IP rate limits (requests per hour)
    pub per_ip_limit: u32,
    /// Per-API-key rate limits (requests per hour) - default if not specified in key
    pub per_api_key_limit: u32,
    /// Admin API rate limits (requests per hour)
    pub admin_api_limit: u32,
    /// Rate limit window in seconds
    pub window_seconds: u64,
    /// Enable/disable rate limiting
    pub enabled: bool,
}

impl Default for GlobalRateLimitConfig {
    fn default() -> Self {
        Self {
            per_ip_limit: 100,           // 100 requests per hour for unauthenticated
            per_api_key_limit: 1000,     // 1000 requests per hour for API keys
            admin_api_limit: 10000,      // 10000 requests per hour for admin
            window_seconds: 3600,        // 1 hour window
            enabled: true,
        }
    }
}

/// Create a configured rate limiter
pub fn create_rate_limiter() -> RateLimiter {
    let config = RateLimitConfig {
        max_requests: 1000,
        window_seconds: 3600,
        enabled: std::env::var("RATE_LIMITING_ENABLED")
            .unwrap_or_else(|_| "true".to_string())
            .parse()
            .unwrap_or(true),
    };

    info!("Initializing rate limiter with config: {:?}", config);
    RateLimiter::new(config)
}

/// Background task to clean up expired rate limit data
pub async fn cleanup_rate_limits(rate_limiter: RateLimiter) {
    let mut interval = tokio::time::interval(Duration::from_secs(300)); // Clean up every 5 minutes
    
    loop {
        interval.tick().await;
        rate_limiter.cleanup().await;
        debug!("Rate limit cleanup completed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rate_limit_allows_within_limit() {
        let rate_limiter = RateLimiter::new(RateLimitConfig {
            max_requests: 10,
            window_seconds: 60,
            enabled: true,
        });

        let client_key = "test_client";

        // Should allow first 10 requests
        for i in 0..10 {
            let result = rate_limiter.check_rate_limit(client_key, None).await;
            match result {
                RateLimitResult::Allowed { remaining, .. } => {
                    assert_eq!(remaining, 10 - i - 1);
                }
                _ => panic!("Request {} should have been allowed", i),
            }
        }
    }

    #[tokio::test]
    async fn test_rate_limit_blocks_when_exceeded() {
        let rate_limiter = RateLimiter::new(RateLimitConfig {
            max_requests: 5,
            window_seconds: 60,
            enabled: true,
        });

        let client_key = "test_client_2";

        // Use up the limit
        for _ in 0..5 {
            rate_limiter.check_rate_limit(client_key, None).await;
        }

        // Next request should be blocked
        let result = rate_limiter.check_rate_limit(client_key, None).await;
        match result {
            RateLimitResult::Exceeded { .. } => {
                // This is expected
            }
            _ => panic!("Request should have been blocked"),
        }
    }

    #[tokio::test]
    async fn test_custom_rate_limit() {
        let rate_limiter = RateLimiter::new(RateLimitConfig {
            max_requests: 10,
            window_seconds: 60,
            enabled: true,
        });

        let client_key = "test_client_3";

        // Use custom limit of 2
        let result = rate_limiter.check_rate_limit(client_key, Some(2)).await;
        assert!(matches!(result, RateLimitResult::Allowed { .. }));

        let result = rate_limiter.check_rate_limit(client_key, Some(2)).await;
        assert!(matches!(result, RateLimitResult::Allowed { .. }));

        // Third request should be blocked
        let result = rate_limiter.check_rate_limit(client_key, Some(2)).await;
        assert!(matches!(result, RateLimitResult::Exceeded { .. }));
    }
}

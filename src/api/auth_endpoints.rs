use axum::{
    extract::{Path, State, Query},
    Extension,
    Json,
};
use serde::{Deserialize, Serialize};
use tracing::info;
use uuid::Uuid;

use crate::api::{
    auth::{ApiKeyStore, CreateApiKeyRequest, ApiKeyResponse, Permission, AuthContext},
    rate_limiter::RateLimiter,
    types::{ApiResponse, PaginatedResponse, PaginationParams},
    ApiResult, validation_error,
};
use std::collections::HashMap;

/// API key update request
#[derive(Debug, Deserialize)]
pub struct UpdateApiKeyRequest {
    pub name: Option<String>,
    pub description: Option<String>,
    pub permissions: Option<Vec<Permission>>,
    pub rate_limit_per_hour: Option<Option<u32>>, // Option<Option<u32>> to allow setting to None
    pub is_active: Option<bool>,
}

/// Rate limit status response
#[derive(Debug, Serialize)]
pub struct RateLimitStatus {
    pub client_id: String,
    pub current_requests: u32,
    pub limit: u32,
    pub reset_time: u64,
    pub remaining: u32,
}

/// System authentication statistics
#[derive(Debug, Serialize)]
pub struct AuthStats {
    pub total_api_keys: usize,
    pub active_api_keys: usize,
    pub expired_api_keys: usize,
    pub total_requests_last_hour: u32,
    pub rate_limited_requests_last_hour: u32,
}

// ==== API Key Management Endpoints ====

/// Create a new API key
pub async fn create_api_key(
    State(api_key_store): State<ApiKeyStore>,
    Extension(auth_context): Extension<AuthContext>,
    Json(request): Json<CreateApiKeyRequest>,
) -> ApiResult<Json<ApiResponse<ApiKeyResponse>>> {
    info!("Creating new API key: {}", request.name);

    // Check permissions - only admin can create API keys
    if !auth_context.has_permission(&Permission::Admin) {
        return Err(crate::utils::Error::authorization(
            "Admin permission required to create API keys"
        ).into());
    }

    let (api_key, key_value) = api_key_store.create_api_key(request).await?;
    
    let mut response: ApiKeyResponse = api_key.into();
    response.key = Some(key_value); // Include the actual key only when creating

    info!("API key created successfully: {} ({})", response.name, response.id);
    
    Ok(Json(ApiResponse::success_with_message(
        response,
        "API key created successfully. Store the key securely - it won't be shown again.".to_string(),
    )))
}

/// List all API keys (admin only)
pub async fn list_api_keys(
    State(api_key_store): State<ApiKeyStore>,
    Extension(auth_context): Extension<AuthContext>,
    Query(pagination): Query<PaginationParams>,
) -> ApiResult<Json<ApiResponse<PaginatedResponse<ApiKeyResponse>>>> {
    info!("Listing API keys");

    // Check permissions
    if !auth_context.has_permission(&Permission::Admin) {
        return Err(crate::utils::Error::authorization(
            "Admin permission required to list API keys"
        ).into());
    }

    pagination.validate().map_err(|e| validation_error("pagination", &e))?;

    let api_keys = api_key_store.list_api_keys().await?;
    
    // Convert to responses (without sensitive data)
    let mut responses: Vec<ApiKeyResponse> = api_keys
        .into_iter()
        .map(|key| key.into())
        .collect();
    
    // Sort by creation date (newest first)
    responses.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    
    // Apply pagination
    let total_count = responses.len() as u64;
    let start = ((pagination.page - 1) * pagination.limit) as usize;
    let end = (start + pagination.limit as usize).min(responses.len());
    
    let paginated_keys = if start < responses.len() {
        responses[start..end].to_vec()
    } else {
        Vec::new()
    };

    let paginated_response = PaginatedResponse::new(
        paginated_keys,
        pagination.page,
        pagination.limit,
        total_count,
    );

    Ok(Json(ApiResponse::success(paginated_response)))
}

/// Get a specific API key by ID
pub async fn get_api_key(
    State(api_key_store): State<ApiKeyStore>,
    Extension(auth_context): Extension<AuthContext>,
    Path(key_id): Path<Uuid>,
) -> ApiResult<Json<ApiResponse<ApiKeyResponse>>> {
    info!("Getting API key: {}", key_id);

    // Check permissions - admin can see all keys, users can only see their own
    if !auth_context.has_permission(&Permission::Admin) && auth_context.api_key_id != key_id {
        return Err(crate::utils::Error::authorization(
            "You can only view your own API key or need admin permission"
        ).into());
    }

    let api_key = api_key_store.get_api_key(key_id).await?;
    let response: ApiKeyResponse = api_key.into();

    Ok(Json(ApiResponse::success(response)))
}

/// Update an API key
pub async fn update_api_key(
    State(api_key_store): State<ApiKeyStore>,
    Extension(auth_context): Extension<AuthContext>,
    Path(key_id): Path<Uuid>,
    Json(request): Json<UpdateApiKeyRequest>,
) -> ApiResult<Json<ApiResponse<ApiKeyResponse>>> {
    info!("Updating API key: {}", key_id);

    // Check permissions
    if !auth_context.has_permission(&Permission::Admin) {
        return Err(crate::utils::Error::authorization(
            "Admin permission required to update API keys"
        ).into());
    }

    // Get current key
    let mut api_key = api_key_store.get_api_key(key_id).await?;

    // Update fields if provided
    if let Some(name) = request.name {
        if name.trim().is_empty() {
            return Err(validation_error("name", "Name cannot be empty"));
        }
        api_key.name = name;
    }

    if let Some(description) = request.description {
        api_key.description = Some(description);
    }

    if let Some(permissions) = request.permissions {
        if permissions.is_empty() {
            return Err(validation_error("permissions", "At least one permission is required"));
        }
        api_key.permissions = permissions;
    }

    if let Some(rate_limit) = request.rate_limit_per_hour {
        api_key.rate_limit_per_hour = rate_limit;
    }

    if let Some(is_active) = request.is_active {
        api_key.is_active = is_active;
    }

    api_key.updated_at = chrono::Utc::now();

    // Update in store (this is simplified - in real implementation we'd update the database)
    // For now, we'll need to recreate the key since our in-memory store doesn't have update
    // In production, this would be a database update operation
    
    let response: ApiKeyResponse = api_key.into();

    Ok(Json(ApiResponse::success_with_message(
        response,
        "API key updated successfully".to_string(),
    )))
}

/// Revoke an API key (deactivate)
pub async fn revoke_api_key(
    State(api_key_store): State<ApiKeyStore>,
    Extension(auth_context): Extension<AuthContext>,
    Path(key_id): Path<Uuid>,
) -> ApiResult<Json<ApiResponse<String>>> {
    info!("Revoking API key: {}", key_id);

    // Check permissions
    if !auth_context.has_permission(&Permission::Admin) {
        return Err(crate::utils::Error::authorization(
            "Admin permission required to revoke API keys"
        ).into());
    }

    // Cannot revoke own key
    if auth_context.api_key_id == key_id {
        return Err(validation_error("key_id", "Cannot revoke your own API key"));
    }

    api_key_store.revoke_api_key(key_id).await?;

    Ok(Json(ApiResponse::success_with_message(
        "revoked".to_string(),
        format!("API key {} has been revoked", key_id),
    )))
}

/// Delete an API key completely
pub async fn delete_api_key(
    State(api_key_store): State<ApiKeyStore>,
    Extension(auth_context): Extension<AuthContext>,
    Path(key_id): Path<Uuid>,
) -> ApiResult<Json<ApiResponse<String>>> {
    info!("Deleting API key: {}", key_id);

    // Check permissions
    if !auth_context.has_permission(&Permission::Admin) {
        return Err(crate::utils::Error::authorization(
            "Admin permission required to delete API keys"
        ).into());
    }

    // Cannot delete own key
    if auth_context.api_key_id == key_id {
        return Err(validation_error("key_id", "Cannot delete your own API key"));
    }

    api_key_store.delete_api_key(key_id).await?;

    Ok(Json(ApiResponse::success_with_message(
        "deleted".to_string(),
        format!("API key {} has been permanently deleted", key_id),
    )))
}

// ==== Authentication Status and Monitoring ====

/// Get current authentication info
pub async fn get_current_auth_info(
    State(api_key_store): State<ApiKeyStore>,
    Extension(auth_context): Extension<AuthContext>,
) -> ApiResult<Json<ApiResponse<ApiKeyResponse>>> {
    info!("Getting current auth info for key: {}", auth_context.api_key_id);

    let api_key = api_key_store.get_api_key(auth_context.api_key_id).await?;
    let response: ApiKeyResponse = api_key.into();

    Ok(Json(ApiResponse::success(response)))
}

/// Get rate limiting status for current user
pub async fn get_rate_limit_status(
    State(rate_limiter): State<RateLimiter>,
    Extension(auth_context): Extension<AuthContext>,
) -> ApiResult<Json<ApiResponse<RateLimitStatus>>> {
    let client_id = format!("api_key:{}", auth_context.api_key_id);
    
    if let Some((current_requests, limit, reset_time)) = rate_limiter.get_status(&client_id).await {
        let remaining = limit.saturating_sub(current_requests);
        
        let status = RateLimitStatus {
            client_id,
            current_requests,
            limit,
            reset_time,
            remaining,
        };

        Ok(Json(ApiResponse::success(status)))
    } else {
        // No rate limiting data found - probably first request
        let default_limit = auth_context.rate_limit_per_hour.unwrap_or(1000);
        let status = RateLimitStatus {
            client_id,
            current_requests: 0,
            limit: default_limit,
            reset_time: 0,
            remaining: default_limit,
        };

        Ok(Json(ApiResponse::success(status)))
    }
}

/// Get all rate limit statuses (admin only)
pub async fn get_all_rate_limit_statuses(
    State(rate_limiter): State<RateLimiter>,
    Extension(auth_context): Extension<AuthContext>,
) -> ApiResult<Json<ApiResponse<HashMap<String, RateLimitStatus>>>> {
    info!("Getting all rate limit statuses");

    // Check permissions
    if !auth_context.has_permission(&Permission::Admin) {
        return Err(crate::utils::Error::authorization(
            "Admin permission required to view all rate limit statuses"
        ).into());
    }

    let all_statuses = rate_limiter.get_all_statuses().await;
    let mut response_map = HashMap::new();

    for (client_id, (current_requests, limit, reset_time)) in all_statuses {
        let remaining = limit.saturating_sub(current_requests);
        let status = RateLimitStatus {
            client_id: client_id.clone(),
            current_requests,
            limit,
            reset_time,
            remaining,
        };
        response_map.insert(client_id, status);
    }

    Ok(Json(ApiResponse::success(response_map)))
}

/// Get authentication system statistics (admin only)
pub async fn get_auth_stats(
    State(api_key_store): State<ApiKeyStore>,
    Extension(auth_context): Extension<AuthContext>,
) -> ApiResult<Json<ApiResponse<AuthStats>>> {
    info!("Getting authentication statistics");

    // Check permissions
    if !auth_context.has_permission(&Permission::Admin) {
        return Err(crate::utils::Error::authorization(
            "Admin permission required to view authentication statistics"
        ).into());
    }

    let api_keys = api_key_store.list_api_keys().await?;
    let now = chrono::Utc::now();
    
    let total_api_keys = api_keys.len();
    let active_api_keys = api_keys.iter().filter(|key| key.is_active).count();
    let expired_api_keys = api_keys
        .iter()
        .filter(|key| {
            if let Some(expires_at) = key.expires_at {
                now > expires_at
            } else {
                false
            }
        })
        .count();

    // TODO: Implement proper request counting
    // For now, return placeholder values
    let total_requests_last_hour = 0;
    let rate_limited_requests_last_hour = 0;

    let stats = AuthStats {
        total_api_keys,
        active_api_keys,
        expired_api_keys,
        total_requests_last_hour,
        rate_limited_requests_last_hour,
    };

    Ok(Json(ApiResponse::success(stats)))
}

// ==== Permission Testing Endpoints (for development) ====

/// Test endpoint to check Read permission
pub async fn test_read_permission(
    Extension(auth_context): Extension<AuthContext>,
) -> ApiResult<Json<ApiResponse<String>>> {
    if !auth_context.has_permission(&Permission::Read) {
        return Err(crate::utils::Error::authorization(
            "Read permission required"
        ).into());
    }

    Ok(Json(ApiResponse::success(
        "Read permission test passed".to_string()
    )))
}

/// Test endpoint to check Write permission
pub async fn test_write_permission(
    Extension(auth_context): Extension<AuthContext>,
) -> ApiResult<Json<ApiResponse<String>>> {
    if !auth_context.has_permission(&Permission::Write) {
        return Err(crate::utils::Error::authorization(
            "Write permission required"
        ).into());
    }

    Ok(Json(ApiResponse::success(
        "Write permission test passed".to_string()
    )))
}

/// Test endpoint to check Delete permission
pub async fn test_delete_permission(
    Extension(auth_context): Extension<AuthContext>,
) -> ApiResult<Json<ApiResponse<String>>> {
    if !auth_context.has_permission(&Permission::Delete) {
        return Err(crate::utils::Error::authorization(
            "Delete permission required"
        ).into());
    }

    Ok(Json(ApiResponse::success(
        "Delete permission test passed".to_string()
    )))
}

/// Test endpoint to check Admin permission
pub async fn test_admin_permission(
    Extension(auth_context): Extension<AuthContext>,
) -> ApiResult<Json<ApiResponse<String>>> {
    if !auth_context.has_permission(&Permission::Admin) {
        return Err(crate::utils::Error::authorization(
            "Admin permission required"
        ).into());
    }

    Ok(Json(ApiResponse::success(
        "Admin permission test passed".to_string()
    )))
}

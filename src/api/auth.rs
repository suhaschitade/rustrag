use axum::{
    extract::{Request, State},
    http::{HeaderMap, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};
use tracing::{error, info, warn};
use uuid::Uuid;

use crate::{
    api::{validation_error, ApiResult, ErrorResponse},
    utils::Error,
};

/// API key permissions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Permission {
    /// Can read documents and query
    Read,
    /// Can upload and modify documents
    Write,
    /// Can delete documents
    Delete,
    /// Can perform administrative operations
    Admin,
    /// Can access all operations (super admin)
    All,
}

/// API key information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKey {
    pub id: Uuid,
    pub key_hash: String,
    pub name: String,
    pub description: Option<String>,
    pub permissions: Vec<Permission>,
    pub rate_limit_per_hour: Option<u32>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub last_used_at: Option<DateTime<Utc>>,
    pub is_active: bool,
    pub usage_count: u64,
}

/// API key creation request
#[derive(Debug, Deserialize)]
pub struct CreateApiKeyRequest {
    pub name: String,
    pub description: Option<String>,
    pub permissions: Vec<Permission>,
    pub rate_limit_per_hour: Option<u32>,
    pub expires_in_days: Option<u32>,
}

/// API key response (without sensitive information)
#[derive(Debug, Serialize, Clone)]
pub struct ApiKeyResponse {
    pub id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub permissions: Vec<Permission>,
    pub rate_limit_per_hour: Option<u32>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub last_used_at: Option<DateTime<Utc>>,
    pub is_active: bool,
    pub usage_count: u64,
    /// Only included when creating a new key
    pub key: Option<String>,
}

impl From<ApiKey> for ApiKeyResponse {
    fn from(api_key: ApiKey) -> Self {
        Self {
            id: api_key.id,
            name: api_key.name,
            description: api_key.description,
            permissions: api_key.permissions,
            rate_limit_per_hour: api_key.rate_limit_per_hour,
            created_at: api_key.created_at,
            updated_at: api_key.updated_at,
            expires_at: api_key.expires_at,
            last_used_at: api_key.last_used_at,
            is_active: api_key.is_active,
            usage_count: api_key.usage_count,
            key: None,
        }
    }
}

/// Authentication context added to request extensions
#[derive(Debug, Clone)]
pub struct AuthContext {
    pub api_key_id: Uuid,
    pub permissions: Vec<Permission>,
    pub rate_limit_per_hour: Option<u32>,
}

impl AuthContext {
    /// Check if the current user has the required permission
    pub fn has_permission(&self, required_permission: &Permission) -> bool {
        self.permissions.contains(&Permission::All) 
            || self.permissions.contains(required_permission)
    }

    /// Check if the current user has any of the required permissions
    pub fn has_any_permission(&self, required_permissions: &[Permission]) -> bool {
        if self.permissions.contains(&Permission::All) {
            return true;
        }
        required_permissions.iter().any(|perm| self.permissions.contains(perm))
    }
}

/// In-memory API key store (in production, this would be a database)
#[derive(Debug, Clone)]
pub struct ApiKeyStore {
    keys: Arc<RwLock<HashMap<String, ApiKey>>>,
    keys_by_id: Arc<RwLock<HashMap<Uuid, String>>>,
}

impl ApiKeyStore {
    pub fn new() -> Self {
        let mut store = Self {
            keys: Arc::new(RwLock::new(HashMap::new())),
            keys_by_id: Arc::new(RwLock::new(HashMap::new())),
        };

        // Create a default admin API key for development
        let admin_key = store.create_default_admin_key();
        info!("Created default admin API key: {}", admin_key);

        store
    }

    /// Create a default admin API key for development
    fn create_default_admin_key(&self) -> String {
        let key = self.generate_api_key();
        let key_hash = self.hash_api_key(&key);
        
        let api_key = ApiKey {
            id: Uuid::new_v4(),
            key_hash: key_hash.clone(),
            name: "Default Admin Key".to_string(),
            description: Some("Default admin key for development - should be replaced in production".to_string()),
            permissions: vec![Permission::All],
            rate_limit_per_hour: None, // No rate limit for admin
            created_at: Utc::now(),
            updated_at: Utc::now(),
            expires_at: None, // Never expires
            last_used_at: None,
            is_active: true,
            usage_count: 0,
        };

        let mut keys = self.keys.write().unwrap();
        let mut keys_by_id = self.keys_by_id.write().unwrap();
        
        keys.insert(key_hash.clone(), api_key.clone());
        keys_by_id.insert(api_key.id, key_hash);

        key
    }

    /// Generate a new API key
    fn generate_api_key(&self) -> String {
        let prefix = "rag_";
        let random_part = Uuid::new_v4().to_string().replace("-", "");
        format!("{}{}", prefix, random_part)
    }

    /// Hash an API key (in production, use a proper hash function like Argon2)
    fn hash_api_key(&self, key: &str) -> String {
        // Simple hash for demonstration - in production, use proper crypto
        format!("hash_{}", sha256::digest(key))
    }

    /// Create a new API key
    pub async fn create_api_key(&self, request: CreateApiKeyRequest) -> ApiResult<(ApiKey, String)> {
        if request.name.trim().is_empty() {
            return Err(validation_error("name", "API key name cannot be empty"));
        }

        if request.permissions.is_empty() {
            return Err(validation_error("permissions", "At least one permission is required"));
        }

        let key = self.generate_api_key();
        let key_hash = self.hash_api_key(&key);
        
        let expires_at = request.expires_in_days.map(|days| 
            Utc::now() + chrono::Duration::days(days as i64)
        );

        let api_key = ApiKey {
            id: Uuid::new_v4(),
            key_hash: key_hash.clone(),
            name: request.name,
            description: request.description,
            permissions: request.permissions,
            rate_limit_per_hour: request.rate_limit_per_hour,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            expires_at,
            last_used_at: None,
            is_active: true,
            usage_count: 0,
        };

        let mut keys = self.keys.write().unwrap();
        let mut keys_by_id = self.keys_by_id.write().unwrap();
        
        keys.insert(key_hash.clone(), api_key.clone());
        keys_by_id.insert(api_key.id, key_hash);

        info!("Created new API key: {} ({})", api_key.name, api_key.id);
        Ok((api_key, key))
    }

    /// Validate an API key and return the associated information
    pub async fn validate_api_key(&self, key: &str) -> ApiResult<ApiKey> {
        let key_hash = self.hash_api_key(key);
        
        let mut keys = self.keys.write().unwrap();
        
        if let Some(api_key) = keys.get_mut(&key_hash) {
            // Check if key is active
            if !api_key.is_active {
                return Err(Error::authentication("API key is deactivated"));
            }

            // Check if key has expired
            if let Some(expires_at) = api_key.expires_at {
                if Utc::now() > expires_at {
                    return Err(Error::authentication("API key has expired"));
                }
            }

            // Update last used time and usage count
            api_key.last_used_at = Some(Utc::now());
            api_key.usage_count += 1;

            Ok(api_key.clone())
        } else {
            Err(Error::authentication("Invalid API key"))
        }
    }

    /// List all API keys (admin only)
    pub async fn list_api_keys(&self) -> ApiResult<Vec<ApiKey>> {
        let keys = self.keys.read().unwrap();
        Ok(keys.values().cloned().collect())
    }

    /// Get API key by ID
    pub async fn get_api_key(&self, key_id: Uuid) -> ApiResult<ApiKey> {
        let keys_by_id = self.keys_by_id.read().unwrap();
        let keys = self.keys.read().unwrap();
        
        if let Some(key_hash) = keys_by_id.get(&key_id) {
            if let Some(api_key) = keys.get(key_hash) {
                return Ok(api_key.clone());
            }
        }
        
        Err(Error::not_found("API key"))
    }

    /// Revoke an API key
    pub async fn revoke_api_key(&self, key_id: Uuid) -> ApiResult<()> {
        let keys_by_id = self.keys_by_id.read().unwrap();
        let mut keys = self.keys.write().unwrap();
        
        if let Some(key_hash) = keys_by_id.get(&key_id) {
            if let Some(api_key) = keys.get_mut(key_hash) {
                api_key.is_active = false;
                api_key.updated_at = Utc::now();
                info!("Revoked API key: {} ({})", api_key.name, api_key.id);
                return Ok(());
            }
        }
        
        Err(Error::not_found("API key"))
    }

    /// Delete an API key completely
    pub async fn delete_api_key(&self, key_id: Uuid) -> ApiResult<()> {
        let mut keys_by_id = self.keys_by_id.write().unwrap();
        let mut keys = self.keys.write().unwrap();
        
        if let Some(key_hash) = keys_by_id.remove(&key_id) {
            if let Some(api_key) = keys.remove(&key_hash) {
                info!("Deleted API key: {} ({})", api_key.name, api_key.id);
                return Ok(());
            }
        }
        
        Err(Error::not_found("API key"))
    }
}

impl Default for ApiKeyStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Authentication middleware
pub async fn auth_middleware(
    State(api_key_store): State<ApiKeyStore>,
    mut request: Request,
    next: Next,
) -> Result<Response, impl IntoResponse> {
    // Skip auth for health endpoints
    let path = request.uri().path();
    if path == "/health" || path == "/api/v1/health" || path.starts_with("/api/v1/health/") {
        return Ok(next.run(request).await);
    }

    // Extract API key from headers
    let api_key = match extract_api_key(request.headers()) {
        Ok(key) => key,
        Err(error_response) => return Err(error_response),
    };

    // Validate API key
    match api_key_store.validate_api_key(&api_key).await {
        Ok(key_info) => {
            let key_id = key_info.id;
            let key_name = key_info.name.clone();
            let permissions = key_info.permissions.clone();
            
            // Add auth context to request extensions
            let auth_context = AuthContext {
                api_key_id: key_info.id,
                permissions: key_info.permissions,
                rate_limit_per_hour: key_info.rate_limit_per_hour,
            };
            
            request.extensions_mut().insert(auth_context);
            
            info!(
                api_key_id = %key_id,
                api_key_name = %key_name,
                permissions = ?permissions,
                "Authentication successful"
            );
            
            Ok(next.run(request).await)
        }
        Err(e) => {
            warn!("Authentication failed: {}", e);
            Err((
                StatusCode::UNAUTHORIZED,
                Json(ErrorResponse::new(
                    "AUTHENTICATION_ERROR",
                    &e.to_string(),
                    401,
                )),
            ))
        }
    }
}

/// Extract API key from request headers
fn extract_api_key(headers: &HeaderMap) -> Result<String, (StatusCode, Json<ErrorResponse>)> {
    // Check Authorization header (Bearer token)
    if let Some(auth_header) = headers.get("authorization") {
        if let Ok(auth_str) = auth_header.to_str() {
            if let Some(token) = auth_str.strip_prefix("Bearer ") {
                if !token.trim().is_empty() {
                    return Ok(token.to_string());
                }
            }
        }
    }

    // Check X-API-Key header
    if let Some(api_key_header) = headers.get("x-api-key") {
        if let Ok(api_key) = api_key_header.to_str() {
            if !api_key.trim().is_empty() {
                return Ok(api_key.to_string());
            }
        }
    }

    Err((
        StatusCode::UNAUTHORIZED,
        Json(ErrorResponse::new(
            "AUTHENTICATION_ERROR",
            "Missing API key. Provide via 'Authorization: Bearer <token>' or 'X-API-Key' header",
            401,
        )),
    ))
}

// Note: Permission middleware functions are simplified for now
// In a production system, these would be implemented as proper middleware layers

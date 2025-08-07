use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Standard API response wrapper
#[derive(Debug, Serialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub message: Option<String>,
    pub timestamp: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
}

impl<T> ApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            message: None,
            timestamp: Utc::now(),
            request_id: None,
        }
    }

    pub fn success_with_message(data: T, message: String) -> Self {
        Self {
            success: true,
            data: Some(data),
            message: Some(message),
            timestamp: Utc::now(),
            request_id: None,
        }
    }

    pub fn with_request_id(mut self, request_id: String) -> Self {
        self.request_id = Some(request_id);
        self
    }
}

/// Pagination parameters for list endpoints
#[derive(Debug, Deserialize)]
pub struct PaginationParams {
    #[serde(default = "default_page")]
    pub page: u64,
    
    #[serde(default = "default_limit")]
    pub limit: u64,
    
    #[serde(default)]
    pub sort_by: Option<String>,
    
    #[serde(default)]
    pub sort_order: Option<SortOrder>,
}

fn default_page() -> u64 { 1 }
fn default_limit() -> u64 { 20 }

#[derive(Debug, Deserialize, Serialize)]
pub enum SortOrder {
    #[serde(rename = "asc")]
    Ascending,
    #[serde(rename = "desc")]
    Descending,
}

impl Default for SortOrder {
    fn default() -> Self {
        Self::Descending
    }
}

impl PaginationParams {
    pub fn validate(&self) -> Result<(), String> {
        if self.page == 0 {
            return Err("Page must be greater than 0".to_string());
        }
        
        if self.limit == 0 {
            return Err("Limit must be greater than 0".to_string());
        }
        
        if self.limit > 100 {
            return Err("Limit cannot exceed 100".to_string());
        }
        
        Ok(())
    }

    pub fn offset(&self) -> u64 {
        (self.page - 1) * self.limit
    }
}

/// Paginated response wrapper
#[derive(Debug, Serialize)]
pub struct PaginatedResponse<T> {
    pub items: Vec<T>,
    pub pagination: PaginationInfo,
}

#[derive(Debug, Serialize)]
pub struct PaginationInfo {
    pub page: u64,
    pub limit: u64,
    pub total_items: u64,
    pub total_pages: u64,
    pub has_next_page: bool,
    pub has_previous_page: bool,
}

impl<T> PaginatedResponse<T> {
    pub fn new(
        items: Vec<T>,
        page: u64,
        limit: u64,
        total_items: u64,
    ) -> Self {
        let total_pages = (total_items as f64 / limit as f64).ceil() as u64;
        let has_next_page = page < total_pages;
        let has_previous_page = page > 1;

        Self {
            items,
            pagination: PaginationInfo {
                page,
                limit,
                total_items,
                total_pages,
                has_next_page,
                has_previous_page,
            },
        }
    }
}

/// Health check response
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub uptime: String,
    pub services: ServiceHealthStatus,
}

#[derive(Debug, Serialize)]
pub struct ServiceHealthStatus {
    pub database: ServiceStatus,
    pub vector_store: ServiceStatus,
    pub llm_provider: ServiceStatus,
}

#[derive(Debug, Serialize)]
pub struct ServiceStatus {
    pub status: String,
    pub response_time_ms: Option<u64>,
    pub last_error: Option<String>,
}

impl ServiceStatus {
    pub fn healthy(response_time_ms: u64) -> Self {
        Self {
            status: "healthy".to_string(),
            response_time_ms: Some(response_time_ms),
            last_error: None,
        }
    }

    pub fn unhealthy(error: String) -> Self {
        Self {
            status: "unhealthy".to_string(),
            response_time_ms: None,
            last_error: Some(error),
        }
    }

    pub fn unknown() -> Self {
        Self {
            status: "unknown".to_string(),
            response_time_ms: None,
            last_error: None,
        }
    }
}

/// Document upload response
#[derive(Debug, Serialize)]
pub struct DocumentUploadResponse {
    pub id: Uuid,
    pub filename: String,
    pub size_bytes: u64,
    pub mime_type: String,
    pub status: DocumentStatus,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum DocumentStatus {
    #[serde(rename = "pending")]
    Pending,
    #[serde(rename = "processing")]
    Processing,
    #[serde(rename = "completed")]
    Completed,
    #[serde(rename = "failed")]
    Failed,
    #[serde(rename = "deleted")]
    Deleted,
}

/// Query processing response
#[derive(Debug, Serialize)]
pub struct QueryProcessingResponse {
    pub query_id: Uuid,
    pub query: String,
    pub answer: String,
    pub confidence_score: f32,
    pub retrieved_chunks: Vec<RetrievedChunk>,
    pub citations: Vec<Citation>,
    pub processing_time_ms: u64,
    pub model_used: String,
}

#[derive(Debug, Serialize)]
pub struct RetrievedChunk {
    pub id: Uuid,
    pub document_id: Uuid,
    pub content: String,
    pub similarity_score: f32,
    pub chunk_index: u32,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct Citation {
    pub document_id: Uuid,
    pub document_title: String,
    pub chunk_id: Uuid,
    pub page_number: Option<u32>,
    pub confidence: f32,
}

/// Batch operation response
#[derive(Debug, Serialize)]
pub struct BatchOperationResponse {
    pub batch_id: Uuid,
    pub total_items: u32,
    pub successful: u32,
    pub failed: u32,
    pub errors: Vec<BatchError>,
    pub status: BatchStatus,
}

#[derive(Debug, Serialize)]
pub struct BatchError {
    pub item_id: String,
    pub error_code: String,
    pub error_message: String,
}

#[derive(Debug, Serialize)]
pub enum BatchStatus {
    #[serde(rename = "pending")]
    Pending,
    #[serde(rename = "running")]
    Running,
    #[serde(rename = "completed")]
    Completed,
    #[serde(rename = "partially_completed")]
    PartiallyCompleted,
    #[serde(rename = "failed")]
    Failed,
}

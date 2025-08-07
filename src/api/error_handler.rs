use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use tracing::error;

use crate::utils::Error;

/// API Error response structure
#[derive(Debug, serde::Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub message: String,
    pub code: u16,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
}

impl ErrorResponse {
    pub fn new(error: &str, message: &str, code: u16) -> Self {
        Self {
            error: error.to_string(),
            message: message.to_string(),
            code,
            details: None,
        }
    }

    pub fn with_details(mut self, details: serde_json::Value) -> Self {
        self.details = Some(details);
        self
    }
}

/// Convert our custom Error type to HTTP responses
impl IntoResponse for Error {
    fn into_response(self) -> Response {
        let (status, error_type, message) = match &self {
            Error::Validation { field, message } => (
                StatusCode::BAD_REQUEST,
                "VALIDATION_ERROR",
                format!("Validation failed for '{}': {}", field, message),
            ),
            Error::NotFound { resource } => (
                StatusCode::NOT_FOUND,
                "NOT_FOUND",
                format!("Resource not found: {}", resource),
            ),
            Error::Authentication { message } => (
                StatusCode::UNAUTHORIZED,
                "AUTHENTICATION_ERROR",
                message.clone(),
            ),
            Error::Authorization { message } => (
                StatusCode::FORBIDDEN,
                "AUTHORIZATION_ERROR",
                message.clone(),
            ),
            Error::DocumentProcessing { message } => (
                StatusCode::UNPROCESSABLE_ENTITY,
                "DOCUMENT_PROCESSING_ERROR",
                message.clone(),
            ),
            Error::VectorDb { message } => (
                StatusCode::SERVICE_UNAVAILABLE,
                "VECTOR_DB_ERROR",
                message.clone(),
            ),
            Error::LlmApi { message } => (
                StatusCode::BAD_GATEWAY,
                "LLM_API_ERROR",
                message.clone(),
            ),
            Error::ExternalService { service, message } => (
                StatusCode::BAD_GATEWAY,
                "EXTERNAL_SERVICE_ERROR",
                format!("{}: {}", service, message),
            ),
            _ => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "INTERNAL_SERVER_ERROR",
                "An internal server error occurred".to_string(),
            ),
        };

        // Log the error
        error!("API Error: {} - {}", error_type, message);

        let error_response = ErrorResponse::new(
            error_type,
            &message,
            status.as_u16(),
        );

        (status, Json(error_response)).into_response()
    }
}

/// Custom result type for API handlers
pub type ApiResult<T> = Result<T, Error>;

/// Helper function to create validation errors
pub fn validation_error(field: &str, message: &str) -> Error {
    Error::validation(field, message)
}

/// Helper function to create not found errors
pub fn not_found_error(resource: &str) -> Error {
    Error::not_found(resource)
}

/// Helper function to create internal server errors
pub fn internal_error(message: &str) -> Error {
    Error::internal(message)
}

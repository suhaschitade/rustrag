pub mod mock;
pub use mock::MockProvider;

#[cfg(feature = "embeddings-openai")]
pub mod openai;
#[cfg(feature = "embeddings-openai")]
pub use openai::OpenAIProvider;

#[cfg(feature = "embeddings-onnx")]
pub mod onnx;
#[cfg(feature = "embeddings-onnx")]
pub use onnx::ONNXProvider;

#[cfg(feature = "embeddings-candle")]
pub mod candle;
#[cfg(feature = "embeddings-candle")]
pub use candle::CandleProvider;

use thiserror::Error;

/// Embedding provider specific errors
#[derive(Error, Debug)]
pub enum EmbeddingError {
    #[error("Provider not found: {0}")]
    ProviderNotFound(String),
    
    #[error("No embedding providers available")]
    NoProvidersAvailable,
    
    #[error("Provider initialization failed: {0}")]
    InitializationFailed(String),
    
    #[error("API request failed: {0}")]
    ApiError(String),
    
    #[error("Authentication failed: {0}")]
    AuthenticationError(String),
    
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    
    #[error("Input too long: {actual} exceeds maximum {max}")]
    InputTooLong { actual: usize, max: usize },
    
    #[error("Batch too large: {actual} exceeds maximum {max}")]
    BatchTooLarge { actual: usize, max: usize },
    
    #[error("Invalid configuration: {0}")]
    ConfigurationError(String),
    
    #[error("Network error: {0}")]
    NetworkError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Provider health check failed: {0}")]
    HealthCheckFailed(String),
    
    #[error("Rate limit exceeded: {0}")]
    RateLimitExceeded(String),
    
    #[error("Unexpected provider error: {0}")]
    Other(String),
}

impl From<EmbeddingError> for crate::utils::Error {
    fn from(e: EmbeddingError) -> Self {
        crate::utils::Error::embedding(e.to_string())
    }
}

/// Common utilities for providers
pub mod utils {
    use super::EmbeddingError;
    
    /// Validate text length against provider limits
    pub fn validate_text_length(text: &str, max_length: usize) -> Result<(), EmbeddingError> {
        let actual_length = text.len();
        if actual_length > max_length {
            return Err(EmbeddingError::InputTooLong {
                actual: actual_length,
                max: max_length,
            });
        }
        Ok(())
    }
    
    /// Validate batch size against provider limits
    pub fn validate_batch_size(batch_size: usize, max_batch_size: usize) -> Result<(), EmbeddingError> {
        if batch_size > max_batch_size {
            return Err(EmbeddingError::BatchTooLarge {
                actual: batch_size,
                max: max_batch_size,
            });
        }
        Ok(())
    }
    
    /// Truncate text to fit within provider limits
    pub fn truncate_text(text: &str, max_length: usize) -> String {
        if text.len() <= max_length {
            text.to_string()
        } else {
            // Try to truncate at word boundary
            let truncated = &text[..max_length];
            if let Some(last_space) = truncated.rfind(' ') {
                truncated[..last_space].to_string()
            } else {
                truncated.to_string()
            }
        }
    }
}

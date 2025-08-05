/// Custom error type for RustRAG operations
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Configuration error: {0}")]
    Config(#[from] config::ConfigError),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Logging error: {0}")]
    Logging(String),

    #[error("Vector database error: {message}")]
    VectorDb { message: String },

    #[error("Document processing error: {message}")]
    DocumentProcessing { message: String },

    #[error("Embedding generation error: {message}")]
    Embedding { message: String },

    #[error("LLM API error: {message}")]
    LlmApi { message: String },

    #[error("Authentication error: {message}")]
    Authentication { message: String },

    #[error("Authorization error: {message}")]
    Authorization { message: String },

    #[error("Validation error: {field}: {message}")]
    Validation { field: String, message: String },

    #[error("Not found: {resource}")]
    NotFound { resource: String },

    #[error("Internal server error: {message}")]
    Internal { message: String },

    #[error("External service error: {service}: {message}")]
    ExternalService { service: String, message: String },
}

impl Error {
    /// Create a new vector database error
    pub fn vector_db<T: Into<String>>(message: T) -> Self {
        Self::VectorDb {
            message: message.into(),
        }
    }

    /// Create a new document processing error
    pub fn document_processing<T: Into<String>>(message: T) -> Self {
        Self::DocumentProcessing {
            message: message.into(),
        }
    }

    /// Create a new embedding error
    pub fn embedding<T: Into<String>>(message: T) -> Self {
        Self::Embedding {
            message: message.into(),
        }
    }

    /// Create a new LLM API error
    pub fn llm_api<T: Into<String>>(message: T) -> Self {
        Self::LlmApi {
            message: message.into(),
        }
    }

    /// Create a new authentication error
    pub fn authentication<T: Into<String>>(message: T) -> Self {
        Self::Authentication {
            message: message.into(),
        }
    }

    /// Create a new authorization error
    pub fn authorization<T: Into<String>>(message: T) -> Self {
        Self::Authorization {
            message: message.into(),
        }
    }

    /// Create a new validation error
    pub fn validation<T: Into<String>, U: Into<String>>(field: T, message: U) -> Self {
        Self::Validation {
            field: field.into(),
            message: message.into(),
        }
    }

    /// Create a new not found error
    pub fn not_found<T: Into<String>>(resource: T) -> Self {
        Self::NotFound {
            resource: resource.into(),
        }
    }

    /// Create a new internal error
    pub fn internal<T: Into<String>>(message: T) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }

    /// Create a new external service error
    pub fn external_service<T: Into<String>, U: Into<String>>(service: T, message: U) -> Self {
        Self::ExternalService {
            service: service.into(),
            message: message.into(),
        }
    }

    /// Get HTTP status code for this error
    pub fn status_code(&self) -> u16 {
        match self {
            Error::Authentication { .. } => 401,
            Error::Authorization { .. } => 403,
            Error::NotFound { .. } => 404,
            Error::Validation { .. } => 400,
            _ => 500,
        }
    }

    /// Check if error is retriable
    pub fn is_retriable(&self) -> bool {
        match self {
            Error::VectorDb { .. } => true,
            Error::LlmApi { .. } => true,
            Error::ExternalService { .. } => true,
            _ => false,
        }
    }
}

// Add conversion from Box<dyn std::error::Error + Send + Sync>
impl From<Box<dyn std::error::Error + Send + Sync>> for Error {
    fn from(err: Box<dyn std::error::Error + Send + Sync>) -> Self {
        Error::Logging(err.to_string())
    }
}

/// Result type alias for RustRAG operations
pub type Result<T> = std::result::Result<T, Error>;

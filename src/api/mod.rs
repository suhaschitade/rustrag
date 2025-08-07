// API modules
pub mod types;
pub mod error_handler;
pub mod auth_endpoints;
pub mod documents;
pub mod health;
pub mod queries;
pub mod upload_document;
pub mod middleware;
pub mod rate_limiter;
pub mod auth;
pub mod rate_limit;
pub mod query_expansion;
pub mod metrics;

// Re-exports
pub use types::*;
// Note: error_handler is used internally but not re-exported to avoid conflicts
pub use auth_endpoints::*;
pub use documents::*;
pub use health::*;
pub use queries::*;
pub use query_expansion::*;
pub use middleware::*;
pub use auth::*;
pub use rate_limiter::*;
pub use metrics::*;

// Core API functionality
pub mod router;

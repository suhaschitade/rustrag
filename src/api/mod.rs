// API modules
pub mod documents;
pub mod health;
pub mod queries;
pub mod middleware;
pub mod error_handler;
pub mod types;
pub mod auth;
pub mod auth_endpoints;
pub mod rate_limiter;

// Re-exports
pub use documents::*;
pub use health::*;
pub use queries::*;
pub use middleware::*;
pub use error_handler::*;
pub use types::*;
pub use auth::*;
pub use auth_endpoints::*;
pub use rate_limiter::*;

// Core API functionality
pub mod router;

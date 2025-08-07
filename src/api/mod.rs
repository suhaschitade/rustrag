// API modules
pub mod documents;
pub mod health;
pub mod queries;
pub mod middleware;
pub mod error_handler;
pub mod types;

// Re-exports
pub use documents::*;
pub use health::*;
pub use queries::*;
pub use middleware::*;
pub use error_handler::*;
pub use types::*;

// Core API functionality
pub mod router;

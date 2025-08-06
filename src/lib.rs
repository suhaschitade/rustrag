//! # RustRAG: Enterprise Contextual AI Assistant Platform
//!
//! A high-performance, privacy-focused Retrieval-Augmented Generation (RAG) platform
//! built in Rust for enterprise applications.

#[cfg(feature = "web-server")]
pub mod api;
pub mod config;
pub mod core;
// pub mod embedding; // TODO: Implement embedding module
pub mod indexing;
pub mod models;
// pub mod processing; // TODO: Implement processing module
pub mod storage;
pub mod utils;

// Re-export commonly used types for convenience
pub use config::Settings;
pub use models::{Document, DocumentChunk, Query, QueryResponse};
pub use utils::{Error, Result};

/// Current version of the RustRAG platform
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Initialize the RustRAG library with default configuration
pub async fn init() -> Result<()> {
    utils::logging::init_tracing()?;
    Ok(())
}

/// Initialize the RustRAG library with custom settings
pub async fn init_with_config(settings: Settings) -> Result<()> {
    utils::logging::init_tracing_with_config(&settings)?;
    Ok(())
}

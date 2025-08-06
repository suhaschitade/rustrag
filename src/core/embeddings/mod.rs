pub mod providers;
pub mod service;
pub mod config;
pub mod builder;
pub mod examples;

// Re-export core types and traits
pub use service::{EmbeddingService, EmbeddingProvider, ProviderStats};
pub use config::{EmbeddingConfig, ProviderConfig, EmbeddingModel};
pub use providers::{MockProvider, EmbeddingError};
pub use builder::EmbeddingServiceBuilder;

#[cfg(feature = "embeddings-openai")]
pub use providers::OpenAIProvider;

#[cfg(feature = "embeddings-onnx")]
pub use providers::ONNXProvider;

#[cfg(feature = "embeddings-candle")]
pub use providers::CandleProvider;

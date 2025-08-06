use super::{EmbeddingService, EmbeddingConfig, ProviderConfig, providers::EmbeddingError};
use super::providers::MockProvider;
use std::sync::Arc;

#[cfg(feature = "embeddings-openai")]
use super::providers::OpenAIProvider;

#[cfg(feature = "embeddings-onnx")]
use super::providers::ONNXProvider;

#[cfg(feature = "embeddings-candle")]
use super::providers::CandleProvider;

/// Builder for creating embedding services with various configurations
pub struct EmbeddingServiceBuilder {
    config: EmbeddingConfig,
}

impl EmbeddingServiceBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: EmbeddingConfig::default(),
        }
    }
    
    /// Create a new builder with custom configuration
    pub fn with_config(config: EmbeddingConfig) -> Self {
        Self { config }
    }
    
    /// Set the primary provider
    pub fn with_primary_provider(mut self, provider_name: String) -> Self {
        self.config.primary_provider = provider_name;
        self
    }
    
    /// Add a fallback provider
    pub fn with_fallback_provider(mut self, provider_name: String) -> Self {
        if !self.config.fallback_providers.contains(&provider_name) {
            self.config.fallback_providers.push(provider_name);
        }
        self
    }
    
    /// Set maximum batch size
    pub fn with_max_batch_size(mut self, max_batch_size: usize) -> Self {
        self.config.max_batch_size = max_batch_size;
        self
    }
    
    /// Set request timeout
    pub fn with_timeout_seconds(mut self, timeout_seconds: u64) -> Self {
        self.config.timeout_seconds = timeout_seconds;
        self
    }
    
    /// Enable or disable caching
    pub fn with_caching(mut self, enable_caching: bool) -> Self {
        self.config.enable_caching = enable_caching;
        self
    }
    
    /// Build the embedding service with automatic provider registration
    pub fn build(self) -> Result<EmbeddingService, EmbeddingError> {
        let mut service = EmbeddingService::new(self.config.clone());
        
        // Register providers based on configuration
        for (provider_name, provider_config) in &self.config.providers {
            if provider_config.enabled {
                let provider = self.create_provider(provider_name, provider_config)?;
                service.register_provider(provider_name.clone(), provider);
            }
        }
        
        Ok(service)
    }
    
    /// Create a provider instance based on configuration
    fn create_provider(
        &self,
        name: &str,
        config: &ProviderConfig,
    ) -> Result<Arc<dyn crate::core::embeddings::service::EmbeddingProvider>, EmbeddingError> {
        match name {
            name if name.starts_with("mock") => {
                // Check if we should simulate errors
                let simulate_errors = config.options.get("simulate_errors")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                
                let provider = MockProvider::with_config(
                    name,
                    config.model.dimension,
                    100, // max_batch_size
                    config.model.max_input_length,
                    simulate_errors,
                );
                Ok(Arc::new(provider))
            }
            
            #[cfg(feature = "embeddings-openai")]
            "openai" => {
                let api_key = config.api_key.as_ref()
                    .ok_or_else(|| EmbeddingError::ConfigurationError(
                        "OpenAI API key is required".to_string()
                    ))?;
                
                let provider = OpenAIProvider::from_config(
                    api_key.clone(),
                    Some(config.model.name.clone()),
                    config.base_url.clone(),
                )?;
                
                Ok(Arc::new(provider))
            }
            
            #[cfg(feature = "embeddings-onnx")]
            "onnx" => {
                let model_path = config.options.get("model_path")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| EmbeddingError::ConfigurationError(
                        "ONNX model path is required".to_string()
                    ))?;
                
                let tokenizer_path = config.options.get("tokenizer_path")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                
                let provider = ONNXProvider::from_config(
                    model_path.to_string(),
                    tokenizer_path,
                    Some(config.model.dimension),
                )?;
                
                Ok(Arc::new(provider))
            }
            
            #[cfg(feature = "embeddings-candle")]
            "candle" => {
                let model_path = config.options.get("model_path")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| EmbeddingError::ConfigurationError(
                        "Candle model path is required".to_string()
                    ))?;
                
                let tokenizer_path = config.options.get("tokenizer_path")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                
                let provider = CandleProvider::from_config(
                    model_path.to_string(),
                    tokenizer_path,
                    Some(config.model.dimension),
                )?;
                
                Ok(Arc::new(provider))
            }
            
            _ => Err(EmbeddingError::ProviderNotFound(name.to_string())),
        }
    }
}

impl Default for EmbeddingServiceBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience functions for common setups
impl EmbeddingServiceBuilder {
    /// Create a service with mock provider (for testing/development)
    pub fn mock() -> Result<EmbeddingService, EmbeddingError> {
        Self::new().build()
    }
    
    /// Create a service with OpenAI provider
    #[cfg(feature = "embeddings-openai")]
    pub fn openai(api_key: String) -> Result<EmbeddingService, EmbeddingError> {
        let config = EmbeddingConfig::with_openai(api_key);
        Self::with_config(config).build()
    }
    
    /// Create a service with ONNX provider
    #[cfg(feature = "embeddings-onnx")]
    pub fn onnx(model_path: String) -> Result<EmbeddingService, EmbeddingError> {
        let config = EmbeddingConfig::with_onnx(model_path);
        Self::with_config(config).build()
    }
    
    /// Create a service with hybrid setup (OpenAI primary, local fallback)
    #[cfg(all(feature = "embeddings-openai", feature = "embeddings-onnx"))]
    pub fn hybrid_openai_onnx(
        openai_api_key: String,
        onnx_model_path: String,
    ) -> Result<EmbeddingService, EmbeddingError> {
        let mut config = EmbeddingConfig::with_openai(openai_api_key);
        
        // Add ONNX as fallback
        let mut onnx_options = std::collections::HashMap::new();
        onnx_options.insert("model_path".to_string(), serde_json::Value::String(onnx_model_path));
        
        config = config.add_fallback(
            "onnx".to_string(),
            ProviderConfig {
                enabled: true,
                api_key: None,
                base_url: None,
                model: super::EmbeddingModel {
                    name: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
                    dimension: 384,
                    max_input_length: 512,
                    parameters: std::collections::HashMap::new(),
                },
                options: onnx_options,
            },
        );
        
        Self::with_config(config).build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_builder_mock() {
        let service = EmbeddingServiceBuilder::mock().unwrap();
        
        let embedding = service.generate_embedding("test").await.unwrap();
        assert_eq!(embedding.len(), 1536);
    }
    
    #[tokio::test]
    async fn test_builder_custom_config() {
        let service = EmbeddingServiceBuilder::new()
            .with_max_batch_size(50)
            .with_timeout_seconds(60)
            .with_caching(false)
            .build()
            .unwrap();
        
        let embedding = service.generate_embedding("test").await.unwrap();
        assert_eq!(embedding.len(), 1536);
    }
}

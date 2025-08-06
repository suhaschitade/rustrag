#[cfg(feature = "embeddings-openai")]
use super::{EmbeddingError, utils};
#[cfg(feature = "embeddings-openai")]
use crate::core::embeddings::service::EmbeddingProvider;
#[cfg(feature = "embeddings-openai")]
use async_trait::async_trait;
#[cfg(feature = "embeddings-openai")]
use async_openai::{Client, types::{CreateEmbeddingRequest, EmbeddingModel}};
#[cfg(feature = "embeddings-openai")]
use tracing::{info, warn};

#[cfg(feature = "embeddings-openai")]
/// OpenAI embedding provider
pub struct OpenAIProvider {
    client: Client,
    model: String,
    dimension: usize,
    max_batch_size: usize,
    max_input_length: usize,
}

#[cfg(feature = "embeddings-openai")]
impl OpenAIProvider {
    /// Create a new OpenAI provider
    pub fn new(api_key: String) -> Result<Self, EmbeddingError> {
        Self::with_model(api_key, "text-embedding-ada-002".to_string())
    }
    
    /// Create a new OpenAI provider with a specific model
    pub fn with_model(api_key: String, model: String) -> Result<Self, EmbeddingError> {
        let client = Client::new(api_key);
        
        // Get model-specific configuration
        let (dimension, max_batch_size, max_input_length) = match model.as_str() {
            "text-embedding-ada-002" => (1536, 2048, 8191),
            "text-embedding-3-small" => (1536, 2048, 8191),
            "text-embedding-3-large" => (3072, 2048, 8191),
            _ => {
                warn!("Unknown OpenAI model: {}, using defaults", model);
                (1536, 2048, 8191)
            }
        };
        
        Ok(Self {
            client,
            model,
            dimension,
            max_batch_size,
            max_input_length,
        })
    }
    
    /// Create OpenAI provider from configuration
    pub fn from_config(
        api_key: String,
        model: Option<String>,
        base_url: Option<String>,
    ) -> Result<Self, EmbeddingError> {
        let model = model.unwrap_or_else(|| "text-embedding-ada-002".to_string());
        
        if base_url.is_some() {
            warn!("Custom base URL not supported for OpenAI provider yet");
        }
        
        Self::with_model(api_key, model)
    }
}

#[cfg(feature = "embeddings-openai")]
#[async_trait]
impl EmbeddingProvider for OpenAIProvider {
    fn name(&self) -> &str {
        "openai"
    }
    
    fn dimension(&self) -> usize {
        self.dimension
    }
    
    async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        utils::validate_text_length(text, self.max_input_length)?;
        
        info!("Generating OpenAI embedding for text: {}", &text[..std::cmp::min(50, text.len())]);
        
        let request = CreateEmbeddingRequest {
            model: self.model.clone().into(),
            input: vec![text.to_string()].into(),
            encoding_format: None,
            dimensions: None,
            user: None,
        };
        
        let response = self.client
            .embeddings()
            .create(request)
            .await
            .map_err(|e| EmbeddingError::ApiError(format!("OpenAI API error: {}", e)))?;
        
        if response.data.is_empty() {
            return Err(EmbeddingError::ApiError("Empty response from OpenAI".to_string()));
        }
        
        Ok(response.data[0].embedding.clone())
    }
    
    async fn generate_embeddings(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        utils::validate_batch_size(texts.len(), self.max_batch_size)?;
        
        for text in texts {
            utils::validate_text_length(text, self.max_input_length)?;
        }
        
        info!("Generating {} OpenAI embeddings", texts.len());
        
        let request = CreateEmbeddingRequest {
            model: self.model.clone().into(),
            input: texts.clone().into(),
            encoding_format: None,
            dimensions: None,
            user: None,
        };
        
        let response = self.client
            .embeddings()
            .create(request)
            .await
            .map_err(|e| {
                let error_msg = format!("OpenAI API error: {}", e);
                if error_msg.contains("rate_limit") {
                    EmbeddingError::RateLimitExceeded(error_msg)
                } else if error_msg.contains("authentication") || error_msg.contains("unauthorized") {
                    EmbeddingError::AuthenticationError(error_msg)
                } else {
                    EmbeddingError::ApiError(error_msg)
                }
            })?;
        
        if response.data.len() != texts.len() {
            return Err(EmbeddingError::ApiError(
                format!("Expected {} embeddings, got {}", texts.len(), response.data.len())
            ));
        }
        
        let embeddings: Vec<Vec<f32>> = response.data
            .into_iter()
            .map(|item| item.embedding)
            .collect();
        
        Ok(embeddings)
    }
    
    async fn health_check(&self) -> Result<(), EmbeddingError> {
        info!("Performing OpenAI provider health check");
        
        // Test with a simple embedding request
        let test_text = "health check";
        match self.generate_embedding(test_text).await {
            Ok(_) => {
                info!("OpenAI provider health check passed");
                Ok(())
            }
            Err(e) => {
                warn!("OpenAI provider health check failed: {}", e);
                Err(EmbeddingError::HealthCheckFailed(e.to_string()))
            }
        }
    }
    
    fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }
    
    fn max_input_length(&self) -> usize {
        self.max_input_length
    }
}

#[cfg(all(test, feature = "embeddings-openai"))]
mod tests {
    use super::*;
    
    // Note: These tests require a valid OpenAI API key
    // They are marked as ignored by default to prevent accidental API usage
    
    #[tokio::test]
    #[ignore]
    async fn test_openai_provider_single_embedding() {
        let api_key = std::env::var("OPENAI_API_KEY")
            .expect("OPENAI_API_KEY environment variable required for tests");
        
        let provider = OpenAIProvider::new(api_key).unwrap();
        
        let text = "This is a test document";
        let embedding = provider.generate_embedding(text).await.unwrap();
        
        assert_eq!(embedding.len(), 1536);
        assert!(embedding.iter().any(|&x| x != 0.0)); // Should not be all zeros
    }
    
    #[tokio::test]
    #[ignore]
    async fn test_openai_provider_batch_embeddings() {
        let api_key = std::env::var("OPENAI_API_KEY")
            .expect("OPENAI_API_KEY environment variable required for tests");
        
        let provider = OpenAIProvider::new(api_key).unwrap();
        
        let texts = vec![
            "First document".to_string(),
            "Second document".to_string(),
            "Third document".to_string(),
        ];
        
        let embeddings = provider.generate_embeddings(&texts).await.unwrap();
        
        assert_eq!(embeddings.len(), 3);
        assert_eq!(embeddings[0].len(), 1536);
        
        // Each embedding should be different
        assert_ne!(embeddings[0], embeddings[1]);
        assert_ne!(embeddings[1], embeddings[2]);
    }
    
    #[tokio::test]
    #[ignore]
    async fn test_openai_provider_health_check() {
        let api_key = std::env::var("OPENAI_API_KEY")
            .expect("OPENAI_API_KEY environment variable required for tests");
        
        let provider = OpenAIProvider::new(api_key).unwrap();
        
        let result = provider.health_check().await;
        assert!(result.is_ok());
    }
}

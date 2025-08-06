use super::{EmbeddingError, utils};
use crate::core::embeddings::service::EmbeddingProvider;
use async_trait::async_trait;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use tracing::info;

/// Mock embedding provider for testing and development
pub struct MockProvider {
    name: String,
    dimension: usize,
    max_batch_size: usize,
    max_input_length: usize,
    simulate_errors: bool,
}

impl MockProvider {
    /// Create a new mock provider
    pub fn new() -> Self {
        Self::with_config("mock", 1536, 100, 8192, false)
    }
    
    /// Create a mock provider with custom configuration
    pub fn with_config(
        name: &str,
        dimension: usize,
        max_batch_size: usize,
        max_input_length: usize,
        simulate_errors: bool,
    ) -> Self {
        Self {
            name: name.to_string(),
            dimension,
            max_batch_size,
            max_input_length,
            simulate_errors,
        }
    }
    
    /// Generate a deterministic but pseudo-random embedding based on text content
    fn generate_mock_embedding(&self, text: &str) -> Vec<f32> {
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let seed = hasher.finish();
        
        // Use a simple linear congruential generator for reproducible results
        let mut rng = seed;
        let mut embedding = Vec::with_capacity(self.dimension);
        
        for _ in 0..self.dimension {
            // Simple LCG: next = (a * current + c) mod m
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let value = ((rng >> 16) as f32 / 32768.0) - 1.0; // Normalize to [-1, 1]
            embedding.push(value * 0.1); // Scale down for more realistic embeddings
        }
        
        // Normalize the embedding vector
        let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for value in &mut embedding {
                *value /= magnitude;
            }
        }
        
        embedding
    }
}

impl Default for MockProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl EmbeddingProvider for MockProvider {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn dimension(&self) -> usize {
        self.dimension
    }
    
    async fn generate_embedding(&self, text: &str) -> crate::utils::Result<Vec<f32>> {
        if self.simulate_errors && text.contains("error") {
            return Err(EmbeddingError::Other("Simulated error".to_string()).into());
        }
        
        utils::validate_text_length(text, self.max_input_length)
            .map_err(|e| crate::utils::Error::embedding(e.to_string()))?;
        
        info!("Generating mock embedding for text: {}", &text[..std::cmp::min(50, text.len())]);
        
        // Simulate some processing time
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        
        Ok(self.generate_mock_embedding(text))
    }
    
    async fn generate_embeddings(&self, texts: &[String]) -> crate::utils::Result<Vec<Vec<f32>>> {
        utils::validate_batch_size(texts.len(), self.max_batch_size)
            .map_err(|e| crate::utils::Error::embedding(e.to_string()))?;
        
        if self.simulate_errors && texts.iter().any(|t| t.contains("error")) {
            return Err(EmbeddingError::Other("Simulated batch error".to_string()).into());
        }
        
        info!("Generating {} mock embeddings", texts.len());
        
        // Simulate batch processing time
        let processing_time = std::cmp::min(texts.len() * 5, 100); // Max 100ms
        tokio::time::sleep(tokio::time::Duration::from_millis(processing_time as u64)).await;
        
        let mut embeddings = Vec::with_capacity(texts.len());
        for text in texts {
            utils::validate_text_length(text, self.max_input_length)
                .map_err(|e| crate::utils::Error::embedding(e.to_string()))?;
            embeddings.push(self.generate_mock_embedding(text));
        }
        
        Ok(embeddings)
    }
    
    async fn health_check(&self) -> crate::utils::Result<()> {
        if self.simulate_errors {
            Err(EmbeddingError::HealthCheckFailed("Mock health check failure".to_string()).into())
        } else {
            info!("Mock provider health check passed");
            Ok(())
        }
    }
    
    fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }
    
    fn max_input_length(&self) -> usize {
        self.max_input_length
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_mock_provider_single_embedding() {
        let provider = MockProvider::new();
        
        let text = "This is a test document";
        let embedding = provider.generate_embedding(text).await.unwrap();
        
        assert_eq!(embedding.len(), 1536);
        
        // Test determinism - same input should produce same output
        let embedding2 = provider.generate_embedding(text).await.unwrap();
        assert_eq!(embedding, embedding2);
        
        // Test different input produces different output
        let embedding3 = provider.generate_embedding("Different text").await.unwrap();
        assert_ne!(embedding, embedding3);
    }
    
    #[tokio::test]
    async fn test_mock_provider_batch_embeddings() {
        let provider = MockProvider::new();
        
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
    async fn test_mock_provider_error_simulation() {
        let provider = MockProvider::with_config("mock", 1536, 100, 8192, true);
        
        let result = provider.generate_embedding("text with error").await;
        assert!(result.is_err());
        
        let health = provider.health_check().await;
        assert!(health.is_err());
    }
    
    #[tokio::test]
    async fn test_mock_provider_input_validation() {
        let provider = MockProvider::with_config("mock", 1536, 2, 10, false);
        
        // Test input too long
        let long_text = "a".repeat(20);
        let result = provider.generate_embedding(&long_text).await;
        assert!(result.is_err());
        
        // Test batch too large
        let texts = vec!["text".to_string(); 5];
        let result = provider.generate_embeddings(&texts).await;
        assert!(result.is_err());
    }
}

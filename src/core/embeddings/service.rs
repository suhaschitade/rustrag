use crate::models::DocumentChunk;
use super::{EmbeddingConfig, providers::EmbeddingError};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{info, warn, error};

/// Trait for embedding providers
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Get the provider name
    fn name(&self) -> &str;
    
    /// Get the embedding dimension
    fn dimension(&self) -> usize;
    
    /// Generate a single embedding
    async fn generate_embedding(&self, text: &str) -> crate::utils::Result<Vec<f32>>;
    
    /// Generate embeddings for a batch of texts
    async fn generate_embeddings(&self, texts: &[String]) -> crate::utils::Result<Vec<Vec<f32>>>;
    
    /// Check if the provider is healthy/available
    async fn health_check(&self) -> crate::utils::Result<()>;
    
    /// Get maximum batch size supported by this provider
    fn max_batch_size(&self) -> usize;
    
    /// Get maximum input length supported by this provider
    fn max_input_length(&self) -> usize;
}

/// Multi-provider embedding service
pub struct EmbeddingService {
    config: EmbeddingConfig,
    providers: HashMap<String, Arc<dyn EmbeddingProvider>>,
}

impl EmbeddingService {
    /// Create a new embedding service with the given configuration
    pub fn new(config: EmbeddingConfig) -> Self {
        Self {
            config,
            providers: HashMap::new(),
        }
    }
    
    /// Register a provider
    pub fn register_provider(&mut self, name: String, provider: Arc<dyn EmbeddingProvider>) {
        info!("Registering embedding provider: {}", name);
        self.providers.insert(name, provider);
    }
    
    /// Get the primary provider
    fn get_primary_provider(&self) -> crate::utils::Result<&Arc<dyn EmbeddingProvider>> {
        self.providers
            .get(&self.config.primary_provider)
            .ok_or_else(|| EmbeddingError::ProviderNotFound(self.config.primary_provider.clone()).into())
    }
    
    /// Get providers in order of preference (primary first, then fallbacks)
    fn get_providers_in_order(&self) -> Vec<&Arc<dyn EmbeddingProvider>> {
        let mut providers = Vec::new();
        
        // Add primary provider first
        if let Some(provider) = self.providers.get(&self.config.primary_provider) {
            providers.push(provider);
        }
        
        // Add fallback providers
        for fallback in &self.config.fallback_providers {
            if let Some(provider) = self.providers.get(fallback) {
                providers.push(provider);
            }
        }
        
        providers
    }
    
    /// Generate embedding with automatic fallback
    pub async fn generate_embedding(&self, text: &str) -> crate::utils::Result<Vec<f32>> {
        let providers = self.get_providers_in_order();
        
        if providers.is_empty() {
            return Err(EmbeddingError::NoProvidersAvailable.into());
        }
        
        let mut last_error = None;
        
        for provider in providers {
            match self.try_generate_embedding(provider, text).await {
                Ok(embedding) => {
                    info!("Successfully generated embedding using provider: {}", provider.name());
                    return Ok(embedding);
                }
                Err(e) => {
                    warn!("Provider {} failed: {}", provider.name(), e);
                    last_error = Some(e);
                    continue;
                }
            }
        }
        
        error!("All embedding providers failed");
        Err(last_error.unwrap_or_else(|| EmbeddingError::NoProvidersAvailable.into()))
    }
    
    /// Try to generate embedding with a specific provider with retries
    async fn try_generate_embedding(
        &self,
        provider: &Arc<dyn EmbeddingProvider>,
        text: &str,
    ) -> crate::utils::Result<Vec<f32>> {
        let mut attempts = 0;
        let max_retries = self.config.max_retries;
        
        while attempts <= max_retries {
            match provider.generate_embedding(text).await {
                Ok(embedding) => return Ok(embedding),
                Err(e) => {
                    attempts += 1;
                    if attempts > max_retries {
                        return Err(e);
                    }
                    
                    warn!(
                        "Embedding generation attempt {} failed for provider {}: {}. Retrying...",
                        attempts, provider.name(), e
                    );
                    
                    // Simple exponential backoff
                    tokio::time::sleep(tokio::time::Duration::from_millis(100 * attempts as u64)).await;
                }
            }
        }
        
        unreachable!()
    }
    
    /// Generate embeddings for multiple texts
    pub async fn generate_embeddings(&self, texts: &[String]) -> crate::utils::Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        
        let providers = self.get_providers_in_order();
        
        if providers.is_empty() {
            return Err(EmbeddingError::NoProvidersAvailable.into());
        }
        
        let mut last_error = None;
        
        for provider in providers {
            match self.try_generate_embeddings(provider, texts).await {
                Ok(embeddings) => {
                    info!("Successfully generated {} embeddings using provider: {}", 
                          embeddings.len(), provider.name());
                    return Ok(embeddings);
                }
                Err(e) => {
                    warn!("Provider {} failed for batch: {}", provider.name(), e);
                    last_error = Some(e);
                    continue;
                }
            }
        }
        
        error!("All embedding providers failed for batch");
        Err(last_error.unwrap_or_else(|| EmbeddingError::NoProvidersAvailable.into()))
    }
    
    /// Try to generate embeddings with a specific provider, handling batching
    async fn try_generate_embeddings(
        &self,
        provider: &Arc<dyn EmbeddingProvider>,
        texts: &[String],
    ) -> crate::utils::Result<Vec<Vec<f32>>> {
        let batch_size = std::cmp::min(provider.max_batch_size(), self.config.max_batch_size);
        let mut all_embeddings = Vec::new();
        
        for chunk in texts.chunks(batch_size) {
            let embeddings = provider.generate_embeddings(chunk).await?;
            all_embeddings.extend(embeddings);
        }
        
        Ok(all_embeddings)
    }
    
    /// Generate embeddings for document chunks
    pub async fn generate_embeddings_for_chunks(
        &self,
        chunks: &mut [DocumentChunk],
    ) -> crate::utils::Result<()> {
        let texts: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();
        let embeddings = self.generate_embeddings(&texts).await?;
        
        for (chunk, embedding) in chunks.iter_mut().zip(embeddings.into_iter()) {
            chunk.embedding = Some(embedding);
        }
        
        Ok(())
    }
    
    /// Generate embedding for a query
    pub async fn generate_query_embedding(&self, query: &str) -> crate::utils::Result<Vec<f32>> {
        self.generate_embedding(query).await
    }
    
    /// Check health of all providers
    pub async fn health_check(&self) -> HashMap<String, crate::utils::Result<()>> {
        let mut results = HashMap::new();
        
        for (name, provider) in &self.providers {
            let result = provider.health_check().await;
            results.insert(name.clone(), result);
        }
        
        results
    }
    
    /// Get embedding dimension from the primary provider
    pub fn embedding_dimension(&self) -> crate::utils::Result<usize> {
        let provider = self.get_primary_provider()
            .map_err(|e| crate::utils::Error::from(e))?;
        Ok(provider.dimension())
    }
    
    /// Get provider statistics
    pub fn get_provider_stats(&self) -> HashMap<String, ProviderStats> {
        let mut stats = HashMap::new();
        
        for (name, provider) in &self.providers {
            stats.insert(
                name.clone(),
                ProviderStats {
                    name: provider.name().to_string(),
                    dimension: provider.dimension(),
                    max_batch_size: provider.max_batch_size(),
                    max_input_length: provider.max_input_length(),
                    is_primary: name == &self.config.primary_provider,
                },
            );
        }
        
        stats
    }
}

/// Statistics for an embedding provider
#[derive(Debug, Clone)]
pub struct ProviderStats {
    pub name: String,
    pub dimension: usize,
    pub max_batch_size: usize,
    pub max_input_length: usize,
    pub is_primary: bool,
}

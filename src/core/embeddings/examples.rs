use super::{EmbeddingServiceBuilder, EmbeddingConfig, ProviderConfig, EmbeddingModel};
use crate::models::DocumentChunk;
use std::collections::HashMap;
use uuid::Uuid;

/// Example: Basic usage with mock provider (default)
pub async fn example_basic_mock_embedding() -> crate::utils::Result<()> {
    println!("=== Basic Mock Embedding Example ===");
    
    // Create embedding service with default mock provider
    let embedding_service = EmbeddingServiceBuilder::mock()
        .map_err(|e| crate::utils::Error::embedding(e.to_string()))?;
    
    // Generate single embedding
    let text = "This is a sample document for embedding generation.";
    let embedding = embedding_service.generate_embedding(text).await?;
    
    println!("Generated embedding with {} dimensions", embedding.len());
    println!("First 5 values: {:?}", &embedding[..5]);
    
    // Generate batch embeddings
    let texts = vec![
        "First document chunk".to_string(),
        "Second document chunk".to_string(),
        "Third document chunk with different content".to_string(),
    ];
    
    let embeddings = embedding_service.generate_embeddings(&texts).await?;
    println!("Generated {} batch embeddings", embeddings.len());
    
    // Generate embeddings for document chunks
    let mut chunks = vec![
        DocumentChunk::new(
            Uuid::new_v4(),
            0,
            "Chapter 1: Introduction to machine learning".to_string(),
        ),
        DocumentChunk::new(
            Uuid::new_v4(),
            1,
            "Chapter 2: Deep learning fundamentals".to_string(),
        ),
    ];
    
    embedding_service.generate_embeddings_for_chunks(&mut chunks).await?;
    println!("Generated embeddings for {} document chunks", chunks.len());
    
    // Check provider health
    let health_results = embedding_service.health_check().await;
    for (provider, result) in health_results {
        println!("Provider {} health: {:?}", provider, result);
    }
    
    Ok(())
}

/// Example: Custom configuration with multiple providers
pub async fn example_multi_provider_config() -> crate::utils::Result<()> {
    println!("=== Multi-Provider Configuration Example ===");
    
    let mut config = EmbeddingConfig::default();
    
    // Add multiple mock providers with different configurations
    config.providers.insert(
        "mock-primary".to_string(),
        ProviderConfig {
            enabled: true,
            api_key: None,
            base_url: None,
            model: EmbeddingModel {
                name: "mock-ada-002".to_string(),
                dimension: 1536,
                max_input_length: 8192,
                parameters: HashMap::new(),
            },
            options: HashMap::new(),
        },
    );
    
    config.providers.insert(
        "mock-fallback".to_string(),
        ProviderConfig {
            enabled: true,
            api_key: None,
            base_url: None,
            model: EmbeddingModel {
                name: "mock-mini".to_string(),
                dimension: 384,
                max_input_length: 512,
                parameters: HashMap::new(),
            },
            options: HashMap::new(),
        },
    );
    
    config.primary_provider = "mock-primary".to_string();
    config.fallback_providers = vec!["mock-fallback".to_string()];
    
    let embedding_service = EmbeddingServiceBuilder::with_config(config)
        .with_max_batch_size(50)
        .with_timeout_seconds(30)
        .with_caching(true)
        .build()
        .map_err(|e| crate::utils::Error::embedding(e.to_string()))?;
    
    // Test embedding generation
    let text = "Multi-provider embedding test";
    let embedding = embedding_service.generate_embedding(text).await?;
    println!("Generated embedding with {} dimensions using multi-provider setup", embedding.len());
    
    // Get provider statistics
    let stats = embedding_service.get_provider_stats();
    for (name, stat) in stats {
        println!("Provider {}: {} dimensions, max batch {}, is primary: {}", 
                 name, stat.dimension, stat.max_batch_size, stat.is_primary);
    }
    
    Ok(())
}

/// Example: Builder pattern with different configurations
pub async fn example_builder_pattern() -> crate::utils::Result<()> {
    println!("=== Builder Pattern Example ===");
    
    // Example 1: Basic mock service
    let service1 = EmbeddingServiceBuilder::new()
        .with_max_batch_size(25)
        .with_timeout_seconds(15)
        .build()
        .map_err(|e| crate::utils::Error::embedding(e.to_string()))?;
    
    let embedding1 = service1.generate_embedding("Builder pattern test 1").await?;
    println!("Service 1: {} dimensions", embedding1.len());
    
    // Example 2: Custom configuration
    let service2 = EmbeddingServiceBuilder::new()
        .with_primary_provider("mock".to_string())
        .with_max_batch_size(100)
        .with_caching(false)
        .build()
        .map_err(|e| crate::utils::Error::embedding(e.to_string()))?;
    
    let embedding2 = service2.generate_embedding("Builder pattern test 2").await?;
    println!("Service 2: {} dimensions", embedding2.len());
    
    Ok(())
}

/// Example: Error handling and fallback behavior
pub async fn example_error_handling_and_fallback() -> crate::utils::Result<()> {
    println!("=== Error Handling and Fallback Example ===");
    
    // Create a service with error simulation
    let mut config = EmbeddingConfig::default();
    config.providers.insert(
        "mock-error".to_string(),
        ProviderConfig {
            enabled: true,
            api_key: None,
            base_url: None,
            model: EmbeddingModel::default(),
            options: {
                let mut opts = HashMap::new();
                opts.insert("simulate_errors".to_string(), serde_json::Value::Bool(true));
                opts
            },
        },
    );
    
    config.providers.insert(
        "mock-backup".to_string(),
        ProviderConfig {
            enabled: true,
            api_key: None,
            base_url: None,
            model: EmbeddingModel::default(),
            options: HashMap::new(),
        },
    );
    
    config.primary_provider = "mock-error".to_string();
    config.fallback_providers = vec!["mock-backup".to_string()];
    config.max_retries = 2;
    
    let embedding_service = EmbeddingServiceBuilder::with_config(config)
        .build()
        .map_err(|e| crate::utils::Error::embedding(e.to_string()))?;
    
    // This should work because it falls back to the backup provider
    match embedding_service.generate_embedding("test fallback").await {
        Ok(embedding) => println!("Fallback successful: {} dimensions", embedding.len()),
        Err(e) => println!("Fallback failed: {}", e),
    }
    
    // Test health check with problematic provider
    let health_results = embedding_service.health_check().await;
    for (provider, result) in health_results {
        match result {
            Ok(_) => println!("Provider {} is healthy", provider),
            Err(e) => println!("Provider {} has issues: {}", provider, e),
        }
    }
    
    Ok(())
}

/// Run all examples
pub async fn run_all_examples() -> crate::utils::Result<()> {
    println!("🚀 Running Multi-Provider Embedding Architecture Examples\n");
    
    example_basic_mock_embedding().await?;
    println!();
    
    example_multi_provider_config().await?;
    println!();
    
    example_builder_pattern().await?;
    println!();
    
    example_error_handling_and_fallback().await?;
    println!();
    
    println!("✅ All embedding examples completed successfully!");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_basic_mock_embedding() {
        example_basic_mock_embedding().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_multi_provider_config() {
        example_multi_provider_config().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_builder_pattern() {
        example_builder_pattern().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_error_handling_and_fallback() {
        example_error_handling_and_fallback().await.unwrap();
    }
}

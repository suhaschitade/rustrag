use rustrag::core::embeddings::{EmbeddingServiceBuilder};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for logging
    tracing_subscriber::fmt::init();
    
    println!("🚀 Multi-Provider Embedding Architecture Demo");
    
    // Create a simple embedding service with mock provider
    match EmbeddingServiceBuilder::mock() {
        Ok(service) => {
            println!("✅ Successfully created mock embedding service");
            
            // Test single embedding
            match service.generate_embedding("Hello, world!").await {
                Ok(embedding) => {
                    println!("✅ Generated embedding with {} dimensions", embedding.len());
                    println!("   First 5 values: {:?}", &embedding[..5]);
                },
                Err(e) => {
                    println!("❌ Failed to generate embedding: {}", e);
                }
            }
        },
        Err(e) => {
            println!("❌ Failed to create embedding service: {}", e);
        }
    }
    
    Ok(())
}

use rustrag::core::embeddings::examples;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for logging
    tracing_subscriber::fmt::init();
    
    // Run all embedding examples
    examples::run_all_examples().await?;
    
    Ok(())
}

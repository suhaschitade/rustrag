use rustrag::{
    api::{router::create_api_router, documents::ensure_uploads_directory},
    utils::logging,
};
use std::net::SocketAddr;
use tokio::net::TcpListener;
use tracing::info;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load environment variables
    dotenvy::dotenv().ok();

    // Initialize logging
    logging::init_tracing().map_err(|e| format!("Failed to initialize logging: {}", e))?;

    info!("Starting RustRAG server v{}", rustrag::VERSION);
    info!("Initializing upload infrastructure...");
    
    // Initialize upload directory
    ensure_uploads_directory().await
        .map_err(|e| format!("Failed to initialize uploads directory: {}", e))?;
    
    info!("Building API router with comprehensive middleware stack...");

    // Build the comprehensive application router
    let app = create_api_router();

    // Server configuration
    let host = std::env::var("SERVER_HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let port = std::env::var("SERVER_PORT")
        .unwrap_or_else(|_| "8000".to_string())
        .parse::<u16>()
        .unwrap_or(8000);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    info!("Server configured to listen on {}:{}", host, port);
    
    // Create listener
    let listener = TcpListener::bind(&addr).await?;
    info!("Server successfully bound to {}", addr);
    
    info!("🚀 RustRAG API Server is ready!");
    info!("📋 Available endpoints:");
    info!("   • GET  /                    - API information");
    info!("   • GET  /api/v1/health       - Basic health check");
    info!("   • GET  /api/v1/health/detailed - Detailed health check");
    info!("   • POST /api/v1/documents    - Upload documents");
    info!("   • GET  /api/v1/documents    - List documents");
    info!("   • POST /api/v1/query        - Process queries");
    info!("   • GET  /api/v1/admin/*      - Admin endpoints");
    info!("🔐 Authentication: API Key required (except /health)");
    
    // Start the server
    axum::serve(listener, app).await?;

    Ok(())
}

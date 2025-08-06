use axum::{
    routing::{get, post},
    Router, Server,
};
use rustrag::{
    api::{health_check, create_document, get_document, list_documents, delete_document, process_query},
    utils::logging,
};
use std::net::SocketAddr;
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;
use tracing::info;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load environment variables
    dotenvy::dotenv().ok();

    // Initialize logging
    logging::init_tracing().map_err(|e| format!("Failed to initialize logging: {}", e))?;

    info!("Starting RustRAG server v{}", rustrag::VERSION);

    // Build the application router
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/documents", post(create_document).get(list_documents))
        .route("/documents/:id", get(get_document).delete(delete_document))
        .route("/query", post(process_query))
        .layer(
            ServiceBuilder::new()
                .layer(CorsLayer::permissive())
        );

    // Server configuration
    let host = std::env::var("SERVER_HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let port = std::env::var("SERVER_PORT")
        .unwrap_or_else(|_| "8000".to_string())
        .parse::<u16>()
        .unwrap_or(8000);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    info!("Server listening on {}:{}", host, port);

    // Start the server
    Server::bind(&addr)
        .serve(app.into_make_service())
        .await?;

    Ok(())
}

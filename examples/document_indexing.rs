use rustrag::{
    embedding::{EmbeddingServiceBuilder, MockProvider, ProviderConfig},
    indexing::{DocumentIndexer, DocumentIndexerBuilder, IndexingConfig, IndexingProgress, IndexingStage},
    models::{Document, DocumentMetadata},
    processing::{DocumentProcessor, ProcessingConfig},
    storage::{QdrantConfig, MockVectorStore},
    Result,
};
use std::sync::Arc;
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    rustrag::init().await?;

    // Example 1: Basic document indexing with mock services
    println!("=== Example 1: Basic Document Indexing with Mock Services ===");
    basic_indexing_example().await?;

    // Example 2: Document indexing with progress reporting
    println!("\n=== Example 2: Document Indexing with Progress Reporting ===");
    indexing_with_progress_example().await?;

    // Example 3: Batch document indexing
    println!("\n=== Example 3: Batch Document Indexing ===");
    batch_indexing_example().await?;

    // Example 4: Indexing with custom configuration
    println!("\n=== Example 4: Custom Configuration Indexing ===");
    custom_config_indexing_example().await?;

    Ok(())
}

async fn basic_indexing_example() -> Result<()> {
    // Create mock embedding service
    let embedding_service = Arc::new(
        EmbeddingServiceBuilder::new()
            .add_provider(Box::new(MockProvider::default()))
            .build()?
    );

    // Create document processor
    let processing_config = ProcessingConfig::default();
    let document_processor = Arc::new(DocumentProcessor::new(processing_config));

    // Create document indexer with mock vector store
    let indexer = DocumentIndexerBuilder::new()
        .with_embedding_service(embedding_service)
        .with_document_processor(document_processor)
        .build_with_mock()?;

    // Create a sample document
    let document = Document {
        id: Uuid::new_v4(),
        title: "Introduction to Rust".to_string(),
        content: "Rust is a systems programming language that runs blazingly fast, prevents segfaults, and guarantees thread safety. It accomplishes these goals by being memory safe without using garbage collection. Rust has excellent documentation, a friendly compiler with useful error messages, and top-notch tooling — an integrated package manager and build tool, smart multi-editor support with auto-completion and type inspections, an auto-formatter, and more.".to_string(),
        metadata: DocumentMetadata {
            source: Some("example.txt".to_string()),
            author: Some("Rust Team".to_string()),
            created_at: chrono::Utc::now(),
            file_type: Some("text".to_string()),
            tags: vec!["programming".to_string(), "rust".to_string()],
            custom_fields: std::collections::HashMap::new(),
        },
        created_at: chrono::Utc::now(),
    };

    // Index the document
    let result = indexer.index_document(&document).await?;
    
    println!("Document indexed successfully!");
    println!("  Document ID: {}", result.document_id);
    println!("  Chunks processed: {}", result.chunks_processed);
    println!("  Chunks embedded: {}", result.chunks_embedded);
    println!("  Chunks stored: {}", result.chunks_stored);
    println!("  Processing time: {}ms", result.processing_time_ms);
    
    if !result.errors.is_empty() {
        println!("  Errors encountered:");
        for error in &result.errors {
            println!("    - {}", error);
        }
    }

    // Get vector store info
    let store_info = indexer.get_store_info().await?;
    println!("Vector store info: {:?}", store_info);

    Ok(())
}

async fn indexing_with_progress_example() -> Result<()> {
    // Create services
    let embedding_service = Arc::new(
        EmbeddingServiceBuilder::new()
            .add_provider(Box::new(MockProvider::default()))
            .build()?
    );

    let processing_config = ProcessingConfig::default();
    let document_processor = Arc::new(DocumentProcessor::new(processing_config));

    let indexer = DocumentIndexerBuilder::new()
        .with_embedding_service(embedding_service)
        .with_document_processor(document_processor)
        .build_with_mock()?;

    // Create a longer document to see progress
    let document = Document {
        id: Uuid::new_v4(),
        title: "Advanced Rust Concepts".to_string(),
        content: "Memory Management in Rust: Rust uses a unique approach to memory management through ownership, borrowing, and lifetimes. This system ensures memory safety without garbage collection. Ownership rules: Each value in Rust has a variable that's called its owner. There can only be one owner at a time. When the owner goes out of scope, the value will be dropped. Borrowing allows you to refer to some value without taking ownership of it. References are immutable by default, but you can make them mutable with the mut keyword. Lifetimes ensure that references are valid for as long as we need them to be. The borrow checker uses lifetimes to ensure that all borrows are valid. Concurrency in Rust is achieved through various abstractions like threads, async/await, and channels. Rust's ownership system prevents data races at compile time. Error handling in Rust uses Result and Option types instead of exceptions. This approach makes error handling explicit and prevents many runtime errors. Pattern matching with match expressions provides powerful ways to handle different cases. Traits in Rust are similar to interfaces in other languages but more powerful. They allow you to define shared behavior across different types. Generic types allow you to write flexible, reusable code while maintaining type safety and performance.".to_string(),
        metadata: DocumentMetadata {
            source: Some("advanced_rust.txt".to_string()),
            author: Some("Rust Expert".to_string()),
            created_at: chrono::Utc::now(),
            file_type: Some("text".to_string()),
            tags: vec!["programming".to_string(), "rust".to_string(), "advanced".to_string()],
            custom_fields: std::collections::HashMap::new(),
        },
        created_at: chrono::Utc::now(),
    };

    // Index with progress callback
    let result = indexer.index_document_with_progress(&document, Some(|progress| {
        match progress.stage {
            IndexingStage::Processing => {
                println!("📝 Processing document...");
            },
            IndexingStage::GeneratingEmbeddings => {
                println!("🧠 Generating embeddings: {}/{}", progress.chunks_processed, progress.total_chunks);
            },
            IndexingStage::StoringVectors => {
                println!("💾 Storing vectors: {}/{}", progress.chunks_processed, progress.total_chunks);
            },
            IndexingStage::Complete => {
                println!("✅ Indexing complete!");
            },
            IndexingStage::Error(ref err) => {
                println!("❌ Error during indexing: {}", err);
            },
        }
    })).await?;

    println!("\nIndexing completed with progress tracking:");
    println!("  Total processing time: {}ms", result.processing_time_ms);

    Ok(())
}

async fn batch_indexing_example() -> Result<()> {
    // Create services
    let embedding_service = Arc::new(
        EmbeddingServiceBuilder::new()
            .add_provider(Box::new(MockProvider::default()))
            .build()?
    );

    let processing_config = ProcessingConfig::default();
    let document_processor = Arc::new(DocumentProcessor::new(processing_config));

    let indexer = DocumentIndexerBuilder::new()
        .with_embedding_service(embedding_service)
        .with_document_processor(document_processor)
        .build_with_mock()?;

    // Create multiple documents
    let documents = vec![
        Document {
            id: Uuid::new_v4(),
            title: "Rust Basics".to_string(),
            content: "Rust is a systems programming language focused on safety, speed, and concurrency.".to_string(),
            metadata: DocumentMetadata {
                source: Some("basics.txt".to_string()),
                author: Some("Beginner Guide".to_string()),
                created_at: chrono::Utc::now(),
                file_type: Some("text".to_string()),
                tags: vec!["rust".to_string(), "basics".to_string()],
                custom_fields: std::collections::HashMap::new(),
            },
            created_at: chrono::Utc::now(),
        },
        Document {
            id: Uuid::new_v4(),
            title: "Rust Performance".to_string(),
            content: "Rust provides zero-cost abstractions, move semantics, guaranteed memory safety, threads without data races, trait-based generics, pattern matching, type inference, and efficient C bindings.".to_string(),
            metadata: DocumentMetadata {
                source: Some("performance.txt".to_string()),
                author: Some("Performance Guide".to_string()),
                created_at: chrono::Utc::now(),
                file_type: Some("text".to_string()),
                tags: vec!["rust".to_string(), "performance".to_string()],
                custom_fields: std::collections::HashMap::new(),
            },
            created_at: chrono::Utc::now(),
        },
        Document {
            id: Uuid::new_v4(),
            title: "Rust Ecosystem".to_string(),
            content: "The Rust ecosystem includes Cargo for package management, crates.io for package distribution, extensive documentation, and a growing community of developers and libraries.".to_string(),
            metadata: DocumentMetadata {
                source: Some("ecosystem.txt".to_string()),
                author: Some("Ecosystem Guide".to_string()),
                created_at: chrono::Utc::now(),
                file_type: Some("text".to_string()),
                tags: vec!["rust".to_string(), "ecosystem".to_string()],
                custom_fields: std::collections::HashMap::new(),
            },
            created_at: chrono::Utc::now(),
        },
    ];

    // Index all documents
    let results = indexer.index_documents(&documents).await?;

    println!("Batch indexing completed:");
    for (i, result) in results.iter().enumerate() {
        println!("  Document {}: {} chunks processed, {} embedded, {} stored",
                 i + 1, result.chunks_processed, result.chunks_embedded, result.chunks_stored);
    }

    let total_chunks: usize = results.iter().map(|r| r.chunks_processed).sum();
    let total_time: u64 = results.iter().map(|r| r.processing_time_ms).sum();
    println!("  Total: {} chunks processed in {}ms", total_chunks, total_time);

    Ok(())
}

async fn custom_config_indexing_example() -> Result<()> {
    // Create services with custom configuration
    let embedding_service = Arc::new(
        EmbeddingServiceBuilder::new()
            .add_provider(Box::new(MockProvider::default()))
            .build()?
    );

    // Custom processing configuration
    let processing_config = ProcessingConfig {
        chunk_size: 200,  // Smaller chunks
        chunk_overlap: 50,
        min_chunk_size: 50,
        ..ProcessingConfig::default()
    };
    let document_processor = Arc::new(DocumentProcessor::new(processing_config));

    // Custom indexing configuration
    let indexing_config = IndexingConfig {
        embedding_batch_size: 16,  // Smaller batch size
        vector_batch_size: 50,
        update_existing: true,
        max_retries: 2,
        enable_progress: true,
    };

    let indexer = DocumentIndexerBuilder::new()
        .with_embedding_service(embedding_service)
        .with_document_processor(document_processor)
        .with_indexing_config(indexing_config)
        .build_with_mock()?;

    // Create a document
    let document = Document {
        id: Uuid::new_v4(),
        title: "Rust Web Development".to_string(),
        content: "Rust is increasingly popular for web development with frameworks like Actix-web, Warp, and Rocket. These frameworks provide high performance and safety guarantees. Actix-web is an actor-based framework that provides excellent performance. Warp is a filter-based framework that emphasizes composability. Rocket provides a more Rails-like experience with procedural macros. Rust's async/await support makes it suitable for building scalable web services. The ecosystem includes crates for JSON handling, database connectivity, authentication, and more. Rust web applications can achieve performance comparable to C++ while maintaining memory safety.".to_string(),
        metadata: DocumentMetadata {
            source: Some("web_dev.txt".to_string()),
            author: Some("Web Dev Guide".to_string()),
            created_at: chrono::Utc::now(),
            file_type: Some("text".to_string()),
            tags: vec!["rust".to_string(), "web".to_string(), "development".to_string()],
            custom_fields: std::collections::HashMap::new(),
        },
        created_at: chrono::Utc::now(),
    };

    // Index the document
    let result = indexer.index_document(&document).await?;

    println!("Custom configuration indexing completed:");
    println!("  Chunks processed: {}", result.chunks_processed);
    println!("  Processing time: {}ms", result.processing_time_ms);

    Ok(())
}

use rustrag::{
    indexing::{
        DocumentIndexerBuilder, IndexingProgress, IndexingStage,
        EmbeddingService, DocumentProcessor, ProcessingConfig,
    },
    models::{Document, DocumentMetadata},
    Result,
};
use std::sync::Arc;
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    rustrag::init().await?;

    println!("=== Document Indexing System Demo ===");

    // Example 1: Basic document indexing with mock services
    basic_indexing_example().await?;

    // Example 2: Document indexing with progress reporting
    indexing_with_progress_example().await?;

    // Example 3: Batch document indexing
    batch_indexing_example().await?;

    Ok(())
}

async fn basic_indexing_example() -> Result<()> {
    println!("\n--- Basic Document Indexing ---");

    // Create mock embedding service and document processor
    let embedding_service = Arc::new(EmbeddingService);
    let processing_config = ProcessingConfig::default();
    let document_processor = Arc::new(DocumentProcessor::new(processing_config));

    // Create document indexer with mock vector store
    let indexer = DocumentIndexerBuilder::new()
        .with_embedding_service(embedding_service)
        .with_document_processor(document_processor)
        .build_with_mock()?;

    // Create a sample document
    let document = create_sample_document(
        "Introduction to Rust",
        "Rust is a systems programming language that runs blazingly fast, prevents segfaults, and guarantees thread safety. It accomplishes these goals by being memory safe without using garbage collection.",
    );

    // Index the document
    println!("Indexing document: {}", document.title);
    let result = indexer.index_document(&document).await?;
    
    println!("✅ Document indexed successfully!");
    println!("   Document ID: {}", result.document_id);
    println!("   Chunks processed: {}", result.chunks_processed);
    println!("   Chunks embedded: {}", result.chunks_embedded);
    println!("   Chunks stored: {}", result.chunks_stored);
    println!("   Processing time: {}ms", result.processing_time_ms);
    
    if !result.errors.is_empty() {
        println!("   ⚠️ Errors encountered:");
        for error in &result.errors {
            println!("     - {}", error);
        }
    }

    // Get vector store info
    let store_info = indexer.get_store_info().await?;
    println!("📊 Vector store: {} (status: {})", store_info.name, store_info.status);

    Ok(())
}

async fn indexing_with_progress_example() -> Result<()> {
    println!("\n--- Indexing with Progress Reporting ---");

    // Create services
    let embedding_service = Arc::new(EmbeddingService);
    let processing_config = ProcessingConfig::default();
    let document_processor = Arc::new(DocumentProcessor::new(processing_config));

    let indexer = DocumentIndexerBuilder::new()
        .with_embedding_service(embedding_service)
        .with_document_processor(document_processor)
        .build_with_mock()?;

    // Create a longer document to see progress
    let document = create_sample_document(
        "Advanced Rust Concepts",
        "Memory Management in Rust: Rust uses a unique approach to memory management through ownership, borrowing, and lifetimes. This system ensures memory safety without garbage collection. Ownership rules: Each value in Rust has a variable that's called its owner. There can only be one owner at a time. When the owner goes out of scope, the value will be dropped. Borrowing allows you to refer to some value without taking ownership of it. References are immutable by default, but you can make them mutable with the mut keyword. Lifetimes ensure that references are valid for as long as we need them to be. The borrow checker uses lifetimes to ensure that all borrows are valid.",
    );

    println!("Indexing document with progress: {}", document.title);

    // Index with progress callback
    let result = indexer.index_document_with_progress(&document, Some(|progress: IndexingProgress| {
        match progress.stage {
            IndexingStage::Processing => {
                println!("  📝 Processing document...");
            },
            IndexingStage::GeneratingEmbeddings => {
                println!("  🧠 Generating embeddings: {}/{}", progress.chunks_processed, progress.total_chunks);
            },
            IndexingStage::StoringVectors => {
                println!("  💾 Storing vectors: {}/{}", progress.chunks_processed, progress.total_chunks);
            },
            IndexingStage::Complete => {
                println!("  ✅ Indexing complete!");
            },
            IndexingStage::Error(ref err) => {
                println!("  ❌ Error during indexing: {}", err);
            },
        }
    })).await?;

    println!("📊 Total processing time: {}ms", result.processing_time_ms);

    Ok(())
}

async fn batch_indexing_example() -> Result<()> {
    println!("\n--- Batch Document Indexing ---");

    // Create services
    let embedding_service = Arc::new(EmbeddingService);
    let processing_config = ProcessingConfig::default();
    let document_processor = Arc::new(DocumentProcessor::new(processing_config));

    let indexer = DocumentIndexerBuilder::new()
        .with_embedding_service(embedding_service)
        .with_document_processor(document_processor)
        .build_with_mock()?;

    // Create multiple documents
    let documents = vec![
        create_sample_document(
            "Rust Basics",
            "Rust is a systems programming language focused on safety, speed, and concurrency.",
        ),
        create_sample_document(
            "Rust Performance",
            "Rust provides zero-cost abstractions, move semantics, guaranteed memory safety, threads without data races, trait-based generics, pattern matching, type inference, and efficient C bindings.",
        ),
        create_sample_document(
            "Rust Ecosystem",
            "The Rust ecosystem includes Cargo for package management, crates.io for package distribution, extensive documentation, and a growing community of developers and libraries.",
        ),
    ];

    println!("Indexing {} documents in batch...", documents.len());

    // Index all documents
    let results = indexer.index_documents(&documents).await?;

    println!("✅ Batch indexing completed:");
    for (i, result) in results.iter().enumerate() {
        println!("  Document {}: {} chunks processed, {} embedded, {} stored",
                 i + 1, result.chunks_processed, result.chunks_embedded, result.chunks_stored);
    }

    let total_chunks: usize = results.iter().map(|r| r.chunks_processed).sum();
    let total_time: u64 = results.iter().map(|r| r.processing_time_ms).sum();
    println!("📊 Total: {} chunks processed in {}ms", total_chunks, total_time);

    Ok(())
}

fn create_sample_document(title: &str, content: &str) -> Document {
    let now = chrono::Utc::now();
    Document {
        id: Uuid::new_v4(),
        title: title.to_string(),
        content: content.to_string(),
        file_path: Some(format!("{}.txt", title.to_lowercase().replace(' ', "_"))),
        file_type: Some("text".to_string()),
        file_size: Some(content.len() as i64),
        metadata: DocumentMetadata {
            author: Some("RustRAG Demo".to_string()),
            created_date: Some(now),
            modified_date: Some(now),
            language: Some("en".to_string()),
            category: Some("documentation".to_string()),
            word_count: Some(content.split_whitespace().count() as u32),
            page_count: Some(1),
            file_type: Some("text".to_string()),
            file_size: Some(content.len() as u64),
            tags: vec!["rust".to_string(), "programming".to_string()],
            custom_fields: serde_json::json!({
                "source": "rustrag_demo",
                "version": "1.0"
            }),
        },
        created_at: now,
        updated_at: now,
    }
}

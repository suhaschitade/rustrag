use rustrag::core::{DocumentProcessor, MetadataExtractor, ChunkingStrategy};
use rustrag::models::DocumentMetadata;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::init();

    println!("🔍 RustRAG Metadata Extraction Demo");
    println!("=====================================\n");

    // Example 1: Extract metadata from content
    demo_content_metadata().await?;
    
    // Example 2: Document processing with metadata
    demo_document_processing().await?;
    
    // Example 3: Different file type metadata extraction
    demo_file_type_metadata().await?;

    Ok(())
}

async fn demo_content_metadata() -> Result<(), Box<dyn std::error::Error>> {
    println!("📄 Demo 1: Content-based Metadata Extraction");
    println!("-------------------------------------------");

    // Test with different content types
    let markdown_content = r#"---
title: "Sample Document"
author: "John Doe"
date: "2024-01-15T10:30:00Z"
tags: "rust, documentation, metadata"
category: "technical"
---

# Introduction

This is a sample markdown document with YAML frontmatter.

## Features

- Metadata extraction
- Content chunking
- Language detection
"#;

    let html_content = r#"<!DOCTYPE html>
<html>
<head>
    <title>Sample HTML Document</title>
    <meta name="author" content="Jane Smith">
    <meta name="keywords" content="html, web, development">
    <meta name="description" content="A sample HTML document for testing">
</head>
<body>
    <h1>Welcome to Our Website</h1>
    <p>This is a sample HTML document with metadata in the head section.</p>
    <p>It demonstrates HTML metadata extraction capabilities.</p>
</body>
</html>"#;

    // Extract metadata from markdown
    println!("🔸 Markdown Content:");
    let md_metadata = MetadataExtractor::extract_from_content(markdown_content, Some("md"))?;
    print_metadata(&md_metadata);

    // Extract metadata from HTML
    println!("\n🔸 HTML Content:");
    let html_metadata = MetadataExtractor::extract_from_content(html_content, Some("html"))?;
    print_metadata(&html_metadata);

    // Extract metadata from plain text
    println!("\n🔸 Plain Text Content:");
    let text_content = "Document Title\n\nThis is a sample text document with multiple paragraphs. It contains several sentences to test the word counting and language detection features of the metadata extraction system.";
    let text_metadata = MetadataExtractor::extract_from_content(text_content, Some("txt"))?;
    print_metadata(&text_metadata);

    Ok(())
}

async fn demo_document_processing() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n\n📊 Demo 2: Complete Document Processing");
    println!("--------------------------------------");

    let processor = DocumentProcessor::new_with_config(500, 100); // Smaller chunks for demo

    // Process document content with metadata
    let content = "# Machine Learning Fundamentals\n\nMachine learning is a subset of artificial intelligence that focuses on algorithms and statistical models. These systems automatically improve their performance through experience without being explicitly programmed.\n\n## Supervised Learning\n\nSupervised learning uses labeled training data to learn a mapping function from inputs to outputs. Common algorithms include linear regression, decision trees, and neural networks.\n\n## Unsupervised Learning\n\nUnsupervised learning finds hidden patterns in data without labeled examples. Clustering and dimensionality reduction are common unsupervised techniques.\n\n## Reinforcement Learning\n\nReinforcement learning involves training agents to make decisions by interacting with an environment and receiving rewards or penalties.";

    let (document, chunks) = processor.process_document_content(
        "Machine Learning Guide".to_string(),
        content.to_string(),
        Some("md"),
        ChunkingStrategy::Semantic,
    )?;

    println!("📄 Processed Document:");
    println!("  ID: {}", document.id);
    println!("  Title: {}", document.title);
    println!("  Content Length: {} characters", document.content.len());
    
    println!("\n📊 Document Metadata:");
    print_metadata(&document.metadata);

    println!("\n📦 Generated Chunks ({} total):", chunks.len());
    for (i, chunk) in chunks.iter().enumerate() {
        println!("  Chunk {}: {} characters", i + 1, chunk.content.len());
        if chunk.content.len() < 200 {
            println!("    Preview: {}", chunk.content.replace('\n', " ").chars().take(150).collect::<String>() + "...");
        }
    }

    Ok(())
}

async fn demo_file_type_metadata() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n\n🔧 Demo 3: File Type Specific Features");
    println!("-------------------------------------");

    // Demo different chunking strategies
    let processor = DocumentProcessor::new();
    let sample_text = "This is sentence one. This is sentence two! This is sentence three? This is sentence four.\n\nThis is a new paragraph with multiple sentences. Another sentence here. And one more sentence.\n\nFinal paragraph with some content. Last sentence of the document.";

    println!("🔸 Chunking Strategy Comparison:");
    
    let strategies = [
        (ChunkingStrategy::FixedSize, "Fixed Size"),
        (ChunkingStrategy::Semantic, "Semantic"),
        (ChunkingStrategy::Sentence, "Sentence-based"),
        (ChunkingStrategy::Paragraph, "Paragraph-based"),
    ];

    for (strategy, name) in strategies {
        let chunks = processor.chunk_text_with_strategy(
            &uuid::Uuid::new_v4(),
            sample_text,
            strategy,
        )?;
        
        println!("  {}: {} chunks", name, chunks.len());
        for (i, chunk) in chunks.iter().enumerate() {
            println!("    Chunk {}: {} chars", i + 1, chunk.content.len());
        }
    }

    // Demo language detection
    println!("\n🔸 Language Detection:");
    let texts = [
        ("The quick brown fox jumps over the lazy dog", "English"),
        ("El perro marrón salta sobre el gato perezoso", "Spanish"),
        ("Le renard brun saute par-dessus le chien paresseux", "French"),
    ];

    for (text, expected) in texts {
        let metadata = MetadataExtractor::extract_from_content(text, Some("txt"))?;
        println!("  Expected: {}, Detected: {}", expected, metadata.language.as_deref().unwrap_or("unknown"));
    }

    Ok(())
}

fn print_metadata(metadata: &DocumentMetadata) {
    println!("  📋 Metadata Summary:");
    
    if let Some(author) = &metadata.author {
        println!("    👤 Author: {}", author);
    }
    
    if let Some(created) = &metadata.created_date {
        println!("    📅 Created: {}", created.format("%Y-%m-%d %H:%M:%S UTC"));
    }
    
    if let Some(modified) = &metadata.modified_date {
        println!("    🔄 Modified: {}", modified.format("%Y-%m-%d %H:%M:%S UTC"));
    }
    
    if let Some(language) = &metadata.language {
        println!("    🌍 Language: {}", language);
    }
    
    if let Some(word_count) = metadata.word_count {
        println!("    📊 Word Count: {}", word_count);
    }
    
    if let Some(page_count) = metadata.page_count {
        println!("    📄 Page Count: {}", page_count);
    }
    
    if let Some(file_type) = &metadata.file_type {
        println!("    📁 File Type: {}", file_type);
    }
    
    if let Some(file_size) = metadata.file_size {
        println!("    💾 File Size: {} bytes", file_size);
    }
    
    if !metadata.tags.is_empty() {
        println!("    🏷️  Tags: {}", metadata.tags.join(", "));
    }
    
    if let Some(category) = &metadata.category {
        println!("    📂 Category: {}", category);
    }
    
    // Print custom fields if any
    if let Ok(custom_map) = serde_json::from_value::<serde_json::Map<String, serde_json::Value>>(metadata.custom_fields.clone()) {
        if !custom_map.is_empty() {
            println!("    🔧 Custom Fields:");
            for (key, value) in custom_map {
                if let Some(str_value) = value.as_str() {
                    println!("      {}: {}", key, str_value);
                }
            }
        }
    }
}

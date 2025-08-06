use rustrag::{
    core::{
        SimilaritySearchEngine, SearchConfig, SearchFilters, DistanceMetric,
    },
    models::QueryOptions,
    storage::MockVectorStore,
    Result,
};
use std::sync::Arc;
use uuid::Uuid;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("🔍 RustRAG Similarity Search Engine Demo");
    println!("========================================\n");

    // Demo 1: Basic similarity search with different distance metrics
    demo_distance_metrics().await?;

    // Demo 2: Advanced search with filters
    demo_search_filters().await?;

    // Demo 3: Batch search operations
    demo_batch_search().await?;

    // Demo 4: Find similar chunks
    demo_similar_chunks().await?;

    // Demo 5: Search statistics and performance metrics
    demo_search_statistics().await?;

    // Demo 6: Query preprocessing and validation
    demo_query_processing().await?;

    Ok(())
}

async fn demo_distance_metrics() -> Result<()> {
    println!("📏 Demo 1: Distance Metrics Comparison");
    println!("-------------------------------------");

    let vector_store = Arc::new(MockVectorStore);
    
    // Test different distance metrics
    let metrics = [
        (DistanceMetric::Cosine, "Cosine Distance (best for normalized vectors)"),
        (DistanceMetric::Euclidean, "Euclidean Distance (L2 norm)"),
        (DistanceMetric::Manhattan, "Manhattan Distance (L1 norm)"),
        (DistanceMetric::DotProduct, "Dot Product (for normalized vectors)"),
    ];

    for (metric, description) in metrics {
        println!("\n🔸 Testing {}: {}", format!("{:?}", metric), description);

        let config = SearchConfig {
            max_results: 5,
            similarity_threshold: 0.3,
            distance_metric: metric,
            include_metadata: true,
            preprocess_query: true,
            max_query_length: 1000,
        };

        let engine = SimilaritySearchEngine::with_config(vector_store.clone(), config);

        // Generate a sample query embedding
        let query_embedding = generate_sample_embedding(384, 0.5);

        let results = engine.search(
            query_embedding.clone(),
            None,
            Some(QueryOptions {
                max_chunks: Some(3),
                similarity_threshold: Some(0.4),
                ..Default::default()
            }),
        ).await?;

        println!("  📊 Found {} results", results.len());
        for (i, result) in results.iter().enumerate() {
            println!("    {}. Score: {:.3}, Distance: {:.3}, Rank: {}", 
                     i + 1, result.similarity_score, result.distance, result.rank);
            if let Some(explanation) = &result.explanation {
                println!("       Explanation: {}", explanation);
            }
        }
    }

    Ok(())
}

async fn demo_search_filters() -> Result<()> {
    println!("\n\n🔍 Demo 2: Advanced Search with Filters");
    println!("---------------------------------------");

    let vector_store = Arc::new(MockVectorStore);
    let engine = SimilaritySearchEngine::new(vector_store);

    // Create various search filters
    let filter_scenarios = [
        ("Document ID Filter", SearchFilters {
            document_ids: Some(vec![Uuid::new_v4(), Uuid::new_v4()]),
            ..Default::default()
        }),
        ("Chunk Index Range", SearchFilters {
            chunk_index_range: Some((0, 10)),
            ..Default::default()
        }),
        ("Content Length Filter", SearchFilters {
            content_length_range: Some((100, 1000)),
            ..Default::default()
        }),
        ("Date Range Filter", SearchFilters {
            created_after: Some(chrono::Utc::now() - chrono::Duration::days(7)),
            created_before: Some(chrono::Utc::now()),
            ..Default::default()
        }),
    ];

    for (name, filters) in filter_scenarios {
        println!("\n🔸 Testing Filter: {}", name);
        
        let query_embedding = generate_sample_embedding(384, 0.7);
        let results = engine.search(
            query_embedding,
            Some(filters),
            Some(QueryOptions {
                max_chunks: Some(5),
                ..Default::default()
            }),
        ).await?;

        println!("  📊 Filtered results: {} chunks", results.len());
        for result in results.iter().take(2) {
            println!("    - Document: {} (Score: {:.3})", 
                     result.chunk.document_title, result.similarity_score);
        }
    }

    // Metadata filters example
    println!("\n🔸 Testing Metadata Filters");
    let mut metadata_filters = HashMap::new();
    metadata_filters.insert("source".to_string(), serde_json::json!("research_papers"));
    metadata_filters.insert("language".to_string(), serde_json::json!("en"));

    let metadata_filter = SearchFilters {
        metadata_filters,
        ..Default::default()
    };

    let query_embedding = generate_sample_embedding(384, 0.8);
    let results = engine.search(
        query_embedding,
        Some(metadata_filter),
        None,
    ).await?;

    println!("  📊 Metadata-filtered results: {} chunks", results.len());

    Ok(())
}

async fn demo_batch_search() -> Result<()> {
    println!("\n\n📦 Demo 3: Batch Search Operations");
    println!("----------------------------------");

    let vector_store = Arc::new(MockVectorStore);
    let engine = SimilaritySearchEngine::new(vector_store);

    // Generate multiple query embeddings
    let query_embeddings = vec![
        generate_sample_embedding(384, 0.1),
        generate_sample_embedding(384, 0.5),
        generate_sample_embedding(384, 0.9),
    ];

    println!("🔸 Performing batch search for {} queries", query_embeddings.len());

    let batch_results = engine.batch_search(
        query_embeddings,
        None,
        Some(QueryOptions {
            max_chunks: Some(3),
            similarity_threshold: Some(0.3),
            ..Default::default()
        }),
    ).await?;

    for (i, results) in batch_results.iter().enumerate() {
        println!("  Query {}: {} results (avg score: {:.3})", 
                 i + 1, 
                 results.len(),
                 results.iter().map(|r| r.similarity_score).sum::<f32>() / results.len() as f32);
    }

    println!("✅ Batch search completed successfully");

    Ok(())
}

async fn demo_similar_chunks() -> Result<()> {
    println!("\n\n🧩 Demo 4: Find Similar Chunks");
    println!("------------------------------");

    let vector_store = Arc::new(MockVectorStore);
    let engine = SimilaritySearchEngine::new(vector_store);

    let source_chunk_id = Uuid::new_v4();

    println!("🔸 Finding chunks similar to source chunk: {}", source_chunk_id);

    let similar_chunks = engine.find_similar_chunks(
        source_chunk_id,
        5, // max_results
        true, // exclude_same_document
    ).await?;

    println!("  📊 Found {} similar chunks", similar_chunks.len());
    for (i, result) in similar_chunks.iter().enumerate() {
        println!("    {}. Document: {} (Score: {:.3})", 
                 i + 1, result.chunk.document_title, result.similarity_score);
        println!("       Content preview: {}...", 
                 result.chunk.content.chars().take(80).collect::<String>());
    }

    Ok(())
}

async fn demo_search_statistics() -> Result<()> {
    println!("\n\n📊 Demo 5: Search Statistics and Performance Metrics");
    println!("---------------------------------------------------");

    let vector_store = Arc::new(MockVectorStore);
    let config = SearchConfig {
        max_results: 20,
        similarity_threshold: 0.2,
        distance_metric: DistanceMetric::Cosine,
        include_metadata: true,
        preprocess_query: true,
        max_query_length: 2000,
    };

    let engine = SimilaritySearchEngine::with_config(vector_store, config);

    let stats = engine.get_search_stats().await?;

    println!("🔸 Vector Store Statistics:");
    println!("  📈 Total vectors: {}", stats.total_vectors);
    println!("  📐 Vector dimension: {}", stats.vector_dimension);
    println!("  🟢 Index status: {}", stats.index_status);
    println!("  📏 Distance metric: {:?}", stats.distance_metric);
    println!("  🎯 Similarity threshold: {}", stats.similarity_threshold);

    // Performance test
    println!("\n🔸 Performance Test:");
    let start_time = std::time::Instant::now();
    
    let query_embedding = generate_sample_embedding(384, 0.6);
    let _results = engine.search(query_embedding, None, None).await?;
    
    let elapsed = start_time.elapsed();
    println!("  ⏱️  Search completed in: {:?}", elapsed);

    Ok(())
}

async fn demo_query_processing() -> Result<()> {
    println!("\n\n🔤 Demo 6: Query Preprocessing and Validation");
    println!("---------------------------------------------");

    let vector_store = Arc::new(MockVectorStore);
    let engine = SimilaritySearchEngine::new(vector_store);

    // Mock embedding generator
    let embedding_generator = |text: &str| -> Result<Vec<f32>> {
        println!("  🧠 Generating embedding for: \"{}\"", 
                 text.chars().take(50).collect::<String>());
        Ok(generate_sample_embedding(384, text.len() as f32 / 1000.0))
    };

    let test_queries = [
        "What is machine learning?",
        "  HELLO, WORLD!!! How are you?  ",
        "Explain neural networks and deep learning algorithms",
        "🚀 Modern AI applications in healthcare",
    ];

    for query in test_queries {
        println!("\n🔸 Processing query: \"{}\"", query);
        
        match engine.search_with_text(
            query,
            &embedding_generator,
            None,
            Some(QueryOptions {
                max_chunks: Some(2),
                ..Default::default()
            }),
        ).await {
            Ok(results) => {
                println!("  ✅ Query processed successfully - {} results", results.len());
                for result in results.iter().take(1) {
                    println!("    Top result: {} (Score: {:.3})", 
                             result.chunk.document_title, result.similarity_score);
                }
            }
            Err(e) => {
                println!("  ❌ Query processing failed: {}", e);
            }
        }
    }

    // Test query length validation
    println!("\n🔸 Testing query length limits:");
    let long_query = "word ".repeat(500); // Very long query
    match engine.search_with_text(
        &long_query,
        &embedding_generator,
        None,
        None,
    ).await {
        Ok(_) => println!("  ✅ Long query processed successfully"),
        Err(e) => println!("  ❌ Long query rejected: {}", e),
    }

    Ok(())
}

// Helper function to generate sample embeddings
fn generate_sample_embedding(dimension: usize, seed: f32) -> Vec<f32> {
    let mut embedding = Vec::with_capacity(dimension);
    for i in 0..dimension {
        // Generate a deterministic but varied embedding
        let value = ((i as f32 * seed).sin() * 0.5 + 0.5) * 2.0 - 1.0;
        embedding.push(value);
    }
    
    // Normalize the embedding for cosine similarity
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut embedding {
            *value /= norm;
        }
    }
    
    embedding
}

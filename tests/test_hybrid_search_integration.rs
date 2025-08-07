use rustrag::{
    models::{Query, DocumentChunk},
    core::{HybridSearchScorer, HybridSearchConfig, build_document_stats},
    utils::Result,
};
use std::sync::Arc;
use uuid::Uuid;
use chrono::Utc;

/// Create a test document chunk
fn create_test_chunk(content: &str, embedding: Option<Vec<f32>>) -> DocumentChunk {
    DocumentChunk {
        id: Uuid::new_v4(),
        document_id: Uuid::new_v4(),
        content: content.to_string(),
        chunk_index: 0,
        embedding,
        metadata: serde_json::json!({}),
        created_at: Utc::now(),
    }
}

/// Create test embedding data
fn create_test_embedding(similarity: f32) -> Vec<f32> {
    // Create a simple embedding that when compared with [1,0,0,0] gives the desired similarity
    vec![similarity, (1.0 - similarity * similarity).sqrt(), 0.0, 0.0]
}


#[tokio::test]
async fn test_hybrid_search_scorer_directly() -> Result<()> {
    let chunks = vec![
        create_test_chunk(
            "Machine learning and artificial intelligence are related fields",
            Some(vec![0.1, 0.2, 0.3, 0.4]),
        ),
        create_test_chunk(
            "The cat sat on the mat quietly",
            Some(vec![0.5, 0.6, 0.7, 0.8]),
        ),
    ];

    // Build document stats
    let stats = Arc::new(build_document_stats(&chunks).await);

    // Create hybrid search configuration
    let config = HybridSearchConfig {
        vector_weight: 0.6,
        keyword_weight: 0.4,
        ..Default::default()
    };

    // Create scorer
    let scorer = HybridSearchScorer::new(config, stats);

    // Test query
    let query = Query::new("machine learning algorithms".to_string());
    let query_embedding = vec![0.1, 0.2, 0.3, 0.4];

    // Perform search
    let results = scorer.search_chunks(&query, &chunks, &query_embedding).await?;

    assert_eq!(results.len(), 2);
    
    // The machine learning chunk should rank higher due to keyword matching
    assert!(results[0].chunk.content.contains("Machine learning"));
    assert!(results[0].hybrid_score > results[1].hybrid_score);

    // Check that scores are reasonable
    assert!(results[0].vector_score >= 0.0);
    assert!(results[0].keyword_score >= 0.0);
    assert!(results[0].hybrid_score >= 0.0);

    println!("Direct hybrid search test results:");
    for result in &results {
        println!("Content: {}", result.chunk.content);
        println!("  Vector: {:.3}, Keyword: {:.3}, Hybrid: {:.3}", 
                result.vector_score, result.keyword_score, result.hybrid_score);
        println!("  Explanation: {}", result.explanation);
    }

    Ok(())
}

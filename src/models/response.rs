use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Query response structure containing generated answer and retrieved chunks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResponse {
    pub id: Uuid,
    pub query_id: Uuid,
    pub answer: String,
    pub retrieved_chunks: Vec<RetrievedChunk>,
    pub citations: Vec<Citation>,
    pub processing_time_ms: u64,
    pub created_at: DateTime<Utc>,
}

/// Retrieved chunk with similarity score and enhanced metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievedChunk {
    pub id: Uuid,  // chunk_id for compatibility
    pub chunk_id: Uuid,  // Keep for backwards compatibility
    pub document_id: Uuid,
    pub document_title: String,
    pub content: String,
    pub similarity_score: f32,
    pub chunk_index: i32,
    pub embedding: Option<Vec<f32>>,
    pub metadata: serde_json::Value,
    pub created_at: DateTime<Utc>,
}

/// Citation information for transparency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    pub document_id: Uuid,
    pub document_title: String,
    pub chunk_id: Uuid,
    pub page_number: Option<u32>,
    pub excerpt: String,
}

impl QueryResponse {
    /// Create a new query response
    pub fn new(
        query_id: Uuid,
        answer: String,
        retrieved_chunks: Vec<RetrievedChunk>,
        processing_time_ms: u64,
    ) -> Self {
        let citations = retrieved_chunks
            .iter()
            .map(|chunk| Citation {
                document_id: chunk.document_id,
                document_title: chunk.document_title.clone(),
                chunk_id: chunk.chunk_id,
                page_number: None, // TODO: Extract from chunk metadata
                excerpt: chunk.content
                    .chars()
                    .take(200)
                    .collect::<String>()
                    .trim()
                    .to_string(),
            })
            .collect();

        Self {
            id: Uuid::new_v4(),
            query_id,
            answer,
            retrieved_chunks,
            citations,
            processing_time_ms,
            created_at: Utc::now(),
        }
    }

    /// Create a new query response without citations
    pub fn new_without_citations(
        query_id: Uuid,
        answer: String,
        retrieved_chunks: Vec<RetrievedChunk>,
        processing_time_ms: u64,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            query_id,
            answer,
            retrieved_chunks,
            citations: Vec::new(),
            processing_time_ms,
            created_at: Utc::now(),
        }
    }
}

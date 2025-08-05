use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Query structure for RAG operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Query {
    pub id: Uuid,
    pub text: String,
    pub options: QueryOptions,
    pub created_at: DateTime<Utc>,
}

/// Query options for controlling retrieval behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryOptions {
    /// Maximum number of chunks to retrieve
    pub max_chunks: Option<usize>,
    /// Minimum similarity threshold for retrieval
    pub similarity_threshold: Option<f32>,
    /// Whether to include citations in the response
    pub include_citations: bool,
    /// Specific document IDs to search within (optional)
    pub document_ids: Option<Vec<Uuid>>,
    /// Tags to filter documents by
    pub filter_tags: Option<Vec<String>>,
    /// Document category to filter by
    pub filter_category: Option<String>,
    /// Temperature setting for LLM generation
    pub temperature: Option<f32>,
    /// Maximum tokens for LLM response
    pub max_tokens: Option<u32>,
}

impl Default for QueryOptions {
    fn default() -> Self {
        Self {
            max_chunks: Some(10),
            similarity_threshold: Some(0.7),
            include_citations: true,
            document_ids: None,
            filter_tags: None,
            filter_category: None,
            temperature: Some(0.7),
            max_tokens: Some(1000),
        }
    }
}

impl Query {
    /// Create a new query with default options
    pub fn new(text: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            text,
            options: QueryOptions::default(),
            created_at: Utc::now(),
        }
    }

    /// Create a new query with custom options
    pub fn new_with_options(text: String, options: QueryOptions) -> Self {
        Self {
            id: Uuid::new_v4(),
            text,
            options,
            created_at: Utc::now(),
        }
    }
}

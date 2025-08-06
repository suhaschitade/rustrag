use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Document model representing a processed document in the system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: Uuid,
    pub title: String,
    pub content: String,
    pub file_path: Option<String>,
    pub file_type: Option<String>,
    pub file_size: Option<i64>,
    pub metadata: DocumentMetadata,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Document chunk for RAG processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentChunk {
    pub id: Uuid,
    pub document_id: Uuid,
    pub chunk_index: i32,
    pub content: String,
    pub embedding: Option<Vec<f32>>,
    pub metadata: serde_json::Value,
    pub created_at: DateTime<Utc>,
}

/// Document metadata structure
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DocumentMetadata {
    pub author: Option<String>,
    pub created_date: Option<DateTime<Utc>>,
    pub modified_date: Option<DateTime<Utc>>,
    pub language: Option<String>,
    pub tags: Vec<String>,
    pub category: Option<String>,
    pub word_count: Option<u32>,
    pub page_count: Option<u32>,
    pub file_type: Option<String>,
    pub file_size: Option<u64>,
    pub custom_fields: serde_json::Value,
}

impl Document {
    /// Create a new document
    pub fn new(title: String, content: String) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            title,
            content,
            file_path: None,
            file_type: None,
            file_size: None,
            metadata: DocumentMetadata::default(),
            created_at: now,
            updated_at: now,
        }
    }

    /// Create a new document with metadata
    pub fn new_with_metadata(
        title: String,
        content: String,
        metadata: DocumentMetadata,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            title,
            content,
            file_path: None,
            file_type: None,
            file_size: None,
            metadata,
            created_at: now,
            updated_at: now,
        }
    }
}

impl DocumentChunk {
    /// Create a new document chunk
    pub fn new(document_id: Uuid, chunk_index: i32, content: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            document_id,
            chunk_index,
            content,
            embedding: None,
            metadata: serde_json::Value::Object(serde_json::Map::new()),
            created_at: Utc::now(),
        }
    }

    /// Create a new document chunk with embedding
    pub fn new_with_embedding(
        document_id: Uuid,
        chunk_index: i32,
        content: String,
        embedding: Vec<f32>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            document_id,
            chunk_index,
            content,
            embedding: Some(embedding),
            metadata: serde_json::Value::Object(serde_json::Map::new()),
            created_at: Utc::now(),
        }
    }
}

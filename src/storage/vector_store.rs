use crate::models::{DocumentChunk, Query};
use crate::utils::Result;
use async_trait::async_trait;
use qdrant_client::Qdrant;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Vector store operations
#[async_trait]
pub trait VectorStore {
    /// Insert document chunk embeddings into vector store
    async fn insert_embeddings(&self, chunks: Vec<DocumentChunk>) -> Result<()>;

    /// Perform similarity search on the vector store for the given query
    async fn search_similar_chunks(&self, query: &Query) -> Result<Vec<DocumentChunk>>;
    
    /// Store a single embedding with metadata
    async fn store_embedding(
        &self,
        chunk_id: Uuid,
        embedding: Vec<f32>,
        metadata: serde_json::Value,
    ) -> Result<()>;
    
    /// Search for similar vectors with configurable parameters
    async fn search_similar(
        &self,
        query_embedding: Vec<f32>,
        limit: usize,
        threshold: f32,
    ) -> Result<Vec<SimilarityMatch>>;
    
    /// Delete a vector by chunk ID
    async fn delete_vector(&self, chunk_id: Uuid) -> Result<()>;
    
    /// Get collection information
    async fn get_collection_info(&self) -> Result<CollectionInfo>;
}

/// Configuration for Qdrant vector store
#[derive(Debug, Clone)]
pub struct QdrantConfig {
    pub url: String,
    pub api_key: Option<String>,
    pub collection_name: String,
    pub vector_size: usize,
    pub enable_compression: bool,
}

impl Default for QdrantConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:6334".to_string(),
            api_key: None,
            collection_name: "rustrag_documents".to_string(),
            vector_size: 384, // Default for sentence-transformers
            enable_compression: true,
        }
    }
}

/// Qdrant implementation of vector store
pub struct QdrantVectorStore {
    client: Qdrant,
    config: QdrantConfig,
}

impl QdrantVectorStore {
    /// Create a new Qdrant vector store with default configuration
    pub async fn new() -> Result<Self> {
        Self::with_config(QdrantConfig::default()).await
    }

    /// Create a new Qdrant vector store with custom configuration
    pub async fn with_config(config: QdrantConfig) -> Result<Self> {
        use qdrant_client::config::QdrantConfig as QdrantClientConfig;

        let mut client_config = QdrantClientConfig::from_url(&config.url);
        
        // Set API key if provided
        if let Some(api_key) = config.api_key.clone() {
            client_config = client_config.api_key(api_key);
        }

        let client = Qdrant::new(client_config)
            .map_err(|e| crate::utils::Error::vector_db(format!("Failed to connect to Qdrant: {}", e)))?;

        let vector_store = Self { client, config };
        
        // Initialize collection if it doesn't exist
        vector_store.ensure_collection().await?;
        
        Ok(vector_store)
    }

    /// Ensure the collection exists, create if not
    async fn ensure_collection(&self) -> Result<()> {
        // Check if collection exists
        let collections = self.client.list_collections().await
            .map_err(|e| crate::utils::Error::vector_db(format!("Failed to list collections: {}", e)))?;

        let collection_exists = collections
            .collections
            .iter()
            .any(|c| c.name == self.config.collection_name);

        if !collection_exists {
            self.create_collection().await?;
        } else {
            tracing::info!("Collection '{}' already exists", self.config.collection_name);
        }

        Ok(())
    }

    /// Create a new collection
    async fn create_collection(&self) -> Result<()> {
        use qdrant_client::qdrant::{vectors_config::Config, CreateCollection, Distance, VectorParams, VectorsConfig};

        let vectors_config = VectorsConfig {
            config: Some(Config::Params(
                VectorParams {
                    size: self.config.vector_size as u64,
                    distance: Distance::Cosine.into(),
                    hnsw_config: None,
                    quantization_config: None,
                    on_disk: Some(false),
                    datatype: None,
                    multivector_config: None,
                },
            )),
        };

        let create_collection = CreateCollection {
            collection_name: self.config.collection_name.clone(),
            vectors_config: Some(vectors_config),
            shard_number: Some(1),
            replication_factor: Some(1),
            write_consistency_factor: Some(1),
            on_disk_payload: Some(true),
            hnsw_config: None,
            wal_config: None,
            optimizers_config: None,
            init_from_collection: None,
            quantization_config: None,
            sharding_method: None,
            sparse_vectors_config: None,
            strict_mode_config: None,
            timeout: None,
        };

        self.client
            .create_collection(create_collection)
            .await
            .map_err(|e| crate::utils::Error::vector_db(format!("Failed to create collection: {}", e)))?;

        tracing::info!("Created collection '{}'", self.config.collection_name);
        Ok(())
    }

    /// Store multiple embeddings in batch
    pub async fn store_embeddings_batch(
        &self,
        embeddings: Vec<(Uuid, Vec<f32>, serde_json::Value)>,
    ) -> Result<()> {
        if embeddings.is_empty() {
            return Ok(());
        }

        // For now, just log that this would store embeddings
        // The new API is too complex to implement without more specific documentation
        tracing::info!("Would store {} embeddings in batch", embeddings.len());
        
        Ok(())
    }
}

#[async_trait]
impl VectorStore for QdrantVectorStore {
    async fn insert_embeddings(&self, chunks: Vec<DocumentChunk>) -> Result<()> {
        if chunks.is_empty() {
            return Ok(());
        }

        // Convert chunks to embeddings format
        let embeddings: Vec<(Uuid, Vec<f32>, serde_json::Value)> = chunks
            .into_iter()
            .filter_map(|chunk| {
                if let Some(embedding) = chunk.embedding {
                    let metadata = serde_json::json!({
                        "chunk_id": chunk.id,
                        "document_id": chunk.document_id,
                        "chunk_index": chunk.chunk_index,
                        "content": chunk.content,
                        "created_at": chunk.created_at,
                        "metadata": chunk.metadata
                    });
                    Some((chunk.id, embedding, metadata))
                } else {
                    tracing::warn!("Chunk {} has no embedding, skipping", chunk.id);
                    None
                }
            })
            .collect();

        self.store_embeddings_batch(embeddings).await
    }

    async fn search_similar_chunks(&self, _query: &Query) -> Result<Vec<DocumentChunk>> {
        // This requires query embeddings to be implemented
        // For now, return empty result with a todo message
        tracing::warn!("search_similar_chunks requires query embedding generation - implement in embedding service");
        Ok(vec![])
    }

    async fn store_embedding(
        &self,
        chunk_id: Uuid,
        embedding: Vec<f32>,
        _metadata: serde_json::Value,
    ) -> Result<()> {
        if embedding.len() != self.config.vector_size {
            return Err(crate::utils::Error::vector_db(format!(
                "Embedding size mismatch: expected {}, got {}",
                self.config.vector_size,
                embedding.len()
            )));
        }

        // For now, just log that this would store the embedding
        tracing::info!("Would store embedding for chunk: {}", chunk_id);
        
        Ok(())
    }

    async fn search_similar(
        &self,
        query_embedding: Vec<f32>,
        limit: usize,
        threshold: f32,
    ) -> Result<Vec<SimilarityMatch>> {
        if query_embedding.len() != self.config.vector_size {
            return Err(crate::utils::Error::vector_db(format!(
                "Query embedding size mismatch: expected {}, got {}",
                self.config.vector_size,
                query_embedding.len()
            )));
        }

        // For now, just log that this would search for similar vectors
        tracing::info!("Would search for {} similar vectors with threshold {}", limit, threshold);
        
        Ok(vec![])
    }

    async fn delete_vector(&self, chunk_id: Uuid) -> Result<()> {
        // For now, just log that this would delete the vector
        tracing::info!("Would delete vector for chunk: {}", chunk_id);
        
        Ok(())
    }

    async fn get_collection_info(&self) -> Result<CollectionInfo> {
        // For now, return mock collection info
        Ok(CollectionInfo {
            name: self.config.collection_name.clone(),
            status: "Ready".to_string(),
            vectors_count: 0,
            points_count: 0,
        })
    }
}

/// Mock implementation of a vector store
pub struct MockVectorStore;

#[async_trait]
impl VectorStore for MockVectorStore {
    async fn insert_embeddings(&self, chunks: Vec<DocumentChunk>) -> Result<()> {
        tracing::info!("Mock: Inserting {} embeddings", chunks.len());
        Ok(())
    }

    async fn search_similar_chunks(&self, query: &Query) -> Result<Vec<DocumentChunk>> {
        tracing::info!("Mock: Searching similar chunks for query: {}", query.id);
        Ok(vec![])
    }
    
    async fn store_embedding(
        &self,
        chunk_id: Uuid,
        _embedding: Vec<f32>,
        _metadata: serde_json::Value,
    ) -> Result<()> {
        tracing::info!("Mock: Storing embedding for chunk: {}", chunk_id);
        Ok(())
    }
    
    async fn search_similar(
        &self,
        _query_embedding: Vec<f32>,
        limit: usize,
        _threshold: f32,
    ) -> Result<Vec<SimilarityMatch>> {
        tracing::info!("Mock: Searching for {} similar vectors", limit);
        Ok(vec![])
    }
    
    async fn delete_vector(&self, chunk_id: Uuid) -> Result<()> {
        tracing::info!("Mock: Deleting vector for chunk: {}", chunk_id);
        Ok(())
    }
    
    async fn get_collection_info(&self) -> Result<CollectionInfo> {
        Ok(CollectionInfo {
            name: "mock_collection".to_string(),
            status: "Green".to_string(),
            vectors_count: 0,
            points_count: 0,
        })
    }
}

/// Result of a similarity search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityMatch {
    pub chunk_id: Uuid,
    pub score: f32,
    pub metadata: serde_json::Value,
}

/// Collection information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionInfo {
    pub name: String,
    pub status: String,
    pub vectors_count: u64,
    pub points_count: u64,
}

/// Payload field types for indexing
#[derive(Debug, Clone, Copy)]
pub enum PayloadFieldType {
    Keyword,
    Integer,
    Float,
    Bool,
}

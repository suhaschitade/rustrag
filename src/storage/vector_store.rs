use crate::models::{DocumentChunk, Query, RetrievedChunk};
use crate::utils::Result;
use crate::core::similarity_search::SearchFilters;
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
    
    /// Advanced similarity search with filters and enhanced results
    async fn similarity_search(
        &self,
        query_embedding: &[f32],
        limit: usize,
        filters: Option<&SearchFilters>,
    ) -> Result<Vec<RetrievedChunk>>;
    
    /// Get a chunk by its ID with embedding
    async fn get_chunk_by_id(&self, chunk_id: &Uuid) -> Result<Option<RetrievedChunk>>;
    
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

        use qdrant_client::qdrant::{PointStruct, UpsertPoints, Vectors};
        use std::collections::HashMap;

        let embeddings_len = embeddings.len();
        
        // Convert embeddings to Qdrant points
        let points: Vec<PointStruct> = embeddings
            .into_iter()
            .map(|(chunk_id, embedding, metadata)| {
                // Convert metadata to HashMap<String, Value>
                let payload: HashMap<String, qdrant_client::qdrant::Value> = metadata
                    .as_object()
                    .unwrap_or(&serde_json::Map::new())
                    .iter()
                    .map(|(k, v)| {
                        let qdrant_value = match v {
                            serde_json::Value::String(s) => qdrant_client::qdrant::Value {
                                kind: Some(qdrant_client::qdrant::value::Kind::StringValue(s.clone())),
                            },
                            serde_json::Value::Number(n) => {
                                if let Some(i) = n.as_i64() {
                                    qdrant_client::qdrant::Value {
                                        kind: Some(qdrant_client::qdrant::value::Kind::IntegerValue(i)),
                                    }
                                } else if let Some(f) = n.as_f64() {
                                    qdrant_client::qdrant::Value {
                                        kind: Some(qdrant_client::qdrant::value::Kind::DoubleValue(f)),
                                    }
                                } else {
                                    qdrant_client::qdrant::Value {
                                        kind: Some(qdrant_client::qdrant::value::Kind::StringValue(n.to_string())),
                                    }
                                }
                            },
                            serde_json::Value::Bool(b) => qdrant_client::qdrant::Value {
                                kind: Some(qdrant_client::qdrant::value::Kind::BoolValue(*b)),
                            },
                            _ => qdrant_client::qdrant::Value {
                                kind: Some(qdrant_client::qdrant::value::Kind::StringValue(v.to_string())),
                            },
                        };
                        (k.clone(), qdrant_value)
                    })
                    .collect();

                PointStruct {
                    id: Some(chunk_id.to_string().into()),
                    vectors: Some(Vectors {
                        vectors_options: Some(qdrant_client::qdrant::vectors::VectorsOptions::Vector(
                            qdrant_client::qdrant::Vector {
                                data: embedding,
                                indices: None,
                                vector: None,
                                vectors_count: None,
                            },
                        )),
                    }),
                    payload,
                }
            })
            .collect();

        let upsert_request = UpsertPoints {
            collection_name: self.config.collection_name.clone(),
            points,
            wait: Some(true),
            ordering: None,
            shard_key_selector: None,
        };

        self.client
            .upsert_points(upsert_request)
            .await
            .map_err(|e| {
                crate::utils::Error::vector_db(format!("Failed to upsert points: {}", e))
            })?;

        tracing::info!("Successfully stored {} embeddings in batch", embeddings_len);
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

    async fn similarity_search(
        &self,
        _query_embedding: &[f32],
        limit: usize,
        _filters: Option<&SearchFilters>,
    ) -> Result<Vec<RetrievedChunk>> {
        // Mock implementation - this would use actual Qdrant search
        tracing::info!("Mock Qdrant: Searching for {} similar vectors", limit);
        
        // For now, return empty results
        Ok(vec![])
    }

    async fn get_chunk_by_id(&self, chunk_id: &Uuid) -> Result<Option<RetrievedChunk>> {
        // Mock implementation - this would query Qdrant for the specific chunk
        tracing::info!("Mock Qdrant: Getting chunk by ID: {}", chunk_id);
        
        Ok(None)
    }

    async fn get_collection_info(&self) -> Result<CollectionInfo> {
        // For now, return mock collection info
        Ok(CollectionInfo {
            name: self.config.collection_name.clone(),
            status: "Ready".to_string(),
            vectors_count: Some(0),
            points_count: 0,
            vectors_config: Some(VectorConfig {
                size: self.config.vector_size as u64,
                distance: "Cosine".to_string(),
            }),
        })
    }
}

/// In-memory vector store for testing and development
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

#[derive(Debug, Clone)]
pub struct InMemoryVector {
    pub id: Uuid,
    pub embedding: Vec<f32>,
    pub metadata: serde_json::Value,
}

pub struct InMemoryVectorStore {
    vectors: Arc<RwLock<HashMap<Uuid, InMemoryVector>>>,
    dimension: usize,
}

impl InMemoryVectorStore {
    pub fn new() -> Self {
        Self {
            vectors: Arc::new(RwLock::new(HashMap::new())),
            dimension: 384, // Default dimension
        }
    }
    
    pub fn with_dimension(dimension: usize) -> Self {
        Self {
            vectors: Arc::new(RwLock::new(HashMap::new())),
            dimension,
        }
    }
    
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }
        
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        
        dot_product / (norm_a * norm_b)
    }
}

#[async_trait]
impl VectorStore for InMemoryVectorStore {
    async fn insert_embeddings(&self, chunks: Vec<DocumentChunk>) -> Result<()> {
        let mut vectors = self.vectors.write().unwrap();
        
        for chunk in chunks {
            if let Some(embedding) = chunk.embedding {
                let vector = InMemoryVector {
                    id: chunk.id,
                    embedding,
                    metadata: serde_json::json!({
                        "chunk_id": chunk.id,
                        "document_id": chunk.document_id,
                        "content": chunk.content,
                        "chunk_index": chunk.chunk_index,
                        "created_at": chunk.created_at,
                        "metadata": chunk.metadata
                    }),
                };
                vectors.insert(chunk.id, vector);
            }
        }
        
        tracing::info!("InMemory: Inserted {} embeddings", vectors.len());
        Ok(())
    }
    
    async fn search_similar_chunks(&self, _query: &Query) -> Result<Vec<DocumentChunk>> {
        // This would need query embedding implementation
        tracing::info!("InMemory: search_similar_chunks not implemented - use similarity_search instead");
        Ok(vec![])
    }
    
    async fn store_embedding(
        &self,
        chunk_id: Uuid,
        embedding: Vec<f32>,
        metadata: serde_json::Value,
    ) -> Result<()> {
        if embedding.len() != self.dimension {
            return Err(crate::utils::Error::vector_db(format!(
                "Embedding dimension mismatch: expected {}, got {}",
                self.dimension, embedding.len()
            )));
        }
        
        let vector = InMemoryVector {
            id: chunk_id,
            embedding,
            metadata,
        };
        
        let mut vectors = self.vectors.write().unwrap();
        vectors.insert(chunk_id, vector);
        
        tracing::info!("InMemory: Stored embedding for chunk: {}", chunk_id);
        Ok(())
    }
    
    async fn search_similar(
        &self,
        query_embedding: Vec<f32>,
        limit: usize,
        threshold: f32,
    ) -> Result<Vec<SimilarityMatch>> {
        let vectors = self.vectors.read().unwrap();
        
        let mut matches: Vec<SimilarityMatch> = vectors
            .values()
            .map(|vector| {
                let score = Self::cosine_similarity(&query_embedding, &vector.embedding);
                SimilarityMatch {
                    chunk_id: vector.id,
                    score,
                    metadata: vector.metadata.clone(),
                }
            })
            .filter(|m| m.score >= threshold)
            .collect();
        
        // Sort by score descending
        matches.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        matches.truncate(limit);
        
        tracing::info!("InMemory: Found {} matches above threshold {}", matches.len(), threshold);
        Ok(matches)
    }
    
    async fn similarity_search(
        &self,
        query_embedding: &[f32],
        limit: usize,
        _filters: Option<&SearchFilters>,
    ) -> Result<Vec<RetrievedChunk>> {
        let vectors = self.vectors.read().unwrap();
        
        let mut matches: Vec<(f32, &InMemoryVector)> = vectors
            .values()
            .map(|vector| {
                let score = Self::cosine_similarity(query_embedding, &vector.embedding);
                (score, vector)
            })
            .collect();
        
        // Sort by score descending
        matches.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        matches.truncate(limit);
        
        let results: Vec<RetrievedChunk> = matches
            .into_iter()
            .enumerate()
            .map(|(i, (score, vector))| {
                let metadata = vector.metadata.as_object().unwrap();
                
                RetrievedChunk {
                    id: vector.id,
                    chunk_id: vector.id,
                    document_id: metadata.get("document_id")
                        .and_then(|v| v.as_str())
                        .and_then(|s| Uuid::parse_str(s).ok())
                        .unwrap_or_else(Uuid::new_v4),
                    document_title: format!("Document {}", i + 1), // TODO: Extract from metadata
                    content: metadata.get("content")
                        .and_then(|v| v.as_str())
                        .unwrap_or("No content")
                        .to_string(),
                    similarity_score: score,
                    chunk_index: metadata.get("chunk_index")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0) as i32,
                    embedding: Some(vector.embedding.clone()),
                    metadata: vector.metadata.clone(),
                    created_at: metadata.get("created_at")
                        .and_then(|v| v.as_str())
                        .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                        .map(|dt| dt.with_timezone(&chrono::Utc))
                        .unwrap_or_else(chrono::Utc::now),
                }
            })
            .collect();
        
        tracing::info!("InMemory: Similarity search returned {} results", results.len());
        Ok(results)
    }
    
    async fn get_chunk_by_id(&self, chunk_id: &Uuid) -> Result<Option<RetrievedChunk>> {
        let vectors = self.vectors.read().unwrap();
        
        if let Some(vector) = vectors.get(chunk_id) {
            let metadata = vector.metadata.as_object().unwrap();
            
            Ok(Some(RetrievedChunk {
                id: vector.id,
                chunk_id: vector.id,
                document_id: metadata.get("document_id")
                    .and_then(|v| v.as_str())
                    .and_then(|s| Uuid::parse_str(s).ok())
                    .unwrap_or_else(Uuid::new_v4),
                document_title: "Document".to_string(),
                content: metadata.get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("No content")
                    .to_string(),
                similarity_score: 1.0,
                chunk_index: metadata.get("chunk_index")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0) as i32,
                embedding: Some(vector.embedding.clone()),
                metadata: vector.metadata.clone(),
                created_at: chrono::Utc::now(),
            }))
        } else {
            Ok(None)
        }
    }
    
    async fn delete_vector(&self, chunk_id: Uuid) -> Result<()> {
        let mut vectors = self.vectors.write().unwrap();
        vectors.remove(&chunk_id);
        
        tracing::info!("InMemory: Deleted vector for chunk: {}", chunk_id);
        Ok(())
    }
    
    async fn get_collection_info(&self) -> Result<CollectionInfo> {
        let vectors = self.vectors.read().unwrap();
        
        Ok(CollectionInfo {
            name: "in_memory_collection".to_string(),
            status: "Ready".to_string(),
            vectors_count: Some(vectors.len() as u64),
            points_count: vectors.len() as u64,
            vectors_config: Some(VectorConfig {
                size: self.dimension as u64,
                distance: "Cosine".to_string(),
            }),
        })
    }
}

impl Default for InMemoryVectorStore {
    fn default() -> Self {
        Self::new()
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
    
    async fn similarity_search(
        &self,
        query_embedding: &[f32],
        limit: usize,
        _filters: Option<&SearchFilters>,
    ) -> Result<Vec<RetrievedChunk>> {
        tracing::info!("Mock: Similarity search for {} results with embedding dim {}", limit, query_embedding.len());
        
        // Generate mock results
        let mock_results = (0..std::cmp::min(limit, 3))
            .map(|i| RetrievedChunk {
                id: Uuid::new_v4(),
                chunk_id: Uuid::new_v4(),
                document_id: Uuid::new_v4(),
                document_title: format!("Mock Document {}", i + 1),
                content: format!("Mock content for chunk {} - this would be actual retrieved content from the vector database based on similarity to the query.", i + 1),
                similarity_score: 0.95 - (i as f32 * 0.1), // Decreasing similarity scores
                chunk_index: i as i32,
                embedding: Some(vec![0.1; query_embedding.len()]), // Mock embedding
                metadata: serde_json::json!({
                    "source": "mock",
                    "chunk_type": "text"
                }),
                created_at: chrono::Utc::now(),
            })
            .collect();
        
        Ok(mock_results)
    }

    async fn get_chunk_by_id(&self, chunk_id: &Uuid) -> Result<Option<RetrievedChunk>> {
        tracing::info!("Mock: Getting chunk by ID: {}", chunk_id);
        
        // Return a mock chunk for testing
        Ok(Some(RetrievedChunk {
            id: *chunk_id,
            chunk_id: *chunk_id,
            document_id: Uuid::new_v4(),
            document_title: "Mock Document".to_string(),
            content: "Mock chunk content for testing similarity search functionality.".to_string(),
            similarity_score: 1.0,
            chunk_index: 0,
            embedding: Some(vec![0.1; 384]), // Standard embedding dimension
            metadata: serde_json::json!({
                "source": "mock",
                "chunk_type": "text"
            }),
            created_at: chrono::Utc::now(),
        }))
    }

    async fn get_collection_info(&self) -> Result<CollectionInfo> {
        Ok(CollectionInfo {
            name: "mock_collection".to_string(),
            status: "Green".to_string(),
            vectors_count: Some(100),
            points_count: 100,
            vectors_config: Some(VectorConfig {
                size: 384,
                distance: "Cosine".to_string(),
            }),
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
    pub vectors_count: Option<u64>,
    pub points_count: u64,
    pub vectors_config: Option<VectorConfig>,
}

/// Vector configuration information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorConfig {
    pub size: u64,
    pub distance: String,
}

/// Payload field types for indexing
#[derive(Debug, Clone, Copy)]
pub enum PayloadFieldType {
    Keyword,
    Integer,
    Float,
    Bool,
}

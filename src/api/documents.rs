use axum::{
    extract::{Path, Query as AxumQuery, Multipart},
    response::Json,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use tokio::fs;
use tracing::{info, warn, error};
use uuid::Uuid;

use crate::api::types::{
    ApiResponse, ApiResult, DocumentUploadResponse, DocumentStatus,
    PaginatedResponse, PaginationParams, validation_error, internal_error,
};
use crate::core::{
    DocumentProcessor, DocumentFormatProcessor, ChunkingStrategy,
};
use crate::storage::file_storage::{FileStorage, SimpleFileStorage};

// Request/Response types
#[derive(Debug, Serialize, Deserialize)]
pub struct CreateDocumentRequest {
    pub title: String,
    pub content: String,
    pub metadata: Option<HashMap<String, Value>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UpdateDocumentRequest {
    pub title: Option<String>,
    pub metadata: Option<HashMap<String, Value>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DocumentResponse {
    pub id: Uuid,
    pub title: String,
    pub filename: Option<String>,
    pub content_type: String,
    pub size_bytes: u64,
    pub status: DocumentStatus,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub metadata: HashMap<String, Value>,
    pub chunk_count: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DocumentChunkResponse {
    pub id: Uuid,
    pub document_id: Uuid,
    pub content: String,
    pub chunk_index: u32,
    pub start_char: u64,
    pub end_char: u64,
    pub embedding_id: Option<String>,
    pub metadata: HashMap<String, Value>,
}

#[derive(Debug, Deserialize)]
pub struct SearchDocumentsQuery {
    pub q: Option<String>,
    pub content_type: Option<String>,
    pub status: Option<DocumentStatus>,
    #[serde(flatten)]
    pub pagination: PaginationParams,
}

#[derive(Debug, Deserialize)]
pub struct BatchDeleteRequest {
    pub document_ids: Vec<Uuid>,
}

// ==== Document CRUD Operations ====

/// Upload a document (file or JSON content)
pub async fn upload_document(
    mut multipart: Multipart,
) -> ApiResult<Json<ApiResponse<DocumentUploadResponse>>> {
    info!("Document upload request received");
    
    // Configuration constants
    const MAX_FILE_SIZE: usize = 50 * 1024 * 1024; // 50MB
    const ALLOWED_EXTENSIONS: &[&str] = &["pdf", "txt", "md", "docx", "html", "htm", "rtf"];
    
    // Initialize upload state
    let mut filename = None;
    let mut file_data = None;
    let mut title = None;
    let mut metadata: HashMap<String, Value> = HashMap::new();
    let mut chunking_strategy = ChunkingStrategy::Semantic;
    
    // Process multipart fields
    while let Some(field) = multipart.next_field().await
        .map_err(|e| validation_error("multipart", &format!("Failed to read multipart field: {}", e)))? {
        
        let field_name = field.name().unwrap_or("").to_string();
        
        match field_name.as_str() {
            "file" => {
                // Extract filename first
                let field_filename = field.file_name()
                    .ok_or_else(|| validation_error("file", "No filename provided"))?
                    .to_string();
                
                // Validate file extension
                let extension = extract_file_extension(&field_filename)?;
                if !ALLOWED_EXTENSIONS.contains(&extension.as_str()) {
                    return Err(validation_error("file", 
                        &format!("Unsupported file type: .{}. Allowed: {}", 
                               extension, ALLOWED_EXTENSIONS.join(", "))));
                }
                
                // Read file data with size limit
                let data = field.bytes().await
                    .map_err(|e| validation_error("file", &format!("Failed to read file data: {}", e)))?;
                
                if data.len() > MAX_FILE_SIZE {
                    return Err(validation_error("file", 
                        &format!("File too large: {} bytes. Maximum allowed: {} bytes", 
                               data.len(), MAX_FILE_SIZE)));
                }
                
                if data.is_empty() {
                    return Err(validation_error("file", "File is empty"));
                }
                
                filename = Some(field_filename);
                file_data = Some(data.to_vec());
            },
            "title" => {
                let field_data = field.text().await
                    .map_err(|e| validation_error("title", &format!("Failed to read title: {}", e)))?;
                title = Some(field_data);
            },
            "chunking_strategy" => {
                let strategy_str = field.text().await
                    .map_err(|e| validation_error("chunking_strategy", &format!("Failed to read chunking strategy: {}", e)))?;
                chunking_strategy = parse_chunking_strategy(&strategy_str)?;
            },
            "metadata" => {
                let metadata_str = field.text().await
                    .map_err(|e| validation_error("metadata", &format!("Failed to read metadata: {}", e)))?;
                
                if !metadata_str.trim().is_empty() {
                    let parsed_metadata: HashMap<String, Value> = serde_json::from_str(&metadata_str)
                        .map_err(|e| validation_error("metadata", &format!("Invalid JSON metadata: {}", e)))?;
                    metadata.extend(parsed_metadata);
                }
            },
            _ => {
                // Store unknown fields as metadata
                let field_data = field.text().await.unwrap_or_default();
                if !field_data.trim().is_empty() {
                    metadata.insert(field_name, json!(field_data));
                }
            }
        }
    }
    
    // Validate required fields
    let filename = filename.ok_or_else(|| validation_error("file", "No file uploaded"))?;
    let file_data = file_data.ok_or_else(|| validation_error("file", "No file data received"))?;
    
    // Use filename as title if not provided
    let title = title.unwrap_or_else(|| {
        PathBuf::from(&filename)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(&filename)
            .to_string()
    });
    
    info!("Processing upload: filename='{}', size={} bytes, title='{}'", 
          filename, file_data.len(), title);
    
    // Process the document
    match process_uploaded_document(filename, file_data, title, metadata, chunking_strategy).await {
        Ok(upload_response) => {
            info!("Document upload completed successfully: ID={}", upload_response.id);
            Ok(Json(ApiResponse::success_with_message(
                upload_response,
                "Document uploaded successfully and queued for processing".to_string(),
            )))
        },
        Err(e) => {
            error!("Document upload failed: {}", e);
            Err(e)
        }
    }
}

/// Create a document from JSON content
pub async fn create_document(
    Json(request): Json<CreateDocumentRequest>,
) -> ApiResult<Json<ApiResponse<DocumentResponse>>> {
    info!("Creating document with title: {}", request.title);
    
    // Validate request
    if request.title.trim().is_empty() {
        return Err(validation_error("title", "Title cannot be empty"));
    }
    
    if request.content.trim().is_empty() {
        return Err(validation_error("content", "Content cannot be empty"));
    }
    
    // TODO: Implement actual document creation
    let document_id = Uuid::new_v4();
    let now = Utc::now();
    
    let document_response = DocumentResponse {
        id: document_id,
        title: request.title,
        filename: None,
        content_type: "text/plain".to_string(),
        size_bytes: request.content.len() as u64,
        status: DocumentStatus::Pending,
        created_at: now,
        updated_at: now,
        metadata: request.metadata.unwrap_or_default(),
        chunk_count: None,
    };
    
    info!("Document created successfully with ID: {}", document_id);
    Ok(Json(ApiResponse::success(document_response)))
}

/// Get a document by ID
pub async fn get_document(
    Path(document_id): Path<Uuid>,
) -> ApiResult<Json<ApiResponse<DocumentResponse>>> {
    info!("Retrieving document: {}", document_id);
    
    // TODO: Implement actual document retrieval from database
    // For now, return a mock document
    let now = Utc::now();
    let document = DocumentResponse {
        id: document_id,
        title: "Sample Document".to_string(),
        filename: Some("sample.pdf".to_string()),
        content_type: "application/pdf".to_string(),
        size_bytes: 2048,
        status: DocumentStatus::Completed,
        created_at: now,
        updated_at: now,
        metadata: HashMap::new(),
        chunk_count: Some(5),
    };
    
    Ok(Json(ApiResponse::success(document)))
}

/// Update document metadata
pub async fn update_document(
    Path(document_id): Path<Uuid>,
    Json(request): Json<UpdateDocumentRequest>,
) -> ApiResult<Json<ApiResponse<DocumentResponse>>> {
    info!("Updating document: {}", document_id);
    
    // TODO: Implement actual document update
    // For now, return updated mock document
    let now = Utc::now();
    let updated_document = DocumentResponse {
        id: document_id,
        title: request.title.unwrap_or("Updated Document".to_string()),
        filename: Some("updated.pdf".to_string()),
        content_type: "application/pdf".to_string(),
        size_bytes: 2048,
        status: DocumentStatus::Completed,
        created_at: now,
        updated_at: now,
        metadata: request.metadata.unwrap_or_default(),
        chunk_count: Some(5),
    };
    
    Ok(Json(ApiResponse::success_with_message(
        updated_document,
        "Document updated successfully".to_string(),
    )))
}

/// List documents with pagination and filtering
pub async fn list_documents(
    AxumQuery(params): AxumQuery<PaginationParams>,
) -> ApiResult<Json<ApiResponse<PaginatedResponse<DocumentResponse>>>> {
    info!("Listing documents with pagination: page={}, limit={}", params.page, params.limit);
    
    // Validate pagination parameters
    params.validate().map_err(|e| validation_error("pagination", &e))?;
    
    // TODO: Implement actual document listing from database
    // For now, return empty list
    let documents = vec![];
    let total_count = 0;
    
    let paginated_response = PaginatedResponse::new(
        documents,
        params.page,
        params.limit,
        total_count,
    );
    
    Ok(Json(ApiResponse::success(paginated_response)))
}

/// Delete a document
pub async fn delete_document(
    Path(document_id): Path<Uuid>,
) -> ApiResult<Json<ApiResponse<String>>> {
    info!("Deleting document: {}", document_id);
    
    // TODO: Implement actual document deletion
    // Check if document exists first
    
    Ok(Json(ApiResponse::success_with_message(
        "deleted".to_string(),
        format!("Document {} deleted successfully", document_id),
    )))
}

// ==== Document Content Operations ====

/// Get document content
pub async fn get_document_content(
    Path(document_id): Path<Uuid>,
) -> ApiResult<Json<ApiResponse<String>>> {
    info!("Retrieving content for document: {}", document_id);
    
    // TODO: Implement actual content retrieval
    let content = "This is sample document content. In a real implementation, this would be retrieved from storage.";
    
    Ok(Json(ApiResponse::success(content.to_string())))
}

/// Get document metadata
pub async fn get_document_metadata(
    Path(document_id): Path<Uuid>,
) -> ApiResult<Json<ApiResponse<HashMap<String, Value>>>> {
    info!("Retrieving metadata for document: {}", document_id);
    
    // TODO: Implement actual metadata retrieval
    let mut metadata = HashMap::new();
    metadata.insert("author".to_string(), json!("John Doe"));
    metadata.insert("created_date".to_string(), json!("2024-01-01"));
    metadata.insert("tags".to_string(), json!(["sample", "document"]));
    
    Ok(Json(ApiResponse::success(metadata)))
}

/// Update document metadata only
pub async fn update_document_metadata(
    Path(document_id): Path<Uuid>,
    Json(metadata): Json<HashMap<String, Value>>,
) -> ApiResult<Json<ApiResponse<HashMap<String, Value>>>> {
    info!("Updating metadata for document: {}", document_id);
    
    // TODO: Implement actual metadata update
    
    Ok(Json(ApiResponse::success_with_message(
        metadata,
        "Document metadata updated successfully".to_string(),
    )))
}

// ==== Document Chunks Operations ====

/// List document chunks
pub async fn list_document_chunks(
    Path(document_id): Path<Uuid>,
    AxumQuery(params): AxumQuery<PaginationParams>,
) -> ApiResult<Json<ApiResponse<PaginatedResponse<DocumentChunkResponse>>>> {
    info!("Listing chunks for document: {}", document_id);
    
    params.validate().map_err(|e| validation_error("pagination", &e))?;
    
    // TODO: Implement actual chunk retrieval
    let chunks = vec![];
    let total_count = 0;
    
    let paginated_response = PaginatedResponse::new(
        chunks,
        params.page,
        params.limit,
        total_count,
    );
    
    Ok(Json(ApiResponse::success(paginated_response)))
}

// ==== Document Processing Operations ====

/// Reprocess a document (re-chunk and re-embed)
pub async fn reprocess_document(
    Path(document_id): Path<Uuid>,
) -> ApiResult<Json<ApiResponse<String>>> {
    info!("Reprocessing document: {}", document_id);
    
    // TODO: Implement actual document reprocessing
    // This should trigger the document processing pipeline
    
    Ok(Json(ApiResponse::success_with_message(
        "processing".to_string(),
        format!("Document {} queued for reprocessing", document_id),
    )))
}

// ==== Batch Operations ====

/// Upload multiple documents
pub async fn batch_upload_documents(
    mut multipart: Multipart,
) -> ApiResult<Json<ApiResponse<Vec<DocumentUploadResponse>>>> {
    info!("Batch document upload requested");
    
    // Configuration constants
    const MAX_FILES: usize = 10;
    const MAX_FILE_SIZE: usize = 50 * 1024 * 1024; // 50MB per file
    const ALLOWED_EXTENSIONS: &[&str] = &["pdf", "txt", "md", "docx", "html", "htm", "rtf"];
    
    // Collect files and metadata
    let mut files_to_process = Vec::new();
    let mut global_metadata: HashMap<String, Value> = HashMap::new();
    let mut chunking_strategy = ChunkingStrategy::Semantic;
    
    // Process multipart fields
    while let Some(field) = multipart.next_field().await
        .map_err(|e| validation_error("multipart", &format!("Failed to read multipart field: {}", e)))? {
        
        let field_name = field.name().unwrap_or("").to_string();
        
        match field_name.as_str() {
            "files" => {
                // Multiple files with the same field name
                let field_filename = field.file_name()
                    .ok_or_else(|| validation_error("files", "No filename provided for one of the files"))?
                    .to_string();
                
                // Validate file extension
                let extension = extract_file_extension(&field_filename)?;
                if !ALLOWED_EXTENSIONS.contains(&extension.as_str()) {
                    return Err(validation_error("files", 
                        &format!("Unsupported file type in batch: .{}. Allowed: {}", 
                               extension, ALLOWED_EXTENSIONS.join(", "))));
                }
                
                // Read file data with size limit
                let data = field.bytes().await
                    .map_err(|e| validation_error("files", &format!("Failed to read file data for {}: {}", field_filename, e)))?;
                
                if data.len() > MAX_FILE_SIZE {
                    return Err(validation_error("files", 
                        &format!("File too large in batch: {} ({} bytes). Maximum allowed: {} bytes", 
                               field_filename, data.len(), MAX_FILE_SIZE)));
                }
                
                if data.is_empty() {
                    return Err(validation_error("files", &format!("File is empty in batch: {}", field_filename)));
                }
                
                files_to_process.push((field_filename, data.to_vec()));
                
                if files_to_process.len() > MAX_FILES {
                    return Err(validation_error("files", 
                        &format!("Too many files in batch: {}. Maximum allowed: {}", 
                               files_to_process.len(), MAX_FILES)));
                }
            },
            "chunking_strategy" => {
                let strategy_str = field.text().await
                    .map_err(|e| validation_error("chunking_strategy", &format!("Failed to read chunking strategy: {}", e)))?;
                chunking_strategy = parse_chunking_strategy(&strategy_str)?;
            },
            "metadata" => {
                let metadata_str = field.text().await
                    .map_err(|e| validation_error("metadata", &format!("Failed to read metadata: {}", e)))?;
                
                if !metadata_str.trim().is_empty() {
                    let parsed_metadata: HashMap<String, Value> = serde_json::from_str(&metadata_str)
                        .map_err(|e| validation_error("metadata", &format!("Invalid JSON metadata: {}", e)))?;
                    global_metadata.extend(parsed_metadata);
                }
            },
            _ => {
                // Store unknown fields as global metadata
                let field_data = field.text().await.unwrap_or_default();
                if !field_data.trim().is_empty() {
                    global_metadata.insert(field_name, json!(field_data));
                }
            }
        }
    }
    
    // Validate that we have files to process
    if files_to_process.is_empty() {
        return Err(validation_error("files", "No files provided for batch upload"));
    }
    
    info!("Processing {} files in batch upload", files_to_process.len());
    
    // Process each file
    let mut upload_responses = Vec::new();
    let mut successful_uploads = 0;
    let mut failed_uploads = 0;
    let total_files = files_to_process.len();
    
    for (filename, file_data) in &files_to_process {
        // Generate title from filename
        let title = PathBuf::from(filename)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(filename)
            .to_string();
        
        // Clone metadata for this file
        let mut file_metadata = global_metadata.clone();
        file_metadata.insert("batch_upload".to_string(), json!(true));
        file_metadata.insert("original_batch_size".to_string(), json!(total_files));
        
        // Process the document
        match process_uploaded_document(filename.clone(), file_data.clone(), title, file_metadata, chunking_strategy).await {
            Ok(upload_response) => {
                info!("Batch upload successful for file: {} (ID: {})", filename, upload_response.id);
                upload_responses.push(upload_response);
                successful_uploads += 1;
            },
            Err(e) => {
                error!("Batch upload failed for file: {} - {}", filename, e);
                failed_uploads += 1;
                
                // Create error response for this file
                let error_response = DocumentUploadResponse {
                    id: Uuid::new_v4(),
                    filename: filename.clone(),
                    size_bytes: 0,
                    mime_type: "application/octet-stream".to_string(),
                    status: DocumentStatus::Failed,
                    created_at: Utc::now(),
                };
                upload_responses.push(error_response);
            }
        }
    }
    
    let message = if failed_uploads == 0 {
        format!("Batch upload completed successfully: {} files processed", successful_uploads)
    } else {
        format!("Batch upload completed with {} successes and {} failures", successful_uploads, failed_uploads)
    };
    
    info!("Batch upload completed: {} successful, {} failed", successful_uploads, failed_uploads);
    
    Ok(Json(ApiResponse::success_with_message(
        upload_responses,
        message,
    )))
}

/// Delete multiple documents
pub async fn batch_delete_documents(
    Json(request): Json<BatchDeleteRequest>,
) -> ApiResult<Json<ApiResponse<String>>> {
    info!("Batch deleting {} documents", request.document_ids.len());
    
    if request.document_ids.is_empty() {
        return Err(validation_error("document_ids", "At least one document ID is required"));
    }
    
    if request.document_ids.len() > 100 {
        return Err(validation_error("document_ids", "Cannot delete more than 100 documents at once"));
    }
    
    // TODO: Implement actual batch deletion
    
    Ok(Json(ApiResponse::success_with_message(
        "deleted".to_string(),
        format!("{} documents deleted successfully", request.document_ids.len()),
    )))
}

// ==== Search Operations ====

/// Search documents
pub async fn search_documents(
    AxumQuery(query): AxumQuery<SearchDocumentsQuery>,
) -> ApiResult<Json<ApiResponse<PaginatedResponse<DocumentResponse>>>> {
    info!("Searching documents with query: {:?}", query.q);
    
    query.pagination.validate().map_err(|e| validation_error("pagination", &e))?;
    
    // TODO: Implement actual document search
    let documents = vec![];
    let total_count = 0;
    
    let paginated_response = PaginatedResponse::new(
        documents,
        query.pagination.page,
        query.pagination.limit,
        total_count,
    );
    
    Ok(Json(ApiResponse::success(paginated_response)))
}

// ==== Helper Functions ====

/// Extract file extension from filename
fn extract_file_extension(filename: &str) -> ApiResult<String> {
    let extension = PathBuf::from(filename)
        .extension()
        .and_then(|ext| ext.to_str())
        .ok_or_else(|| validation_error("file", "No file extension found"))?
        .to_lowercase();
    
    Ok(extension)
}

/// Parse chunking strategy from string
fn parse_chunking_strategy(strategy_str: &str) -> ApiResult<ChunkingStrategy> {
    match strategy_str.to_lowercase().as_str() {
        "fixed_size" | "fixed" => Ok(ChunkingStrategy::FixedSize),
        "semantic" => Ok(ChunkingStrategy::Semantic),
        "sentence" => Ok(ChunkingStrategy::Sentence),
        "paragraph" => Ok(ChunkingStrategy::Paragraph),
        _ => Err(validation_error("chunking_strategy", 
            "Invalid chunking strategy. Allowed: fixed_size, semantic, sentence, paragraph"))
    }
}

/// Generate a safe filename for storage
fn generate_safe_filename(original_filename: &str, document_id: &Uuid) -> String {
    let path = PathBuf::from(original_filename);
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("bin");
    
    format!("{}.{}", document_id, extension)
}

/// Detect MIME type from file extension
fn detect_mime_type(filename: &str) -> String {
    let extension = PathBuf::from(filename)
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_lowercase();
    
    match extension.as_str() {
        "pdf" => "application/pdf".to_string(),
        "txt" => "text/plain".to_string(),
        "md" | "markdown" => "text/markdown".to_string(),
        "docx" => "application/vnd.openxmlformats-officedocument.wordprocessingml.document".to_string(),
        "html" | "htm" => "text/html".to_string(),
        "rtf" => "application/rtf".to_string(),
        _ => "application/octet-stream".to_string(),
    }
}

/// Process uploaded document through the document processing pipeline
async fn process_uploaded_document(
    filename: String,
    file_data: Vec<u8>,
    _title: String,
    _metadata: HashMap<String, Value>,
    _chunking_strategy: ChunkingStrategy,
) -> ApiResult<DocumentUploadResponse> {
    info!("Starting document processing pipeline for: {}", filename);
    
    // Generate unique document ID
    let document_id = Uuid::new_v4();
    
    // Create safe filename for storage
    let safe_filename = generate_safe_filename(&filename, &document_id);
    
    // Initialize storage and processors
    let file_storage = SimpleFileStorage;
    let document_processor = DocumentProcessor::new();
    
    // Save file to storage
    let file_path = file_storage.save_file(&file_data, &safe_filename)
        .await
        .map_err(|e| internal_error(&format!("Failed to save file: {}", e)))?;
    
    info!("File saved to storage: {}", file_path);
    
    // Process document content and extract chunks
    match DocumentFormatProcessor::process_document(&file_path, &file_data, &document_processor).await {
        Ok((_document, chunks)) => {
            info!("Document processed successfully: {} chunks extracted", chunks.len());
            
            // TODO: Save document and chunks to database
            // TODO: Queue for embedding generation
            
            // Create upload response
            let upload_response = DocumentUploadResponse {
                id: document_id,
                filename: filename.clone(),
                size_bytes: file_data.len() as u64,
                mime_type: detect_mime_type(&filename),
                status: DocumentStatus::Processing,
                created_at: Utc::now(),
            };
            
            Ok(upload_response)
        },
        Err(e) => {
            error!("Failed to process document {}: {}", filename, e);
            
            // Clean up saved file on processing error
            if let Err(cleanup_err) = file_storage.delete_file(&file_path).await {
                warn!("Failed to cleanup file after processing error: {}", cleanup_err);
            }
            
            Err(internal_error(&format!("Document processing failed: {}", e)))
        }
    }
}

/// Initialize uploads directory if it doesn't exist
pub async fn ensure_uploads_directory() -> ApiResult<()> {
    let uploads_dir = "./uploads";
    
    if !PathBuf::from(uploads_dir).exists() {
        fs::create_dir_all(uploads_dir)
            .await
            .map_err(|e| internal_error(&format!("Failed to create uploads directory: {}", e)))?;
        
        info!("Created uploads directory: {}", uploads_dir);
    }
    
    Ok(())
}

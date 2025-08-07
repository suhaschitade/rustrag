use crate::models::{Document, DocumentChunk, DocumentMetadata};
use crate::utils::Result;
use crate::core::{ContentValidator, ValidationConfig};
use regex::Regex;
use std::path::Path;

/// Different strategies for chunking text documents
#[derive(Debug, Clone, Copy)]
pub enum ChunkingStrategy {
    /// Fixed-size chunking with overlap
    FixedSize,
    /// Semantic chunking based on document structure
    Semantic,
    /// Sentence-based chunking
    Sentence,
    /// Paragraph-based chunking
    Paragraph,
}

pub struct DocumentProcessor {
    chunk_size: usize,
    chunk_overlap: usize,
}

impl DocumentProcessor {
    pub fn new() -> Self {
        Self {
            chunk_size: 1000,
            chunk_overlap: 200,
        }
    }

    pub fn new_with_config(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            chunk_size,
            chunk_overlap,
        }
    }

    /// Chunk text with a specific strategy
    pub fn chunk_text_with_strategy(
        &self,
        document_id: &uuid::Uuid,
        text: &str,
        strategy: ChunkingStrategy,
    ) -> Result<Vec<DocumentChunk>> {
        match strategy {
            ChunkingStrategy::FixedSize => self.chunk_text_fixed_size(document_id, text),
            ChunkingStrategy::Semantic => self.chunk_text_semantic(document_id, text),
            ChunkingStrategy::Sentence => self.chunk_text_by_sentences(document_id, text),
            ChunkingStrategy::Paragraph => self.chunk_text_by_paragraphs(document_id, text),
        }
    }

    /// Fixed-size chunking with overlap
    pub fn chunk_text_fixed_size(&self, document_id: &uuid::Uuid, text: &str) -> Result<Vec<DocumentChunk>> {
        let mut chunks = Vec::new();
        let mut start = 0;
        let mut chunk_index = 0;

        while start < text.len() {
            let end = std::cmp::min(start + self.chunk_size, text.len());
            let chunk_text = text[start..end].to_string();

            let chunk = DocumentChunk::new(*document_id, chunk_index, chunk_text);
            chunks.push(chunk);

            // Move start position with overlap
            start += self.chunk_size - self.chunk_overlap;
            chunk_index += 1;

            // Avoid infinite loop if chunk_overlap >= chunk_size
            if self.chunk_overlap >= self.chunk_size {
                break;
            }
        }

        Ok(chunks)
    }

    /// Semantic chunking based on document structure
    pub fn chunk_text_semantic(&self, document_id: &uuid::Uuid, text: &str) -> Result<Vec<DocumentChunk>> {
        let section_regex = Regex::new(r"(?m)^(#{1,6}\s+.+|\d+\.\s+.+|[A-Z][A-Z\s]+:)")
            .map_err(|e| crate::utils::Error::document_processing(format!("Regex error: {}", e)))?;
        let mut chunks = Vec::new();
        let mut chunk_index = 0;
        let mut last_end = 0;
        let mut current_section = String::new();
        
        for mat in section_regex.find_iter(text) {
            if !current_section.trim().is_empty() {
                let chunk = DocumentChunk::new(*document_id, chunk_index, current_section.trim().to_string());
                chunks.push(chunk);
                chunk_index += 1;
            }
            current_section = text[last_end..mat.start()].to_string();
            current_section.push_str(&text[mat.start()..mat.end()]);
            last_end = mat.end();
        }
        
        if !current_section.trim().is_empty() {
            let chunk = DocumentChunk::new(*document_id, chunk_index, current_section.trim().to_string());
            chunks.push(chunk);
        }
        
        // Fallback to paragraph chunking if no sections found
        if chunks.is_empty() {
            return self.chunk_text_by_paragraphs(document_id, text);
        }
        
        Ok(chunks)
    }

    /// Sentence-based chunking
    pub fn chunk_text_by_sentences(&self, document_id: &uuid::Uuid, text: &str) -> Result<Vec<DocumentChunk>> {
        let mut chunks = Vec::new();
        let mut chunk_index = 0;
        let mut current_chunk = String::new();
        let mut current_size = 0;

        // Simple sentence boundary detection
        let sentence_regex = Regex::new(r"[.!?]+\s+")
            .map_err(|e| crate::utils::Error::document_processing(format!("Regex error: {}", e)))?;

        let mut last_end = 0;
        for mat in sentence_regex.find_iter(text) {
            let sentence = &text[last_end..mat.end()];
            
            // Check if adding this sentence would exceed chunk size
            if current_size + sentence.len() > self.chunk_size && !current_chunk.is_empty() {
                let chunk = DocumentChunk::new(*document_id, chunk_index, current_chunk.trim().to_string());
                chunks.push(chunk);
                chunk_index += 1;
                current_chunk.clear();
                current_size = 0;
            }

            current_chunk.push_str(sentence);
            current_size += sentence.len();
            last_end = mat.end();
        }

        // Add remaining text as final chunk
        if last_end < text.len() {
            current_chunk.push_str(&text[last_end..]);
        }

        if !current_chunk.trim().is_empty() {
            let chunk = DocumentChunk::new(*document_id, chunk_index, current_chunk.trim().to_string());
            chunks.push(chunk);
        }

        // Fallback to fixed-size if no sentences found
        if chunks.is_empty() {
            return self.chunk_text_fixed_size(document_id, text);
        }

        Ok(chunks)
    }

    /// Paragraph-based chunking
    pub fn chunk_text_by_paragraphs(&self, document_id: &uuid::Uuid, text: &str) -> Result<Vec<DocumentChunk>> {
        let mut chunks = Vec::new();
        let mut chunk_index = 0;
        let mut current_chunk = String::new();
        let mut current_size = 0;

        for paragraph in text.split("\n\n") {
            let paragraph = paragraph.trim();
            if paragraph.is_empty() {
                continue;
            }

            // Check if adding this paragraph would exceed chunk size
            if current_size + paragraph.len() > self.chunk_size && !current_chunk.is_empty() {
                let chunk = DocumentChunk::new(*document_id, chunk_index, current_chunk.trim().to_string());
                chunks.push(chunk);
                chunk_index += 1;
                current_chunk.clear();
                current_size = 0;
            }

            if !current_chunk.is_empty() {
                current_chunk.push_str("\n\n");
                current_size += 2;
            }
            current_chunk.push_str(paragraph);
            current_size += paragraph.len();
        }

        // Add final chunk
        if !current_chunk.trim().is_empty() {
            let chunk = DocumentChunk::new(*document_id, chunk_index, current_chunk.trim().to_string());
            chunks.push(chunk);
        }

        // Fallback to fixed-size if no paragraphs found
        if chunks.is_empty() {
            return self.chunk_text_fixed_size(document_id, text);
        }

        Ok(chunks)
    }

    /// Process a document file and extract both content and metadata
    pub async fn process_document_file(
        &self,
        file_path: &Path,
        title: Option<String>,
        strategy: ChunkingStrategy,
    ) -> Result<(Document, Vec<DocumentChunk>)> {
        // TODO: Extract metadata from file when MetadataExtractor is implemented
        let metadata = DocumentMetadata::default();
        
        // Read file content (this would normally use DocumentFormatProcessor)
        let content = std::fs::read_to_string(file_path)
            .map_err(|e| crate::utils::Error::document_processing(
                format!("Failed to read file: {}", e)
            ))?;

        // Extract additional metadata from content
        let file_extension = file_path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|s| s.to_lowercase());
        
        // TODO: Extract content metadata when MetadataExtractor is implemented
        let content_metadata = DocumentMetadata::default();

        // Merge metadata (file metadata takes precedence)
        let merged_metadata = self.merge_metadata(metadata, content_metadata);

        // Determine document title
        let doc_title = title
            .or_else(|| {
                // Try to get title from metadata
                if let Ok(custom_map) = serde_json::from_value::<serde_json::Map<String, serde_json::Value>>(merged_metadata.custom_fields.clone()) {
                    custom_map.get("title")
                        .or_else(|| custom_map.get("pdf_title"))
                        .or_else(|| custom_map.get("html_title"))
                        .or_else(|| custom_map.get("inferred_title"))
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                } else {
                    None
                }
            })
            .or_else(|| {
                // Fallback to filename
                file_path.file_stem()
                    .and_then(|stem| stem.to_str())
                    .map(|s| s.to_string())
            })
            .unwrap_or_else(|| "Untitled Document".to_string());

        // Create document with metadata
        let mut document = Document::new_with_metadata(doc_title, content.clone(), merged_metadata);
        
        // Set file-related fields
        document.file_path = Some(file_path.to_string_lossy().to_string());
        if let Some(ext) = file_extension {
            document.file_type = Some(ext);
        }
        if let Ok(file_metadata) = std::fs::metadata(file_path) {
            document.file_size = Some(file_metadata.len() as i64);
        }

        // Generate chunks
        let chunks = self.chunk_text_with_strategy(&document.id, &content, strategy)?;

        Ok((document, chunks))
    }

    /// Process document content with metadata extraction
    pub fn process_document_content(
        &self,
        title: String,
        content: String,
        file_type: Option<&str>,
        strategy: ChunkingStrategy,
    ) -> Result<(Document, Vec<DocumentChunk>)> {
        // TODO: Extract metadata from content when MetadataExtractor is implemented
        let metadata = DocumentMetadata::default();

        // Create document with metadata
        let document = Document::new_with_metadata(title, content.clone(), metadata);

        // Generate chunks
        let chunks = self.chunk_text_with_strategy(&document.id, &content, strategy)?;

        Ok((document, chunks))
    }

    /// Merge two DocumentMetadata structs, with priority given to the first one
    fn merge_metadata(&self, primary: DocumentMetadata, secondary: DocumentMetadata) -> DocumentMetadata {
        DocumentMetadata {
            author: primary.author.or(secondary.author),
            created_date: primary.created_date.or(secondary.created_date),
            modified_date: primary.modified_date.or(secondary.modified_date),
            language: primary.language.or(secondary.language),
            tags: if primary.tags.is_empty() { secondary.tags } else { primary.tags },
            category: primary.category.or(secondary.category),
            word_count: primary.word_count.or(secondary.word_count),
            page_count: primary.page_count.or(secondary.page_count),
            file_type: primary.file_type.or(secondary.file_type),
            file_size: primary.file_size.or(secondary.file_size),
            custom_fields: self.merge_custom_fields(primary.custom_fields, secondary.custom_fields),
        }
    }

    /// Merge custom fields JSON objects
    fn merge_custom_fields(
        &self,
        primary: serde_json::Value,
        secondary: serde_json::Value,
    ) -> serde_json::Value {
        match (primary, secondary) {
            (serde_json::Value::Object(mut primary_map), serde_json::Value::Object(secondary_map)) => {
                // Add fields from secondary that don't exist in primary
                for (key, value) in secondary_map {
                    primary_map.entry(key).or_insert(value);
                }
                serde_json::Value::Object(primary_map)
            }
            (primary, _) if !primary.is_null() => primary,
            (_, secondary) => secondary,
        }
    }

    /// Process document with validation and sanitization
    pub async fn process_document_with_validation(
        &self,
        content: String,
        title: String,
        file_type: Option<&str>,
        strategy: ChunkingStrategy,
        validation_config: Option<ValidationConfig>,
    ) -> Result<(Document, Vec<DocumentChunk>, crate::core::ValidationResult)> {
        // Create content validator
        let validator = if let Some(config) = validation_config {
            ContentValidator::with_config(config)?
        } else {
            ContentValidator::new()?
        };

        // Validate and sanitize content
        let validation_result = validator.validate_and_sanitize(&content, file_type)?;

        // Use sanitized content if validation passed
        let processed_content = if let Some(sanitized) = &validation_result.sanitized_content {
            sanitized.clone()
        } else {
            // If validation failed completely, we still might want to process with original content
            // This depends on business requirements - here we allow it with warnings
            content.clone()
        };

        // Process the content normally
        let (document, chunks) = self.process_document_content(
            title,
            processed_content,
            file_type,
            strategy,
        )?;

        Ok((document, chunks, validation_result))
    }

    /// Process file with validation and sanitization
    pub async fn process_file_with_validation(
        &self,
        file_path: &Path,
        title: Option<String>,
        strategy: ChunkingStrategy,
        validation_config: Option<ValidationConfig>,
    ) -> Result<(Document, Vec<DocumentChunk>, crate::core::ValidationResult)> {
        // Read file content first
        let content = std::fs::read_to_string(file_path)
            .map_err(|e| crate::utils::Error::document_processing(
                format!("Failed to read file: {}", e)
            ))?;

        // Get file type
        let file_type = file_path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|s| s.to_lowercase());

        // Create content validator
        let validator = if let Some(config) = validation_config {
            ContentValidator::with_config(config)?
        } else {
            ContentValidator::new()?
        };

        // Validate and sanitize content
        let validation_result = validator.validate_and_sanitize(&content, file_type.as_deref())?;

        // Use sanitized content if available
        let processed_content = if let Some(sanitized) = &validation_result.sanitized_content {
            sanitized.clone()
        } else {
            content.clone()
        };

        // TODO: Extract metadata from file when MetadataExtractor is implemented
        let metadata = DocumentMetadata::default();
        
        // TODO: Extract content metadata when MetadataExtractor is implemented
        let content_metadata = DocumentMetadata::default();

        // Merge metadata
        let merged_metadata = self.merge_metadata(metadata, content_metadata);

        // Determine document title
        let doc_title = title
            .or_else(|| {
                // Try to get title from metadata
                if let Ok(custom_map) = serde_json::from_value::<serde_json::Map<String, serde_json::Value>>(merged_metadata.custom_fields.clone()) {
                    custom_map.get("title")
                        .or_else(|| custom_map.get("pdf_title"))
                        .or_else(|| custom_map.get("html_title"))
                        .or_else(|| custom_map.get("inferred_title"))
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                } else {
                    None
                }
            })
            .or_else(|| {
                // Fallback to filename
                file_path.file_stem()
                    .and_then(|stem| stem.to_str())
                    .map(|s| s.to_string())
            })
            .unwrap_or_else(|| "Untitled Document".to_string());

        // Create document with metadata
        let mut document = Document::new_with_metadata(doc_title, processed_content.clone(), merged_metadata);
        
        // Set file-related fields
        document.file_path = Some(file_path.to_string_lossy().to_string());
        if let Some(ext) = file_type {
            document.file_type = Some(ext);
        }
        if let Ok(file_metadata) = std::fs::metadata(file_path) {
            document.file_size = Some(file_metadata.len() as i64);
        }

        // Generate chunks from processed content
        let chunks = self.chunk_text_with_strategy(&document.id, &processed_content, strategy)?;

        Ok((document, chunks, validation_result))
    }

    /// Quick validation check for content before processing
    pub fn quick_validate_content(&self, content: &str, file_type: Option<&str>) -> Result<bool> {
        let validator = ContentValidator::new()?;
        validator.quick_validate(content, file_type)
    }

    /// Get validation statistics for a batch of documents
    pub fn batch_validate_documents(&self, documents: &[(String, Option<String>)]) -> Result<crate::core::content_validator::BatchValidationStats> {
        let validator = ContentValidator::new()?;
        let contents: Vec<(&str, Option<&str>)> = documents
            .iter()
            .map(|(content, file_type)| (content.as_str(), file_type.as_deref()))
            .collect();
        
        validator.batch_validate_stats(&contents)
    }
}

use crate::models::{Document, DocumentChunk, DocumentMetadata};
use crate::utils::{Error, Result};

/// Document processing service for parsing and chunking documents
pub struct DocumentProcessor {
    chunk_size: usize,
    chunk_overlap: usize,
}

impl DocumentProcessor {
    /// Create a new document processor with default settings
    pub fn new() -> Self {
        Self {
            chunk_size: 1000,
            chunk_overlap: 200,
        }
    }

    /// Create a new document processor with custom settings
    pub fn new_with_config(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            chunk_size,
            chunk_overlap,
        }
    }

    /// Process a text document into chunks
    pub async fn process_text_document(
        &self,
        title: String,
        content: String,
        metadata: Option<DocumentMetadata>,
    ) -> Result<(Document, Vec<DocumentChunk>)> {
        let document = match metadata {
            Some(meta) => Document::new_with_metadata(title, content.clone(), meta),
            None => Document::new(title, content.clone()),
        };

        let chunks = self.chunk_text(&document.id, &content)?;

        Ok((document, chunks))
    }

    /// Process a PDF document (placeholder implementation)
    pub async fn process_pdf_document(
        &self,
        file_path: &str,
        _data: &[u8],
    ) -> Result<(Document, Vec<DocumentChunk>)> {
        // TODO: Implement PDF processing using pdf-extract or lopdf
        let title = std::path::Path::new(file_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("Unknown")
            .to_string();

        let content = "PDF processing not yet implemented".to_string();
        let document = Document::new(title, content.clone());
        let chunks = self.chunk_text(&document.id, &content)?;

        Ok((document, chunks))
    }

    /// Chunk text into smaller pieces
    fn chunk_text(&self, document_id: &uuid::Uuid, text: &str) -> Result<Vec<DocumentChunk>> {
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

    /// Extract metadata from document content (basic implementation)
    pub fn extract_metadata(&self, content: &str) -> DocumentMetadata {
        let word_count = content.split_whitespace().count() as u32;

        DocumentMetadata {
            word_count: Some(word_count),
            language: Some("en".to_string()), // Default to English
            ..Default::default()
        }
    }
}

impl Default for DocumentProcessor {
    fn default() -> Self {
        Self::new()
    }
}

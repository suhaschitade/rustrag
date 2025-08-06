use crate::models::DocumentChunk;
use crate::utils::Result;
use regex::Regex;

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
}
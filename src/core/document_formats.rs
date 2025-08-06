use crate::models::{Document, DocumentChunk, DocumentMetadata};
use crate::utils::{Error, Result};
use std::path::Path;

/// Document format processor for various file types
pub struct DocumentFormatProcessor;

impl DocumentFormatProcessor {
    /// Process any document based on file extension
    pub async fn process_document(
        file_path: &str,
        data: &[u8],
        processor: &crate::core::DocumentProcessor,
    ) -> Result<(Document, Vec<DocumentChunk>)> {
        let path = Path::new(file_path);
        let extension = path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_lowercase();

        match extension.as_str() {
            "pdf" => Self::process_pdf_document(file_path, data, processor).await,
            "txt" | "text" => Self::process_plain_text_document(file_path, data, processor).await,
            "md" | "markdown" => Self::process_markdown_document(file_path, data, processor).await,
            "docx" => Self::process_docx_document(file_path, data, processor).await,
            "html" | "htm" => Self::process_html_document(file_path, data, processor).await,
            "rtf" => Self::process_rtf_document(file_path, data, processor).await,
            _ => {
                // Try to detect if it's a text-based document
                if Self::is_likely_text(data) {
                    Self::process_plain_text_document(file_path, data, processor).await
                } else {
                    Err(Error::document_processing(
                        format!("Unsupported document type for file: {}", file_path)
                    ))
                }
            }
        }
    }

    /// Process PDF documents using lopdf and pdf-extract as fallback
    pub async fn process_pdf_document(
        file_path: &str,
        data: &[u8],
        processor: &crate::core::DocumentProcessor,
    ) -> Result<(Document, Vec<DocumentChunk>)> {
        let title = Path::new(file_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("Unknown")
            .to_string();

        // Try lopdf first, then pdf-extract as fallback
        let content = Self::extract_pdf_text_lopdf(data)
            .or_else(|_| Self::extract_pdf_text_fallback(data))
            .map_err(|e| Error::document_processing(format!("Failed to extract PDF text: {}", e)))?;

        // Extract metadata from PDF
        let mut metadata = Self::extract_basic_metadata(&content);
        metadata.file_type = Some("pdf".to_string());
        metadata.file_size = Some(data.len() as u64);

        let document = Document::new_with_metadata(title, content.clone(), metadata);
        let chunks = processor.chunk_text_semantic(&document.id, &content)?;

        Ok((document, chunks))
    }

    /// Process plain text documents (TXT)
    pub async fn process_plain_text_document(
        file_path: &str,
        data: &[u8],
        processor: &crate::core::DocumentProcessor,
    ) -> Result<(Document, Vec<DocumentChunk>)> {
        let title = Path::new(file_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("Unknown")
            .to_string();

        // Detect encoding and convert to UTF-8
        let content = Self::decode_text_with_detection(data)?;
        
        // Extract metadata
        let mut metadata = Self::extract_basic_metadata(&content);
        metadata.file_type = Some("txt".to_string());
        metadata.file_size = Some(data.len() as u64);

        let document = Document::new_with_metadata(title, content.clone(), metadata);
        let chunks = processor.chunk_text_semantic(&document.id, &content)?;

        Ok((document, chunks))
    }

    /// Process Markdown documents (MD)
    pub async fn process_markdown_document(
        file_path: &str,
        data: &[u8],
        processor: &crate::core::DocumentProcessor,
    ) -> Result<(Document, Vec<DocumentChunk>)> {
        let title = Path::new(file_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("Unknown")
            .to_string();

        // Decode text content
        let markdown_content = Self::decode_text_with_detection(data)?;
        
        // Convert Markdown to plain text for processing
        let content = Self::markdown_to_text(&markdown_content)?;
        
        // Extract metadata including markdown-specific info
        let mut metadata = Self::extract_basic_metadata(&content);
        metadata.file_type = Some("markdown".to_string());
        metadata.file_size = Some(data.len() as u64);
        
        // Try to extract title from markdown headers
        if let Some(md_title) = Self::extract_markdown_title(&markdown_content) {
            metadata.custom_fields = serde_json::json!({ "original_title": md_title });
        }

        let document = Document::new_with_metadata(title, content.clone(), metadata);
        let chunks = processor.chunk_text_semantic(&document.id, &content)?;

        Ok((document, chunks))
    }

    /// Process DOCX documents
    pub async fn process_docx_document(
        file_path: &str,
        data: &[u8],
        processor: &crate::core::DocumentProcessor,
    ) -> Result<(Document, Vec<DocumentChunk>)> {
        let title = Path::new(file_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("Unknown")
            .to_string();

        // Extract text from DOCX
        let content = Self::extract_docx_text(data)?;
        
        // Extract metadata
        let mut metadata = Self::extract_basic_metadata(&content);
        metadata.file_type = Some("docx".to_string());
        metadata.file_size = Some(data.len() as u64);

        let document = Document::new_with_metadata(title, content.clone(), metadata);
        let chunks = processor.chunk_text_semantic(&document.id, &content)?;

        Ok((document, chunks))
    }

    /// Process HTML documents
    pub async fn process_html_document(
        file_path: &str,
        data: &[u8],
        processor: &crate::core::DocumentProcessor,
    ) -> Result<(Document, Vec<DocumentChunk>)> {
        let title = Path::new(file_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("Unknown")
            .to_string();

        // Decode and extract text from HTML
        let html_content = Self::decode_text_with_detection(data)?;
        let content = Self::extract_html_text(&html_content)?;
        
        // Extract metadata
        let mut metadata = Self::extract_basic_metadata(&content);
        metadata.file_type = Some("html".to_string());
        metadata.file_size = Some(data.len() as u64);
        
        // Try to extract title from HTML
        if let Some(html_title) = Self::extract_html_title(&html_content) {
            metadata.custom_fields = serde_json::json!({ "html_title": html_title });
        }

        let document = Document::new_with_metadata(title, content.clone(), metadata);
        let chunks = processor.chunk_text_semantic(&document.id, &content)?;

        Ok((document, chunks))
    }

    /// Process RTF documents (basic implementation)
    pub async fn process_rtf_document(
        file_path: &str,
        data: &[u8],
        processor: &crate::core::DocumentProcessor,
    ) -> Result<(Document, Vec<DocumentChunk>)> {
        let title = Path::new(file_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("Unknown")
            .to_string();

        // Basic RTF text extraction (remove RTF control codes)
        let rtf_content = Self::decode_text_with_detection(data)?;
        let content = Self::extract_rtf_text(&rtf_content)?;
        
        // Extract metadata
        let mut metadata = Self::extract_basic_metadata(&content);
        metadata.file_type = Some("rtf".to_string());
        metadata.file_size = Some(data.len() as u64);

        let document = Document::new_with_metadata(title, content.clone(), metadata);
        let chunks = processor.chunk_text_semantic(&document.id, &content)?;

        Ok((document, chunks))
    }

    // ========== Text Extraction Methods ==========

    /// Extract text from PDF using lopdf (primary method)
    fn extract_pdf_text_lopdf(data: &[u8]) -> Result<String> {
        use lopdf::Document as PdfDocument;
        use std::io::Cursor;

        let mut cursor = Cursor::new(data);
        let doc = PdfDocument::load_from(&mut cursor)
            .map_err(|e| Error::document_processing(format!("Failed to load PDF: {}", e)))?;

        let mut text = String::new();
        let pages = doc.get_pages();

        for (page_num, _page_id) in pages {
            // lopdf extract_text expects page numbers, not page IDs
            if let Ok(page_text) = doc.extract_text(&[page_num]) {
                text.push_str(&page_text);
                text.push('\n');
            }
        }

        if text.trim().is_empty() {
            return Err(Error::document_processing("No text found in PDF"));
        }

        Ok(Self::clean_extracted_text(text))
    }

    /// Extract text from PDF using pdf-extract (fallback method)
    fn extract_pdf_text_fallback(data: &[u8]) -> Result<String> {
        let text = pdf_extract::extract_text_from_mem(data)
            .map_err(|e| Error::document_processing(format!("pdf-extract failed: {}", e)))?;

        if text.trim().is_empty() {
            return Err(Error::document_processing("No text found in PDF"));
        }

        Ok(Self::clean_extracted_text(text))
    }

    /// Detect text encoding and convert it to UTF-8
    fn decode_text_with_detection(data: &[u8]) -> Result<String> {
        use encoding_rs::UTF_8;
        use chardet::detect;

        // Try to detect encoding
        let detected = detect(data);
        let encoding_name = if detected.0.eq_ignore_ascii_case("utf-8") {
            UTF_8
        } else if detected.0.eq_ignore_ascii_case("windows-1252") {
            encoding_rs::WINDOWS_1252
        } else if detected.0.eq_ignore_ascii_case("iso-8859-1") {
            encoding_rs::WINDOWS_1252 // Use Windows-1252 as a superset
        } else {
            UTF_8 // Default to UTF-8
        };

        let (cow, _encoding, _had_errors) = encoding_name.decode(data);
        Ok(cow.to_string())
    }

    /// Convert Markdown content to plain text
    fn markdown_to_text(markdown_content: &str) -> Result<String> {
        use pulldown_cmark::{html, Options, Parser};

        let parser = Parser::new_ext(markdown_content, Options::all());
        let mut html_output = String::new();
        html::push_html(&mut html_output, parser);

        Self::extract_html_text(&html_output)
    }

    /// Extract text from DOCX content
    fn extract_docx_text(data: &[u8]) -> Result<String> {
        let docx = docx_rs::read_docx(data)
            .map_err(|e| Error::document_processing(format!("Failed to read DOCX: {}", e)))?;
        
        let mut text = String::new();
        
        // Extract text from document body
        let document = docx.document;
        for child in document.children {
            if let docx_rs::DocumentChild::Paragraph(paragraph) = child {
                for run in paragraph.children {
                    if let docx_rs::ParagraphChild::Run(run_child) = run {
                        for run_child_inner in run_child.children {
                            if let docx_rs::RunChild::Text(text_elem) = run_child_inner {
                                text.push_str(&text_elem.text);
                            }
                        }
                    }
                }
                text.push('\n');
            }
        }

        Ok(Self::clean_extracted_text(text))
    }

    /// Extract text from HTML content
    fn extract_html_text(html_content: &str) -> Result<String> {
        use scraper::{Html, Selector};

        let fragment = Html::parse_fragment(html_content);
        let selector = Selector::parse("*")
            .map_err(|e| Error::document_processing(format!("Selector error: {}", e)))?;

        let mut text = String::new();
        for element in fragment.select(&selector) {
            text.push_str(&element.text().collect::<Vec<_>>().join(" "));
            text.push(' ');
        }

        Ok(Self::clean_extracted_text(text))
    }

    /// Extract title from HTML content
    fn extract_html_title(html_content: &str) -> Option<String> {
        use scraper::{Html, Selector};

        let fragment = Html::parse_fragment(html_content);
        let selector = Selector::parse("title").ok()?;
        fragment.select(&selector).next()?.text().next().map(|s| s.to_string())
    }

    /// Extract RTF text content
    fn extract_rtf_text(rtf_content: &str) -> Result<String> {
        use regex::Regex;

        let re_control = Regex::new(r"\\[^\s]*")
            .map_err(|e| Error::document_processing(format!("Regex error: {}", e)))?;
        let cleaned = re_control.replace_all(rtf_content, "");
        Ok(Self::clean_extracted_text(cleaned.into_owned()))
    }

    /// Extract title from Markdown content
    fn extract_markdown_title(markdown_content: &str) -> Option<String> {
        markdown_content.lines().find_map(|line| {
            if line.starts_with("# ") {
                Some(line.trim_start_matches("# ").to_string())
            } else {
                None
            }
        })
    }

    /// Clean and normalize extracted text
    fn clean_extracted_text(text: String) -> String {
        use regex::Regex;
        
        // Remove excessive whitespace and normalize line breaks
        let re_whitespace = Regex::new(r"\s+").unwrap();
        let cleaned = re_whitespace.replace_all(&text, " ");
        
        // Remove control characters except for line breaks and tabs
        let re_control = Regex::new(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]").unwrap();
        let cleaned = re_control.replace_all(&cleaned, "");
        
        cleaned.trim().to_string()
    }

    /// Extract basic metadata from document content
    fn extract_basic_metadata(content: &str) -> DocumentMetadata {
        let word_count = content.split_whitespace().count() as u32;

        DocumentMetadata {
            word_count: Some(word_count),
            language: Some("en".to_string()), // Default to English
            ..Default::default()
        }
    }

    /// Check if data is likely text based on content analysis
    fn is_likely_text(data: &[u8]) -> bool {
        let text = String::from_utf8_lossy(data);
        let non_text_ratio = text.chars().filter(|c| c.is_control()).count() as f64 / text.len() as f64;
        non_text_ratio < 0.05
    }
}

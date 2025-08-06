use crate::utils::{Error, Result};
use async_trait::async_trait;

/// File storage operations
#[async_trait]
pub trait FileStorage {
    /// Save file and return its path
    async fn save_file(&self, data: &[u8], filename: &str) -> Result<String>;

    /// Delete file by path
    async fn delete_file(&self, file_path: &str) -> Result<()>;
}

/// Simple file storage implementation
pub struct SimpleFileStorage;

#[async_trait]
impl FileStorage for SimpleFileStorage {
    async fn save_file(&self, data: &[u8], filename: &str) -> Result<String> {
        let path = format!("./uploads/{}", filename);

        tokio::fs::write(&path, data).await.map_err(|e| Error::Io(e))?;

        Ok(path)
    }

    async fn delete_file(&self, file_path: &str) -> Result<()> {
        tokio::fs::remove_file(file_path).await.map_err(|e| Error::Io(e))?;

        Ok(())
    }
}

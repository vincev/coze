use anyhow::{anyhow, bail, Result};
use hf_hub::api::sync::ApiBuilder;
use std::{
    fs, io,
    path::{Path, PathBuf},
};

use crate::models::{ModelId, ModelSpecs};

const MODELS_PATH: &str = "models";

/// Models files cache.
#[derive(Debug)]
pub struct ModelsCache {
    cache_dir: PathBuf,
}

impl ModelsCache {
    /// Creates a new cache instance.
    pub fn new() -> Result<Self> {
        let mut cache_dir =
            dirs::home_dir().ok_or_else(|| anyhow!("Home directory cannot be found"))?;
        cache_dir.push(".cache");
        cache_dir.push("coze");

        fs::create_dir_all(&cache_dir).map_err(|e| anyhow!("Unable to create cache dir: {e}"))?;
        Ok(Self { cache_dir })
    }

    /// Gets a cached model.
    ///
    /// The model may be empty and needs to be downloaded.
    pub fn cached_model(&self, model_id: ModelId) -> CachedModel {
        let specs = model_id.specs();

        let cache_path = self.cache_dir.join(MODELS_PATH).join(specs.cache_dir);
        let model_path = cache_path.join(specs.model_filename);
        let tokenizer_path = if !specs.tokenizer_filename.is_empty() {
            cache_path.join(specs.tokenizer_filename)
        } else {
            PathBuf::new()
        };

        CachedModel {
            cache_path,
            model_path,
            tokenizer_path,
            specs,
        }
    }
}

/// A model files cached on disk.
#[derive(Debug)]
pub struct CachedModel {
    /// Cache folder path.
    pub cache_path: PathBuf,
    /// Weights file
    pub model_path: PathBuf,
    /// Tokenizer file path, may be empty for models without a tokenizer.
    pub tokenizer_path: PathBuf,
    /// Model specifications.
    pub specs: ModelSpecs,
}

impl CachedModel {
    /// Checks if this model has been cached to disk.
    pub fn cached(&self) -> bool {
        if self.tokenizer_path.as_os_str().is_empty() {
            self.model_path.exists()
        } else {
            self.model_path.exists() && self.tokenizer_path.exists()
        }
    }

    /// Downloads model file from Hugging Face.
    ///
    /// The update_fn reports percentage progress to the caller.
    pub fn download_model(&self, update_fn: impl Fn(f32) -> bool + 'static) -> Result<()> {
        fs::create_dir_all(&self.cache_path)
            .map_err(|e| anyhow!("Unable to create model cache dir: {e}"))?;

        let api = ApiBuilder::new()
            .with_progress(false)
            .build()
            .map_err(|e| anyhow!("Hub api error: {e}"))?;

        let weights_url = api
            .model(self.specs.model_repo.to_string())
            .url(self.specs.model_filename);

        download_from_repo(weights_url, &self.model_path, update_fn)
    }

    /// Downloads tokenizer file from Hugging Face.
    ///
    /// The update_fn reports percentage progress to the caller.
    pub fn download_tokenizer(&self, update_fn: impl Fn(f32) -> bool + 'static) -> Result<()> {
        if self.has_tokenizer() {
            // If the spec has a tokenizer the path should not be empty.
            assert!(!self.tokenizer_path.as_os_str().is_empty());

            fs::create_dir_all(&self.cache_path)
                .map_err(|e| anyhow!("Unable to create model cache dir: {e}"))?;

            let api = ApiBuilder::new()
                .with_progress(false)
                .build()
                .map_err(|e| anyhow!("Hub api error: {e}"))?;

            let weights_url = api
                .model(self.specs.tokenizer_repo.to_string())
                .url(self.specs.tokenizer_filename);

            download_from_repo(weights_url, &self.tokenizer_path, update_fn)?;
        }

        Ok(())
    }

    /// Check if this model has a tokenizer
    pub fn has_tokenizer(&self) -> bool {
        !self.specs.tokenizer_filename.is_empty()
    }
}

pub fn download_from_repo(
    url: String,
    dest_filename: &Path,
    update_fn: impl Fn(f32) -> bool + 'static,
) -> Result<()> {
    let agent = ureq::builder().try_proxy_from_env(true).build();

    let response = agent.get(&url).call()?;
    let content_length = response
        .header("content-length")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);

    let reader = response.into_reader();
    let mut reader = ProgressReader::new(reader, content_length, update_fn);

    let temp_filepath = dest_filename.with_extension("tmp");
    let mut temp_file = fs::File::create(&temp_filepath)?;

    if let Err(e) = io::copy(&mut reader, &mut temp_file) {
        let _ = fs::remove_file(&temp_filepath);
        bail!("File copy error: {e}");
    }

    temp_file.sync_all()?;
    drop(temp_file);

    fs::rename(temp_filepath, dest_filename)?;

    Ok(())
}

struct ProgressReader {
    reader: Box<dyn io::Read + Send + Sync>,
    length: usize,
    bytes_read: usize,
    batch_read: usize,
    update_fn: Box<dyn Fn(f32) -> bool + 'static>,
}

impl ProgressReader {
    fn new(
        reader: Box<dyn io::Read + Send + Sync>,
        length: usize,
        update_fn: impl Fn(f32) -> bool + 'static,
    ) -> Self {
        Self {
            reader,
            length,
            bytes_read: 0,
            batch_read: 0,
            update_fn: Box::new(update_fn),
        }
    }

    fn update(&mut self, n: usize) -> io::Result<()> {
        self.batch_read += n;

        let pct = if self.length == 0 {
            // If we didn't get the file length cycle every 100 reads.
            self.bytes_read += 1;
            (self.bytes_read % 100) as f32 / 100.0
        } else {
            self.bytes_read += n;
            self.bytes_read as f32 / self.length as f32
        };

        // Notify UI every half percent.
        if self.batch_read > self.length / 200 {
            self.batch_read = 0;
            if (*self.update_fn)(pct) {
                Ok(())
            } else {
                Err(io::Error::new(io::ErrorKind::BrokenPipe, "User interrupt"))
            }
        } else {
            Ok(())
        }
    }
}

impl std::io::Read for ProgressReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let nread = self.reader.read(buf)?;
        self.update(nread)?;
        Ok(nread)
    }

    fn read_vectored(&mut self, bufs: &mut [io::IoSliceMut<'_>]) -> io::Result<usize> {
        let nread = self.reader.read_vectored(bufs)?;
        self.update(nread)?;
        Ok(nread)
    }

    fn read_to_string(&mut self, buf: &mut String) -> io::Result<usize> {
        let nread = self.reader.read_to_string(buf)?;
        self.update(nread)?;
        Ok(nread)
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> io::Result<()> {
        self.reader.read_exact(buf)?;
        self.update(buf.len())?;
        Ok(())
    }
}

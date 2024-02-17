use anyhow::{anyhow, bail, Result};
use hf_hub::api::sync::ApiBuilder;
use std::{fs, io, path::PathBuf};

const MODEL_ID: &str = "vincevas/coze-stablelm-2-1_6b";
const WEIGHTS_FILENAME: &str = "stablelm-2-zephyr-1_6b-Q4_1.gguf";

/// Cache for model weights files.
///
/// The first time weights are requested they are downloaded from Huggingface.
pub struct WeightsCache {
    cache_dir: PathBuf,
    weights_url: String,
}

impl WeightsCache {
    pub fn new() -> Result<Self> {
        let mut cache_dir =
            dirs::home_dir().ok_or_else(|| anyhow!("Home directory cannot be found"))?;
        cache_dir.push(".cache");
        cache_dir.push("coze");
        cache_dir.push(MODEL_ID);

        fs::create_dir_all(&cache_dir).map_err(|e| anyhow!("Unable to create cache dir: {e}"))?;

        let api = ApiBuilder::new()
            .with_progress(false)
            .build()
            .map_err(|e| anyhow!("Hub api error: {e}"))?;

        let weights_url = api.model(MODEL_ID.to_string()).url(WEIGHTS_FILENAME);

        Ok(Self {
            cache_dir,
            weights_url,
        })
    }

    /// Returns the weights path.
    pub fn weights_path(&self) -> PathBuf {
        self.cache_dir.join(WEIGHTS_FILENAME)
    }

    /// Dowload weights
    pub fn download_weights(&self, update_fn: impl Fn(f32) -> bool + 'static) -> Result<()> {
        let agent = ureq::builder().try_proxy_from_env(true).build();
        let response = agent.get(&self.weights_url).call()?;
        let content_length = response
            .header("content-length")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);

        let reader = response.into_reader();
        let mut reader = ProgressReader::new(reader, content_length, update_fn);

        let temp_filename = format!("{WEIGHTS_FILENAME}.tmp");
        let temp_filepath = self.cache_dir.join(temp_filename);
        let mut temp_file = fs::File::create(&temp_filepath)?;

        if let Err(e) = io::copy(&mut reader, &mut temp_file) {
            let _ = fs::remove_file(&temp_filepath);
            bail!("File copy error: {e}");
        }

        temp_file.sync_all()?;
        drop(temp_file);

        fs::rename(temp_filepath, self.weights_path())?;

        Ok(())
    }
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
            self.bytes_read as f32 / self.length as f32 * 100.0
        };

        // Notify UI every 1MB
        if self.batch_read > 1_048_576 {
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

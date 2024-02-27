//! Models configuration and loading.
use anyhow::Result;

pub use cache::ModelsCache;
pub use config::ModelConfig;

mod cache;
mod config;
mod llama;
mod stablelm;

/// A model specification used to loading and UI.
#[derive(Debug, Clone, Copy)]
pub struct ModelSpecs {
    /// The model model
    pub name: &'static str,
    /// The model size in GB
    pub size: usize,
    /// True if the model data is cached on local disk.
    pub cached: bool,
    /// Cache dir
    pub cache_dir: &'static str,
    /// Repo identifier
    pub model_repo: &'static str,
    /// Model path.
    pub model_filename: &'static str,
    /// Tokenizer repo
    pub tokenizer_repo: &'static str,
    /// Tokenizer path
    pub tokenizer_filename: &'static str,
}

/// Interface to an inference model.
pub trait Model {
    /// Initialize the model with a prompt.
    fn prompt(&mut self) -> Result<()>;

    /// Get the next token.
    ///
    /// Returns Ok(None) when tokens generation is complete.
    fn next(&mut self) -> Result<Option<String>>;
}

#[derive(Debug, Clone, Copy)]
pub enum ModelId {
    StableLm2Zephyr,
    Mistral7bInstructV02,
}

impl ModelId {
    /// Get the model specification.
    pub fn specs(&self) -> ModelSpecs {
        match self {
            ModelId::StableLm2Zephyr => ModelSpecs {
                name: "Stablelm 2 Zephyr 1.6B",
                size: 1029022272,
                cached: true,
                cache_dir: "stablelm2_zephyr_1_6b",
                model_repo: "vincevas/coze-stablelm-2-1_6b",
                model_filename: "stablelm-2-zephyr-1_6b-Q4_1.gguf",
                tokenizer_repo: "",
                tokenizer_filename: "",
            },
            ModelId::Mistral7bInstructV02 => ModelSpecs {
                name: "Mistral Instruct 7B (v0.2)",
                size: 4140374304,
                cached: false,
                cache_dir: "mistral_instruct_7b_v02",
                model_repo: "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
                model_filename: "mistral-7b-instruct-v0.2.Q4_K_S.gguf",
                tokenizer_repo: "mistralai/Mistral-7B-Instruct-v0.2",
                tokenizer_filename: "tokenizer.json",
            },
        }
    }

    /// Create a model instance.
    pub fn model(&self) -> impl Model {
        match self {
            ModelId::StableLm2Zephyr => DummyModel,
            ModelId::Mistral7bInstructV02 => DummyModel,
        }
    }
}

/// Returns a list of available models.
pub fn list_models() -> Vec<ModelId> {
    use ModelId::*;
    vec![StableLm2Zephyr, Mistral7bInstructV02]
}

struct DummyModel;

impl Model for DummyModel {
    fn prompt(&mut self) -> Result<()> {
        todo!()
    }

    fn next(&mut self) -> Result<Option<String>> {
        todo!()
    }
}

//! Models configuration and loading.
use anyhow::Result;
use strum::{EnumIter, IntoEnumIterator};

pub use cache::ModelsCache;
pub use config::ModelConfig;

mod cache;
mod config;
mod llama;
mod stablelm;

#[derive(Debug, Clone, Copy, EnumIter)]
pub enum ModelId {
    StableLm2Zephyr,
    Mistral7bInstructV02,
    Zephyr7bBeta,
}

impl ModelId {
    /// Get the model specification.
    pub fn spec(&self) -> ModelSpec {
        match self {
            ModelId::StableLm2Zephyr => ModelSpec {
                model_id: *self,
                name: "Stablelm 2 Zephyr 1.6B",
                size: 1029022272,
                cache_dir: "stablelm2_zephyr_1_6b",
                model_repo: "vincevas/coze-stablelm-2-1_6b",
                model_filename: "stablelm-2-zephyr-1_6b-Q4_1.gguf",
                tokenizer_repo: "",
                tokenizer_filename: "",
            },
            ModelId::Mistral7bInstructV02 => ModelSpec {
                model_id: *self,
                name: "Mistral Instruct 7B (v0.2)",
                size: 4140374304,
                cache_dir: "mistral_instruct_7b_v02",
                model_repo: "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
                model_filename: "mistral-7b-instruct-v0.2.Q4_K_S.gguf",
                tokenizer_repo: "mistralai/Mistral-7B-Instruct-v0.2",
                tokenizer_filename: "tokenizer.json",
            },
            ModelId::Zephyr7bBeta => ModelSpec {
                model_id: *self,
                name: "Zephyr 7B β",
                size: 4368438976,
                cache_dir: "zephyr-7b-beta",
                model_repo: "TheBloke/zephyr-7B-beta-GGUF",
                model_filename: "zephyr-7b-beta.Q4_K_M.gguf",
                tokenizer_repo: "mistralai/Mistral-7B-Instruct-v0.2",
                tokenizer_filename: "tokenizer.json",
            },
        }
    }

    /// Returns the list of models.
    pub fn models() -> Vec<Self> {
        Self::iter().collect()
    }

    /// Create a model instance.
    pub fn model(&self) -> impl Model {
        match self {
            ModelId::StableLm2Zephyr => DummyModel,
            ModelId::Mistral7bInstructV02 => DummyModel,
            ModelId::Zephyr7bBeta => todo!(),
        }
    }
}

/// A model specification used to loading and UI.
#[derive(Debug, Clone, Copy)]
pub struct ModelSpec {
    /// The model identifier.
    pub model_id: ModelId,
    /// The model model
    pub name: &'static str,
    /// The model size in GB
    pub size: usize,
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

struct DummyModel;

impl Model for DummyModel {
    fn prompt(&mut self) -> Result<()> {
        todo!()
    }

    fn next(&mut self) -> Result<Option<String>> {
        todo!()
    }
}

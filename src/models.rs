//! Models configuration and loading.
pub use config::ModelConfig;

mod config;

/// A model specification used to loading and UI.
#[derive(Debug, Clone, Copy)]
pub struct ModelSpec {
    /// The model model
    pub name: &'static str,
    /// The model size in GB
    pub size: usize,
    /// True if the model data is cached on local disk.
    pub cached: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum ModelId {
    StableLm2Zephyr,
    Mistral7bInstructV02,
}

impl ModelId {
    /// Get the model specification.
    pub fn spec(&self) -> ModelSpec {
        match self {
            ModelId::StableLm2Zephyr => ModelSpec {
                name: "Stablelm 2 Zephyr 1.6B",
                size: 1029022272,
                cached: true,
            },
            ModelId::Mistral7bInstructV02 => ModelSpec {
                name: "Mistral Instruct 7B (v0.2)",
                size: 4140374304,
                cached: false,
            },
        }
    }
}

/// Returns a list of available models.
pub fn list_models() -> Vec<ModelId> {
    use ModelId::*;
    vec![StableLm2Zephyr, Mistral7bInstructV02]
}

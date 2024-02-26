use serde::{Deserialize, Serialize};

/// The model configuration that defines how tokens are generated.
#[derive(Debug, Default, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ModelConfig {
    /// Choose the token with highest probability
    #[default]
    Careful,
    /// Choose from a small number of best tokens,
    Creative,
    /// Choose at random from more tokens.
    Deranged,
}

impl ModelConfig {
    /// Gets the value description.
    pub fn description(&self) -> &'static str {
        match self {
            ModelConfig::Careful => "Careful",
            ModelConfig::Creative => "Creative",
            ModelConfig::Deranged => "Deranged",
        }
    }

    fn config(&self) -> ModelParams {
        match self {
            ModelConfig::Careful => ModelParams::careful(),
            ModelConfig::Creative => ModelParams::creative(),
            ModelConfig::Deranged => ModelParams::deranged(),
        }
    }
}

/// Model configuration parameters.
#[derive(Debug, Clone, Copy)]
struct ModelParams {
    /// Best K tokens
    top_k: usize,
    /// Temperature (higher value flattens token probabilities).
    temperature: f32,
    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    repeat_penalty: f32,
    /// The context size to consider for the repeat penalty.
    repeat_last_n: usize,
    /// The maximum sample length.
    sample_max: u32,
}

impl ModelParams {
    fn careful() -> Self {
        Self {
            top_k: 1,
            temperature: 1.,
            repeat_penalty: 1.2,
            repeat_last_n: 64,
            sample_max: 2048,
        }
    }

    fn creative() -> Self {
        Self {
            top_k: 5,
            temperature: 2.,
            repeat_penalty: 1.2,
            repeat_last_n: 64,
            sample_max: 2048,
        }
    }

    fn deranged() -> Self {
        Self {
            top_k: 10,
            temperature: 5.,
            repeat_penalty: 2.,
            repeat_last_n: 128,
            sample_max: 2048,
        }
    }
}

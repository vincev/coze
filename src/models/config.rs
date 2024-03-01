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

    pub fn params(&self) -> ModelParams {
        match self {
            ModelConfig::Careful => ModelParams::careful(),
            ModelConfig::Creative => ModelParams::creative(),
            ModelConfig::Deranged => ModelParams::deranged(),
        }
    }
}

/// Model configuration parameters.
#[derive(Debug, Clone, Copy)]
pub struct ModelParams {
    /// Best K tokens
    pub top_k: usize,
    /// Temperature (higher value flattens token probabilities).
    pub temperature: f32,
    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    pub repeat_penalty: f32,
    /// The context size to consider for the repeat penalty.
    pub repeat_last_n: usize,
}

impl ModelParams {
    fn careful() -> Self {
        Self {
            top_k: 1,
            temperature: 1.,
            repeat_penalty: 1.2,
            repeat_last_n: 64,
        }
    }

    fn creative() -> Self {
        Self {
            top_k: 5,
            temperature: 2.,
            repeat_penalty: 1.2,
            repeat_last_n: 64,
        }
    }

    fn deranged() -> Self {
        Self {
            top_k: 10,
            temperature: 5.,
            repeat_penalty: 2.,
            repeat_last_n: 128,
        }
    }
}

//! Models configuration and loading.
use anyhow::Result;
use candle::{DType, Tensor};
use rand::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use strum::{EnumIter, IntoEnumIterator};

pub use cache::ModelsCache;
pub use config::{ModelConfig, ModelParams};

mod cache;
mod config;
mod qmistral;
mod qstablelm;
mod qzephyr;
mod transformers;

#[derive(Debug, Clone, Copy, EnumIter)]
pub enum ModelId {
    Mistral7bInstructV02,
    Zephyr7bBeta,
    StableLm2Zephyr,
}

impl ModelId {
    /// Get the model specification.
    pub fn spec(&self) -> ModelSpec {
        match self {
            ModelId::Mistral7bInstructV02 => ModelSpec {
                model_id: *self,
                name: "Mistral Instruct 7B v0.2",
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
            ModelId::StableLm2Zephyr => ModelSpec {
                model_id: *self,
                name: "Stablelm 2 Zephyr 1.6B",
                size: 1029022272,
                cache_dir: "stablelm2_zephyr_1_6b",
                model_repo: "vincevas/coze-stablelm-2-1_6b",
                model_filename: "stablelm-2-zephyr-1_6b-Q4_1.gguf",
                tokenizer_repo: "stabilityai/stablelm-2-zephyr-1_6b",
                tokenizer_filename: "tokenizer.json",
            },
        }
    }

    /// Returns the list of models.
    pub fn models() -> Vec<Self> {
        Self::iter().collect()
    }

    /// Create a model instance.
    pub fn model(&self, params: ModelParams) -> Result<Box<dyn Model>> {
        match self {
            ModelId::StableLm2Zephyr => Ok(Box::new(qstablelm::QuantizedStableLM::new(params)?)),
            ModelId::Zephyr7bBeta => Ok(Box::new(qzephyr::QuantizedZephyr::new(params)?)),
            ModelId::Mistral7bInstructV02 => Ok(Box::new(qmistral::QuantizedMistral::new(params)?)),
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
    fn prompt(&mut self, prompt: &str, params: &ModelParams) -> Result<TokensStream>;

    /// Runs the forward step for the given tokens.
    fn forward(&mut self, tokens: &[u32], pos: usize) -> Result<u32>;

    /// Decode the given tokens.
    fn decode(&mut self, tokens: &[u32]) -> Result<String>;
}

/// Generates tokens for a model.
#[derive(Debug)]
pub struct TokensStream {
    eos_token: u32,
    prompt_tokens_len: usize,
    tokens: Vec<u32>,
    consumed: bool,
}

impl TokensStream {
    /// Creates a new stream.
    pub fn new(eos_token: u32, prompt_tokens_len: usize) -> Self {
        Self {
            eos_token,
            prompt_tokens_len,
            tokens: vec![0],
            consumed: false,
        }
    }

    /// Generates the next token.
    pub fn next(&mut self, model: &mut dyn Model) -> Result<Option<String>> {
        if self.consumed {
            Ok(None)
        } else {
            let decode_idx = self.tokens.len().saturating_sub(5);
            let prev_text = model.decode(&self.tokens[decode_idx..])?;
            loop {
                let token = self.next_token(model)?;
                if token == self.eos_token {
                    self.consumed = true;
                    return Ok(None);
                }

                self.tokens.push(token);
                let text = model.decode(&self.tokens[decode_idx..])?;
                if text.len() > prev_text.len() {
                    return Ok(Some(text.trim_start_matches(&prev_text).to_string()));
                }
            }
        }
    }

    fn next_token(&mut self, model: &mut dyn Model) -> Result<u32> {
        let last_idx = self.tokens.len().saturating_sub(1);
        model.forward(
            &self.tokens[last_idx..],
            self.prompt_tokens_len + self.tokens.len(),
        )
    }
}

/// Sample a token from the given logits tensor and tokens history.
pub fn sample_token(logits: Tensor, tokens: &[u32], params: &ModelParams) -> Result<u32> {
    #[derive(PartialEq, Debug)]
    struct HeapVal(f32);

    impl Eq for HeapVal {}

    impl PartialOrd for HeapVal {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for HeapVal {
        fn cmp(&self, other: &HeapVal) -> Ordering {
            other.0.partial_cmp(&self.0).unwrap_or(Ordering::Greater)
        }
    }

    let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
    let logits = if params.repeat_penalty == 1. {
        logits
    } else {
        let start_at = tokens.len().saturating_sub(params.repeat_last_n);
        candle_transformers::utils::apply_repeat_penalty(
            &logits,
            params.repeat_penalty,
            &tokens[start_at..],
        )?
    };

    let logits_v: Vec<f32> = logits.to_vec1()?;

    let mut heap = BinaryHeap::with_capacity(params.top_k);
    for (idx, v) in logits_v.iter().enumerate() {
        heap.push((HeapVal(*v), idx));
        if heap.len() > params.top_k {
            heap.pop();
        }
    }

    let max_logit = heap
        .iter()
        .max_by(|(u, _), (v, _)| u.cmp(v))
        .map(|(l, _)| l.0)
        .unwrap();

    let (exp_logits, tokens): (Vec<_>, Vec<_>) = heap
        .into_iter()
        .map(|(l, t)| (((l.0 - max_logit) / params.temperature).exp(), t))
        .unzip();

    let total = exp_logits.iter().sum::<f32>();
    let softmax = exp_logits
        .into_iter()
        .map(|v| v / total)
        .collect::<Vec<_>>();

    let mut rng = rand::thread_rng();
    let distr = rand::distributions::WeightedIndex::new(softmax)?;
    let next_token = tokens[distr.sample(&mut rng)];
    Ok(next_token as u32)
}

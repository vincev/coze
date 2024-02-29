use anyhow::Result;
use candle::{DType, Device, Tensor};
use candle_transformers::quantized_var_builder::VarBuilder;

use crate::models::{sample_token, Model, ModelId, ModelParams, ModelsCache};

mod arcade100k;
mod transformer;

/// Quantized StableLM model.
pub struct QStableLM {
    model: transformer::Transformer,
    params: ModelParams,
    tokenizer: arcade100k::Arcade100k,
    tokens: Vec<u32>,
    consumed: bool,
    begin_prompt: bool,
    decoded_index: usize,
    eos_token: u32,
}

impl QStableLM {
    pub fn new(params: ModelParams) -> Result<Self> {
        let cache = ModelsCache::new()?;
        let cached_model = cache.cached_model(ModelId::StableLm2Zephyr);

        let device = Device::Cpu;
        let vb = VarBuilder::from_gguf(cached_model.model_path, &device)?;
        let model = transformer::Transformer::new(vb)?;
        let tokenizer = arcade100k::Arcade100k::new();
        let eos_token = tokenizer.get_token("<|endoftext|>").unwrap();

        Ok(Self {
            model,
            params,
            tokenizer,
            tokens: Default::default(),
            consumed: false,
            begin_prompt: true,
            decoded_index: 0,
            eos_token,
        })
    }

    fn next_token(&mut self) -> Result<bool> {
        let start_pos = if self.begin_prompt {
            self.begin_prompt = false;
            0
        } else {
            self.tokens.len().saturating_sub(1)
        };

        let input = Tensor::new(&self.tokens[start_pos..], &Device::Cpu)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, start_pos)?;
        let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
        let logits = if self.params.repeat_penalty == 1. {
            logits
        } else {
            let start_at = self.tokens.len().saturating_sub(self.params.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                self.params.repeat_penalty,
                &self.tokens[start_at..],
            )?
        };

        let next_token = sample_token(&logits, &self.params)?;
        if next_token == self.eos_token {
            self.consumed = true;
            Ok(false)
        } else {
            self.tokens.push(next_token);
            Ok(true)
        }
    }
}

impl Model for QStableLM {
    fn prompt(&mut self, prompt: &str, params: &ModelParams) -> Result<()> {
        self.params = *params;
        self.begin_prompt = true;
        self.decoded_index = 0;
        self.consumed = false;
        self.model.clear_kv_cache();

        let template = format!("<|user|>\n{prompt}<|endoftext|>\n<|assistant|>\n");
        self.tokens = self.tokenizer.encode(&template);
        self.decoded_index = self.tokens.len();

        Ok(())
    }

    fn next(&mut self) -> Result<Option<String>> {
        if !self.consumed && self.next_token()? {
            let decoded = self.tokenizer.decode(&self.tokens[self.decoded_index..])?;
            self.decoded_index = self.tokens.len();
            Ok(Some(decoded))
        } else {
            Ok(None)
        }
    }
}

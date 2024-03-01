use anyhow::Result;
use candle::{quantized::gguf_file, DType, Device, Tensor};

use crate::models::{
    sample_token, transformers::quantized_llama, Generator, ModelId, ModelParams, ModelsCache,
};

/// Quantized StableLM model.
pub struct Model {
    model: quantized_llama::Transformer,
    params: ModelParams,
    tokenizer: tokenizers::Tokenizer,
    tokens: Vec<u32>,
    consumed: bool,
    begin_prompt: bool,
    decoded_index: usize,
    eos_token: u32,
}

impl Model {
    pub fn new(params: ModelParams) -> Result<Self> {
        let cache = ModelsCache::new()?;
        let cached_model = cache.cached_model(ModelId::Mistral7bInstructV02);

        let device = Device::Cpu;

        let mut file = std::fs::File::open(&cached_model.model_path)?;
        let gguf_content = gguf_file::Content::read(&mut file)
            .map_err(|e| e.with_path(cached_model.model_path))?;
        let model = quantized_llama::Transformer::from_gguf(gguf_content, &mut file, &device)?;

        let tokenizer = tokenizers::Tokenizer::from_file(cached_model.tokenizer_path)
            .map_err(anyhow::Error::msg)?;

        let eos_token = *tokenizer.get_vocab(true).get("</s>").unwrap();

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

impl Generator for Model {
    fn prompt(&mut self, prompt: &str, params: &ModelParams) -> Result<()> {
        self.params = *params;
        self.begin_prompt = true;
        self.decoded_index = 0;
        self.consumed = false;
        self.model.clear_kv_cache();

        let template = format!("<s>[INST] {prompt} [/INST] ");
        self.tokens = self
            .tokenizer
            .encode(template, true)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();
        self.decoded_index = self.tokens.len();
        Ok(())
    }

    fn next(&mut self) -> Result<Option<String>> {
        if self.consumed {
            Ok(None)
        } else {
            let prev_text = self
                .tokenizer
                .decode(&self.tokens[self.decoded_index.saturating_sub(1)..], true)
                .map_err(anyhow::Error::msg)?;

            while self.next_token()? {
                let text = self
                    .tokenizer
                    .decode(&self.tokens[self.decoded_index.saturating_sub(1)..], true)
                    .map_err(anyhow::Error::msg)?;

                if text.len() > prev_text.len() {
                    self.decoded_index = self.tokens.len();
                    let text = text.split_at(prev_text.len());
                    return Ok(Some(text.1.to_string()));
                }
            }

            Ok(None)
        }
    }
}

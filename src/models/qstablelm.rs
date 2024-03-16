use anyhow::Result;
use candle::{Device, Tensor};
use candle_transformers::quantized_var_builder::VarBuilder;

use crate::models::{
    sample_token, transformers::quantized_stable_lm, Model, ModelId, ModelParams, ModelsCache,
    TokensStream,
};

/// Quantized StableLM model.
pub struct QuantizedStableLM {
    model: quantized_stable_lm::Transformer,
    params: ModelParams,
    tokenizer: tokenizers::Tokenizer,
    eos_token: u32,
}

impl QuantizedStableLM {
    pub fn new(params: ModelParams) -> Result<Self> {
        let cache = ModelsCache::new()?;
        let cached_model = cache.cached_model(ModelId::StableLm2Zephyr);

        let device = Device::Cpu;
        let vb = VarBuilder::from_gguf(cached_model.model_path, &device)?;
        let model = quantized_stable_lm::Transformer::new(vb)?;
        let tokenizer = tokenizers::Tokenizer::from_file(cached_model.tokenizer_path)
            .map_err(anyhow::Error::msg)?;
        let eos_token = *tokenizer.get_vocab(true).get("<|endoftext|>").unwrap();

        Ok(Self {
            model,
            params,
            tokenizer,
            eos_token,
        })
    }
}

impl Model for QuantizedStableLM {
    fn prompt(&mut self, prompt: &str, params: &ModelParams) -> Result<TokensStream> {
        self.params = *params;
        self.model.clear_kv_cache();

        let template = format!("<|user|>\n{prompt}<|endoftext|>\n");
        let tokens = self
            .tokenizer
            .encode(template, true)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();
        self.forward(&tokens, 0)?;

        Ok(TokensStream::new(self.eos_token, tokens.len()))
    }

    fn forward(&mut self, tokens: &[u32], pos: usize) -> Result<u32> {
        let input = Tensor::new(tokens, &Device::Cpu)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, pos)?;
        sample_token(logits, tokens, &self.params)
    }

    fn decode(&mut self, tokens: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(tokens, false)
            .map_err(anyhow::Error::msg)
    }
}

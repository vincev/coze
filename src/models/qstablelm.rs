use anyhow::Result;
use candle::{Device, Tensor};
use candle_transformers::quantized_var_builder::VarBuilder;

use crate::models::{
    sample_token, transformers::quantized_stable_lm, Generator, ModelId, ModelParams, ModelsCache,
    TokensStream,
};

mod arcade100k;

/// Quantized StableLM model.
pub struct Model {
    model: quantized_stable_lm::Transformer,
    params: ModelParams,
    tokenizer: arcade100k::Arcade100k,
    eos_token: u32,
}

impl Model {
    pub fn new(params: ModelParams) -> Result<Self> {
        let cache = ModelsCache::new()?;
        let cached_model = cache.cached_model(ModelId::StableLm2Zephyr);

        let device = Device::Cpu;
        let vb = VarBuilder::from_gguf(cached_model.model_path, &device)?;
        let model = quantized_stable_lm::Transformer::new(vb)?;
        let tokenizer = arcade100k::Arcade100k::new();
        let eos_token = tokenizer.get_token("<|endoftext|>").unwrap();

        Ok(Self {
            model,
            params,
            tokenizer,
            eos_token,
        })
    }
}

impl Generator for Model {
    fn prompt(&mut self, prompt: &str, params: &ModelParams) -> Result<TokensStream> {
        self.params = *params;
        self.model.clear_kv_cache();

        let template = format!("<|user|>\n{prompt}<|endoftext|>\n<|assistant|>\n");
        let tokens = self.tokenizer.encode(&template);
        self.forward(&tokens, 0)?;

        Ok(TokensStream::new(self.eos_token, tokens.len()))
    }

    fn forward(&mut self, tokens: &[u32], pos: usize) -> Result<u32> {
        let input = Tensor::new(tokens, &Device::Cpu)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, pos)?;
        sample_token(logits, tokens, &self.params)
    }

    fn decode(&mut self, tokens: &[u32]) -> Result<String> {
        self.tokenizer.decode(tokens)
    }
}

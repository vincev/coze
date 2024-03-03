use anyhow::Result;
use candle::{quantized::gguf_file, Device, Tensor};

use crate::models::{
    sample_token, transformers::quantized_llama, Generator, ModelId, ModelParams, ModelsCache,
    TokensStream,
};

/// Quantized StableLM model.
pub struct Model {
    model: quantized_llama::Transformer,
    params: ModelParams,
    tokenizer: tokenizers::Tokenizer,
    eos_token: u32,
}

impl Model {
    pub fn new(params: ModelParams) -> Result<Self> {
        let cache = ModelsCache::new()?;
        let cached_model = cache.cached_model(ModelId::Zephyr7bBeta);

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
            eos_token,
        })
    }
}

impl Generator for Model {
    fn prompt(&mut self, prompt: &str, params: &ModelParams) -> Result<TokensStream> {
        self.params = *params;
        self.model.clear_kv_cache();

        let template = format!("<|system|>\n</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n");
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
            .decode(tokens, true)
            .map_err(anyhow::Error::msg)
    }
}

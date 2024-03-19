use anyhow::Result;
use candle::{quantized::gguf_file, Device, Tensor};
use candle_transformers::{
    models::mistral, models::quantized_mistral, quantized_var_builder::VarBuilder,
};

use crate::models::{
    sample_token, transformers::quantized_llama, Model, ModelId, ModelParams, ModelsCache,
    TokensStream,
};

/// Quantized Mistral instruct model.
pub struct QuantizedMistralInstruct {
    model: quantized_llama::Transformer,
    params: ModelParams,
    tokenizer: tokenizers::Tokenizer,
    eos_token: u32,
}

impl QuantizedMistralInstruct {
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
            eos_token,
        })
    }
}

impl Model for QuantizedMistralInstruct {
    fn prompt(&mut self, prompt: &str, params: &ModelParams) -> Result<TokensStream> {
        self.params = *params;
        self.model.clear_kv_cache();

        let template = format!("[INST] {prompt} [/INST]");
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

/// Quantized Mistral 7B model.
pub struct QuantizedMistral7B {
    model: quantized_mistral::Model,
    params: ModelParams,
    tokenizer: tokenizers::Tokenizer,
    eos_token: u32,
}

impl QuantizedMistral7B {
    pub fn new(params: ModelParams) -> Result<Self> {
        let cache = ModelsCache::new()?;
        let cached_model = cache.cached_model(ModelId::Mistral7B);

        let device = Device::Cpu;

        let vb = VarBuilder::from_gguf(cached_model.model_path, &device)?;
        let config = mistral::Config::config_7b_v0_1(false);
        let model = quantized_mistral::Model::new(&config, vb)?;

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

impl Model for QuantizedMistral7B {
    fn prompt(&mut self, prompt: &str, params: &ModelParams) -> Result<TokensStream> {
        self.params = *params;
        self.model.clear_kv_cache();

        let tokens = self
            .tokenizer
            .encode(prompt, true)
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

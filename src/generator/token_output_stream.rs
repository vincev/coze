// This is a copy of the token output stream at:
//
//  https://github.com/huggingface/candle/blob/main/candle-examples/src/token_output_stream.rs
//
// modified to use the arcade100k tokenizer.
use anyhow::{bail, Result};

use super::arcade100k;

pub struct TokenOutputStream {
    tokenizer: arcade100k::Arcade100k,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
    eos_token: u32,
}

impl TokenOutputStream {
    pub fn new() -> Self {
        let tokenizer = arcade100k::Arcade100k::new();
        let eos_token = tokenizer.get_token("<|endoftext|>").unwrap();
        Self {
            tokenizer,
            tokens: Vec::new(),
            prev_index: 0,
            current_index: 0,
            eos_token,
        }
    }

    pub fn eos_token(&self) -> u32 {
        self.eos_token
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        match self.tokenizer.decode(tokens) {
            Ok(str) => Ok(str),
            Err(err) => bail!("cannot decode: {err}"),
        }
    }

    pub fn encode(&mut self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text)
    }

    // https://github.com/huggingface/text-generation-inference/blob/5ba53d44a18983a4de32d122f4cb46f4a17d9ef6/server/text_generation_server/models/model.py#L68
    pub fn next_token(&mut self, token: u32) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        self.tokens.push(token);
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() && text.chars().last().unwrap().is_ascii() {
            let text = text.split_at(prev_text.len());
            self.prev_index = self.current_index;
            self.current_index = self.tokens.len();
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    pub fn decode_rest(&self) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() {
            let text = text.split_at(prev_text.len());
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    pub fn clear(&mut self) {
        self.tokens.clear();
        self.prev_index = 0;
        self.current_index = 0;
    }
}

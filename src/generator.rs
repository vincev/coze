use anyhow::Result;
use candle::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use crossbeam_channel::{bounded, Receiver, Sender};
use rand::prelude::*;
use std::thread;
use token_output_stream::TokenOutputStream;

use transformer::Transformer;

mod arcade100k;
mod token_output_stream;
mod transformer;

/// Generator configuration.
#[derive(Debug)]
pub struct Config {
    /// The temperature used to generate samples.
    pub temperature: Option<f64>,
    /// Nucleus sampling probability cutoff.
    pub top_p: Option<f64>,
    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    pub repeat_penalty: f32,
    /// The context size to consider for the repeat penalty.
    pub repeat_last_n: usize,
    /// The maximum sample length.
    pub sample_max: u32,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            temperature: None,
            top_p: None,
            repeat_penalty: 1.1,
            repeat_last_n: 128,
            sample_max: 2048,
            seed: None,
        }
    }
}

/// Command for the generator.
enum Command {
    /// Process the given prompt.
    Prompt(String),
    /// Update the generator configuration.
    Config(Config),
    /// Shutdown generator thread.
    Shutdown,
}

/// A message sent by the generator task
pub enum Message {
    /// A generated token.
    Token(String),
    /// An error message.
    Error(String),
}

/// Tokens generator.
///
/// Runs the model on another thread, processes incoming `Command` through a channel
/// and sends back generated tokens to the UI.
#[derive(Debug)]
pub struct Generator {
    command_tx: Sender<Command>,
    message_rx: Receiver<Message>,
    task: Option<thread::JoinHandle<()>>,
}

impl Generator {
    /// Creates a new generator with the given configuration.
    pub fn new(config: Config) -> Self {
        let (command_tx, command_rx) = bounded(1024);
        let (message_tx, message_rx) = bounded(1024);

        let task = thread::spawn(move || {
            generator(config, command_rx, message_tx);
        });

        Self {
            command_tx,
            message_rx,
            task: Some(task),
        }
    }

    /// Sends a new prompt to the mode.
    pub fn send_prompt(&self, prompt: &str) {
        let template = format!("<|user|>\n{prompt}<|endoftext|>\n<|assistant|>\n");
        let _ = self.command_tx.send(Command::Prompt(template));
    }

    /// Get the next available generator message.
    pub fn next_message(&self) -> Option<Message> {
        self.message_rx.try_recv().ok()
    }

    /// Shutdown generator thread
    pub fn shutdown(&mut self) {
        let _ = self.command_tx.send(Command::Shutdown);
        self.task.take().map(|h| h.join());
    }
}

fn generator(config: Config, command_rx: Receiver<Command>, message_tx: Sender<Message>) {
    let mut model = match transformer::Transformer::new() {
        Ok(model) => model,
        Err(e) => {
            let _ = message_tx.send(Message::Error(e.to_string()));
            return;
        }
    };

    let mut tokenizer = TokenOutputStream::new();
    let seed = config.seed.unwrap_or_else(|| thread_rng().gen());
    let mut logits_processor = LogitsProcessor::new(seed, config.temperature, config.top_p);

    while let Ok(cmd) = command_rx.recv() {
        match cmd {
            Command::Prompt(prompt) => {
                tokenizer.clear();
                model.reset();

                let mut tokens = tokenizer.encode(&prompt);

                for idx in 0..config.sample_max {
                    let context_size = if idx > 0 { 1 } else { tokens.len() };
                    let start_pos = tokens.len().saturating_sub(context_size);
                    let result = generate_token(
                        &tokens,
                        start_pos,
                        &config,
                        &mut model,
                        &mut logits_processor,
                        tokenizer.eos_token(),
                    );

                    match result {
                        Ok(None) => {
                            // Generated eos_token
                            break;
                        }
                        Ok(Some(token)) => {
                            tokens.push(token);
                            if let Ok(Some(token_str)) = tokenizer.next_token(token) {
                                let _ = message_tx.send(Message::Token(token_str));
                            } else {
                                break;
                            }
                        }
                        Err(err) => {
                            let _ = message_tx.send(Message::Error(err.to_string()));
                        }
                    }
                }

                if let Ok(Some(token_str)) = tokenizer.decode_rest() {
                    let _ = message_tx.send(Message::Token(token_str));
                }
            }
            Command::Config(_config) => todo!(),
            Command::Shutdown => break,
        }
    }
}

fn generate_token(
    tokens: &[u32],
    start_pos: usize,
    config: &Config,
    model: &mut Transformer,
    logits_processor: &mut LogitsProcessor,
    eos_token: u32,
) -> Result<Option<u32>> {
    let device = Device::Cpu;

    let ctx = &tokens[start_pos..];
    let input = Tensor::new(ctx, &device)?.unsqueeze(0)?;
    let logits = model.forward(&input, start_pos)?;
    let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
    let logits = if config.repeat_penalty == 1. {
        logits
    } else {
        let start_at = tokens.len().saturating_sub(config.repeat_last_n);
        candle_transformers::utils::apply_repeat_penalty(
            &logits,
            config.repeat_penalty,
            &tokens[start_at..],
        )?
    };

    let next_token = logits_processor.sample(&logits)?;
    if next_token == eos_token {
        Ok(None)
    } else {
        Ok(Some(next_token))
    }
}

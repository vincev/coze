use anyhow::Result;
use candle::{DType, Device, Tensor};
use crossbeam_channel::{bounded, Receiver, Sender};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::thread;
use token_output_stream::TokenOutputStream;

use transformer::Transformer;
use weights_cache::WeightsCache;

mod arcade100k;
mod token_output_stream;
mod transformer;
mod weights_cache;

/// Generator mode defines how tokens are choosen.
#[derive(Debug, Default, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum GeneratorMode {
    /// Choose the token with highest probability
    #[default]
    Careful,
    /// Choose from a small number of best tokens,
    Creative,
    /// Choose at random from more tokens.
    Deranged,
}

impl GeneratorMode {
    /// Gets the value description.
    pub fn description(&self) -> &'static str {
        match self {
            GeneratorMode::Careful => "Careful",
            GeneratorMode::Creative => "Creative",
            GeneratorMode::Deranged => "Deranged",
        }
    }

    fn config(&self) -> Config {
        match self {
            GeneratorMode::Careful => Config::careful(),
            GeneratorMode::Creative => Config::creative(),
            GeneratorMode::Deranged => Config::deranged(),
        }
    }
}

/// Generator configuration.
#[derive(Debug, Clone, Copy)]
struct Config {
    /// Best K tokens
    top_k: usize,
    /// Temperature (higher value flattens token probabilities).
    temperature: f32,
    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    repeat_penalty: f32,
    /// The context size to consider for the repeat penalty.
    repeat_last_n: usize,
    /// The maximum sample length.
    sample_max: u32,
}

impl Config {
    fn careful() -> Self {
        Self {
            top_k: 1,
            temperature: 1.,
            repeat_penalty: 1.2,
            repeat_last_n: 64,
            sample_max: 2048,
        }
    }

    fn creative() -> Self {
        Self {
            top_k: 5,
            temperature: 2.,
            repeat_penalty: 1.2,
            repeat_last_n: 64,
            sample_max: 2048,
        }
    }

    fn deranged() -> Self {
        Self {
            top_k: 10,
            temperature: 5.,
            repeat_penalty: 2.,
            repeat_last_n: 128,
            sample_max: 2048,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PromptId(u32);

impl PromptId {
    fn inc(&self) -> PromptId {
        PromptId(self.0 + 1)
    }
}

/// Command for the generator.
enum Command {
    /// Process the given prompt.
    Prompt(PromptId, String),
    /// Update the generator configuration.
    Config(GeneratorMode),
    /// Refresh weights
    ReloadWeights,
    /// Stops token generation.
    Stop,
    /// Shutdown generator thread.
    Shutdown,
}

/// A message sent by the generator task
pub enum Message {
    /// A generated token.
    Token(PromptId, String),
    /// An error message.
    Error(String),
    /// Weights download has started for a model.
    WeightsDownloadBegin(String),
    /// Weights download percent progress.
    WeightsDownloadProgress(f32),
    /// Weights download has completed.
    WeightsDownloadComplete,
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
    last_prompt_id: PromptId,
    mode: GeneratorMode,
}

impl Generator {
    /// Creates a new generator with the given configuration.
    pub fn new(mode: GeneratorMode) -> Self {
        let (command_tx, command_rx) = bounded(1024);
        let (message_tx, message_rx) = bounded(1024);

        let task = thread::spawn(move || {
            generator(mode, command_rx, message_tx);
        });

        Self {
            command_tx,
            message_rx,
            task: Some(task),
            last_prompt_id: PromptId::default(),
            mode,
        }
    }

    /// Sends a new prompt to the mode.
    pub fn send_prompt(&mut self, prompt: &str) -> PromptId {
        self.last_prompt_id = self.last_prompt_id.inc();

        let template = format!("<|user|>\n{prompt}<|endoftext|>\n<|assistant|>\n");
        let _ = self
            .command_tx
            .send(Command::Prompt(self.last_prompt_id, template));

        self.last_prompt_id
    }

    /// Refresh weights
    pub fn reload_weights(&self) {
        let _ = self.command_tx.send(Command::ReloadWeights);
    }

    /// Returns the current config.
    pub fn mode(&self) -> GeneratorMode {
        self.mode
    }

    /// Sets the generator config
    pub fn set_config(&mut self, config: GeneratorMode) {
        self.mode = config;
        let _ = self.command_tx.send(Command::Config(config));
    }

    /// Get the next available generator message.
    pub fn next_message(&self) -> Option<Message> {
        self.message_rx.try_recv().ok()
    }

    /// Stops token generation.
    ///
    /// This may be useful when the generator is in deranged mode and it keeps
    /// generating text we are not interested in.
    pub fn stop(&self) {
        let _ = self.command_tx.send(Command::Stop);
    }

    /// Shutdown generator thread
    pub fn shutdown(&mut self) {
        let _ = self.command_tx.send(Command::Shutdown);
        self.task.take().map(|h| h.join());
    }
}

fn generator(
    config_value: GeneratorMode,
    command_rx: Receiver<Command>,
    message_tx: Sender<Message>,
) {
    let mut model = match load_model(&command_rx, &message_tx, false) {
        Ok(model) => Some(model),
        Err(e) => {
            let _ = message_tx.send(Message::Error(e.to_string()));
            None
        }
    };

    let mut tokenizer = TokenOutputStream::new();
    let mut config = config_value.config();

    while let Ok(cmd) = command_rx.recv() {
        match cmd {
            Command::Prompt(prompt_id, prompt) => 'prompt: {
                if let Some(model) = model.as_mut() {
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
                            model,
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
                                    let _ = message_tx.send(Message::Token(prompt_id, token_str));
                                }
                            }
                            Err(err) => {
                                let _ = message_tx.send(Message::Error(err.to_string()));
                            }
                        }

                        // Skip remainining tokens if there is a new command.
                        if !command_rx.is_empty() {
                            break 'prompt;
                        }
                    }

                    if let Ok(Some(token_str)) = tokenizer.decode_rest() {
                        let _ = message_tx.send(Message::Token(prompt_id, token_str));
                    }
                }
            }
            Command::Config(value) => config = value.config(),
            Command::Stop => {}
            Command::ReloadWeights => {
                model = match load_model(&command_rx, &message_tx, true) {
                    Ok(model) => Some(model),
                    Err(e) => {
                        let _ = message_tx.send(Message::Error(e.to_string()));
                        model.or(None)
                    }
                };
            }
            Command::Shutdown => break,
        }
    }
}

fn load_model(
    command_rx: &Receiver<Command>,
    message_tx: &Sender<Message>,
    reload: bool,
) -> Result<Transformer> {
    let cache = WeightsCache::new()?;
    let _ = message_tx.send(Message::WeightsDownloadBegin(cache.model_name()));

    let cache = WeightsCache::new()?;
    let weights_path = cache.weights_path();
    if !weights_path.exists() || reload {
        cache.download_weights({
            let message_tx = message_tx.clone();
            let command_rx = command_rx.clone();
            move |pct| {
                if command_rx.is_empty() {
                    let _ = message_tx.send(Message::WeightsDownloadProgress(pct));
                    true
                } else {
                    false
                }
            }
        })?;
    } else {
        for pct in 0..=100 {
            if command_rx.is_empty() {
                let _ = message_tx.send(Message::WeightsDownloadProgress(pct as f32 / 100.0));
            } else {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    let _ = message_tx.send(Message::WeightsDownloadComplete);
    Transformer::new(&weights_path)
}

fn generate_token(
    tokens: &[u32],
    start_pos: usize,
    config: &Config,
    model: &mut Transformer,
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

    let next_token = top_k(config, &logits)?;
    if next_token == eos_token {
        Ok(None)
    } else {
        Ok(Some(next_token))
    }
}

fn top_k(config: &Config, logits: &Tensor) -> Result<u32> {
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

    let logits_v: Vec<f32> = logits.to_vec1()?;

    let mut heap = BinaryHeap::with_capacity(config.top_k);
    for (idx, v) in logits_v.iter().enumerate() {
        heap.push((HeapVal(*v), idx));
        if heap.len() > config.top_k {
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
        .map(|(l, t)| (((l.0 - max_logit) / config.temperature).exp(), t))
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

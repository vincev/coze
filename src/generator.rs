use anyhow::Result;
use candle::{DType, Device, Tensor};
use crossbeam_channel::{bounded, Receiver, Sender};
use rand::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::thread;
use token_output_stream::TokenOutputStream;

use transformer::Transformer;

mod arcade100k;
mod token_output_stream;
mod transformer;

/// Generator configuration.
#[derive(Debug)]
pub struct Config {
    /// Best K tokens
    pub top_k: Option<usize>,
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
            top_k: Some(5),
            repeat_penalty: 1.1,
            repeat_last_n: 128,
            sample_max: 2048,
            seed: None,
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
    Config(Config),
    /// Shutdown generator thread.
    Shutdown,
}

/// A message sent by the generator task
pub enum Message {
    /// A generated token.
    Token(PromptId, String),
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
    last_prompt_id: PromptId,
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
            last_prompt_id: PromptId::default(),
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
    let mut model = match Transformer::new() {
        Ok(model) => model,
        Err(e) => {
            let _ = message_tx.send(Message::Error(e.to_string()));
            return;
        }
    };

    let mut tokenizer = TokenOutputStream::new();

    while let Ok(cmd) = command_rx.recv() {
        match cmd {
            Command::Prompt(prompt_id, prompt) => 'prompt: {
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

    let next_token = top_k(config.top_k, &logits)?;
    if next_token == eos_token {
        Ok(None)
    } else {
        Ok(Some(next_token))
    }
}

fn top_k(top_k: Option<usize>, logits: &Tensor) -> Result<u32> {
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

    let top_k = top_k.unwrap_or(5);
    let logits_v: Vec<f32> = logits.to_vec1()?;

    let mut heap = BinaryHeap::with_capacity(top_k);
    for (idx, v) in logits_v.iter().enumerate() {
        heap.push((HeapVal(*v), idx));
        if heap.len() > top_k {
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
        .map(|(l, t)| ((l.0 - max_logit).exp(), t))
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

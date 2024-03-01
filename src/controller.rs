use anyhow::Result;
use crossbeam_channel::{bounded, Receiver, Sender};
use std::thread;

use crate::models::{Generator, ModelConfig, ModelId, ModelParams, ModelsCache};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PromptId(u32);

impl PromptId {
    fn inc(&self) -> PromptId {
        PromptId(self.0 + 1)
    }
}

/// Command for the controller.
enum Command {
    /// Load the given model.
    LoadModel(ModelId),
    /// Process the given prompt.
    Prompt(PromptId, String),
    /// Update the model configuration.
    Config(ModelConfig),
    /// Refresh weights for the given model.
    ReloadWeights(ModelId),
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
    DownloadBegin(String),
    /// Weights download connection.
    DownloadConnecting,
    /// Weights download percent progress.
    DownloadProgress(f32),
    /// Weights download has completed.
    DownloadComplete,
}

/// Tokens generator.
///
/// Runs the model on another thread, processes incoming `Command` through a channel
/// and sends back generated tokens to the UI.
#[derive(Debug)]
pub struct Controller {
    command_tx: Sender<Command>,
    message_rx: Receiver<Message>,
    task: Option<thread::JoinHandle<()>>,
    last_prompt_id: PromptId,
    model_config: ModelConfig,
}

impl Controller {
    /// Creates a new generator with the given configuration.
    pub fn new(model_config: ModelConfig) -> Self {
        let (command_tx, command_rx) = bounded(1024);
        let (message_tx, message_rx) = bounded(1024);

        let task = thread::spawn(move || {
            message_loop(model_config, command_rx, message_tx);
        });

        Self {
            command_tx,
            message_rx,
            task: Some(task),
            last_prompt_id: PromptId::default(),
            model_config,
        }
    }

    /// Sends a new prompt to the mode.
    pub fn send_prompt(&mut self, prompt: &str) -> PromptId {
        self.last_prompt_id = self.last_prompt_id.inc();

        let _ = self
            .command_tx
            .send(Command::Prompt(self.last_prompt_id, prompt.to_string()));

        self.last_prompt_id
    }

    /// Refreshes weights
    pub fn reload_weights(&self, model_id: ModelId) {
        let _ = self.command_tx.send(Command::ReloadWeights(model_id));
    }

    /// Loads the a model.
    pub fn load_model(&self, model_id: ModelId) {
        let _ = self.command_tx.send(Command::LoadModel(model_id));
    }

    /// Returns the current config.
    pub fn model_config(&self) -> ModelConfig {
        self.model_config
    }

    /// Sets the generator config
    pub fn set_config(&mut self, config: ModelConfig) {
        self.model_config = config;
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

fn message_loop(
    model_config: ModelConfig,
    command_rx: Receiver<Command>,
    message_tx: Sender<Message>,
) {
    let mut generator: Option<Box<dyn Generator>> = None;
    let mut model_params = model_config.params();

    while let Ok(cmd) = command_rx.recv() {
        match cmd {
            Command::LoadModel(model_id) => {
                match load_model(
                    model_id,
                    model_config.params(),
                    &command_rx,
                    &message_tx,
                    false,
                ) {
                    Ok(m) => generator = Some(m),
                    Err(e) => {
                        let _ = message_tx.send(Message::Error(e.to_string()));
                    }
                };
            }
            Command::Prompt(prompt_id, prompt) => {
                if let Some(model) = generator.as_mut() {
                    if let Err(e) = model.prompt(&prompt, &model_params) {
                        let _ = message_tx.send(Message::Error(e.to_string()));
                    } else {
                        loop {
                            match model.next() {
                                Ok(Some(token_str)) => {
                                    let _ = message_tx.send(Message::Token(prompt_id, token_str));
                                }
                                Ok(None) => break,
                                Err(e) => {
                                    let _ = message_tx.send(Message::Error(e.to_string()));
                                    break;
                                }
                            }

                            // Skip remainining tokens if there is a new command.
                            if !command_rx.is_empty() {
                                break;
                            }
                        }
                    }
                }
            }
            Command::Config(config) => model_params = config.params(),
            Command::Stop => {}
            Command::ReloadWeights(model_id) => {
                match load_model(
                    model_id,
                    model_config.params(),
                    &command_rx,
                    &message_tx,
                    true,
                ) {
                    Ok(m) => generator = Some(m),
                    Err(e) => {
                        let _ = message_tx.send(Message::Error(e.to_string()));
                    }
                };
            }
            Command::Shutdown => break,
        }
    }
}

fn load_model(
    model_id: ModelId,
    params: ModelParams,
    command_rx: &Receiver<Command>,
    message_tx: &Sender<Message>,
    reload: bool,
) -> Result<Box<dyn Generator>> {
    let cache = ModelsCache::new()?;
    let cached_model = cache.cached_model(model_id);

    if !cached_model.is_model_cached() || reload {
        let _ = message_tx.send(Message::DownloadBegin("Downloading Model".to_string()));
        let _ = message_tx.send(Message::DownloadConnecting);

        cached_model.download_model({
            let message_tx = message_tx.clone();
            let command_rx = command_rx.clone();
            move |pct| {
                if command_rx.is_empty() {
                    let _ = message_tx.send(Message::DownloadProgress(pct));
                    true
                } else {
                    false
                }
            }
        })?;
    }

    if !cached_model.is_tokenizer_cached() || reload {
        let _ = message_tx.send(Message::DownloadBegin("Downloading Tokenizer".to_string()));
        let _ = message_tx.send(Message::DownloadConnecting);

        cached_model.download_tokenizer({
            let message_tx = message_tx.clone();
            let command_rx = command_rx.clone();
            move |pct| {
                if command_rx.is_empty() {
                    let _ = message_tx.send(Message::DownloadProgress(pct));
                    true
                } else {
                    false
                }
            }
        })?;
    }

    let _ = message_tx.send(Message::DownloadComplete);
    model_id.model(params)
}

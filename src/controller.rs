use crossbeam_channel::{bounded, Receiver, Sender};
use std::thread;

use crate::models::{ModelConfig, ModelId};

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
    /// Weights download connection.
    WeightsDownloadConnecting,
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
            generator(model_config, command_rx, message_tx);
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

        let template = format!("<|user|>\n{prompt}<|endoftext|>\n<|assistant|>\n");
        let _ = self
            .command_tx
            .send(Command::Prompt(self.last_prompt_id, template));

        self.last_prompt_id
    }

    /// Refreshes weights
    pub fn reload_weights(&self) {
        let _ = self.command_tx.send(Command::ReloadWeights);
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

fn generator(
    mut model_config: ModelConfig,
    command_rx: Receiver<Command>,
    message_tx: Sender<Message>,
) {
    while let Ok(cmd) = command_rx.recv() {
        match cmd {
            Command::LoadModel(model_id) => {}
            Command::Prompt(prompt_id, prompt) => {}
            Command::Config(value) => model_config = value,
            Command::Stop => {}
            Command::ReloadWeights => {}
            Command::Shutdown => break,
        }
    }
}

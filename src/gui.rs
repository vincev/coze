use std::fmt::Debug;

use eframe::egui::*;
use serde::{Deserialize, Serialize};

use crate::generator::{Generator, GeneratorMode, Message, PromptId};
use bubble::{Bubble, BubbleContent};
use history::HistoryNavigator;
use prompt_panel::PromptPanel;

mod bubble;
mod config;
mod error;
mod help;
mod history;
mod prompt_panel;

const TEXT_FONT: FontId = FontId::new(15.0, FontFamily::Monospace);
const ROUNDING: f32 = 8.0;

#[derive(Clone, Copy, Deserialize, Serialize, Debug, Default, PartialEq)]
enum UiMode {
    #[default]
    Light,
    Dark,
}

impl UiMode {
    fn visuals(&self) -> Visuals {
        match self {
            UiMode::Light => Visuals::light(),
            UiMode::Dark => Visuals::dark(),
        }
    }

    fn description(&self) -> &'static str {
        match self {
            UiMode::Light => "Light",
            UiMode::Dark => "Dark",
        }
    }

    fn fill_color(&self) -> Color32 {
        match &self {
            UiMode::Light => Color32::from_gray(230),
            UiMode::Dark => Color32::from_gray(50),
        }
    }
}

/// State persisted by egui.
#[derive(Deserialize, Serialize, Debug, Default)]
struct PersistedState {
    history: Vec<Prompt>,
    generator_mode: GeneratorMode,
    ui_mode: UiMode,
}

#[derive(Deserialize, Serialize, Debug)]
struct Prompt {
    prompt: String,
    reply: String,
}

trait Panel: Debug {
    fn update(&mut self, ctx: &Context, app: &mut AppContext);
    fn process_input(&mut self, ctx: &Context, app: &mut AppContext);
    fn message(&mut self, _app: &mut AppContext, _msg: &Message) {}
}

#[derive(Debug)]
struct AppContext {
    state: PersistedState,
    generator: Generator,
}

#[derive(Debug)]
pub struct App {
    ctx: AppContext,
    error: Option<String>,
    show_config: bool,
    show_help: bool,
    active_panel: Box<dyn Panel>,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let state: PersistedState = if let Some(storage) = cc.storage {
            // Load previous app state (if any).
            eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default()
        } else {
            Default::default()
        };

        cc.egui_ctx.set_visuals(state.ui_mode.visuals());

        let generator = Generator::new(state.generator_mode);
        let state = AppContext { state, generator };

        Self {
            ctx: state,
            error: None,
            show_config: false,
            show_help: false,
            active_panel: Box::new(PromptPanel::new()),
        }
    }
}

impl eframe::App for App {
    /// Called by the framework to save state before shutdown.
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, &self.ctx.state);
    }

    /// Handle input and repaint screen.
    fn update(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        ctx.send_viewport_cmd(ViewportCommand::Title(format!(
            "Coze ({})",
            self.ctx.generator.mode().description()
        )));

        match self.ctx.generator.next_message() {
            Some(m @ Message::Token(_, _)) => {
                self.active_panel.message(&mut self.ctx, &m);
            }
            Some(Message::WeightsDownloadBegin) => {
                println!("WeightsDownloadBegin");
            }
            Some(Message::WeightsDownloadProgress(pct)) => {
                println!("WeightsDownloadProgress({pct})");
            }
            Some(Message::WeightsDownloadComplete) => {
                println!("WeightsDownloadComplete");
            }
            Some(Message::Error(msg)) => {
                self.error = Some(msg);
            }
            None => (),
        };

        self.active_panel.process_input(ctx, &mut self.ctx);

        // Render menu
        TopBottomPanel::top("top_panel").show(ctx, |ui| {
            menu::bar(ui, |ui| {
                ui.menu_button("Edit", |ui| {
                    if ui.button("Config").clicked() {
                        self.show_config = true;
                        ui.close_menu();
                    }

                    if ui.button("Clear history").clicked() {
                        self.ctx.state.history.clear();
                        ui.close_menu();
                    }

                    if ui.button("Reload weights").clicked() {
                        self.ctx.generator.reload_weights();
                        ui.close_menu();
                    }
                });

                if ui.button("Help").clicked() {
                    self.show_help = true;
                    ui.close_menu();
                }
            });
        });

        self.active_panel.update(ctx, &mut self.ctx);

        self.config_window(ctx);
        self.error_window(ctx);
        self.help_window(ctx);

        // Run 20 frames per second.
        ctx.request_repaint_after(std::time::Duration::from_millis(50));
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.ctx.generator.shutdown();
    }
}

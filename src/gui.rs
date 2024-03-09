use eframe::egui::*;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use crate::{
    controller::{Controller, Message},
    models::ModelConfig,
};

mod bubble;
mod config;
mod gauge;
mod help;
mod history;
mod load_panel;
mod models_panel;
mod prompt_panel;

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
    model_config: ModelConfig,
    ui_mode: UiMode,
}

#[derive(Deserialize, Serialize, Debug)]
struct Prompt {
    prompt: String,
    reply: String,
    info: String,
}

trait Panel: Debug {
    fn update(&mut self, ctx: &mut AppContext);

    fn handle_input(&mut self, _ctx: &mut AppContext) {}

    fn handle_message(&mut self, _ctx: &mut AppContext, _msg: Message) {}

    fn next_panel(&mut self, _ctx: &mut AppContext) -> Option<Box<dyn Panel>> {
        None
    }

    fn is_start_panel(&mut self) -> bool {
        false
    }
}

#[derive(Debug)]
struct AppContext {
    state: PersistedState,
    controller: Controller,
    egui_ctx: Context,
}

#[derive(Debug)]
pub struct App {
    ctx: AppContext,
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

        let controller = Controller::new(state.model_config);
        let state = AppContext {
            state,
            controller,
            egui_ctx: cc.egui_ctx.clone(),
        };

        Self {
            ctx: state,
            show_config: false,
            show_help: false,
            active_panel: Box::new(models_panel::ModelsPanel::new()),
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
            self.ctx.controller.model_config().description()
        )));

        if let Some(m) = self.ctx.controller.next_message() {
            self.active_panel.handle_message(&mut self.ctx, m);
        };

        self.active_panel.handle_input(&mut self.ctx);

        // Render menu
        TopBottomPanel::top("top_panel").show(ctx, |ui| {
            menu::bar(ui, |ui| {
                if !self.active_panel.is_start_panel() {
                    let arrow = RichText::new("⬅").font(FontId::new(24.0, FontFamily::Monospace));
                    if ui.add(Button::new(arrow).frame(false)).clicked() {
                        self.ctx.controller.stop();
                        self.active_panel = Box::new(models_panel::ModelsPanel::new());
                    }
                }

                ui.menu_button("Edit", |ui| {
                    if ui.button("Config").clicked() {
                        self.show_config = true;
                        ui.close_menu();
                    }

                    if ui.button("Clear history").clicked() {
                        self.ctx.state.history.clear();
                        ui.close_menu();
                    }
                });

                if ui.button("Help").clicked() {
                    self.show_help = true;
                    ui.close_menu();
                }
            });
        });

        self.active_panel.update(&mut self.ctx);

        self.config_window(ctx);
        self.help_window(ctx);

        if let Some(panel) = self.active_panel.next_panel(&mut self.ctx) {
            self.active_panel = panel;
        }

        // Run 20 frames per second.
        ctx.request_repaint_after(std::time::Duration::from_millis(50));
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.ctx.controller.shutdown();
    }
}

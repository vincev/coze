use eframe::egui::*;
use serde::{Deserialize, Serialize};

use crate::generator::{Generator, GeneratorMode, Message, PromptId};
use bubble::{Bubble, BubbleContent};
use history::HistoryNavigator;

mod bubble;
mod config;
mod error;
mod help;
mod history;

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

#[derive(Debug)]
pub struct App {
    prompt: String,
    prompt_field_id: Id,
    last_prompt_id: PromptId,
    state: PersistedState,
    generator: Generator,
    error: Option<String>,
    show_config: bool,
    show_help: bool,
    history: HistoryNavigator,
    ctx: Context,
    frame_counter: usize,
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

        Self {
            prompt_field_id: Id::new("prompt-id"),
            last_prompt_id: PromptId::default(),
            prompt: Default::default(),
            state,
            generator,
            error: None,
            show_config: false,
            show_help: false,
            history: HistoryNavigator::new(),
            ctx: cc.egui_ctx.clone(),
            frame_counter: 0,
        }
    }

    fn send_prompt(&mut self) {
        let prompt = self.prompt.trim();
        if !prompt.is_empty() {
            // Flush tokens from previous prompt
            while self.generator.next_message().is_some() {}

            self.last_prompt_id = self.generator.send_prompt(prompt);
            self.state.history.push(Prompt {
                prompt: prompt.to_owned(),
                reply: Default::default(),
            });
        }

        self.reset_prompt("".to_string());
        self.history.reset(&self.prompt);
    }

    fn reset_prompt(&mut self, prompt: String) {
        self.prompt = prompt;

        let state = text_edit::TextEditState::default();
        state.store(&self.ctx, self.prompt_field_id);
    }

    fn process_input(&mut self) {
        // Stops tokens generation for the current prompt.
        if self
            .ctx
            .input_mut(|i| i.consume_key(Modifiers::NONE, Key::Escape))
        {
            self.generator.stop();
            self.reset_prompt("".to_string());
            self.history.reset(&self.prompt);
        }

        // Manage history
        if self
            .ctx
            .input_mut(|i| i.consume_key(Modifiers::NONE, Key::ArrowUp))
        {
            if let Some(prompt) = self.history.up(&self.state.history) {
                self.reset_prompt(prompt);
            }
        }

        if self
            .ctx
            .input_mut(|i| i.consume_key(Modifiers::NONE, Key::ArrowDown))
        {
            if let Some(prompt) = self.history.down(&self.state.history) {
                self.reset_prompt(prompt);
            }
        }
    }
}

impl eframe::App for App {
    /// Called by the framework to save state before shutdown.
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, &self.state);
    }

    /// Handle input and repaint screen.
    fn update(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        self.frame_counter += 1;
        let mut scroll_to_bottom = false;

        ctx.send_viewport_cmd(ViewportCommand::Title(format!(
            "Coze ({})",
            self.generator.mode().description()
        )));

        match self.generator.next_message() {
            Some(Message::Token(prompt_id, s)) => {
                // Skip tokens from a previous prompt.
                if self.last_prompt_id == prompt_id {
                    if let Some(prompt) = self.state.history.last_mut() {
                        prompt.reply.push_str(&s);
                        scroll_to_bottom = true;
                    }
                }
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

        self.process_input();

        // Render menu
        TopBottomPanel::top("top_panel").show(ctx, |ui| {
            menu::bar(ui, |ui| {
                ui.menu_button("Edit", |ui| {
                    if ui.button("Config").clicked() {
                        self.show_config = true;
                        ui.close_menu();
                    }

                    if ui.button("Clear history").clicked() {
                        self.state.history.clear();
                        ui.close_menu();
                    }

                    if ui.button("Reload weights").clicked() {
                        self.generator.reload_weights();
                        ui.close_menu();
                    }
                });

                if ui.button("Help").clicked() {
                    self.show_help = true;
                    ui.close_menu();
                }
            });
        });

        let prompt_frame = Frame::none()
            .fill(ctx.style().visuals.window_fill)
            .outer_margin(Margin::same(0.0))
            .inner_margin(Margin::same(10.0));

        // Render prompt panel.
        TopBottomPanel::bottom("bottom_panel")
            .show_separator_line(false)
            .frame(prompt_frame)
            .show(ctx, |ui| {
                Frame::group(ui.style())
                    .rounding(Rounding::same(ROUNDING))
                    .fill(self.state.ui_mode.fill_color())
                    .show(ui, |ui| {
                        ctx.memory_mut(|m| m.request_focus(self.prompt_field_id));

                        // Override multiline Enter behavior
                        if ui.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Enter)) {
                            self.send_prompt();
                            scroll_to_bottom = true;
                        }

                        let text = TextEdit::multiline(&mut self.prompt)
                            .id(self.prompt_field_id)
                            .cursor_at_end(true)
                            .font(TEXT_FONT)
                            .frame(false)
                            .margin(Vec2::new(5.0, 5.0))
                            .desired_rows(1)
                            .hint_text("Prompt me! (Enter to send)");

                        let r = ui.add_sized([ui.available_width(), 10.0], text);
                        if r.changed() {
                            self.history.reset(&self.prompt);
                        }
                    })
            });

        // Render message panel.
        CentralPanel::default().show(ctx, |ui| {
            ScrollArea::vertical()
                .auto_shrink(false)
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for prompt in &self.state.history {
                        let r = ui.add(Bubble::new(
                            &prompt.prompt,
                            BubbleContent::Prompt,
                            self.state.ui_mode,
                        ));
                        if r.clicked() {
                            ui.ctx().copy_text(prompt.prompt.clone());
                        }

                        if r.double_clicked() {
                            self.prompt = prompt.prompt.clone();
                            scroll_to_bottom = true;
                        }

                        ui.add_space(ui.spacing().item_spacing.y);

                        if !prompt.reply.is_empty() {
                            let r = ui.add(Bubble::new(
                                &prompt.reply,
                                BubbleContent::Reply,
                                self.state.ui_mode,
                            ));
                            if r.clicked() {
                                ui.ctx().copy_text(prompt.reply.clone());
                            }

                            ui.add_space(ui.spacing().item_spacing.y * 2.5);
                        } else {
                            let dots = ["⏺   ", " ⏺  ", "  ⏺ ", "   ⏺", "  ⏺ ", " ⏺  "];
                            ui.add(Bubble::new(
                                dots[(self.frame_counter / 18) % dots.len()],
                                BubbleContent::Reply,
                                self.state.ui_mode,
                            ));
                            ui.add_space(ui.spacing().item_spacing.y * 2.5);
                        }
                    }

                    if scroll_to_bottom {
                        ui.scroll_to_cursor(Some(Align::BOTTOM));
                    }
                });
            ui.allocate_space(ui.available_size());
        });

        self.config_window();
        self.error_window();
        self.help_window();

        // Run 20 frames per second.
        ctx.request_repaint_after(std::time::Duration::from_millis(50));
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.generator.shutdown();
    }
}

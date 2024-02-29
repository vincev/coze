use eframe::egui::*;

use crate::{
    controller::{Message, PromptId},
    gui::{
        bubble::{Bubble, BubbleContent},
        history::HistoryNavigator,
        AppContext, Panel, Prompt,
    },
};

const TEXT_FONT: FontId = FontId::new(15.0, FontFamily::Monospace);
const ROUNDING: f32 = 8.0;

#[derive(Debug)]
pub struct PromptPanel {
    prompt: String,
    prompt_field_id: Id,
    last_prompt_id: PromptId,
    error: Option<String>,
    history: HistoryNavigator,
    frame_counter: usize,
    scroll_to_bottom: bool,
}

impl PromptPanel {
    pub fn new() -> Self {
        Self {
            prompt_field_id: Id::new("prompt-id"),
            last_prompt_id: PromptId::default(),
            error: None,
            prompt: Default::default(),
            history: HistoryNavigator::new(),
            frame_counter: 0,
            scroll_to_bottom: false,
        }
    }

    fn send_prompt(&mut self, ctx: &mut AppContext) {
        let prompt = self.prompt.trim();
        if !prompt.is_empty() {
            // Flush tokens from previous prompt
            while ctx.controller.next_message().is_some() {}

            self.last_prompt_id = ctx.controller.send_prompt(prompt);
            ctx.state.history.push(Prompt {
                prompt: prompt.to_owned(),
                reply: Default::default(),
            });
        }

        self.reset_prompt(&ctx.egui_ctx, "".to_string());
        self.history.reset(&self.prompt);
    }

    fn reset_prompt(&mut self, ctx: &Context, prompt: String) {
        self.prompt = prompt;

        let state = text_edit::TextEditState::default();
        state.store(ctx, self.prompt_field_id);
    }

    fn error_window(&mut self, ctx: &Context) {
        // Show error window if any.
        if self.error.is_some() {
            Window::new("Error")
                .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.with_layout(Layout::top_down(Align::Center), |ui| {
                        let msg = self.error.as_ref().unwrap();
                        ui.label(RichText::new(msg).font(TEXT_FONT));
                        ui.add_space(ui.spacing().item_spacing.y * 2.5);
                        if ui.button("Close").clicked() {
                            self.error = None;
                        }
                    });
                });
        }
    }
}

impl Panel for PromptPanel {
    fn update(&mut self, ctx: &mut AppContext) {
        self.frame_counter += 1;

        let egui_ctx = ctx.egui_ctx.clone();
        let prompt_frame = Frame::none()
            .fill(ctx.egui_ctx.style().visuals.window_fill)
            .outer_margin(Margin::same(0.0))
            .inner_margin(Margin::same(10.0));

        // Render prompt panel.
        TopBottomPanel::bottom("bottom_panel")
            .show_separator_line(false)
            .frame(prompt_frame)
            .show(&egui_ctx, |ui| {
                Frame::group(ui.style())
                    .rounding(Rounding::same(ROUNDING))
                    .fill(ctx.state.ui_mode.fill_color())
                    .show(ui, |ui| {
                        egui_ctx.memory_mut(|m| m.request_focus(self.prompt_field_id));

                        // Override multiline Enter behavior
                        if ui.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Enter)) {
                            self.send_prompt(ctx);
                            self.scroll_to_bottom = true;
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
        CentralPanel::default().show(&egui_ctx, |ui| {
            ScrollArea::vertical()
                .auto_shrink(false)
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for prompt in &ctx.state.history {
                        let r = ui.add(Bubble::new(
                            &prompt.prompt,
                            BubbleContent::Prompt,
                            ctx.state.ui_mode,
                        ));
                        if r.clicked() {
                            ui.ctx().copy_text(prompt.prompt.clone());
                        }

                        if r.double_clicked() {
                            self.prompt = prompt.prompt.clone();
                            self.scroll_to_bottom = true;
                        }

                        ui.add_space(ui.spacing().item_spacing.y);

                        if !prompt.reply.is_empty() {
                            let r = ui.add(Bubble::new(
                                &prompt.reply,
                                BubbleContent::Reply,
                                ctx.state.ui_mode,
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
                                ctx.state.ui_mode,
                            ));
                            ui.add_space(ui.spacing().item_spacing.y * 2.5);
                        }
                    }

                    if self.scroll_to_bottom {
                        ui.scroll_to_cursor(Some(Align::BOTTOM));
                    }
                });
            ui.allocate_space(ui.available_size());
        });

        self.error_window(&egui_ctx);

        self.scroll_to_bottom = false;
    }

    fn handle_input(&mut self, app: &mut AppContext) {
        if app
            .egui_ctx
            .input_mut(|i| i.consume_key(Modifiers::NONE, Key::Escape))
        {
            app.controller.stop();
            self.reset_prompt(&app.egui_ctx, "".to_string());
            self.history.reset(&self.prompt);
        }

        // Manage history
        if app
            .egui_ctx
            .input_mut(|i| i.consume_key(Modifiers::NONE, Key::ArrowUp))
        {
            if let Some(prompt) = self.history.up(&app.state.history) {
                self.reset_prompt(&app.egui_ctx, prompt);
            }
        }

        if app
            .egui_ctx
            .input_mut(|i| i.consume_key(Modifiers::NONE, Key::ArrowDown))
        {
            if let Some(prompt) = self.history.down(&app.state.history) {
                self.reset_prompt(&app.egui_ctx, prompt);
            }
        }
    }

    fn handle_message(&mut self, app: &mut AppContext, msg: Message) {
        match msg {
            Message::Token(prompt_id, s) => {
                // Skip tokens from a previous prompt.
                if self.last_prompt_id == prompt_id {
                    if let Some(prompt) = app.state.history.last_mut() {
                        prompt.reply.push_str(&s);
                        self.scroll_to_bottom = true;
                    }
                }
            }
            Message::Error(s) => self.error = Some(s),
            _ => {}
        }
    }
}

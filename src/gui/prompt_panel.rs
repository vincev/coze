use super::*;

#[derive(Debug)]
pub struct PromptPanel {
    prompt: String,
    prompt_field_id: Id,
    last_prompt_id: PromptId,
    history: HistoryNavigator,
    frame_counter: usize,
    scroll_to_bottom: bool,
}

impl PromptPanel {
    pub fn new() -> Self {
        Self {
            prompt_field_id: Id::new("prompt-id"),
            last_prompt_id: PromptId::default(),
            prompt: Default::default(),
            history: HistoryNavigator::new(),
            frame_counter: 0,
            scroll_to_bottom: false,
        }
    }

    fn send_prompt(&mut self, ctx: &Context, app: &mut AppContext) {
        let prompt = self.prompt.trim();
        if !prompt.is_empty() {
            // Flush tokens from previous prompt
            while app.generator.next_message().is_some() {}

            self.last_prompt_id = app.generator.send_prompt(prompt);
            app.state.history.push(Prompt {
                prompt: prompt.to_owned(),
                reply: Default::default(),
            });
        }

        self.reset_prompt(ctx, "".to_string());
        self.history.reset(&self.prompt);
    }

    fn reset_prompt(&mut self, ctx: &Context, prompt: String) {
        self.prompt = prompt;

        let state = text_edit::TextEditState::default();
        state.store(ctx, self.prompt_field_id);
    }
}

impl Panel for PromptPanel {
    fn update(&mut self, ctx: &Context, app: &mut AppContext) {
        self.frame_counter += 1;

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
                    .fill(app.state.ui_mode.fill_color())
                    .show(ui, |ui| {
                        ctx.memory_mut(|m| m.request_focus(self.prompt_field_id));

                        // Override multiline Enter behavior
                        if ui.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Enter)) {
                            self.send_prompt(ctx, app);
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
        CentralPanel::default().show(ctx, |ui| {
            ScrollArea::vertical()
                .auto_shrink(false)
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for prompt in &app.state.history {
                        let r = ui.add(Bubble::new(
                            &prompt.prompt,
                            BubbleContent::Prompt,
                            app.state.ui_mode,
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
                                app.state.ui_mode,
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
                                app.state.ui_mode,
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

        self.scroll_to_bottom = false;
    }

    fn process_input(&mut self, ctx: &Context, app: &mut AppContext) {
        if ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Escape)) {
            app.generator.stop();
            self.reset_prompt(ctx, "".to_string());
            self.history.reset(&self.prompt);
        }

        // Manage history
        if ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::ArrowUp)) {
            if let Some(prompt) = self.history.up(&app.state.history) {
                self.reset_prompt(ctx, prompt);
            }
        }

        if ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::ArrowDown)) {
            if let Some(prompt) = self.history.down(&app.state.history) {
                self.reset_prompt(ctx, prompt);
            }
        }
    }

    fn message(&mut self, app: &mut AppContext, msg: &Message) {
        if let Message::Token(prompt_id, s) = msg {
            // Skip tokens from a previous prompt.
            if self.last_prompt_id == *prompt_id {
                if let Some(prompt) = app.state.history.last_mut() {
                    prompt.reply.push_str(s);
                    self.scroll_to_bottom = true;
                }
            }
        }
    }
}

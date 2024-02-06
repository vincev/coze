use egui::*;

use crate::generator::{Generator, Message};

const TEXT_FONT: FontId = FontId::new(15.0, FontFamily::Monospace);
const ROUNDING: f32 = 8.0;

#[derive(Debug)]
pub struct App {
    prompt: String,
    prompt_id: Id,
    history: History,
    generator: Generator,
}

#[derive(serde::Deserialize, serde::Serialize, Default, Debug)]
#[serde(default)]
struct History {
    prompts: Vec<Prompt>,
}

#[derive(serde::Deserialize, serde::Serialize, Debug)]
struct Prompt {
    prompt: String,
    reply: String,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // let model = generator::Generator::new(None, None, 1.1, 64).unwrap();

        let history = if let Some(storage) = cc.storage {
            // Load previous app state (if any).
            eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default()
        } else {
            Default::default()
        };

        Self {
            prompt_id: Id::new("prompt-id"),
            prompt: Default::default(),
            history,
            generator: Generator::new(Default::default()),
        }
    }

    fn send_prompt(&mut self) {
        let prompt = self.prompt.trim();
        if !prompt.is_empty() {
            self.generator.send_prompt(prompt);
            self.history.prompts.push(Prompt {
                prompt: prompt.to_owned(),
                reply: Default::default(),
            });
        }

        self.prompt.clear();
    }
}

impl eframe::App for App {
    /// Called by the framework to save state before shutdown.
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, &self.history);
    }

    /// Handle input and repaint screen.
    fn update(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        match self.generator.next_message() {
            Some(Message::Token(s)) => {
                if let Some(prompt) = self.history.prompts.last_mut() {
                    prompt.reply.push_str(&s);
                }
            }
            Some(Message::Error(s)) => todo!(),
            None => (),
        };

        ctx.memory_mut(|m| m.request_focus(self.prompt_id));

        TopBottomPanel::top("top_panel").show(ctx, |ui| {
            menu::bar(ui, |ui| {
                ui.menu_button("Edit", |ui| {
                    if ui.button("Clear history").clicked() {
                        self.history.prompts.clear();
                        ui.close_menu();
                    }
                });
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
                    .fill(Color32::from_gray(230))
                    .show(ui, |ui| {
                        let text = TextEdit::multiline(&mut self.prompt)
                            .id(self.prompt_id)
                            .font(TEXT_FONT)
                            .frame(false)
                            .margin(Vec2::new(5.0, 5.0))
                            .desired_rows(1)
                            .hint_text("Prompt me! (Enter to send)");

                        ui.add_sized([ui.available_width(), 10.0], text);
                        // Override multiline Enter behavior
                        if ui.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Enter)) {
                            self.send_prompt();
                        }
                    })
            });

        // Render message panel.
        CentralPanel::default().show(ctx, |ui| {
            ScrollArea::vertical()
                .auto_shrink(false)
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for prompt in &self.history.prompts {
                        ui.add(Bubble::new(&prompt.prompt, BubbleContent::Prompt));
                        if !prompt.reply.is_empty() {
                            ui.add(Bubble::new(&prompt.reply, BubbleContent::Reply));
                            ui.add_space(ui.spacing().item_spacing.y * 2.0);
                        }
                    }
                });
            ui.allocate_space(ui.available_size());
        });

        // Run 20 frames per second.
        ctx.request_repaint_after(std::time::Duration::from_millis(50));
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.generator.shutdown();
    }
}

enum BubbleContent {
    Prompt,
    Reply,
}

struct Bubble {
    text: WidgetText,
    content: BubbleContent,
}

impl Bubble {
    fn new(text: &str, content: BubbleContent) -> Self {
        let text = WidgetText::from(RichText::new(text).font(TEXT_FONT).monospace());
        Self { text, content }
    }
}

impl Widget for Bubble {
    fn ui(self, ui: &mut Ui) -> Response {
        const PADDING: f32 = 10.0;
        const WIDTH_PCT: f32 = 0.80;

        let Bubble {
            text,
            content: bubble_type,
        } = self;

        let text_wrap_width = ui.available_width() * WIDTH_PCT - 2.0 * PADDING;
        let galley = text.into_galley(ui, Some(true), text_wrap_width, TextStyle::Monospace);

        let desired_size = galley.size() + Vec2::splat(2.0 * PADDING);
        let (rect, response) =
            ui.allocate_at_least(desired_size, Sense::focusable_noninteractive());

        let paint_rect = if matches!(bubble_type, BubbleContent::Prompt) {
            // Move prompt to the right
            let dx = ui.available_width() - desired_size.x;
            rect.translate(Vec2::new(dx, 0.0))
        } else {
            rect
        };

        if ui.is_rect_visible(rect) {
            let fill_color = if matches!(bubble_type, BubbleContent::Prompt) {
                Color32::from_rgb(15, 85, 235)
            } else {
                Color32::from_gray(230)
            };

            let text_color = if matches!(bubble_type, BubbleContent::Prompt) {
                Color32::from_rgb(210, 225, 250)
            } else {
                Color32::from_gray(55)
            };

            ui.painter().rect(
                paint_rect,
                Rounding::same(ROUNDING),
                fill_color,
                Stroke::default(),
            );

            let text_pos = ui
                .layout()
                .align_size_within_rect(galley.size(), paint_rect.shrink2(Vec2::splat(PADDING)))
                .min;

            ui.painter().galley(text_pos, galley, text_color);
        }

        response
    }
}

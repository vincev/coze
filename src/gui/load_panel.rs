use super::*;

const TEXT_FONT: FontId = FontId::new(20.0, FontFamily::Monospace);
const PROGRESS_FONT: FontId = FontId::new(28.0, FontFamily::Monospace);

#[derive(Debug)]
pub struct LoadPanel {
    load_pct: f32,
    connecting: bool,
    model_name: String,
    error: Option<String>,
    complete: bool,
    frame_counter: usize,
}

impl LoadPanel {
    pub fn new() -> Self {
        Self {
            load_pct: 0.0,
            connecting: false,
            model_name: Default::default(),
            error: None,
            complete: false,
            frame_counter: 0,
        }
    }
}

impl Panel for LoadPanel {
    fn update(&mut self, ctx: &Context, app: &mut AppContext) {
        const INFO_COLOR: Color32 = Color32::from_rgb(20, 140, 255);

        self.frame_counter += 1;

        CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.label(RichText::new("Loading Model").font(TEXT_FONT));
                ui.label(RichText::new(&self.model_name).font(TEXT_FONT));

                ui.add_space(ui.spacing().item_spacing.y * 2.5);

                if self.connecting {
                    ui.add_space(ui.spacing().item_spacing.y * 10.0);
                    ui.label(
                        RichText::new("Connecting to Hugging Face")
                            .font(TEXT_FONT)
                            .color(INFO_COLOR),
                    );

                    ui.add_space(ui.spacing().item_spacing.y * 5.0);

                    let pos = ((self.frame_counter / 5) % 100) as f32 / 100.0;
                    let pos = ((std::f32::consts::TAU * pos).sin() + 1.0) / 2.0;
                    ui.horizontal(|ui| {
                        let w = ui.available_width();
                        ui.add_space(w * 0.1 + w * 0.7 * pos);
                        ui.label(RichText::new("⏺").font(PROGRESS_FONT).color(INFO_COLOR));
                    });
                } else {
                    let width = ui.available_width() * 0.9;
                    ui.add(
                        gauge::Gauge::new(self.load_pct)
                            .color(INFO_COLOR)
                            .width(width),
                    );
                }

                if let Some(error) = &self.error {
                    let error_color = Color32::LIGHT_RED;
                    ui.add_space(ui.spacing().item_spacing.y * 2.5);
                    ui.label(
                        RichText::new("Error loading model:")
                            .color(error_color)
                            .font(TEXT_FONT),
                    );

                    ui.label(
                        RichText::new(error)
                            .color(error_color)
                            .font(FontId::new(14.0, FontFamily::Monospace)),
                    );

                    ui.add_space(ui.spacing().item_spacing.y * 2.5);

                    let button = Button::new(
                        RichText::new("Try Reload").font(FontId::new(14.0, FontFamily::Monospace)),
                    )
                    .rounding(4.0);

                    if ui.add(button).clicked() {
                        app.generator.reload_weights();
                        self.error = None;
                    }
                }
            });
        });
    }

    fn handle_input(&mut self, _ctx: &Context, _app: &mut AppContext) {}

    fn handle_message(&mut self, _app: &mut AppContext, msg: Message) {
        match msg {
            Message::WeightsDownloadBegin(s) => self.model_name = s,
            Message::WeightsDownloadConnecting => self.connecting = true,
            Message::WeightsDownloadProgress(pct) => {
                self.connecting = false;
                self.load_pct = pct;
            }
            Message::WeightsDownloadComplete => self.complete = true,
            Message::Error(s) => self.error = Some(s),
            _ => {}
        }
    }

    fn next_panel(&mut self) -> Option<Box<dyn Panel>> {
        if self.complete {
            Some(Box::new(prompt_panel::PromptPanel::new()))
        } else {
            None
        }
    }
}

use super::*;

const TEXT_FONT: FontId = FontId::new(20.0, FontFamily::Monospace);

#[derive(Debug)]
pub struct LoadPanel {
    load_pct: f32,
    model_name: String,
    error: Option<String>,
    complete: bool,
}

impl LoadPanel {
    pub fn new() -> Self {
        Self {
            load_pct: 0.0,
            model_name: Default::default(),
            error: None,
            complete: false,
        }
    }
}

impl Panel for LoadPanel {
    fn update(&mut self, ctx: &Context, app: &mut AppContext) {
        const INFO_COLOR: Color32 = Color32::from_rgb(20, 140, 255);

        CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.label(RichText::new("Loading Model").font(TEXT_FONT));
                ui.label(RichText::new(&self.model_name).font(TEXT_FONT));

                ui.add_space(ui.spacing().item_spacing.y * 2.5);

                let width = ui.available_width() * 0.9;
                ui.add(
                    gauge::Gauge::new(self.load_pct)
                        .color(INFO_COLOR)
                        .width(width),
                );
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
            Message::WeightsDownloadProgress(pct) => self.load_pct = pct,
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

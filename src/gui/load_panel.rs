use eframe::egui::*;

use crate::{
    controller::Message,
    gui::{gauge::Gauge, AppContext, Panel},
    models::ModelId,
};

const TEXT_FONT: FontId = FontId::new(20.0, FontFamily::Monospace);
const PROGRESS_FONT: FontId = FontId::new(28.0, FontFamily::Monospace);

#[derive(Debug)]
pub struct LoadPanel {
    load_pct: f32,
    connecting: bool,
    download_msg: String,
    error: Option<String>,
    complete: bool,
    frame_counter: usize,
    model_name: String,
    model_id: ModelId,
}

impl LoadPanel {
    pub fn new(model_id: ModelId, ctx: &mut AppContext) -> Self {
        ctx.controller.load_model(model_id);

        Self {
            load_pct: 0.0,
            connecting: false,
            download_msg: Default::default(),
            error: None,
            complete: false,
            frame_counter: 0,
            model_name: model_id.spec().name.to_string(),
            model_id,
        }
    }
}

impl Panel for LoadPanel {
    fn update(&mut self, ctx: &mut AppContext) {
        const INFO_COLOR: Color32 = Color32::from_rgb(20, 140, 255);

        self.frame_counter += 1;

        CentralPanel::default().show(&ctx.egui_ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.label(RichText::new(&self.model_name).font(TEXT_FONT));
                ui.label(RichText::new(&self.download_msg).font(TEXT_FONT));

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
                    ui.add(Gauge::new(self.load_pct).color(INFO_COLOR).width(width));
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
                        ctx.controller.reload_weights(self.model_id);
                        self.error = None;
                    }
                }
            });
        });
    }

    fn handle_message(&mut self, _ctx: &mut AppContext, msg: Message) {
        match msg {
            Message::DownloadBegin(s) => self.download_msg = s,
            Message::DownloadConnecting => self.connecting = true,
            Message::DownloadProgress(pct) => {
                self.connecting = false;
                self.load_pct = pct;
            }
            Message::DownloadComplete => self.complete = true,
            Message::Error(s) => self.error = Some(s),
            _ => {}
        }
    }

    fn next_panel(&mut self, ctx: &mut AppContext) -> Option<Box<dyn Panel>> {
        // if self.complete {
        //     Some(Box::new(prompt_panel::PromptPanel::new()))
        // } else {
        None
        // }
    }
}

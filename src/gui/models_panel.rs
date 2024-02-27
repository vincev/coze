use eframe::egui::*;

use crate::{
    gui::{load_panel::LoadPanel, AppContext, Panel},
    models::{list_models, ModelId, ModelSpecs},
};

const ROUNDING: f32 = 8.0;

#[derive(Debug)]
pub struct ModelsPanel {
    selected: Option<ModelId>,
}

impl ModelsPanel {
    pub fn new() -> Self {
        Self { selected: None }
    }
}

impl Panel for ModelsPanel {
    fn update(&mut self, ctx: &mut AppContext) {
        CentralPanel::default().show(&ctx.egui_ctx, |ui| {
            ScrollArea::vertical()
                .auto_shrink(false)
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    let width = ui.available_width();
                    for model_id in list_models() {
                        let spec = model_id.specs();
                        let r = ui.add(
                            Button::new(spec_to_text(ui, spec))
                                .rounding(ROUNDING)
                                .wrap(true)
                                .min_size(Vec2::new(width, 120.0)),
                        );
                        if r.clicked() {
                            self.selected = Some(model_id);
                        }
                    }
                })
        });
    }

    fn next_panel(&mut self, ctx: &mut AppContext) -> Option<Box<dyn Panel>> {
        if let Some(model_id) = self.selected {
            Some(Box::new(LoadPanel::new(model_id, ctx)))
        } else {
            None
        }
    }

    fn is_start_panel(&mut self) -> bool {
        true
    }
}

fn spec_to_text(ui: &Ui, spec: ModelSpecs) -> text::LayoutJob {
    const PADDING: f32 = 10.0;

    let mut job = text::LayoutJob::default();

    let font_id = FontId::new(22.0, FontFamily::Monospace);
    job.append(
        spec.name,
        PADDING,
        TextFormat {
            font_id: font_id.clone(),
            color: ui.visuals().text_color(),
            ..Default::default()
        },
    );

    job.append(
        "\n\n",
        PADDING,
        TextFormat {
            font_id,
            color: ui.visuals().text_color(),
            ..Default::default()
        },
    );

    let font_id = FontId::new(18.0, FontFamily::Monospace);

    job.append(
        &format!("Size: {}G", spec.size / (1 << 20)),
        PADDING,
        TextFormat {
            font_id: font_id.clone(),
            color: ui.visuals().text_color(),
            ..Default::default()
        },
    );

    if spec.cached {
        job.append(
            "(Cached)",
            PADDING,
            TextFormat {
                font_id,
                color: ui.visuals().text_color(),
                ..Default::default()
            },
        );
    }

    job
}

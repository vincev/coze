use eframe::egui::*;

use crate::{
    gui::{load_panel::LoadPanel, AppContext, Panel},
    models::{ModelId, ModelSpec, ModelsCache},
};

const ROUNDING: f32 = 8.0;

#[derive(Debug)]
pub struct ModelsPanel {
    selected: Option<ModelId>,
    models: Vec<ModelData>,
}

impl ModelsPanel {
    pub fn new() -> Self {
        let models = ModelId::models()
            .into_iter()
            .map(|model_id| {
                let spec = model_id.spec();
                // Checks if this model is cached on disk, this is done once at
                // construction time to avoid accessing the disk at every frame.
                let cached = ModelsCache::new()
                    .map(|c| c.cached_model(model_id).is_cached())
                    .unwrap_or(false);
                ModelData { spec, cached }
            })
            .collect();

        Self {
            selected: None,
            models,
        }
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
                    for model in &self.models {
                        let r = ui.add(model.button(ui).min_size(Vec2::new(width, 120.0)));
                        if r.clicked() {
                            self.selected = Some(model.spec.model_id);
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

#[derive(Debug)]
struct ModelData {
    spec: ModelSpec,
    cached: bool,
}

impl ModelData {
    fn button(&self, ui: &Ui) -> Button<'_> {
        const PADDING: f32 = 10.0;

        let mut job = text::LayoutJob::default();

        let font_id = FontId::new(22.0, FontFamily::Monospace);
        job.append(
            self.spec.name,
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
            &format!("Size: {}M", self.spec.size / (1 << 20)),
            PADDING,
            TextFormat {
                font_id: font_id.clone(),
                color: ui.visuals().text_color(),
                ..Default::default()
            },
        );

        if self.cached {
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

        Button::new(job).rounding(ROUNDING).wrap(true)
    }
}

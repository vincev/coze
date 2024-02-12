use crate::generator::GeneratorMode;

use super::*;

impl App {
    pub fn config_window(&mut self) {
        // Show config dialog.
        if self.show_config {
            Window::new("Config")
                .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
                .max_width(200.0)
                .collapsible(false)
                .resizable(false)
                .show(&self.ctx, |ui| {
                    Grid::new("TextLayoutDemo")
                        .num_columns(2)
                        .spacing([20.0, 4.0])
                        .show(ui, |ui| {
                            ui.label("Generator mode: ");
                            ComboBox::from_id_source("gm")
                                .selected_text(self.state.generator_mode.description())
                                .show_ui(ui, |ui| {
                                    ui.style_mut().wrap = Some(false);
                                    ui.set_min_width(60.0);
                                    ui.selectable_value(
                                        &mut self.state.generator_mode,
                                        GeneratorMode::Careful,
                                        GeneratorMode::Careful.description(),
                                    );
                                    ui.selectable_value(
                                        &mut self.state.generator_mode,
                                        GeneratorMode::Creative,
                                        GeneratorMode::Creative.description(),
                                    );
                                    ui.selectable_value(
                                        &mut self.state.generator_mode,
                                        GeneratorMode::Deranged,
                                        GeneratorMode::Deranged.description(),
                                    );
                                });
                            ui.end_row();

                            ui.label("Ui mode: ");
                            ComboBox::from_id_source("um")
                                .selected_text(self.state.ui_mode.description())
                                .show_ui(ui, |ui| {
                                    ui.style_mut().wrap = Some(false);
                                    ui.set_min_width(60.0);
                                    ui.selectable_value(
                                        &mut self.state.ui_mode,
                                        UiMode::Light,
                                        UiMode::Light.description(),
                                    );
                                    ui.selectable_value(
                                        &mut self.state.ui_mode,
                                        UiMode::Dark,
                                        UiMode::Dark.description(),
                                    );
                                });
                            self.ctx.set_visuals(self.state.ui_mode.visuals());
                            ui.end_row();
                        });

                    ui.separator();

                    ui.vertical_centered(|ui| {
                        if ui.button("Close").clicked() {
                            self.generator.set_config(self.state.generator_mode);
                            self.show_config = false;
                        }
                    });
                });
        }
    }
}

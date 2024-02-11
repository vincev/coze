use crate::generator::ConfigValue;

use super::*;

impl App {
    pub fn config_window(&mut self) {
        // Show config dialog.
        if self.config.is_some() {
            Window::new("Config")
                .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
                .collapsible(false)
                .resizable(false)
                .show(&self.ctx, |ui| {
                    ui.with_layout(Layout::top_down(Align::Center), |ui| {
                        let mut config = self.config.take().unwrap();
                        ui.horizontal(|ui| {
                            ui.label("Generator mode: ");
                            ComboBox::from_label("")
                                .selected_text(config.description())
                                .show_ui(ui, |ui| {
                                    ui.style_mut().wrap = Some(false);
                                    ui.set_min_width(60.0);
                                    ui.selectable_value(
                                        &mut config,
                                        ConfigValue::Careful,
                                        ConfigValue::Careful.description(),
                                    );
                                    ui.selectable_value(
                                        &mut config,
                                        ConfigValue::Creative,
                                        ConfigValue::Creative.description(),
                                    );
                                    ui.selectable_value(
                                        &mut config,
                                        ConfigValue::Deranged,
                                        ConfigValue::Deranged.description(),
                                    );
                                });
                        });

                        self.config = Some(config);

                        ui.add_space(ui.spacing().item_spacing.y * 2.5);
                        if ui.button("Close").clicked() {
                            self.state.config = config;
                            self.generator.set_config(config);
                            self.config = None;
                        }
                    });
                });
        }
    }
}

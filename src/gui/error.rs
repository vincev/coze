use super::*;

impl App {
    pub fn error_window(&mut self) {
        // Show error window if any.
        if self.error.is_some() {
            Window::new("Error")
                .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
                .collapsible(false)
                .resizable(false)
                .show(&self.ctx, |ui| {
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

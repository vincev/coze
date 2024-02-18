use super::*;

const HELP_TEXT: &str = "# Prompt field

Enter a prompt and press return to generate reply tokens. The prompts appear as
blue bubbles in the history area while the replies as gray bubbles.

Press Escape at any time to stop the replies generation and clear the prompt field.

Click on any bubble to copy its text to the clipboard, double click on a prompt
bubble to copy its text to the prompt field.

Use the up and down arrows to navigate the prompt history, if the prompt field
contains some text it is used to filter the history using fuzzy matching.

# Edit menu

The `Config` menu item shows a dialog with two combo boxes, one for choosing the
token generation randomness and the other for choosing the UI light mode.

The `Clear history` menu item removes all the prompts and replies from the history
area.

The history and window position is saved using the `egui` storage system.";

impl App {
    pub fn help_window(&mut self, ctx: &Context) {
        if self.show_help {
            let ui_rect = ctx.used_rect();

            Window::new("Help")
                .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
                .min_width(ui_rect.width() * 0.5)
                .max_height(ui_rect.height() * 0.8)
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ScrollArea::vertical()
                        .max_height(ui_rect.height() * 0.8)
                        .show(ui, |ui| {
                            render_text(ui);
                        });

                    ui.vertical_centered(|ui| {
                        ui.add_space(ui.spacing().item_spacing.y * 2.0);
                        if ui.button("Close").clicked() {
                            self.show_help = false;
                        }
                    });
                });
        }
    }
}

fn render_text(ui: &mut Ui) {
    let row_height = ui.text_style_height(&TextStyle::Body);

    let layout = Layout::top_down(Align::LEFT);
    ui.with_layout(layout, |ui| {
        for line in HELP_TEXT.split("\n\n") {
            let line = line.replace('\n', " ");
            let rich_text = if line.starts_with('#') {
                RichText::new(line.trim_start_matches(['#', ' ']))
                    .font(FontId::monospace(18.0))
                    .heading()
                    .strong()
            } else {
                RichText::new(&line).font(TEXT_FONT)
            };

            ui.add(Label::new(rich_text).wrap(true));
            ui.add_space(row_height);
        }
    });
}

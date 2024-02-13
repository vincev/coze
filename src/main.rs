#![warn(clippy::all, rust_2018_idioms)]
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() -> eframe::Result<()> {
    const INIT_SIZE: [f32; 2] = [400.0, 600.0];
    let native_options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size(INIT_SIZE)
            .with_min_inner_size(INIT_SIZE)
            .with_title("Coze"),
        ..Default::default()
    };
    eframe::run_native(
        "coze",
        native_options,
        Box::new(|cc| Box::new(coze::App::new(cc))),
    )
}

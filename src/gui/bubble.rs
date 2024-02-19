use eframe::egui::*;

use super::UiMode;

const TEXT_FONT: FontId = FontId::new(15.0, FontFamily::Monospace);
const ROUNDING: f32 = 8.0;

pub enum BubbleContent {
    Prompt,
    Reply,
}

pub struct Bubble {
    text: WidgetText,
    content: BubbleContent,
    ui_mode: UiMode,
}

impl Bubble {
    pub fn new(text: &str, content: BubbleContent, ui_mode: UiMode) -> Self {
        let text = WidgetText::from(RichText::new(text).font(TEXT_FONT).monospace());
        Self {
            text,
            content,
            ui_mode,
        }
    }

    fn fill_color(content: &BubbleContent, ui_mode: UiMode) -> Color32 {
        match content {
            BubbleContent::Prompt => Color32::from_rgb(15, 85, 235),
            BubbleContent::Reply => ui_mode.fill_color(),
        }
    }

    fn text_color(content: &BubbleContent, ui_mode: UiMode) -> Color32 {
        match content {
            BubbleContent::Prompt => Color32::from_rgb(210, 225, 250),
            BubbleContent::Reply => match ui_mode {
                UiMode::Light => Color32::from_gray(60),
                UiMode::Dark => Color32::from_gray(180),
            },
        }
    }
}

impl Widget for Bubble {
    fn ui(self, ui: &mut Ui) -> Response {
        const PADDING: f32 = 10.0;
        const WIDTH_PCT: f32 = 0.9;

        let Bubble {
            text,
            content,
            ui_mode,
        } = self;

        let text_wrap_width = ui.available_width() * WIDTH_PCT - 2.0 * PADDING;
        let galley = text.into_galley(ui, Some(true), text_wrap_width, TextStyle::Monospace);
        let bubble_size = galley.size() + Vec2::splat(2.0 * PADDING);

        let desired_size = Vec2::new(ui.available_width(), bubble_size.y);
        let (rect, response) = ui.allocate_at_least(desired_size, Sense::click());

        let dx = ui.available_width() - bubble_size.x;
        let paint_rect = if matches!(content, BubbleContent::Prompt) {
            // Move prompt to the right
            Rect::from_min_max(Pos2::new(rect.min.x + dx, rect.min.y), rect.max)
        } else {
            Rect::from_min_max(rect.min, Pos2::new(rect.max.x - dx, rect.max.y))
        };

        if ui.is_rect_visible(rect) {
            let fill_color = Self::fill_color(&content, ui_mode);
            let text_color = Self::text_color(&content, ui_mode);

            // On click expand animation.
            let expand = ui
                .ctx()
                .animate_value_with_time(response.id, std::f32::consts::PI, 0.5)
                .sin()
                * ui.spacing().item_spacing.y;
            let paint_rect = paint_rect.expand(expand);

            if response.clicked() {
                ui.ctx().clear_animations();
                ui.ctx().animate_value_with_time(response.id, 0.0, 0.5);
            }

            ui.painter().rect(
                paint_rect,
                Rounding::same(ROUNDING),
                fill_color,
                Stroke::default(),
            );

            let text_pos = ui
                .layout()
                .align_size_within_rect(
                    galley.size(),
                    paint_rect.shrink2(Vec2::splat(PADDING + expand)),
                )
                .min;

            ui.painter().galley(text_pos, galley, text_color);
        }

        response
    }
}

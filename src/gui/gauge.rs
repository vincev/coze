use eframe::{egui::*, epaint::*};

const ASPECT: f32 = 1.5;

pub struct Gauge {
    progress: f32,
    min_size: Vec2,
    color: Color32,
    font_size: f32,
}

impl Gauge {
    pub fn new(progress: f32) -> Self {
        Self {
            progress,
            min_size: Vec2::new(64.0, 64.0 / ASPECT),
            color: Color32::from_rgb(15, 85, 235),
            font_size: 48.0,
        }
    }

    pub fn width(mut self, width: f32) -> Self {
        self.min_size = Vec2::new(width, width / ASPECT);
        self
    }

    pub fn color(mut self, color: Color32) -> Self {
        self.color = color;
        self
    }
}

impl Widget for Gauge {
    fn ui(self, ui: &mut Ui) -> Response {
        use std::f32::consts;

        const PADDING: f32 = 10.0;
        const GAUGE_ANGLE: f32 = consts::FRAC_PI_6 * 8.0;
        const THETA_START: f32 = consts::FRAC_PI_6 * 7.0;
        const THETA_END: f32 = consts::FRAC_PI_6 * 11.0;

        let Gauge {
            progress,
            min_size,
            color,
            font_size,
        } = self;

        let (rect, response) = ui.allocate_at_least(min_size, Sense::click());

        if ui.is_rect_visible(rect) {
            let rect = rect.shrink(PADDING);
            let painter = ui.painter().with_clip_rect(rect);

            let mut center = rect.center();
            center.y = rect.top() + rect.height() * 2.0 / 3.0;

            // Paint full gauge.
            let outer_r = rect.height() * 2.0 / 3.0;
            let inner_r = outer_r * 0.75;
            painter.circle(center, outer_r, color, Stroke::NONE);

            let bg_color = ui.visuals().window_fill;
            painter.circle(center, inner_r, bg_color, Stroke::NONE);

            // Mask the beginning and end of the gauge.
            let base_mask = vec![
                center,
                Pos2::new(
                    center.x + outer_r * THETA_END.cos() * 1.1,
                    center.y - outer_r * THETA_END.sin() * 1.1,
                ),
                Pos2::new(
                    center.x + outer_r * THETA_START.cos() * 1.1,
                    center.y - outer_r * THETA_START.sin() * 1.1,
                ),
            ];

            painter.add(PathShape::convex_polygon(base_mask, bg_color, Stroke::NONE));

            // Mask remaining progress.
            let progress_angle = GAUGE_ANGLE * progress;
            let mut progress_mask = vec![center];
            for step in 0..12 {
                let theta = progress_angle + (step as f32) * consts::FRAC_PI_8;
                if theta > GAUGE_ANGLE + consts::FRAC_PI_8 {
                    break;
                }

                let theta = theta - consts::FRAC_PI_8;
                progress_mask.push(Pos2::new(
                    center.x - outer_r * theta.cos() * 1.1,
                    center.y - outer_r * theta.sin() * 1.1,
                ));
            }

            let shape = PathShape::convex_polygon(progress_mask, bg_color, Stroke::NONE);
            painter.add(shape);

            painter.text(
                center,
                Align2::CENTER_CENTER,
                format!("{:.0}%", progress * 100.0),
                FontId::monospace(font_size),
                color,
            );
        }

        response
    }
}

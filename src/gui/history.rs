use super::Prompt;

#[derive(Debug)]
pub struct HistoryNavigator {
    pattern: String,
    cursor: usize,
}

impl HistoryNavigator {
    pub fn new() -> Self {
        Self {
            pattern: Default::default(),
            cursor: usize::MAX,
        }
    }

    pub fn reset(&mut self, pattern: &str) {
        self.pattern = pattern.to_lowercase();
        self.cursor = usize::MAX;
    }

    pub fn up(&mut self, history: &[Prompt]) -> Option<String> {
        if history.is_empty() {
            return None;
        }

        let mut cursor = self.cursor.min(history.len());

        loop {
            cursor = cursor.saturating_sub(1);
            if let Some(prompt) = history.get(cursor) {
                if self.is_match(history, &prompt.prompt) {
                    self.cursor = cursor;
                    return Some(prompt.prompt.clone());
                }
            }

            if cursor == 0 {
                return None;
            }
        }
    }

    pub fn down(&mut self, history: &[Prompt]) -> Option<String> {
        if history.is_empty() {
            return None;
        }

        let mut cursor = self.cursor.min(history.len() - 1);

        loop {
            cursor = cursor.saturating_add(1);
            if let Some(prompt) = history.get(cursor) {
                if self.is_match(history, &prompt.prompt) {
                    self.cursor = cursor;
                    return Some(prompt.prompt.clone());
                }
            } else {
                return None;
            }
        }
    }

    fn is_match(&self, history: &[Prompt], text: &str) -> bool {
        // Skip repeated prompts.
        let match_current = history
            .get(self.cursor)
            .map(|p| text.eq_ignore_ascii_case(&p.prompt))
            .unwrap_or_default();

        if match_current {
            return false;
        }

        let mut pit = self.pattern.chars().peekable();

        for c in text.chars() {
            if let Some(p) = pit.peek() {
                if p.eq_ignore_ascii_case(&c) {
                    pit.next();
                }
            } else {
                break;
            }
        }

        pit.peek().is_none()
    }
}

use super::logging::LogEntry;
use log::Level;
use numfmt::{Formatter, Precision, Scales};
use ratatui::Frame;
use ratatui::buffer::CellWidth;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line as TextLine, Span};
use ratatui::widgets::{Block, Gauge, Paragraph};
use sparrow::config::{ShrinkDecayStrategy, SparrowConfig};
use sparrow::util::listener::{
    OptimizationPhase, ReportType, SeparationProgress, SeparationResult,
};
use std::time::{Duration, Instant};

pub(super) enum DashboardUpdate {
    Solution {
        report: ReportType,
        width: f32,
        density: f32,
    },
    Phase(OptimizationPhase),
    Separation(SeparationProgress),
    SeparationResult(SeparationResult),
    Compression(f32),
}

#[derive(Clone, Copy)]
pub(super) struct SearchBudget {
    total_duration: Duration,
    max_attempts: Option<usize>,
    shrink_range: Option<(f32, f32)>,
    exploration_boundary: Option<f64>,
}

impl SearchBudget {
    pub(super) fn from_config(config: &SparrowConfig) -> Self {
        let total_duration = config.expl_cfg.time_limit + config.cmpr_cfg.time_limit;
        Self {
            total_duration,
            max_attempts: config.expl_cfg.max_conseq_failed_attempts,
            shrink_range: matches!(
                config.cmpr_cfg.shrink_decay,
                ShrinkDecayStrategy::FailureBased(_)
            )
            .then_some(config.cmpr_cfg.shrink_range),
            exploration_boundary: (config.expl_cfg.max_conseq_failed_attempts.is_none()
                && !total_duration.is_zero())
            .then(|| config.expl_cfg.time_limit.as_secs_f64() / total_duration.as_secs_f64()),
        }
    }
}

pub(super) struct Dashboard {
    report: Option<ReportType>,
    phase: &'static str,
    width: Option<f32>,
    density: Option<f32>,
    separation_width: Option<f32>,
    attempt: usize,
    iteration: usize,
    attempt_initial_loss: Option<f32>,
    loss_remaining: Option<f32>,
    last_separation_result: Option<(SeparationResult, Instant)>,
    shrink_step: Option<f32>,
    started: Instant,
    budget: SearchBudget,
    logs: Vec<LogEntry>,
    log_scroll: usize,
    log_view_height: usize,
    log_view_width: u16,
    finished: bool,
    finished_elapsed: Option<Duration>,
    quit_requested: bool,
}

impl Dashboard {
    pub(super) fn new(budget: SearchBudget) -> Self {
        Self {
            report: None,
            phase: "starting",
            width: None,
            density: None,
            separation_width: None,
            attempt: 0,
            iteration: 0,
            attempt_initial_loss: None,
            loss_remaining: None,
            last_separation_result: None,
            shrink_step: None,
            started: Instant::now(),
            budget,
            logs: Vec::new(),
            log_scroll: 0,
            log_view_height: 1,
            log_view_width: u16::MAX,
            finished: false,
            finished_elapsed: None,
            quit_requested: false,
        }
    }

    pub(super) fn apply(&mut self, update: DashboardUpdate) {
        match update {
            DashboardUpdate::Solution {
                report,
                width,
                density,
            } => {
                if report == ReportType::Final {
                    self.phase = "final";
                }
                if report_is_feasible(&report) {
                    self.loss_remaining = Some(0.0);
                }
                self.width = Some(width);
                self.density = Some(density);
                self.report = Some(report);
            }
            DashboardUpdate::Phase(phase) => self.phase = phase_label(phase),
            DashboardUpdate::Separation(progress) => self.apply_separation_progress(progress),
            DashboardUpdate::SeparationResult(result) => {
                self.last_separation_result = Some((result, Instant::now()));
            }
            DashboardUpdate::Compression(shrink_step) => self.shrink_step = Some(shrink_step),
        }
    }

    pub(super) fn finish(&mut self) {
        self.finished = true;
        self.finished_elapsed = Some(self.started.elapsed());
    }

    pub(super) fn request_quit(&mut self) {
        self.quit_requested = true;
    }

    pub(super) fn quit_requested(&self) -> bool {
        self.quit_requested
    }

    pub(super) fn show_latest_log(&mut self) {
        self.log_scroll = 0;
    }

    fn apply_separation_progress(&mut self, progress: SeparationProgress) {
        if progress.iteration == 0 {
            self.attempt = match self.separation_width == Some(progress.strip_width) {
                true => self.attempt + 1,
                false => 1,
            };
            self.separation_width = Some(progress.strip_width);
            self.attempt_initial_loss = Some(progress.min_loss);
        }
        let initial_loss = self
            .attempt_initial_loss
            .expect("separation progress must start at iteration zero");
        let loss_remaining = match initial_loss {
            0.0 => 0.0,
            _ => (progress.min_loss / initial_loss * 100.0).clamp(0.0, 100.0),
        };
        self.width = Some(progress.strip_width);
        self.density = Some(progress.density);
        self.iteration = progress.iteration;
        self.loss_remaining = Some(loss_remaining);
    }

    pub(super) fn push_log(&mut self, log: LogEntry) {
        let keep_position = self.log_scroll > 0;
        let new_lines = wrapped_log_lines(&log.message, self.log_view_width).count();
        self.logs.push(log);
        if keep_position {
            self.log_scroll = self
                .log_scroll
                .saturating_add(new_lines)
                .min(self.max_log_scroll());
        }
    }

    pub(super) fn scroll_logs_up(&mut self, lines: usize) {
        self.log_scroll = self
            .log_scroll
            .saturating_add(lines)
            .min(self.max_log_scroll());
    }

    pub(super) fn scroll_logs_down(&mut self, lines: usize) {
        self.log_scroll = self.log_scroll.saturating_sub(lines);
    }

    fn max_log_scroll(&self) -> usize {
        self.logs
            .iter()
            .map(|entry| wrapped_log_lines(&entry.message, self.log_view_width).count())
            .sum::<usize>()
            .saturating_sub(self.log_view_height)
    }

    fn separation_result_color(&self) -> Option<Color> {
        match self.last_separation_result {
            Some((result, reported)) if reported.elapsed() < RESULT_FLASH_DURATION => {
                Some(if result.success {
                    COLOR_ACCENT
                } else {
                    COLOR_FAILURE
                })
            }
            _ => None,
        }
    }

    pub(super) fn render(&mut self, frame: &mut Frame) {
        let [summary_area, logs_area, help_area] = Layout::vertical([
            Constraint::Length(10),
            Constraint::Min(5),
            Constraint::Length(1),
        ])
        .areas(frame.area());

        self.render_summary(frame, summary_area);
        self.render_logs(frame, logs_area);

        let help = match (self.finished, self.quit_requested) {
            (true, _) => "Finished. Press q or Esc to exit.",
            (false, true) => "Stopping optimizer...",
            (false, false) => {
                "↑/↓ PgUp/PgDn: scroll logs   Ctrl-C: skip phase   q / Esc: stop and exit"
            }
        };
        frame.render_widget(
            Paragraph::new(help).style(Style::default().fg(COLOR_MUTED)),
            help_area,
        );
    }

    fn render_summary(&self, frame: &mut Frame, area: Rect) {
        let block = Block::bordered()
            .title(" sparrow dashboard ")
            .border_style(Style::default().fg(COLOR_ACCENT));
        let inner = block.inner(area);
        frame.render_widget(block, area);
        let [logo_area, _, metrics_area, _, progress_space] = Layout::horizontal([
            Constraint::Length(LOGO_WIDTH),
            Constraint::Length(1),
            Constraint::Length(METRICS_WIDTH),
            Constraint::Length(3),
            Constraint::Min(0),
        ])
        .areas(inner);
        let progress_area = Rect::new(
            progress_space.x,
            progress_space.y,
            progress_space
                .width
                .saturating_sub(2)
                .min(PROGRESS_MAX_WIDTH),
            progress_space.height,
        );
        frame.render_widget(Paragraph::new(sparrow_logo()), logo_area);

        self.render_metrics(frame, metrics_area);
        self.render_progress(frame, progress_area);
    }

    fn render_metrics(&self, frame: &mut Frame, area: Rect) {
        let [_, metrics_area, _] = Layout::vertical([
            Constraint::Length(2),
            Constraint::Length(5),
            Constraint::Fill(1),
        ])
        .areas(area);

        let (state, state_color) = match (&self.report, self.loss_remaining) {
            (Some(ReportType::Final), _) => ("FINISHED", COLOR_ACCENT),
            (_, Some(0.0)) => ("FEASIBLE", COLOR_ACCENT),
            (_, Some(_)) => ("NESTING", COLOR_ACTIVE),
            (Some(report), None) if report_is_feasible(report) => ("FEASIBLE", COLOR_ACCENT),
            (Some(_), None) => ("INFEASIBLE", COLOR_FAILURE),
            (None, None) => ("STARTING", COLOR_ACTIVE),
        };
        let activity = match self.finished {
            true => " ✓ ",
            false => {
                let frame = (self.started.elapsed().as_millis() / ACTIVITY_INTERVAL.as_millis())
                    % ACTIVITY_FRAMES.len() as u128;
                ACTIVITY_FRAMES[frame as usize]
            }
        };
        let phase = TextLine::from(vec![
            Span::styled(
                format!("{state:<11}"),
                Style::default()
                    .fg(state_color)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("[", Style::default().fg(COLOR_MUTED)),
            Span::styled(
                activity,
                Style::default()
                    .fg(match self.finished {
                        true => COLOR_ACCENT,
                        false => COLOR_TEXT,
                    })
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("]  ", Style::default().fg(COLOR_MUTED)),
            Span::styled(
                self.phase,
                Style::default()
                    .fg(COLOR_ACCENT)
                    .add_modifier(Modifier::BOLD),
            ),
        ]);
        let width = self
            .width
            .map_or("-".to_owned(), |width| format!("{width:.3}"));
        let dimensions = TextLine::from(vec![
            Span::styled("width ", Style::default().fg(COLOR_MUTED)),
            Span::styled(format!("{width:<12}"), Style::default().fg(COLOR_TEXT)),
            Span::styled("density ", Style::default().fg(COLOR_MUTED)),
            Span::styled(
                self.density
                    .map_or("-".to_owned(), |density| format!("{density:.3}%")),
                Style::default().fg(COLOR_ACCENT),
            ),
        ]);
        let iteration = TextLine::from(vec![
            Span::styled("separation iteration ", Style::default().fg(COLOR_MUTED)),
            Span::styled(
                self.iteration.to_string(),
                Style::default().fg(COLOR_ACCENT),
            ),
        ]);
        let viewer = TextLine::from(vec![
            Span::styled("viewer  ", Style::default().fg(COLOR_MUTED)),
            Span::styled(
                LIVE_VIEWER_PATH,
                Style::default()
                    .fg(COLOR_LINK)
                    .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
            ),
        ]);
        let (evals_per_second, moves_per_second, iterations_per_second) =
            self.last_separation_result.map_or_else(
                || ("-".to_owned(), "-".to_owned(), "-".to_owned()),
                |(result, _)| {
                    (
                        format!(
                            "{} K",
                            (result.total_evals as f32 / (1000.0 * result.elapsed_seconds))
                                as usize
                        ),
                        format_rate(result.total_moves as f32 / result.elapsed_seconds),
                        format_rate(result.iterations as f32 / result.elapsed_seconds),
                    )
                },
            );
        let throughput = TextLine::from(vec![
            Span::styled("evals/s ", Style::default().fg(COLOR_MUTED)),
            Span::styled(evals_per_second, Style::default().fg(COLOR_TEXT)),
            Span::styled("  moves/s ", Style::default().fg(COLOR_MUTED)),
            Span::styled(moves_per_second, Style::default().fg(COLOR_TEXT)),
            Span::styled("  iter/s ", Style::default().fg(COLOR_MUTED)),
            Span::styled(iterations_per_second, Style::default().fg(COLOR_TEXT)),
        ]);
        frame.render_widget(
            Paragraph::new(vec![phase, dimensions, iteration, viewer, throughput]),
            metrics_area,
        );
    }

    fn render_progress(&self, frame: &mut Frame, progress_area: Rect) {
        let elapsed = self
            .finished_elapsed
            .unwrap_or_else(|| self.started.elapsed());
        let phase_progress = match self.phase {
            "exploring" => self.budget.max_attempts.map(|max_attempts| {
                let attempt = self.attempt.min(max_attempts);
                (
                    attempt as f64 / max_attempts as f64,
                    format!("attempt  {attempt} / {max_attempts}"),
                )
            }),
            "compressing" => self.budget.shrink_range.map(|range| {
                let shrink_step = self.shrink_step.unwrap_or(range.0);
                (
                    shrink_progress(shrink_step, range),
                    format!(
                        "shrink step  {:.3}% → {:.3}%",
                        shrink_step * 100.0,
                        range.1 * 100.0
                    ),
                )
            }),
            _ => None,
        };
        let has_phase_progress = phase_progress.is_some();
        let stack_height = if has_phase_progress { 5 } else { 3 };
        let stack_y = progress_area.y
            + progress_area
                .height
                .saturating_sub(stack_height)
                .div_ceil(2);
        let time_area = Rect::new(progress_area.x, stack_y, progress_area.width, 1);
        let phase_progress_area = Rect::new(progress_area.x, stack_y + 2, progress_area.width, 1);
        let loss_area = Rect::new(
            progress_area.x,
            stack_y + if has_phase_progress { 4 } else { 2 },
            progress_area.width,
            1,
        );
        if let Some((ratio, label)) = phase_progress {
            frame.render_widget(
                Gauge::default().ratio(ratio).label(label).gauge_style(
                    Style::default()
                        .fg(COLOR_ACTIVE)
                        .bg(COLOR_TRACK)
                        .add_modifier(Modifier::BOLD),
                ),
                phase_progress_area,
            );
        }
        let time_progress = match self.finished {
            true => 1.0,
            false => elapsed.as_secs_f64() / self.budget.total_duration.as_secs_f64().max(0.001),
        }
        .clamp(0.0, 1.0);
        let time_label = format!(
            "time  {}s / {}s",
            elapsed.as_secs(),
            self.budget.total_duration.as_secs()
        );
        let (time_label, boundary_marker_x) = match self.budget.exploration_boundary {
            Some(boundary) if time_area.width > 0 => {
                let marker_offset =
                    (f64::from(time_area.width.saturating_sub(1)) * boundary).round() as u16;
                let label_width = time_label.len();
                let space_left = usize::from(marker_offset);
                let space_right = usize::from(time_area.width.saturating_sub(marker_offset + 1));
                match (space_left >= label_width, space_right >= label_width) {
                    (true, _) => (
                        format!("{time_label:<width$}", width = time_area.width.into()),
                        Some(time_area.x + marker_offset),
                    ),
                    (_, true) => (
                        format!("{time_label:>width$}", width = time_area.width.into()),
                        Some(time_area.x + marker_offset),
                    ),
                    _ => (time_label, None),
                }
            }
            _ => (time_label, None),
        };
        frame.render_widget(
            Gauge::default()
                .ratio(time_progress)
                .label(time_label)
                .gauge_style(
                    Style::default()
                        .fg(COLOR_ACCENT)
                        .bg(COLOR_TRACK)
                        .add_modifier(Modifier::BOLD),
                ),
            time_area,
        );
        if let Some(marker_x) = boundary_marker_x {
            frame.buffer_mut()[(marker_x, time_area.y)]
                .set_symbol("│")
                .set_style(
                    Style::default()
                        .fg(COLOR_ACTIVE)
                        .add_modifier(Modifier::BOLD),
                );
        }
        let collision_progress = self.loss_remaining.map(|loss| 100.0 - loss);
        let separation_label = match collision_progress {
            Some(progress) => format!("separation progress  {progress:.1}%"),
            None => "separation progress  -".to_owned(),
        };
        let separation_label = Span::styled(
            separation_label,
            self.separation_result_color()
                .map_or(Style::default(), |color| Style::default().fg(color)),
        );
        frame.render_widget(
            Gauge::default()
                .ratio((collision_progress.unwrap_or(0.0) / 100.0) as f64)
                .label(separation_label)
                .gauge_style(
                    Style::default()
                        .fg(COLOR_LOSS)
                        .bg(COLOR_TRACK)
                        .add_modifier(Modifier::BOLD),
                ),
            loss_area,
        );
    }

    fn render_logs(&mut self, frame: &mut Frame, area: Rect) {
        let visible_lines = area.height.saturating_sub(2) as usize;
        self.log_view_height = visible_lines.max(1);
        self.log_view_width = area.width.saturating_sub(2).max(1);
        let lines = self
            .logs
            .iter()
            .flat_map(|entry| {
                wrapped_log_lines(&entry.message, self.log_view_width)
                    .map(|line| TextLine::styled(line, log_style(entry)))
            })
            .collect::<Vec<_>>();
        self.log_scroll = self
            .log_scroll
            .min(lines.len().saturating_sub(self.log_view_height));
        let lines = lines
            .into_iter()
            .rev()
            .skip(self.log_scroll)
            .take(visible_lines)
            .rev()
            .collect::<Vec<_>>();
        let title = match self.log_scroll {
            0 => " Logs ".to_owned(),
            lines => format!(" Logs · {lines} lines above latest "),
        };
        frame.render_widget(
            Paragraph::new(lines).block(
                Block::bordered()
                    .title(title)
                    .border_style(Style::default().fg(COLOR_MUTED)),
            ),
            area,
        );
    }
}

fn wrapped_log_lines(message: &str, width: u16) -> impl Iterator<Item = &str> {
    let width = width.max(1);
    let mut remaining = Some(message);

    std::iter::from_fn(move || {
        let message = remaining?;
        if message.is_empty() {
            remaining = None;
            return Some("");
        }

        let mut line_width = 0;
        let split = message
            .char_indices()
            .find_map(|(index, character)| {
                let mut encoded = [0; 4];
                let character_width = character.encode_utf8(&mut encoded).cell_width();
                match line_width > 0 && line_width + character_width > width {
                    true => Some(index),
                    false => {
                        line_width += character_width;
                        None
                    }
                }
            })
            .unwrap_or(message.len());
        let (line, rest) = message.split_at(split);
        remaining = (!rest.is_empty()).then_some(rest);
        Some(line)
    })
}

fn log_style(entry: &LogEntry) -> Style {
    if entry.message.contains("[EXPL] feasible solution found!")
        || entry.message.contains("[CMPR] success at")
    {
        Style::default()
            .fg(COLOR_LOG_SUCCESS)
            .add_modifier(Modifier::BOLD)
    } else if entry.message.contains("[EXPL] unable to reach feasibility")
        || entry.message.contains("[CMPR] failed at")
    {
        Style::default()
            .fg(COLOR_LOG_FAILURE)
            .add_modifier(Modifier::BOLD)
    } else if entry.message.contains("[SEP] finished") {
        Style::default().fg(COLOR_TEXT).add_modifier(Modifier::BOLD)
    } else {
        match entry.level {
            Level::Error => Style::default()
                .fg(COLOR_LOG_FAILURE)
                .add_modifier(Modifier::BOLD),
            Level::Warn => Style::default().fg(COLOR_ACTIVE),
            Level::Info => Style::default().fg(COLOR_TEXT),
            Level::Debug | Level::Trace => Style::default().fg(COLOR_MUTED),
        }
    }
}

fn report_is_feasible(report: &ReportType) -> bool {
    match report {
        ReportType::ExplFeas | ReportType::CmprFeas | ReportType::Final => true,
        ReportType::ExplInfeas | ReportType::ExplImproving => false,
    }
}

fn shrink_progress(shrink_step: f32, range: (f32, f32)) -> f64 {
    ((range.0 - shrink_step) / (range.0 - range.1)).clamp(0.0, 1.0) as f64
}

fn format_rate(rate: f32) -> String {
    Formatter::new()
        .scales(Scales::short())
        .precision(Precision::Significance(3))
        .fmt2(rate)
        .to_owned()
}

fn phase_label(phase: OptimizationPhase) -> &'static str {
    match phase {
        OptimizationPhase::Exploration => "exploring",
        OptimizationPhase::Compression => "compressing",
    }
}

const LIVE_VIEWER_PATH: &str = "data/live/live_viewer.html";
const RESULT_FLASH_DURATION: Duration = Duration::from_millis(100);
const LOGO_WIDTH: u16 = 19;
const METRICS_WIDTH: u16 = 44;
const PROGRESS_MAX_WIDTH: u16 = 60;
const ACTIVITY_INTERVAL: Duration = Duration::from_millis(120);
const ACTIVITY_FRAMES: &[&str] = &["⠀⠶⠀", "⠰⣿⠆", "⢾⣉⡷", "⣏⠀⣹", "⡁⠀⢈"];
const COLOR_ACCENT: Color = Color::LightGreen;
const COLOR_ACTIVE: Color = Color::LightYellow;
const COLOR_LOSS: Color = Color::LightBlue;
const COLOR_FAILURE: Color = Color::LightRed;
const COLOR_LOG_SUCCESS: Color = Color::Rgb(140, 195, 150);
const COLOR_LOG_FAILURE: Color = Color::Rgb(220, 140, 125);
const COLOR_LINK: Color = Color::LightCyan;
const COLOR_TEXT: Color = Color::White;
const COLOR_MUTED: Color = Color::DarkGray;
const COLOR_TRACK: Color = Color::Black;
const COLOR_SPARROW_RUST: Color = Color::Rgb(190, 66, 35);
const COLOR_SPARROW_BROWN: Color = Color::Rgb(150, 84, 44);
const COLOR_SPARROW_GRAY: Color = Color::Rgb(160, 160, 160);
const COLOR_SPARROW_OCHRE: Color = Color::Rgb(195, 139, 60);
const COLOR_SPARROW_DARK: Color = Color::Gray;
const SPARROW_LOGO: &[&[(&str, Color)]] = &[
    &[
        ("   ", Color::Reset),
        ("⣴⣾", COLOR_SPARROW_DARK),
        ("⣿⣿⣿⣶", COLOR_SPARROW_RUST),
        ("⣄", COLOR_SPARROW_DARK),
    ],
    &[
        (" ", Color::Reset),
        ("⠺", COLOR_SPARROW_DARK),
        ("⢿", COLOR_SPARROW_BROWN),
        ("⣿", COLOR_SPARROW_DARK),
        ("⠐", COLOR_TEXT),
        ("⣿⣿⣿", COLOR_SPARROW_GRAY),
        ("⣿", COLOR_SPARROW_RUST),
        ("⣿", COLOR_SPARROW_BROWN),
        ("⣆⡀", COLOR_SPARROW_DARK),
    ],
    &[
        ("  ", Color::Reset),
        ("⣾⣿⣿", COLOR_SPARROW_DARK),
        ("⣿", COLOR_SPARROW_GRAY),
        ("⣿", COLOR_TEXT),
        ("⣿", COLOR_SPARROW_GRAY),
        ("⣿⣿⣿⣿⣷", COLOR_SPARROW_BROWN),
        ("⣄", COLOR_SPARROW_DARK),
    ],
    &[
        ("  ", Color::Reset),
        ("⣿", COLOR_SPARROW_BROWN),
        ("⣿⣿⣿⣿⣿⣿", COLOR_SPARROW_GRAY),
        ("⣿⣿⣿⣿⣿", COLOR_SPARROW_BROWN),
        ("⣆", COLOR_SPARROW_DARK),
    ],
    &[
        ("  ", Color::Reset),
        ("⠘", COLOR_SPARROW_DARK),
        ("⢿⣿⣿⣿⣿⣿⣿⣿⣿", COLOR_SPARROW_GRAY),
        ("⣿⠿", COLOR_SPARROW_BROWN),
        ("⠿⣿⣶⣤", COLOR_SPARROW_DARK),
    ],
    &[
        ("   ", Color::Reset),
        ("⢠⣽", COLOR_SPARROW_DARK),
        ("⣿⣿", COLOR_SPARROW_BROWN),
        ("⣿⣿⣿", COLOR_SPARROW_GRAY),
        ("⣿", COLOR_SPARROW_BROWN),
        ("⡋⠁", COLOR_SPARROW_DARK),
        ("   ", Color::Reset),
        ("⠉⠁", COLOR_SPARROW_DARK),
    ],
    &[
        ("   ", Color::Reset),
        ("⣿", COLOR_SPARROW_DARK),
        ("⣿⣿⣿⣿", COLOR_SPARROW_OCHRE),
        ("⣿", COLOR_SPARROW_BROWN),
        ("⣿⣿", COLOR_SPARROW_OCHRE),
        ("⣿", COLOR_SPARROW_DARK),
    ],
    &[
        ("   ", Color::Reset),
        ("⢿", COLOR_SPARROW_DARK),
        ("⣿⣿", COLOR_SPARROW_OCHRE),
        ("⣿", COLOR_SPARROW_BROWN),
        ("⣿⣿⣿⣿", COLOR_SPARROW_OCHRE),
        ("⡿", COLOR_SPARROW_DARK),
    ],
];

fn sparrow_logo() -> Vec<TextLine<'static>> {
    SPARROW_LOGO
        .iter()
        .map(|line| {
            TextLine::from(
                line.iter()
                    .map(|&(text, color)| Span::styled(text, Style::default().fg(color)))
                    .collect::<Vec<_>>(),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests;

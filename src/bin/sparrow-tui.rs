use anyhow::{Result, anyhow, bail};
use clap::Parser;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use jagua_rs::Instant as CdeInstant;
use jagua_rs::io::import::Importer;
use jagua_rs::io::svg::s_layout_to_svg;
use jagua_rs::probs::spp::entities::{SPInstance, SPSolution};
use jagua_rs::probs::spp::io::ext_repr::ExtSPInstance;
use log::{Level, Log, Metadata, Record};
use rand::SeedableRng;
use rand::rngs::Xoshiro256PlusPlus;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line as TextLine, Span};
use ratatui::widgets::{Block, Gauge, Paragraph};
use ratatui::{DefaultTerminal, Frame};
use sparrow::EPOCH;
use sparrow::config::{DEFAULT_SPARROW_CONFIG, ShrinkDecayStrategy, SparrowConfig};
use sparrow::consts::{
    DEFAULT_COMPRESS_TIME_RATIO, DEFAULT_EXPLORE_TIME_RATIO, DEFAULT_FAIL_DECAY_RATIO_CMPR,
    DEFAULT_MAX_CONSEQ_FAILS_EXPL, DRAW_OPTIONS, LOG_LEVEL_FILTER_DEBUG, LOG_LEVEL_FILTER_RELEASE,
};
use sparrow::optimizer::optimize;
use sparrow::util::io::{self, ExtSPOutput, MainCli};
use sparrow::util::listener::{
    OptimizationPhase, ReportType, SeparationProgress, SolutionListener,
};
use sparrow::util::svg_exporter::SvgExporter;
use sparrow::util::terminator::Terminator;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, SyncSender};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const OUTPUT_DIR: &str = "output";
const LIVE_VIEWER_PATH: &str = "data/live/live_viewer.html";
const LIVE_SVG_PATH: &str = "data/live/.live_solution.svg";
const FRAME_INTERVAL: Duration = Duration::from_millis(100);
const SNAPSHOT_INTERVAL: Duration = Duration::from_millis(100);
const COLOR_ACCENT: Color = Color::LightGreen;
const COLOR_ACTIVE: Color = Color::LightYellow;
const COLOR_LOSS: Color = Color::LightBlue;
const COLOR_FAILURE: Color = Color::LightRed;
const COLOR_LINK: Color = Color::LightCyan;
const COLOR_TEXT: Color = Color::White;
const COLOR_MUTED: Color = Color::DarkGray;
const COLOR_TRACK: Color = Color::Black;

fn main() -> Result<()> {
    let args = MainCli::parse();
    fs::create_dir_all(OUTPUT_DIR)?;
    let (logs_tx, logs_rx) = mpsc::sync_channel(512);
    let log_level = match cfg!(debug_assertions) {
        true => LOG_LEVEL_FILTER_DEBUG,
        false => LOG_LEVEL_FILTER_RELEASE,
    };
    init_tui_logger(log_level, Path::new("output/log.txt"), logs_tx)?;

    let config = configure(&args)?;
    let total_duration = config.expl_cfg.time_limit + config.cmpr_cfg.time_limit;
    let exploration_boundary = (config.expl_cfg.max_conseq_failed_attempts.is_none()
        && !total_duration.is_zero())
    .then(|| config.expl_cfg.time_limit.as_secs_f64() / total_duration.as_secs_f64());
    let budget = SearchBudget {
        total_duration,
        max_attempts: config.expl_cfg.max_conseq_failed_attempts,
        shrink_range: matches!(
            config.cmpr_cfg.shrink_decay,
            ShrinkDecayStrategy::FailureBased(_)
        )
        .then_some(config.cmpr_cfg.shrink_range),
        exploration_boundary,
    };
    let rng = Xoshiro256PlusPlus::seed_from_u64(
        config
            .rng_seed
            .map_or_else(rand::random, |seed| seed as u64),
    );

    let (ext_instance, ext_solution) = io::read_spp_input(Path::new(&args.input))?;
    let importer = Importer::new(
        config.cde_config,
        config.poly_simpl_tolerance,
        config.min_item_separation,
        config.narrow_concavity_cutoff_ratio,
    );
    let instance = jagua_rs::probs::spp::io::import_instance(&importer, &ext_instance)?;
    let initial_solution = ext_solution
        .map(|solution| jagua_rs::probs::spp::io::import_solution(&instance, &solution));

    let signals = TuiSignals::new();
    let (updates_tx, updates_rx) = mpsc::channel();
    let worker = start_optimizer(
        instance.clone(),
        initial_solution,
        config,
        rng,
        signals.clone(),
        updates_tx,
    );

    let solution = ratatui::run(|terminal| {
        run_tui(
            terminal,
            updates_rx,
            logs_rx,
            worker,
            signals,
            (&instance, &ext_instance),
            budget,
        )
    })?;

    let svg_path = format!("{OUTPUT_DIR}/final_{}.svg", ext_instance.name);
    let json_path = format!("{OUTPUT_DIR}/final_{}.json", ext_instance.name);
    println!(
        "Finished at width {:.3}, density {:.3}%\n{svg_path}\n{json_path}",
        solution.strip_width(),
        solution.density(&instance) * 100.0,
    );
    Ok(())
}

fn configure(args: &MainCli) -> Result<SparrowConfig> {
    let mut config = DEFAULT_SPARROW_CONFIG;
    let (exploration, compression) = match (args.global_time, args.exploration, args.compression) {
        (Some(total), None, None) => (
            Duration::from_secs(total).mul_f32(DEFAULT_EXPLORE_TIME_RATIO),
            Duration::from_secs(total).mul_f32(DEFAULT_COMPRESS_TIME_RATIO),
        ),
        (None, Some(exploration), Some(compression)) => (
            Duration::from_secs(exploration),
            Duration::from_secs(compression),
        ),
        (None, None, None) => (
            Duration::from_secs(600).mul_f32(DEFAULT_EXPLORE_TIME_RATIO),
            Duration::from_secs(600).mul_f32(DEFAULT_COMPRESS_TIME_RATIO),
        ),
        _ => bail!("invalid time limit arguments"),
    };
    config.expl_cfg.time_limit = exploration;
    config.cmpr_cfg.time_limit = compression;
    if args.early_termination {
        config.expl_cfg.max_conseq_failed_attempts = Some(DEFAULT_MAX_CONSEQ_FAILS_EXPL);
        config.cmpr_cfg.shrink_decay =
            ShrinkDecayStrategy::FailureBased(DEFAULT_FAIL_DECAY_RATIO_CMPR);
    }
    args.apply_config_overrides(&mut config);
    Ok(config)
}

fn start_optimizer(
    instance: SPInstance,
    initial_solution: Option<SPSolution>,
    config: SparrowConfig,
    rng: Xoshiro256PlusPlus,
    signals: TuiSignals,
    updates: Sender<Update>,
) -> JoinHandle<SPSolution> {
    thread::Builder::new()
        .name("optimizer".into())
        .spawn(move || {
            optimize(
                instance,
                rng,
                &mut TuiListener::new(updates),
                &mut TuiTerminator::new(signals),
                &config.expl_cfg,
                &config.cmpr_cfg,
                initial_solution.as_ref(),
            )
        })
        .expect("failed to start optimizer")
}

fn run_tui(
    terminal: &mut DefaultTerminal,
    updates: Receiver<Update>,
    logs: Receiver<LogEntry>,
    worker: JoinHandle<SPSolution>,
    signals: TuiSignals,
    final_output: (&SPInstance, &ExtSPInstance),
    budget: SearchBudget,
) -> Result<SPSolution> {
    let (instance, ext_instance) = final_output;
    let mut app = App::new(budget);
    let mut worker = Some(worker);
    let mut solution = None;

    loop {
        for log in logs.try_iter() {
            app.push_log(log);
        }
        for update in updates.try_iter() {
            app.apply(update);
        }
        if worker.as_ref().is_some_and(JoinHandle::is_finished) {
            let final_solution = worker
                .take()
                .unwrap()
                .join()
                .map_err(|_| anyhow!("optimizer thread panicked"))?;
            export_final_solution(&final_solution, instance, ext_instance)?;
            solution = Some(final_solution);
            app.finished = true;
            app.finished_elapsed = Some(app.started.elapsed());
        }

        terminal.draw(|frame| app.render(frame))?;

        if app.quit_requested
            && let Some(solution) = solution
        {
            return Ok(solution);
        }
        if event::poll(FRAME_INTERVAL)?
            && let Event::Key(key) = event::read()?
            && key.kind == KeyEventKind::Press
        {
            match key.code {
                KeyCode::Esc | KeyCode::Char('q') => {
                    app.quit_requested = true;
                    signals.quit.store(true, Ordering::Relaxed);
                }
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    signals.interrupt_phase.store(true, Ordering::Relaxed);
                }
                KeyCode::Up => app.scroll_logs_up(1),
                KeyCode::Down => app.scroll_logs_down(1),
                KeyCode::PageUp => app.scroll_logs_up(10),
                KeyCode::PageDown => app.scroll_logs_down(10),
                KeyCode::Home => app.scroll_logs_up(usize::MAX),
                KeyCode::End => app.log_scroll = 0,
                _ => {}
            }
        }
    }
}

#[derive(Clone, Copy)]
struct SearchBudget {
    total_duration: Duration,
    max_attempts: Option<usize>,
    shrink_range: Option<(f32, f32)>,
    exploration_boundary: Option<f64>,
}

struct App {
    report: Option<ReportType>,
    phase: &'static str,
    width: Option<f32>,
    density: Option<f32>,
    separation_width: Option<f32>,
    attempt: usize,
    iteration: usize,
    attempt_initial_loss: Option<f32>,
    loss_remaining: Option<f32>,
    shrink_step: Option<f32>,
    started: Instant,
    budget: SearchBudget,
    logs: Vec<LogEntry>,
    log_scroll: usize,
    log_view_height: usize,
    finished: bool,
    finished_elapsed: Option<Duration>,
    quit_requested: bool,
}

impl App {
    fn new(budget: SearchBudget) -> Self {
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
            shrink_step: None,
            started: Instant::now(),
            budget,
            logs: Vec::new(),
            log_scroll: 0,
            log_view_height: 1,
            finished: false,
            finished_elapsed: None,
            quit_requested: false,
        }
    }

    fn apply(&mut self, update: Update) {
        match update {
            Update::Solution {
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
            Update::Phase(phase) => self.phase = phase_label(phase),
            Update::Separation(progress) => self.apply_separation_progress(progress),
            Update::Compression(shrink_step) => self.shrink_step = Some(shrink_step),
        }
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

    fn push_log(&mut self, log: LogEntry) {
        let keep_position = self.log_scroll > 0;
        self.logs.push(log);
        if keep_position {
            self.log_scroll = self.log_scroll.saturating_add(1).min(self.max_log_scroll());
        }
    }

    fn scroll_logs_up(&mut self, lines: usize) {
        self.log_scroll = self
            .log_scroll
            .saturating_add(lines)
            .min(self.max_log_scroll());
    }

    fn scroll_logs_down(&mut self, lines: usize) {
        self.log_scroll = self.log_scroll.saturating_sub(lines);
    }

    fn max_log_scroll(&self) -> usize {
        self.logs.len().saturating_sub(self.log_view_height)
    }

    fn render(&mut self, frame: &mut Frame) {
        let [summary_area, logs_area, help_area] = Layout::vertical([
            Constraint::Length(6),
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
            .title(" sparrow search overview ")
            .border_style(Style::default().fg(COLOR_ACCENT));
        let inner = block.inner(area);
        frame.render_widget(block, area);
        let [metrics_area, _, progress_area] = Layout::horizontal([
            Constraint::Percentage(50),
            Constraint::Length(2),
            Constraint::Min(20),
        ])
        .areas(inner);

        let (state, state_color) = match (&self.report, self.loss_remaining) {
            (Some(ReportType::Final), _) => ("FINISHED", COLOR_ACCENT),
            (_, Some(0.0)) => ("FEASIBLE", COLOR_ACCENT),
            (_, Some(_)) => ("SEPARATING", COLOR_ACTIVE),
            (Some(report), None) if report_is_feasible(report) => ("FEASIBLE", COLOR_ACCENT),
            (Some(_), None) => ("INFEASIBLE", COLOR_FAILURE),
            (None, None) => ("STARTING", COLOR_ACTIVE),
        };
        let phase = TextLine::from(vec![
            Span::styled(
                format!(" {} ", self.phase),
                Style::default()
                    .fg(COLOR_ACCENT)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled(
                state,
                Style::default()
                    .fg(state_color)
                    .add_modifier(Modifier::BOLD),
            ),
        ]);
        let elapsed = self
            .finished_elapsed
            .unwrap_or_else(|| self.started.elapsed());
        let dimensions = TextLine::from(vec![
            Span::styled("width ", Style::default().fg(COLOR_MUTED)),
            Span::styled(
                self.width
                    .map_or("-".to_owned(), |width| format!("{width:.3}")),
                Style::default().fg(COLOR_TEXT),
            ),
            Span::styled("   density ", Style::default().fg(COLOR_MUTED)),
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
        frame.render_widget(
            Paragraph::new(vec![phase, dimensions, iteration, viewer]),
            metrics_area,
        );

        let [time_area, phase_progress_area, loss_area] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .areas(progress_area);
        let phase_progress = match self.phase {
            "exploration" => self.budget.max_attempts.map(|max_attempts| {
                let attempt = self.attempt.min(max_attempts);
                (
                    attempt as f64 / max_attempts as f64,
                    format!("attempt  {attempt} / {max_attempts}"),
                )
            }),
            "compression" => self.budget.shrink_range.map(|range| {
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
        let loss_area = match phase_progress {
            Some((ratio, label)) => {
                frame.render_widget(
                    Gauge::default().ratio(ratio).label(label).gauge_style(
                        Style::default()
                            .fg(COLOR_ACTIVE)
                            .bg(COLOR_TRACK)
                            .add_modifier(Modifier::BOLD),
                    ),
                    phase_progress_area,
                );
                loss_area
            }
            None => phase_progress_area,
        };
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
        frame.render_widget(
            Gauge::default()
                .ratio(time_progress)
                .label(match self.budget.exploration_boundary {
                    Some(_) => String::new(),
                    None => time_label.clone(),
                })
                .gauge_style(
                    Style::default()
                        .fg(COLOR_ACCENT)
                        .bg(COLOR_TRACK)
                        .add_modifier(Modifier::BOLD),
                ),
            time_area,
        );
        if let Some(boundary) = self.budget.exploration_boundary
            && time_area.width > 0
        {
            let marker_x = time_area.x
                + (f64::from(time_area.width.saturating_sub(1)) * boundary).round() as u16;
            frame.buffer_mut()[(marker_x, time_area.y)]
                .set_symbol("│")
                .set_style(
                    Style::default()
                        .fg(COLOR_ACTIVE)
                        .add_modifier(Modifier::BOLD),
                );
            let label_width = time_label.len().min(time_area.width.into()) as u16;
            let label_x = match boundary < 0.5 {
                true => time_area.right().saturating_sub(label_width),
                false => time_area.x,
            };
            frame.buffer_mut().set_stringn(
                label_x,
                time_area.y,
                time_label,
                label_width.into(),
                Style::default().fg(COLOR_TEXT).add_modifier(Modifier::BOLD),
            );
        }
        let collision_progress = self.loss_remaining.map(|loss| 100.0 - loss);
        frame.render_widget(
            Gauge::default()
                .ratio((collision_progress.unwrap_or(0.0) / 100.0) as f64)
                .label(match collision_progress {
                    Some(progress) => format!("collision progress  {progress:.1}%"),
                    None => "collision progress  -".to_owned(),
                })
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
        self.log_scroll = self.log_scroll.min(self.max_log_scroll());
        let lines = self
            .logs
            .iter()
            .rev()
            .skip(self.log_scroll)
            .take(visible_lines)
            .rev()
            .map(|entry| TextLine::styled(entry.message.as_str(), log_style(entry)))
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

fn log_style(entry: &LogEntry) -> Style {
    if entry.message.contains("[EXPL] feasible solution found!")
        || entry.message.contains("[CMPR] success at")
    {
        Style::default()
            .fg(COLOR_ACCENT)
            .add_modifier(Modifier::BOLD)
    } else if entry.message.contains("[EXPL] unable to reach feasibility")
        || entry.message.contains("[CMPR] failed at")
    {
        Style::default()
            .fg(COLOR_FAILURE)
            .add_modifier(Modifier::BOLD)
    } else if entry.message.contains("[SEP] finished") {
        Style::default().fg(COLOR_TEXT).add_modifier(Modifier::BOLD)
    } else {
        match entry.level {
            Level::Error => Style::default()
                .fg(COLOR_FAILURE)
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

fn export_final_solution(
    solution: &SPSolution,
    instance: &SPInstance,
    ext_instance: &ExtSPInstance,
) -> Result<()> {
    let svg_path = format!("{OUTPUT_DIR}/final_{}.svg", ext_instance.name);
    io::write_svg(
        &s_layout_to_svg(&solution.layout_snapshot, instance, DRAW_OPTIONS, "final"),
        Path::new(&svg_path),
        Level::Info,
    )?;

    let json_path = format!("{OUTPUT_DIR}/final_{}.json", ext_instance.name);
    io::write_json(
        &ExtSPOutput {
            instance: ext_instance.clone(),
            solution: jagua_rs::probs::spp::io::export(instance, solution, *EPOCH),
        },
        Path::new(&json_path),
        Level::Info,
    )
}

struct LogEntry {
    level: Level,
    message: String,
}

struct TuiLogSink {
    logs: SyncSender<LogEntry>,
}

impl Log for TuiLogSink {
    fn enabled(&self, _metadata: &Metadata) -> bool {
        true
    }

    fn log(&self, record: &Record) {
        let _ = self.logs.try_send(LogEntry {
            level: record.level(),
            message: record.args().to_string(),
        });
    }

    fn flush(&self) {}
}

fn init_tui_logger(
    level: log::LevelFilter,
    log_file_path: &Path,
    logs: SyncSender<LogEntry>,
) -> Result<()> {
    let _ = fs::remove_file(log_file_path);
    fern::Dispatch::new()
        .format(|out, message, record| {
            let elapsed = EPOCH.elapsed();
            let seconds = elapsed.as_secs() % 60;
            let minutes = (elapsed.as_secs() / 60) % 60;
            let hours = elapsed.as_secs() / 3600;
            out.finish(format_args!(
                "[{}] [{hours:02}:{minutes:02}:{seconds:02}] {}",
                record.level(),
                message,
            ));
        })
        .level(level)
        .chain(Box::new(TuiLogSink { logs }) as Box<dyn Log>)
        .chain(fern::log_file(log_file_path)?)
        .apply()?;
    Ok(())
}

enum Update {
    Solution {
        report: ReportType,
        width: f32,
        density: f32,
    },
    Phase(OptimizationPhase),
    Separation(SeparationProgress),
    Compression(f32),
}

struct TuiListener {
    updates: Sender<Update>,
    live_svg: SvgExporter,
    last_snapshot: Option<Instant>,
}

impl TuiListener {
    fn new(updates: Sender<Update>) -> Self {
        Self {
            updates,
            live_svg: SvgExporter::new(None, None, Some(LIVE_SVG_PATH.to_owned())),
            last_snapshot: None,
        }
    }
}

impl SolutionListener for TuiListener {
    fn report(&mut self, report: ReportType, solution: &SPSolution, instance: &SPInstance) {
        let now = Instant::now();
        if report != ReportType::Final
            && self
                .last_snapshot
                .is_some_and(|last| now.duration_since(last) < SNAPSHOT_INTERVAL)
        {
            return;
        }

        self.live_svg.report(report.clone(), solution, instance);
        self.last_snapshot = Some(now);
        let update = Update::Solution {
            report: report.clone(),
            width: solution.strip_width(),
            density: solution.density(instance) * 100.0,
        };
        let _ = self.updates.send(update);
    }

    fn report_phase(&mut self, phase: OptimizationPhase) {
        let _ = self.updates.send(Update::Phase(phase));
    }

    fn report_separation_progress(&mut self, progress: SeparationProgress) {
        let _ = self.updates.send(Update::Separation(progress));
    }

    fn report_compression_progress(&mut self, shrink_step: f32) {
        let _ = self.updates.send(Update::Compression(shrink_step));
    }
}

struct TuiTerminator {
    timeout: Option<CdeInstant>,
    signals: TuiSignals,
}

impl TuiTerminator {
    fn new(signals: TuiSignals) -> Self {
        Self {
            timeout: None,
            signals,
        }
    }
}

impl Terminator for TuiTerminator {
    fn kill(&self) -> bool {
        self.signals.quit.load(Ordering::Relaxed)
            || self.signals.interrupt_phase.load(Ordering::Relaxed)
            || self
                .timeout
                .is_some_and(|timeout| CdeInstant::now() > timeout)
    }

    fn new_timeout(&mut self, timeout: Duration) {
        self.signals.interrupt_phase.store(false, Ordering::Relaxed);
        self.timeout = Some(CdeInstant::now() + timeout);
    }

    fn timeout_at(&self) -> Option<CdeInstant> {
        self.timeout
    }
}

#[derive(Clone)]
struct TuiSignals {
    quit: Arc<AtomicBool>,
    interrupt_phase: Arc<AtomicBool>,
}

impl TuiSignals {
    fn new() -> Self {
        Self {
            quit: Arc::new(AtomicBool::new(false)),
            interrupt_phase: Arc::new(AtomicBool::new(false)),
        }
    }
}

fn phase_label(phase: OptimizationPhase) -> &'static str {
    match phase {
        OptimizationPhase::Exploration => "exploration",
        OptimizationPhase::Compression => "compression",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app() -> App {
        App::new(SearchBudget {
            total_duration: Duration::ZERO,
            max_attempts: None,
            shrink_range: None,
            exploration_boundary: None,
        })
    }

    #[test]
    fn groups_separation_attempts_by_width() {
        let mut app = app();
        let progress = |strip_width, iteration, min_loss| {
            Update::Separation(SeparationProgress {
                strip_width,
                density: 80.0,
                iteration,
                min_loss,
            })
        };

        app.apply(progress(100.0, 0, 20.0));
        app.apply(progress(100.0, 1, 10.0));
        assert_eq!(app.attempt, 1);
        assert_eq!(app.loss_remaining, Some(50.0));

        app.apply(progress(100.0, 0, 15.0));
        assert_eq!(app.attempt, 2);
        assert_eq!(app.loss_remaining, Some(100.0));

        app.apply(progress(99.0, 0, 12.0));
        assert_eq!(app.attempt, 1);
    }

    #[test]
    fn keeps_scrolled_logs_in_place_as_new_lines_arrive() {
        let mut app = app();
        app.log_view_height = 2;
        for line in 0..4 {
            app.push_log(LogEntry {
                level: Level::Info,
                message: line.to_string(),
            });
        }

        app.scroll_logs_up(usize::MAX);
        assert_eq!(app.log_scroll, 2);
        app.push_log(LogEntry {
            level: Level::Info,
            message: "new".to_owned(),
        });
        assert_eq!(app.log_scroll, 3);
        app.scroll_logs_down(usize::MAX);
        assert_eq!(app.log_scroll, 0);
    }

    #[test]
    fn phase_interrupt_resets_but_quit_does_not() {
        let signals = TuiSignals::new();
        signals.interrupt_phase.store(true, Ordering::Relaxed);
        let mut terminator = TuiTerminator::new(signals.clone());

        assert!(terminator.kill());
        terminator.new_timeout(Duration::from_secs(1));
        assert!(!terminator.kill());

        signals.quit.store(true, Ordering::Relaxed);
        terminator.new_timeout(Duration::from_secs(1));
        assert!(terminator.kill());
    }

    #[test]
    fn shrink_progress_runs_from_initial_to_final_step() {
        let range = (0.05, 0.01);

        assert_eq!(shrink_progress(range.0, range), 0.0);
        assert_eq!(shrink_progress(range.1, range), 1.0);
        assert!((shrink_progress(0.03, range) - 0.5).abs() < 1e-6);
    }
}

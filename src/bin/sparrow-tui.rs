use anyhow::{Result, anyhow, bail};
use clap::Parser;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use jagua_rs::Instant as CdeInstant;
use jagua_rs::io::import::Importer;
use jagua_rs::io::svg::s_layout_to_svg;
use jagua_rs::probs::spp::entities::{SPInstance, SPSolution};
use log::{Level, Log, Metadata, Record};
use rand::SeedableRng;
use rand::rngs::Xoshiro256PlusPlus;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols::Marker;
use ratatui::text::{Line as TextLine, Span};
use ratatui::widgets::{Axis, Block, Chart, Dataset, Gauge, GraphType, Paragraph};
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
use std::collections::VecDeque;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, SyncSender};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const OUTPUT_DIR: &str = "output";
const LIVE_SVG_PATH: &str = "data/live/.live_solution.svg";
const FRAME_INTERVAL: Duration = Duration::from_millis(100);
const SNAPSHOT_INTERVAL: Duration = Duration::from_millis(100);
const MAX_LOG_LINES: usize = 200;

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

    let stop = Arc::new(AtomicBool::new(false));
    let (updates_tx, updates_rx) = mpsc::channel();
    let worker = start_optimizer(
        instance.clone(),
        initial_solution,
        config,
        rng,
        stop.clone(),
        updates_tx,
    );

    let solution = ratatui::run(|terminal| {
        run_tui(terminal, updates_rx, logs_rx, worker, stop, total_duration)
    })?;

    let svg_path = format!("{OUTPUT_DIR}/final_{}.svg", ext_instance.name);
    io::write_svg(
        &s_layout_to_svg(&solution.layout_snapshot, &instance, DRAW_OPTIONS, "final"),
        Path::new(&svg_path),
        Level::Info,
    )?;
    let json_path = format!("{OUTPUT_DIR}/final_{}.json", ext_instance.name);
    io::write_json(
        &ExtSPOutput {
            instance: ext_instance,
            solution: jagua_rs::probs::spp::io::export(&instance, &solution, *EPOCH),
        },
        Path::new(&json_path),
        Level::Info,
    )?;

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
    stop: Arc<AtomicBool>,
    updates: Sender<Update>,
) -> JoinHandle<SPSolution> {
    thread::Builder::new()
        .name("optimizer".into())
        .spawn(move || {
            optimize(
                instance,
                rng,
                &mut TuiListener::new(updates),
                &mut TuiTerminator::new(stop),
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
    stop: Arc<AtomicBool>,
    total_duration: Duration,
) -> Result<SPSolution> {
    let mut app = App::new(total_duration);
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
            solution = Some(
                worker
                    .take()
                    .unwrap()
                    .join()
                    .map_err(|_| anyhow!("optimizer thread panicked"))?,
            );
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
            && (matches!(key.code, KeyCode::Esc | KeyCode::Char('q'))
                || key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL))
        {
            app.quit_requested = true;
            stop.store(true, Ordering::Relaxed);
        }
    }
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
    loss_history: Vec<(f64, f64)>,
    started: Instant,
    total_duration: Duration,
    logs: VecDeque<LogEntry>,
    finished: bool,
    finished_elapsed: Option<Duration>,
    quit_requested: bool,
}

impl App {
    fn new(total_duration: Duration) -> Self {
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
            loss_history: Vec::new(),
            started: Instant::now(),
            total_duration,
            logs: VecDeque::new(),
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
                self.width = Some(width);
                self.density = Some(density);
                self.report = Some(report);
            }
            Update::Phase(phase) => self.phase = phase_label(phase),
            Update::Separation(progress) => self.apply_separation_progress(progress),
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
            self.loss_history.clear();
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
        self.loss_history
            .push((progress.iteration as f64, loss_remaining as f64));
    }

    fn push_log(&mut self, log: LogEntry) {
        if self.logs.len() == MAX_LOG_LINES {
            self.logs.pop_front();
        }
        self.logs.push_back(log);
    }

    fn render(&mut self, frame: &mut Frame) {
        let [summary_area, loss_area, logs_area, help_area] = Layout::vertical([
            Constraint::Length(6),
            Constraint::Length(11),
            Constraint::Min(5),
            Constraint::Length(1),
        ])
        .areas(frame.area());

        self.render_summary(frame, summary_area);
        self.render_loss_chart(frame, loss_area);
        self.render_logs(frame, logs_area);

        let help = match (self.finished, self.quit_requested) {
            (true, _) => "Finished. Press q or Esc to exit.",
            (false, true) => "Stopping optimizer...",
            (false, false) => {
                "Viewer: data/live/live_viewer.html   q / Esc / Ctrl-C: stop and exit"
            }
        };
        frame.render_widget(
            Paragraph::new(help).style(Style::default().fg(Color::DarkGray)),
            help_area,
        );
    }

    fn render_summary(&self, frame: &mut Frame, area: Rect) {
        let block = Block::bordered()
            .title(" Sparrow search ")
            .border_style(Style::default().fg(Color::Cyan));
        let inner = block.inner(area);
        frame.render_widget(block, area);
        let [metrics_area, progress_area] =
            Layout::vertical([Constraint::Length(3), Constraint::Length(1)]).areas(inner);

        let (state, state_color) = match (&self.report, self.loss_remaining) {
            (Some(ReportType::Final), _) => ("FINISHED", Color::Green),
            (_, Some(0.0)) => ("FEASIBLE", Color::Green),
            (_, Some(_)) => ("SEPARATING", Color::Yellow),
            (Some(report), None) if report_is_feasible(report) => ("FEASIBLE", Color::Green),
            (Some(_), None) => ("INFEASIBLE", Color::Red),
            (None, None) => ("STARTING", Color::Yellow),
        };
        let phase_color = match self.phase {
            "exploration" => Color::Cyan,
            "compression" | "final" => Color::Magenta,
            _ => Color::DarkGray,
        };
        let phase = TextLine::from(vec![
            Span::styled(
                format!(" {} ", self.phase),
                Style::default()
                    .fg(Color::Black)
                    .bg(phase_color)
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
            Span::styled("width ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                self.width
                    .map_or("-".to_owned(), |width| format!("{width:.3}")),
                Style::default().fg(Color::White),
            ),
            Span::styled("   density ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                self.density
                    .map_or("-".to_owned(), |density| format!("{density:.3}%")),
                Style::default().fg(Color::Green),
            ),
            Span::styled("   loss remaining ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                self.loss_remaining
                    .map_or("-".to_owned(), |loss| format!("{loss:.1}%")),
                Style::default().fg(match self.loss_remaining {
                    Some(0.0) => Color::Green,
                    Some(_) => Color::Yellow,
                    None => Color::DarkGray,
                }),
            ),
        ]);
        let attempt = TextLine::from(vec![
            Span::styled("attempt ", Style::default().fg(Color::DarkGray)),
            Span::styled(self.attempt.to_string(), Style::default().fg(Color::Yellow)),
            Span::styled(
                " at this width   iteration ",
                Style::default().fg(Color::DarkGray),
            ),
            Span::styled(
                self.iteration.to_string(),
                Style::default().fg(Color::LightCyan),
            ),
        ]);
        frame.render_widget(
            Paragraph::new(vec![phase, dimensions, attempt]),
            metrics_area,
        );

        let progress = match self.finished {
            true => 1.0,
            false => elapsed.as_secs_f64() / self.total_duration.as_secs_f64().max(0.001),
        }
        .clamp(0.0, 1.0);
        frame.render_widget(
            Gauge::default()
                .ratio(progress)
                .label(format!(
                    "{}s / {}s  {:>3.0}%",
                    elapsed.as_secs(),
                    self.total_duration.as_secs(),
                    progress * 100.0
                ))
                .gauge_style(
                    Style::default()
                        .fg(Color::LightCyan)
                        .bg(Color::DarkGray)
                        .add_modifier(Modifier::BOLD),
                ),
            progress_area,
        );
    }

    fn render_loss_chart(&self, frame: &mut Frame, area: Rect) {
        let title = match self.separation_width {
            Some(width) => format!(
                " Relative collision loss · attempt {} at width {width:.3} ",
                self.attempt
            ),
            None => " Relative collision loss ".to_owned(),
        };
        let block = Block::bordered()
            .title(title)
            .border_style(Style::default().fg(Color::DarkGray));
        if self.loss_history.is_empty() {
            frame.render_widget(
                Paragraph::new("Waiting for the first separation attempt...")
                    .style(Style::default().fg(Color::DarkGray))
                    .block(block),
                area,
            );
            return;
        }

        let max_iteration = self.iteration.max(1) as f64;
        let dataset = Dataset::default()
            .marker(Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::LightCyan))
            .data(&self.loss_history);
        let chart = Chart::new(vec![dataset])
            .block(block)
            .x_axis(
                Axis::default()
                    .title(" iteration ")
                    .style(Style::default().fg(Color::DarkGray))
                    .bounds([0.0, max_iteration])
                    .labels(vec![
                        TextLine::from("0"),
                        TextLine::from(self.iteration.to_string()),
                    ]),
            )
            .y_axis(
                Axis::default()
                    .title(" loss remaining ")
                    .style(Style::default().fg(Color::DarkGray))
                    .bounds([0.0, 100.0])
                    .labels(vec![TextLine::from("0%"), TextLine::from("100%")]),
            );
        frame.render_widget(chart, area);
    }

    fn render_logs(&self, frame: &mut Frame, area: Rect) {
        let visible_lines = area.height.saturating_sub(2) as usize;
        let lines = self
            .logs
            .iter()
            .rev()
            .take(visible_lines)
            .rev()
            .map(|entry| TextLine::styled(entry.message.as_str(), log_style(entry)))
            .collect::<Vec<_>>();
        frame.render_widget(
            Paragraph::new(lines).block(
                Block::bordered()
                    .title(" Logs ")
                    .border_style(Style::default().fg(Color::DarkGray)),
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
            .fg(Color::LightGreen)
            .add_modifier(Modifier::BOLD)
    } else if entry.message.contains("[EXPL] unable to reach feasibility")
        || entry.message.contains("[CMPR] failed at")
    {
        Style::default()
            .fg(Color::LightYellow)
            .add_modifier(Modifier::BOLD)
    } else if entry.message.contains("[SEP] finished") {
        Style::default()
            .fg(Color::White)
            .add_modifier(Modifier::BOLD)
    } else {
        match entry.level {
            Level::Error => Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            Level::Warn => Style::default().fg(Color::Yellow),
            Level::Info => Style::default().fg(Color::Gray),
            Level::Debug | Level::Trace => Style::default().fg(Color::DarkGray),
        }
    }
}

fn report_is_feasible(report: &ReportType) -> bool {
    match report {
        ReportType::ExplFeas | ReportType::CmprFeas | ReportType::Final => true,
        ReportType::ExplInfeas | ReportType::ExplImproving => false,
    }
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
}

struct TuiTerminator {
    timeout: Option<CdeInstant>,
    stop: Arc<AtomicBool>,
}

impl TuiTerminator {
    fn new(stop: Arc<AtomicBool>) -> Self {
        Self {
            timeout: None,
            stop,
        }
    }
}

impl Terminator for TuiTerminator {
    fn kill(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
            || self
                .timeout
                .is_some_and(|timeout| CdeInstant::now() > timeout)
    }

    fn new_timeout(&mut self, timeout: Duration) {
        self.timeout = Some(CdeInstant::now() + timeout);
    }

    fn timeout_at(&self) -> Option<CdeInstant> {
        self.timeout
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

    #[test]
    fn groups_separation_attempts_by_width() {
        let mut app = App::new(Duration::ZERO);
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
        assert_eq!(app.loss_history, vec![(0.0, 100.0), (1.0, 50.0)]);

        app.apply(progress(100.0, 0, 15.0));
        assert_eq!(app.attempt, 2);
        assert_eq!(app.loss_history, vec![(0.0, 100.0)]);

        app.apply(progress(99.0, 0, 12.0));
        assert_eq!(app.attempt, 1);
    }
}

use anyhow::{Result, anyhow, bail};
use clap::Parser;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use image::{DynamicImage, RgbaImage};
use jagua_rs::Instant as CdeInstant;
use jagua_rs::io::import::Importer;
use jagua_rs::io::svg::s_layout_to_svg;
use jagua_rs::probs::spp::entities::{SPInstance, SPSolution};
use log::{Level, Log, Metadata, Record};
use rand::SeedableRng;
use rand::rngs::Xoshiro256PlusPlus;
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line as TextLine, Span};
use ratatui::widgets::{Block, Gauge, Paragraph};
use ratatui::{DefaultTerminal, Frame};
use ratatui_image::picker::{Picker, ProtocolType};
use ratatui_image::protocol::StatefulProtocol;
use ratatui_image::{Resize, StatefulImage};
use resvg::{tiny_skia, usvg};
use sparrow::EPOCH;
use sparrow::config::{DEFAULT_SPARROW_CONFIG, ShrinkDecayStrategy, SparrowConfig};
use sparrow::consts::{
    DEFAULT_COMPRESS_TIME_RATIO, DEFAULT_EXPLORE_TIME_RATIO, DEFAULT_FAIL_DECAY_RATIO_CMPR,
    DEFAULT_MAX_CONSEQ_FAILS_EXPL, DRAW_OPTIONS, LOG_LEVEL_FILTER_DEBUG, LOG_LEVEL_FILTER_RELEASE,
};
use sparrow::optimizer::optimize;
use sparrow::util::io::{self, ExtSPOutput, MainCli};
use sparrow::util::listener::{ReportType, SolutionListener};
use sparrow::util::terminator::Terminator;
use std::collections::VecDeque;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const OUTPUT_DIR: &str = "output";
const FRAME_INTERVAL: Duration = Duration::from_millis(100);
const SNAPSHOT_INTERVAL: Duration = Duration::from_millis(100);
const MAX_LOG_LINES: usize = 200;
const MAX_RASTER_SIZE: (u32, u32) = (1600, 1000);

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
    let (updates_tx, updates_rx) = mpsc::sync_channel(1);
    let worker = start_optimizer(
        instance.clone(),
        initial_solution,
        config,
        rng,
        stop.clone(),
        updates_tx,
    );

    let solution = ratatui::run(|terminal| {
        run_tui(
            terminal,
            instance.clone(),
            updates_rx,
            logs_rx,
            worker,
            stop,
            total_duration,
        )
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
    updates: SyncSender<Update>,
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
    instance: SPInstance,
    updates: Receiver<Update>,
    logs: Receiver<LogEntry>,
    worker: JoinHandle<SPSolution>,
    stop: Arc<AtomicBool>,
    total_duration: Duration,
) -> Result<SPSolution> {
    let picker = Picker::from_query_stdio()?;
    if picker.protocol_type() == ProtocolType::Halfblocks {
        bail!("this terminal does not support Kitty, Sixel, or iTerm2 images");
    }
    let mut app = App::new(picker, total_duration);
    let mut worker = Some(worker);
    let mut solution = None;

    loop {
        for log in logs.try_iter() {
            app.push_log(log);
        }
        if let Some(update) = updates.try_iter().last() {
            app.apply(update, &instance);
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
    picker: Picker,
    svg_options: usvg::Options<'static>,
    image: Option<StatefulProtocol>,
    report: Option<ReportType>,
    phase: &'static str,
    width: Option<f32>,
    density: Option<f32>,
    started: Instant,
    total_duration: Duration,
    n_updates: usize,
    logs: VecDeque<LogEntry>,
    finished: bool,
    quit_requested: bool,
    render_error: Option<String>,
}

impl App {
    fn new(picker: Picker, total_duration: Duration) -> Self {
        Self {
            picker,
            svg_options: svg_options(),
            image: None,
            report: None,
            phase: "starting",
            width: None,
            density: None,
            started: Instant::now(),
            total_duration,
            n_updates: 0,
            logs: VecDeque::new(),
            finished: false,
            quit_requested: false,
            render_error: None,
        }
    }

    fn apply(&mut self, update: Update, instance: &SPInstance) {
        let Update { report, solution } = update;
        self.phase = report_label(&report);
        self.width = Some(solution.strip_width());
        self.density = Some(solution.density(instance) * 100.0);
        self.report = Some(report);
        self.n_updates += 1;

        let svg = s_layout_to_svg(
            &solution.layout_snapshot,
            instance,
            DRAW_OPTIONS,
            self.phase,
        );
        match rasterize_svg(&svg.to_string(), &self.svg_options) {
            Ok(image) => {
                self.image = Some(self.picker.new_resize_protocol(image));
                self.render_error = None;
            }
            Err(error) => self.render_error = Some(error.to_string()),
        }
    }

    fn push_log(&mut self, log: LogEntry) {
        if self.logs.len() == MAX_LOG_LINES {
            self.logs.pop_front();
        }
        self.logs.push_back(log);
    }

    fn render(&mut self, frame: &mut Frame) {
        let [dashboard_area, canvas_area, logs_area, help_area] = Layout::vertical([
            Constraint::Length(5),
            Constraint::Min(5),
            Constraint::Length(8),
            Constraint::Length(1),
        ])
        .areas(frame.area());

        self.render_dashboard(frame, dashboard_area);
        self.render_image(frame, canvas_area);
        self.render_logs(frame, logs_area);

        let help = match (self.finished, self.quit_requested) {
            (true, _) => "Finished. Press q or Esc to exit.",
            (false, true) => "Stopping optimizer...",
            (false, false) => "q / Esc / Ctrl-C: stop and exit",
        };
        frame.render_widget(
            Paragraph::new(help).style(Style::default().fg(Color::DarkGray)),
            help_area,
        );
    }

    fn render_dashboard(&self, frame: &mut Frame, area: ratatui::layout::Rect) {
        let block = Block::bordered()
            .title(" Sparrow search ")
            .border_style(Style::default().fg(Color::Cyan));
        let inner = block.inner(area);
        frame.render_widget(block, area);
        let [metrics_area, progress_area] =
            Layout::vertical([Constraint::Length(2), Constraint::Length(1)]).areas(inner);

        let (state, state_color) = match self.report.as_ref() {
            Some(report) if report_is_feasible(report) => ("FEASIBLE", Color::Green),
            Some(_) => ("INFEASIBLE", Color::Red),
            None => ("STARTING", Color::Yellow),
        };
        let phase_color = match self.report {
            Some(ReportType::CmprFeas | ReportType::Final) => Color::Magenta,
            Some(_) => Color::Cyan,
            None => Color::DarkGray,
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
        let elapsed = self.started.elapsed();
        let update_rate = self.n_updates as f64 / elapsed.as_secs_f64().max(0.001);
        let metrics = TextLine::from(vec![
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
            Span::styled("   updates ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{update_rate:.1}/s"),
                Style::default().fg(Color::Yellow),
            ),
            Span::styled("   elapsed ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{}s", elapsed.as_secs()),
                Style::default().fg(Color::White),
            ),
        ]);
        frame.render_widget(Paragraph::new(vec![phase, metrics]), metrics_area);

        let progress = match self.finished {
            true => 1.0,
            false => elapsed.as_secs_f64() / self.total_duration.as_secs_f64().max(0.001),
        }
        .clamp(0.0, 1.0);
        frame.render_widget(
            Gauge::default()
                .ratio(progress)
                .label(format!("time budget {:>3.0}%", progress * 100.0))
                .gauge_style(
                    Style::default()
                        .fg(Color::LightCyan)
                        .bg(Color::DarkGray)
                        .add_modifier(Modifier::BOLD),
                ),
            progress_area,
        );
    }

    fn render_image(&mut self, frame: &mut Frame, area: ratatui::layout::Rect) {
        let block = Block::bordered()
            .title(" Live packing ")
            .border_style(Style::default().fg(Color::DarkGray));
        let inner = block.inner(area);
        frame.render_widget(block, area);
        match &mut self.image {
            Some(image) => frame.render_stateful_widget(
                StatefulImage::new().resize(Resize::Fit(None)),
                inner,
                image,
            ),
            None => frame.render_widget(
                Paragraph::new(
                    self.render_error
                        .as_deref()
                        .unwrap_or("Waiting for the initial layout..."),
                ),
                inner,
            ),
        }
    }

    fn render_logs(&self, frame: &mut Frame, area: ratatui::layout::Rect) {
        let visible_lines = area.height.saturating_sub(2) as usize;
        let lines = self
            .logs
            .iter()
            .rev()
            .take(visible_lines)
            .rev()
            .map(|entry| {
                TextLine::styled(
                    entry.message.as_str(),
                    match entry.level {
                        Level::Error => {
                            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
                        }
                        Level::Warn => Style::default().fg(Color::Yellow),
                        Level::Info => Style::default().fg(Color::Gray),
                        Level::Debug | Level::Trace => Style::default().fg(Color::DarkGray),
                    },
                )
            })
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

struct Update {
    report: ReportType,
    solution: SPSolution,
}

struct TuiListener {
    updates: SyncSender<Update>,
    last_snapshot: Option<Instant>,
}

impl TuiListener {
    fn new(updates: SyncSender<Update>) -> Self {
        Self {
            updates,
            last_snapshot: None,
        }
    }
}

impl SolutionListener for TuiListener {
    fn report(&mut self, report: ReportType, solution: &SPSolution, _instance: &SPInstance) {
        let now = Instant::now();
        if report != ReportType::Final
            && self
                .last_snapshot
                .is_some_and(|last| now.duration_since(last) < SNAPSHOT_INTERVAL)
        {
            return;
        }

        let update = Update {
            report: report.clone(),
            solution: solution.clone(),
        };
        let sent = match report {
            ReportType::Final => self.updates.send(update).map_err(|_| ()),
            _ => self.updates.try_send(update).map_err(|_| ()),
        };
        if sent.is_ok() {
            self.last_snapshot = Some(now);
        }
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

fn report_label(report: &ReportType) -> &'static str {
    match report {
        ReportType::ExplFeas | ReportType::ExplInfeas | ReportType::ExplImproving => "exploration",
        ReportType::CmprFeas => "compression",
        ReportType::Final => "final",
    }
}

fn svg_options() -> usvg::Options<'static> {
    let mut options = usvg::Options::default();
    options.fontdb_mut().load_system_fonts();
    options.style_sheet =
        Some("text { fill: #D8DEE9; } [stroke=black] { stroke: #D8DEE9; }".to_owned());
    options
}

fn rasterize_svg(svg: &str, options: &usvg::Options) -> Result<DynamicImage> {
    let tree = usvg::Tree::from_str(svg, options)?;
    let source = tree.size();
    let scale =
        (MAX_RASTER_SIZE.0 as f32 / source.width()).min(MAX_RASTER_SIZE.1 as f32 / source.height());
    let width = (source.width() * scale).round().max(1.0) as u32;
    let height = (source.height() * scale).round().max(1.0) as u32;
    let mut pixmap =
        tiny_skia::Pixmap::new(width, height).ok_or_else(|| anyhow!("SVG raster is too large"))?;
    resvg::render(
        &tree,
        tiny_skia::Transform::from_scale(scale, scale),
        &mut pixmap.as_mut(),
    );
    let pixels = RgbaImage::from_raw(width, height, pixmap.data().to_vec())
        .ok_or_else(|| anyhow!("invalid SVG raster buffer"))?;
    Ok(DynamicImage::ImageRgba8(pixels))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rasterizes_svg_with_its_aspect_ratio() {
        let image = rasterize_svg(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100"><rect x="50" y="25" width="100" height="50" fill="red" stroke="black" stroke-width="10"/></svg>"#,
            &svg_options(),
        )
        .unwrap();

        assert_eq!((image.width(), image.height()), (1600, 800));
        let image = image.to_rgba8();
        assert_eq!(image.get_pixel(0, 0).0[3], 0);
        assert_eq!(image.get_pixel(400, 400).0, [216, 222, 233, 255]);
    }
}

use anyhow::{Result, anyhow, bail};
use clap::Parser;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use image::{DynamicImage, RgbaImage};
use jagua_rs::Instant as CdeInstant;
use jagua_rs::io::import::Importer;
use jagua_rs::io::svg::s_layout_to_svg;
use jagua_rs::probs::spp::entities::{SPInstance, SPSolution};
use log::Level;
use rand::SeedableRng;
use rand::rngs::Xoshiro256PlusPlus;
use ratatui::layout::{Constraint, Layout};
use ratatui::widgets::{Block, Paragraph};
use ratatui::{DefaultTerminal, Frame};
use ratatui_image::picker::Picker;
use ratatui_image::protocol::StatefulProtocol;
use ratatui_image::{Resize, StatefulImage};
use resvg::{tiny_skia, usvg};
use sparrow::EPOCH;
use sparrow::config::{DEFAULT_SPARROW_CONFIG, ShrinkDecayStrategy, SparrowConfig};
use sparrow::consts::{
    DEFAULT_COMPRESS_TIME_RATIO, DEFAULT_EXPLORE_TIME_RATIO, DEFAULT_FAIL_DECAY_RATIO_CMPR,
    DEFAULT_MAX_CONSEQ_FAILS_EXPL, DRAW_OPTIONS,
};
use sparrow::optimizer::optimize;
use sparrow::util::io::{self, ExtSPOutput, MainCli};
use sparrow::util::listener::{ReportType, SolutionListener};
use sparrow::util::terminator::Terminator;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const OUTPUT_DIR: &str = "output";
const FRAME_INTERVAL: Duration = Duration::from_millis(50);
const SNAPSHOT_INTERVAL: Duration = Duration::from_millis(250);
const MAX_RASTER_SIZE: (u32, u32) = (1600, 1000);

fn main() -> Result<()> {
    let args = MainCli::parse();
    let config = configure(&args)?;
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

    let picker = Picker::from_query_stdio().unwrap_or_else(|_| Picker::halfblocks());
    let solution = ratatui::run(|terminal| {
        run_tui(terminal, picker, instance.clone(), updates_rx, worker, stop)
    })?;

    fs::create_dir_all(OUTPUT_DIR)?;
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
    picker: Picker,
    instance: SPInstance,
    updates: Receiver<Update>,
    worker: JoinHandle<SPSolution>,
    stop: Arc<AtomicBool>,
) -> Result<SPSolution> {
    let mut app = App::new(picker);
    let mut worker = Some(worker);
    let mut solution = None;

    loop {
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
    phase: &'static str,
    width: Option<f32>,
    density: Option<f32>,
    started: Instant,
    finished: bool,
    quit_requested: bool,
    render_error: Option<String>,
}

impl App {
    fn new(picker: Picker) -> Self {
        Self {
            picker,
            svg_options: svg_options(),
            image: None,
            phase: "starting",
            width: None,
            density: None,
            started: Instant::now(),
            finished: false,
            quit_requested: false,
            render_error: None,
        }
    }

    fn apply(&mut self, update: Update, instance: &SPInstance) {
        self.phase = report_label(&update.report);
        self.width = Some(update.solution.strip_width());
        self.density = Some(update.solution.density(instance) * 100.0);

        let svg = s_layout_to_svg(
            &update.solution.layout_snapshot,
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

    fn render(&mut self, frame: &mut Frame) {
        let [status_area, image_area, help_area] = Layout::vertical([
            Constraint::Length(3),
            Constraint::Min(1),
            Constraint::Length(1),
        ])
        .areas(frame.area());

        let status = match (self.width, self.density) {
            (Some(width), Some(density)) => format!(
                "{}  |  width {width:.3}  |  density {density:.3}%  |  elapsed {}s",
                self.phase,
                self.started.elapsed().as_secs(),
            ),
            _ => "Waiting for the initial layout...".to_owned(),
        };
        frame.render_widget(
            Paragraph::new(status).block(Block::bordered().title(" Sparrow ")),
            status_area,
        );

        let image_block = Block::bordered().title(" Live layout ");
        let image_inner = image_block.inner(image_area);
        frame.render_widget(image_block, image_area);
        match &mut self.image {
            Some(image) => frame.render_stateful_widget(
                StatefulImage::new().resize(Resize::Fit(None)),
                image_inner,
                image,
            ),
            None => frame.render_widget(
                Paragraph::new(self.render_error.as_deref().unwrap_or("No layout yet")),
                image_inner,
            ),
        }

        let help = match (self.finished, self.quit_requested) {
            (true, _) => "Finished. Press q or Esc to exit.",
            (false, true) => "Stopping optimizer...",
            (false, false) => "q / Esc / Ctrl-C: stop and exit",
        };
        frame.render_widget(Paragraph::new(help), help_area);
    }
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
        ReportType::ExplFeas => "exploration / feasible",
        ReportType::ExplInfeas => "exploration / infeasible",
        ReportType::ExplImproving => "exploration / improving",
        ReportType::CmprFeas => "compression / feasible",
        ReportType::Final => "final",
    }
}

fn svg_options() -> usvg::Options<'static> {
    let mut options = usvg::Options::default();
    options.fontdb_mut().load_system_fonts();
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
    pixmap.fill(tiny_skia::Color::WHITE);
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
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100"><rect width="200" height="100" fill="red"/></svg>"#,
            &svg_options(),
        )
        .unwrap();

        assert_eq!((image.width(), image.height()), (1600, 800));
    }
}

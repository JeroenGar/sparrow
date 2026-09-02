use super::OUTPUT_DIR;
use super::dashboard::{Dashboard, DashboardUpdate, SearchBudget};
use super::logging::LogEntry;
use anyhow::{Result, anyhow};
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use jagua_rs::Instant as CdeInstant;
use jagua_rs::io::svg::s_layout_to_svg;
use jagua_rs::probs::spp::entities::{SPInstance, SPSolution};
use jagua_rs::probs::spp::io::ext_repr::ExtSPInstance;
use log::Level;
use rand::rngs::Xoshiro256PlusPlus;
use ratatui::DefaultTerminal;
use sparrow::EPOCH;
use sparrow::config::SparrowConfig;
use sparrow::consts::DRAW_OPTIONS;
use sparrow::optimizer::optimize;
use sparrow::util::io::{self, ExtSPOutput};
use sparrow::util::listener::{
    OptimizationPhase, ReportType, SeparationProgress, SeparationResult, SolutionListener,
};
use sparrow::util::svg_exporter::SvgExporter;
use sparrow::util::terminator::Terminator;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const LIVE_SVG_PATH: &str = "data/live/.live_solution.svg";
const FRAME_INTERVAL: Duration = Duration::from_millis(100);
const SNAPSHOT_INTERVAL: Duration = Duration::from_millis(100);

pub(super) fn run(
    instance: SPInstance,
    ext_instance: &ExtSPInstance,
    initial_solution: Option<SPSolution>,
    config: SparrowConfig,
    rng: Xoshiro256PlusPlus,
    logs: Receiver<LogEntry>,
    budget: SearchBudget,
) -> Result<SPSolution> {
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

    ratatui::run(|terminal| {
        run_dashboard(
            terminal,
            updates_rx,
            logs,
            worker,
            signals,
            (&instance, ext_instance),
            budget,
        )
    })
}

fn start_optimizer(
    instance: SPInstance,
    initial_solution: Option<SPSolution>,
    config: SparrowConfig,
    rng: Xoshiro256PlusPlus,
    signals: TuiSignals,
    updates: Sender<DashboardUpdate>,
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

fn run_dashboard(
    terminal: &mut DefaultTerminal,
    updates: Receiver<DashboardUpdate>,
    logs: Receiver<LogEntry>,
    worker: JoinHandle<SPSolution>,
    signals: TuiSignals,
    final_output: (&SPInstance, &ExtSPInstance),
    budget: SearchBudget,
) -> Result<SPSolution> {
    let (instance, ext_instance) = final_output;
    let mut dashboard = Dashboard::new(budget);
    let mut worker = Some(worker);
    let mut solution = None;

    loop {
        for log in logs.try_iter() {
            dashboard.push_log(log);
        }
        for update in updates.try_iter() {
            dashboard.apply(update);
        }
        if worker.as_ref().is_some_and(JoinHandle::is_finished) {
            let final_solution = worker
                .take()
                .unwrap()
                .join()
                .map_err(|_| anyhow!("optimizer thread panicked"))?;
            export_final_solution(&final_solution, instance, ext_instance)?;
            solution = Some(final_solution);
            dashboard.finish();
        }

        terminal.draw(|frame| dashboard.render(frame))?;

        if dashboard.quit_requested()
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
                    dashboard.request_quit();
                    signals.request_quit();
                }
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    signals.interrupt_phase();
                }
                KeyCode::Up => dashboard.scroll_logs_up(1),
                KeyCode::Down => dashboard.scroll_logs_down(1),
                KeyCode::PageUp => dashboard.scroll_logs_up(10),
                KeyCode::PageDown => dashboard.scroll_logs_down(10),
                KeyCode::Home => dashboard.scroll_logs_up(usize::MAX),
                KeyCode::End => dashboard.show_latest_log(),
                _ => {}
            }
        }
    }
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

struct TuiListener {
    updates: Sender<DashboardUpdate>,
    live_svg: SvgExporter,
    last_snapshot: Option<Instant>,
}

impl TuiListener {
    fn new(updates: Sender<DashboardUpdate>) -> Self {
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
        let update = DashboardUpdate::Solution {
            report: report.clone(),
            width: solution.strip_width(),
            density: solution.density(instance) * 100.0,
        };
        let _ = self.updates.send(update);
    }

    fn report_phase(&mut self, phase: OptimizationPhase) {
        let _ = self.updates.send(DashboardUpdate::Phase(phase));
    }

    fn report_separation_progress(&mut self, progress: SeparationProgress) {
        let _ = self.updates.send(DashboardUpdate::Separation(progress));
    }

    fn report_separation_result(&mut self, result: SeparationResult) {
        let _ = self.updates.send(DashboardUpdate::SeparationResult(result));
    }

    fn report_compression_progress(&mut self, shrink_step: f32) {
        let _ = self.updates.send(DashboardUpdate::Compression(shrink_step));
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

    fn request_quit(&self) {
        self.quit.store(true, Ordering::Relaxed);
    }

    fn interrupt_phase(&self) {
        self.interrupt_phase.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests;

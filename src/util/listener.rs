use jagua_rs::probs::spp::entities::{SPInstance, SPSolution};

/// Trait for listeners that can receive solutions during the optimization process
pub trait SolutionListener {
    fn report(&mut self, report: ReportType, solution: &SPSolution, instance: &SPInstance);

    fn report_phase(&mut self, _phase: OptimizationPhase) {}

    fn report_separation_progress(&mut self, _progress: SeparationProgress) {}

    fn report_compression_progress(&mut self, _shrink_step: f32) {}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationPhase {
    Exploration,
    Compression,
}

/// Progress within one call to [`crate::optimizer::separator::Separator::separate`].
/// Iteration zero describes the initial layout; later values describe completed iterations.
#[derive(Debug, Clone, Copy)]
pub struct SeparationProgress {
    pub strip_width: f32,
    pub density: f32,
    pub iteration: usize,
    pub min_loss: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReportType {
    /// Report contains a feasible solution reached by the exploration phase.
    ExplFeas,
    /// Report contains an infeasible solution reached by the exploration phase.
    ExplInfeas,
    /// Report contains an intermediate solution from the exploration phase that is closer to feasibility than the previous one.
    ExplImproving,
    /// Report contains a feasible solution from the comparison phase.
    CmprFeas,
    /// Report contains the final solution
    Final,
}

/// A dummy implementation of the `SolutionListener` trait that does nothing.
pub struct DummySolListener;

impl SolutionListener for DummySolListener {
    fn report(&mut self, _report: ReportType, _solution: &SPSolution, _instance: &SPInstance) {
        // Do nothing
    }
}

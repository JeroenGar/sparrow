use crate::config::*;
use crate::consts::LBF_SAMPLE_CONFIG;
use crate::optimizer::compress::compression_phase;
use crate::optimizer::explore::exploration_phase;
use crate::optimizer::lbf::LBFBuilder;
use crate::optimizer::separator::Separator;
use crate::sample::uniform_sampler::allowed_poses;
use crate::util::listener::{OptimizationPhase, ReportType, SolutionListener};
use crate::util::terminator::Terminator;
use jagua_rs::geometry::geo_enums::RotationRange;
use jagua_rs::geometry::geo_traits::TransformableFrom;
use jagua_rs::geometry::DTransformation;
use jagua_rs::probs::spp::entities::{SPInstance, SPProblem, SPSolution};
use log::info;
use rand::{Rng, SeedableRng};
use std::time::Duration;
use rand::rngs::Xoshiro256PlusPlus;

pub mod lbf;
pub mod separator;
mod worker;
pub mod explore;
pub mod compress;

/// Algorithm 11 from https://doi.org/10.48550/arXiv.2509.13329
///
/// Returns a construction error if no initial solution is supplied and the
/// initial-placement heuristic cannot build one within its strip growth limit.
pub fn optimize(
    instance: SPInstance,
    mut rng: Xoshiro256PlusPlus,
    sol_listener: &mut impl SolutionListener,
    terminator: &mut impl Terminator,
    expl_config: &ExplorationConfig,
    cmpr_config: &CompressionConfig,
    initial_solution: Option<&SPSolution>
) -> anyhow::Result<SPSolution> {
    let mut next_rng = || Xoshiro256PlusPlus::seed_from_u64(rng.next_u64());
    
    // First build an initial solution if none is provided
    let start_prob = match initial_solution {
        None => {
            let builder = LBFBuilder::new(instance, next_rng(), LBF_SAMPLE_CONFIG)?.construct()?;
            builder.prob
        }
        Some(init_sol) => {
            info!("[OPT] warm starting from provided initial solution");
            let mut prob = jagua_rs::probs::spp::entities::SPProblem::new(instance)?;
            prob.restore(init_sol);
            prob
        }
    };

    // Begin by executing the exploration phase
    sol_listener.report_phase(OptimizationPhase::Exploration);
    terminator.new_timeout(expl_config.time_limit);
    let mut expl_separator = Separator::new(start_prob, next_rng(), expl_config.separator_config);
    let solutions = exploration_phase(
        &mut expl_separator,
        sol_listener,
        terminator,
        expl_config,
    );
    let final_explore_sol = solutions.into_iter().last().unwrap();

    // Start the compression phase from the final solution from the exploration phase
    sol_listener.report_phase(OptimizationPhase::Compression);
    terminator.new_timeout(cmpr_config.time_limit);
    let mut cmpr_separator = Separator::new(expl_separator.prob, next_rng(), cmpr_config.separator_config);
    let cmpr_sol = compression_phase(
        &mut cmpr_separator,
        &final_explore_sol,
        sol_listener,
        terminator,
        cmpr_config,
    );

    sol_listener.report(ReportType::Final, &cmpr_sol);

    // Return the final compressed solution
    Ok(cmpr_sol)
}

/// Conservative lower bound on strip width, including the container's inset.
/// Leave relative slack for f32 rotation and bounding-box rounding near exact fits.
fn strip_width_lower_bound(prob: &SPProblem) -> f32 {
    const REL_TOL: f32 = 1e-6;
    let container = prob.layout().container().outer_cd().bbox();
    let width_inset = prob.strip_width() - container.width();
    let area: f64 = prob.instance().items.iter()
        .map(|(item, qty)| f64::from(item.shape_cd().area()) * *qty as f64)
        .sum();
    let mut width_lower_bound = (area / f64::from(container.height())) as f32 * (1.0 - REL_TOL);

    for (item, _) in &prob.instance().items {
        // ponytail: continuous rotations use only the area bound; add exact rotational bounds if this is too weak.
        // Reflected rotations are continuous exactly when unreflected ones are.
        if item.allowed_orientations().rotations() == &RotationRange::Continuous {
            continue;
        }
        let tolerance = item.shape_cd().diameter() * REL_TOL;
        let mut shape = item.shape_cd().as_ref().clone();
        // Reflected poses count too: one can be narrower than every unreflected pose.
        let item_width = allowed_poses(item)
            .filter_map(|(reflected, rotation)| {
                let pose = DTransformation::new(rotation, (0.0, 0.0)).with_reflection(reflected);
                let bbox = shape.transform_from(item.shape_cd().as_ref(), &pose.compose()).bbox();
                (bbox.height() <= container.height() + tolerance)
                    .then_some((bbox.width() - tolerance).max(0.0))
            })
            .fold(f32::INFINITY, f32::min);
        width_lower_bound = width_lower_bound.max(item_width);
    }
    width_lower_bound + width_inset
}

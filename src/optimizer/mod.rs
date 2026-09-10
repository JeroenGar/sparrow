use crate::config::*;
use crate::consts::LBF_SAMPLE_CONFIG;
use crate::optimizer::compress::compression_phase;
use crate::optimizer::explore::exploration_phase;
use crate::optimizer::lbf::{ConstructionError, LBFBuilder};
use crate::optimizer::separator::Separator;
use crate::util::listener::{OptimizationPhase, ReportType, SolutionListener};
use crate::util::terminator::Terminator;
use jagua_rs::geometry::geo_enums::RotationRange;
use jagua_rs::geometry::geo_traits::TransformableFrom;
use jagua_rs::geometry::Transformation;
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
) -> Result<SPSolution, ConstructionError> {
    let mut next_rng = || Xoshiro256PlusPlus::seed_from_u64(rng.next_u64());
    
    // First build an initial solution if none is provided
    let start_prob = match initial_solution {
        None => {
            let builder = LBFBuilder::new(instance.clone(), next_rng(), LBF_SAMPLE_CONFIG).construct()?;
            builder.prob
        }
        Some(init_sol) => {
            info!("[OPT] warm starting from provided initial solution");
            let mut prob = jagua_rs::probs::spp::entities::SPProblem::new(instance.clone());
            prob.restore(init_sol);
            prob
        }
    };

    // Begin by executing the exploration phase
    sol_listener.report_phase(OptimizationPhase::Exploration);
    terminator.new_timeout(expl_config.time_limit);
    let mut expl_separator = Separator::new(instance.clone(), start_prob, next_rng(), expl_config.separator_config);
    let solutions = exploration_phase(
        &instance,
        &mut expl_separator,
        sol_listener,
        terminator,
        expl_config,
    );
    let final_explore_sol = solutions.last().unwrap().clone();

    // Start the compression phase from the final solution from the exploration phase
    sol_listener.report_phase(OptimizationPhase::Compression);
    terminator.new_timeout(cmpr_config.time_limit);
    let mut cmpr_separator = Separator::new(expl_separator.instance, expl_separator.prob, next_rng(), cmpr_config.separator_config);
    let cmpr_sol = compression_phase(
        &instance,
        &mut cmpr_separator,
        &final_explore_sol,
        sol_listener,
        terminator,
        cmpr_config,
    );

    sol_listener.report(ReportType::Final, &cmpr_sol, &instance);

    // Return the final compressed solution
    Ok(cmpr_sol)
}

/// Necessary width for the collision geometry, including the container's inset.
/// Leave relative slack for f32 rotation and bounding-box rounding near exact fits.
fn minimum_strip_width(prob: &SPProblem) -> f32 {
    const REL_TOL: f32 = 1e-6;
    let container = prob.layout.container.outer_cd.bbox;
    let width_inset = prob.strip_width() - container.width();
    let area: f64 = prob.instance.items.iter()
        .map(|(item, qty)| f64::from(item.shape_cd.area) * *qty as f64)
        .sum();
    let mut min_width = (area / f64::from(container.height())) as f32 * (1.0 - REL_TOL);

    for (item, _) in &prob.instance.items {
        let rotations = match &item.allowed_rotation {
            RotationRange::None => &[0.0][..],
            RotationRange::Discrete(rotations) => rotations.as_slice(),
            // ponytail: continuous rotations use only the area bound; add exact rotational bounds if this is too weak.
            RotationRange::Continuous => continue,
        };
        let tolerance = item.shape_cd.diameter * REL_TOL;
        let mut shape = item.shape_cd.as_ref().clone();
        let item_width = rotations.iter()
            .filter_map(|&rotation| {
                let bbox = shape.transform_from(item.shape_cd.as_ref(), &Transformation::from_rotation(rotation)).bbox;
                (bbox.height() <= container.height() + tolerance)
                    .then_some((bbox.width() - tolerance).max(0.0))
            })
            .fold(f32::INFINITY, f32::min);
        min_width = min_width.max(item_width);
    }
    min_width + width_inset
}

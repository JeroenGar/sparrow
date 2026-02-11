use crate::config::{CompressionConfig, ShrinkDecayStrategy};
use crate::optimizer::separator::Separator;
use crate::util::listener::{ReportType, SolutionListener};
use crate::util::terminator::Terminator;
use jagua_rs::probs::spp::entities::{SPInstance, SPSolution};
use jagua_rs::Instant;
use log::info;
use rand::{Rng, RngExt};

/// Algorithm 13 from https://doi.org/10.48550/arXiv.2509.13329
pub fn compression_phase(
    instance: &SPInstance,
    sep: &mut Separator,
    init_sol: &SPSolution,
    sol_listener: &mut impl SolutionListener,
    term: &impl Terminator,
    config: &CompressionConfig
) -> SPSolution {
    let mut best_sol = init_sol.clone();
    let start = Instant::now();
    let mut n_failed_attempts = 0;

    // Create the function to calculate the shrink step size.
    let shrink_step_size = |n_failed_attempts: i32| -> f32 {
        match config.shrink_decay {
            ShrinkDecayStrategy::TimeBased => {
                let range = config.shrink_range.1 - config.shrink_range.0;
                let elapsed = start.elapsed();
                let ratio = elapsed.as_secs_f32() / config.time_limit.as_secs_f32();
                config.shrink_range.0 + ratio * range
            }
            ShrinkDecayStrategy::FailureBased(r) => {
                config.shrink_range.0 * r.powi(n_failed_attempts)
            }
        }
    };

    // As long as the shrink step size is above the minimum, keep attempting to compress
    while !term.kill() && let step = shrink_step_size(n_failed_attempts) && step >= config.shrink_range.1 {
        match attempt_to_compress(sep, &best_sol, step, term, sol_listener) {
            Some(compacted_sol) => {
                info!("[CMPR] success at {:.3}% ({:.3} | {:.3}%)", step * 100.0, compacted_sol.strip_width(), compacted_sol.density(instance) * 100.0);
                sol_listener.report(ReportType::CmprFeas, &compacted_sol, instance);
                best_sol = compacted_sol;
            }
            None => {
                info!("[CMPR] failed at {:.3}%", step * 100.0);
                n_failed_attempts += 1;
            }
        }
    }
    info!("[CMPR] finished, compressed from {:.3}% to {:.3}% (+{:.3}%)", init_sol.density(instance) * 100.0, best_sol.density(instance) * 100.0, (best_sol.density(instance) - init_sol.density(instance)) * 100.0);
    best_sol
}


fn attempt_to_compress(sep: &mut Separator, init_sol: &SPSolution, r_shrink: f32, term: &impl Terminator, sol_listener: &mut impl SolutionListener) -> Option<SPSolution> {
    // Restore to the initial solution and width
    sep.change_strip_width(init_sol.strip_width(), None);
    sep.rollback(init_sol, None);

    // Shrink the container by the provided amount at a random position
    let new_width = init_sol.strip_width() * (1.0 - r_shrink);
    let split_pos = sep.rng.random_range(0.0..sep.prob.strip_width());
    sep.change_strip_width(new_width, Some(split_pos));

    // Try to separate layout, if all collisions are eliminated, return the solution
    let (compacted_sol, ot) = sep.separate(term, sol_listener);
    match ot.get_total_loss() == 0.0 {
        true => Some(compacted_sol),
        false => None,
    }
}
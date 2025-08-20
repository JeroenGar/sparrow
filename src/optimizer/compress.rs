use jagua_rs::Instant;
use jagua_rs::probs::spp::entities::{SPInstance, SPSolution};
use log::info;
use rand::Rng;
use crate::config::{CompressionConfig, ShrinkDecayStrategy};
use crate::optimizer::separator::Separator;
use crate::util::listener::{ReportType, SolutionListener};
use crate::util::terminator::Terminator;

pub fn compression_phase(
    instance: &SPInstance, 
    sep: &mut Separator, 
    init: &SPSolution,
    sol_listener: &mut impl SolutionListener, 
    term: &impl Terminator,
    config: &CompressionConfig
) -> SPSolution {
    let mut best = init.clone();
    let start = Instant::now();
    let mut n_failed_attempts = 0;

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
    while !term.kill() && let step = shrink_step_size(n_failed_attempts) && step >= config.shrink_range.1 {
        match attempt_to_compress(sep, &best, step, term, sol_listener) {
            Some(compacted_sol) => {
                info!("[CMPR] success at {:.3}% ({:.3} | {:.3}%)", step * 100.0, compacted_sol.strip_width(), compacted_sol.density(instance) * 100.0);
                sol_listener.report(ReportType::CmprFeas, &compacted_sol, instance);
                best = compacted_sol;
            }
            None => {
                info!("[CMPR] failed at {:.3}%", step * 100.0);
                n_failed_attempts += 1;
            }
        }
    }
    info!("[CMPR] finished, compressed from {:.3}% to {:.3}% (+{:.3}%)", init.density(instance) * 100.0, best.density(instance) * 100.0, (best.density(instance) - init.density(instance)) * 100.0);
    best
}


fn attempt_to_compress(sep: &mut Separator, init: &SPSolution, r_shrink: f32, term: &impl Terminator, sol_listener: &mut impl SolutionListener) -> Option<SPSolution> {
    //restore to the initial solution and width
    sep.change_strip_width(init.strip_width(), None);
    sep.rollback(&init, None);

    //shrink the container at a random position
    let new_width = init.strip_width() * (1.0 - r_shrink);
    let split_pos = sep.rng.random_range(0.0..sep.prob.strip_width());
    sep.change_strip_width(new_width, Some(split_pos));

    //try to separate layout, if all collisions are eliminated, return the solution
    let (compacted_sol, ot) = sep.separate(term, sol_listener);
    match ot.get_total_loss() == 0.0 {
        true => Some(compacted_sol),
        false => None,
    }
}
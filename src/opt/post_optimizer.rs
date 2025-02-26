/*
    General idea: during the last 20% of time, take very small steps to reduce the bin size. Aggressively restoring from previous solutions.
    Run this on top of gls_orchestrator.rs, do a random split position
 */
use std::time::Instant;
use jagua_rs::entities::solution::Solution;
use jagua_rs::fsize;
use log::{debug, info};
use rand::Rng;
use crate::opt::gls_orchestrator::{GLSOrchestrator, R_SHRINK};

pub const SHRINK_STEP: fsize = R_SHRINK / 10.0; // one tenth of the normal shrink

pub fn compress(gls: &mut GLSOrchestrator, initial_solution: &Solution, time_out: Instant) -> Solution {

    gls.change_strip_width(initial_solution.layout_snapshots[0].bin.bbox().width(), None);
    gls.rollback(&initial_solution, None);

    let new_width = gls.master_prob.strip_width() * (1.0 - SHRINK_STEP);
    let split_pos = gls.rng.random_range(0.0..gls.master_prob.strip_width());

    gls.change_strip_width(new_width, Some(split_pos));
    info!("[POST] attempting to reduce width to {}", new_width);

    let separated = gls.separate_layout(time_out);
    let best_solution = match separated.1.total_overlap == 0.0 {
        true => {
            info!("[POST] reached improved solution with {} width ({:.3}%)", new_width, separated.0.usage);
            gls.write_to_disk(Some(separated.0.clone()), "post_c", true);
            separated.0
        },
        false => {
            gls.write_to_disk(Some(separated.0.clone()), "post_o", false);
            debug!("[POST] no improvement, returning initial solution");
            initial_solution.clone()
        },
    };

    match Instant::now() < time_out {
        true => compress(gls, &best_solution, time_out),
        false => best_solution,
    }
}
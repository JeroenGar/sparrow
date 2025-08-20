use crate::config::*;
use crate::optimizer::lbf::LBFBuilder;
use crate::optimizer::separator::Separator;
use jagua_rs::probs::spp::entities::{SPInstance, SPSolution};
use rand::prelude::SmallRng;
use rand::{RngCore, SeedableRng};
use std::time::Duration;
use crate::consts::LBF_SAMPLE_CONFIG;
use crate::optimizer::compress::compression_phase;
use crate::optimizer::explore::exploration_phase;
use crate::util::listener::{ReportType, SolutionListener};
use crate::util::terminator::Terminator;

pub mod lbf;
pub mod separator;
mod worker;
pub mod explore;
pub mod compress;

pub fn optimize(instance: SPInstance, mut rng: SmallRng, sol_listener: &mut impl SolutionListener, terminator: &mut impl Terminator, expl_config: &ExplorationConfig, cmpr_config: &CompressionConfig) -> SPSolution {
    let mut next_rng = || SmallRng::seed_from_u64(rng.next_u64());
    let builder = LBFBuilder::new(instance.clone(), next_rng(), LBF_SAMPLE_CONFIG).construct();

    terminator.new_timeout(expl_config.time_limit);
    let mut expl_separator = Separator::new(builder.instance, builder.prob, next_rng(), expl_config.separator_config);
    let solutions = exploration_phase(
        &instance,
        &mut expl_separator,
        sol_listener,
        terminator,
        expl_config,
    );
    let final_explore_sol = solutions.last().unwrap().clone();

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

    cmpr_sol
}
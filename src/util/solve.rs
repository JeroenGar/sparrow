use crate::config::SparrowConfig;
use crate::optimizer::optimize;
use crate::util::listener::DummySolListener;
use crate::util::terminator::BasicTerminator;
use jagua_rs::Instant;
use jagua_rs::io::import::Importer;
use jagua_rs::probs::spp::io::ext_repr::{ExtSPInstance, ExtSPSolution};
use jagua_rs::probs::spp::io::{export, import_instance, import_solution};
use rand::SeedableRng;
use rand::rngs::Xoshiro256PlusPlus;

/// Nests all items of `instance` in a strip of minimal width.
///
/// Runs the complete heuristic (construction, exploration and compression) with `config`,
/// and returns the best feasible solution found. The run takes as long as the time limits in
/// `config` allow, unless [`SparrowConfig::with_early_termination`] lets it stop sooner.
///
/// Returns an error if `instance` is invalid, or if no initial solution can be constructed.
pub fn solve(instance: &ExtSPInstance, config: &SparrowConfig) -> anyhow::Result<ExtSPSolution> {
    run(instance, None, config)
}

/// Like [`solve`], but starts from `initial` instead of constructing a first solution.
///
/// `initial` should be a feasible solution for `instance` that places every item, such as the
/// result of an earlier run: sparrow treats it as its first feasible solution and searches for
/// narrower ones from there.
///
/// Returns an error if `instance` is invalid, or if `initial` does not fit it.
pub fn solve_from(instance: &ExtSPInstance, initial: &ExtSPSolution, config: &SparrowConfig) -> anyhow::Result<ExtSPSolution> {
    run(instance, Some(initial), config)
}

fn run(instance: &ExtSPInstance, initial: Option<&ExtSPSolution>, config: &SparrowConfig) -> anyhow::Result<ExtSPSolution> {
    let start = Instant::now();
    let importer = Importer::new(config.cde_config, config.poly_simpl_tolerance, config.narrow_concavity_cutoff_ratio);
    let instance = import_instance(&importer, instance)?;
    let initial = initial.map(|solution| import_solution(&instance, solution)).transpose()?;
    let seed = config.rng_seed.map_or_else(rand::random, |seed| seed as u64);

    let solution = optimize(
        instance,
        Xoshiro256PlusPlus::seed_from_u64(seed),
        &mut DummySolListener,
        &mut BasicTerminator::new(),
        &config.expl_cfg,
        &config.cmpr_cfg,
        initial.as_ref(),
    )?;
    Ok(export(&solution, start))
}

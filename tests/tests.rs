#[cfg(test)]
mod integration_tests {
    use sparrow::util::terminator::Terminator;
use rand::prelude::SmallRng;
    use rand::SeedableRng;
    use sparrow::config::{CDE_CONFIG, LBF_SAMPLE_CONFIG, OUTPUT_DIR, SEP_CFG_EXPLORE, SIMPL_TOLERANCE, MIN_ITEM_SEPARATION};
    use sparrow::optimizer::lbf::LBFBuilder;
    use sparrow::optimizer::separator::Separator;
    use sparrow::util::io;
    use std::path::Path;
    use std::time::Duration;
    use test_case::test_case;
    use anyhow::Result;
    use jagua_rs::io::import::Importer;
    use sparrow::optimizer::compress::{compression_phase, ShrinkDecayStrategy};
    use sparrow::optimizer::explore::exploration_phase;
    use sparrow::util::svg_exporter::SvgExporter;
    use sparrow::util::terminator::BasicTerminator;

    const EXPLORE_TIMEOUT: Duration = Duration::from_secs(10);
    const COMPRESS_TIMEOUT: Duration = Duration::from_secs(10);
    const INSTANCE_BASE_PATH: &str = "data/input";
    const RNG_SEED: Option<usize> = Some(0); // fix seed for reproducibility
    
    #[test_case("swim.json"; "swim")]
    #[test_case("shirts.json"; "shirts")]
    #[test_case("trousers.json"; "trousers")]
    fn simulate_optimization(path: &str) -> Result<()> {
        let input_file_path = format!("{INSTANCE_BASE_PATH}/{path}");
        let json_instance = io::read_spp_instance_json(Path::new(&input_file_path))?;

        let importer = Importer::new(CDE_CONFIG, SIMPL_TOLERANCE, MIN_ITEM_SEPARATION);
        let instance = jagua_rs::probs::spp::io::import(&importer, &json_instance)?;

        println!("[TEST] loaded instance: {}", json_instance.name);
        
        let rng = match RNG_SEED {
            Some(seed) => {
                println!("[TEST] using provided seed: {}", seed);
                SmallRng::seed_from_u64(seed as u64)
            }
            None => {
                let seed = rand::random();
                println!("[TEST] no seed provided, using: {}", seed);
                SmallRng::seed_from_u64(seed)
            }
        };

        let mut terminator = BasicTerminator::new();
        let mut sol_listener = SvgExporter::new(
            Some(format!("{OUTPUT_DIR}/tests_{}", json_instance.name)),
            None,
            None
        );
        terminator.new_timeout(EXPLORE_TIMEOUT);

        let builder = LBFBuilder::new(instance.clone(), rng, LBF_SAMPLE_CONFIG).construct();
        let mut separator = Separator::new(builder.instance, builder.prob, builder.rng,SEP_CFG_EXPLORE);

        let sols = exploration_phase(&instance, &mut separator, &mut sol_listener, &terminator, None);
        let final_explore_sol = sols.last().expect("no solutions found during exploration");

        terminator.new_timeout(COMPRESS_TIMEOUT);
        compression_phase(&instance, &mut separator, final_explore_sol, &mut sol_listener, &terminator, ShrinkDecayStrategy::TimeBased(terminator.timeout_at().unwrap()));
        Ok(())
    }
}
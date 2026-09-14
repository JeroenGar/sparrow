#[cfg(test)]
mod integration_tests {
    use anyhow::Result;
    use jagua_rs::io::import::Importer;
    use sparrow::config::DEFAULT_SPARROW_CONFIG;
    use sparrow::consts::LBF_SAMPLE_CONFIG;
    use sparrow::optimizer::compress::compression_phase;
    use sparrow::optimizer::explore::exploration_phase;
    use sparrow::optimizer::lbf::LBFBuilder;
    use sparrow::optimizer::separator::Separator;
    use sparrow::util::io;
    use sparrow::util::terminator::BasicTerminator;
    use sparrow::util::terminator::Terminator;
    use std::path::Path;
    use std::time::Duration;
    use rand::rngs::Xoshiro256PlusPlus;
    use rand::SeedableRng;
    use test_case::test_case;
    use sparrow::util::listener::DummySolListener;

    const EXPLORE_TIMEOUT: Duration = Duration::from_secs(10);
    const COMPRESS_TIMEOUT: Duration = Duration::from_secs(10);
    const INSTANCE_BASE_PATH: &str = "data/input";
    const RNG_SEED: Option<usize> = Some(0); // fix seed for reproducibility

    #[test_case("swim.json"; "swim")]
    #[test_case("shirts.json"; "shirts")]
    #[test_case("trousers.json"; "trousers")]
    fn simulate_optimization(path: &str) -> Result<()> {
        let config = DEFAULT_SPARROW_CONFIG;
        let input_file_path = format!("{INSTANCE_BASE_PATH}/{path}");
        let (mut json_instance, _) = io::read_spp_input(Path::new(&input_file_path))?;

        for (idx, item) in json_instance.items.iter_mut().enumerate() {
            item.base.id = 100 + idx as u64 * 7;
        }
        let importer = Importer::new(config.cde_config, config.poly_simpl_tolerance, config.narrow_concavity_cutoff_ratio);
        let instance = jagua_rs::probs::spp::io::import_instance(&importer, &json_instance)?;

        println!("[TEST] loaded instance: {}", json_instance.name);

        let rng = match RNG_SEED {
            Some(seed) => {
                println!("[TEST] using provided seed: {}", seed);
                Xoshiro256PlusPlus::seed_from_u64(seed as u64)
            }
            None => {
                let seed = rand::random();
                println!("[TEST] no seed provided, using: {}", seed);
                Xoshiro256PlusPlus::seed_from_u64(seed)
            }
        };

        let mut terminator = BasicTerminator::new();
        let mut sol_listener = DummySolListener;
        terminator.new_timeout(EXPLORE_TIMEOUT);

        let builder = LBFBuilder::new(instance.clone(), rng, LBF_SAMPLE_CONFIG).construct()?;
        let exported = jagua_rs::probs::spp::io::export(&builder.prob.save(), *sparrow::EPOCH);
        let restored = jagua_rs::probs::spp::io::import_solution(&instance, &exported)?;
        assert!(jagua_rs::entities::Layout::from_snapshot(&restored.layout_snapshot).is_feasible());
        assert_eq!(restored.layout_snapshot.placed_items.len(), instance.total_item_qty());
        let mut separator = Separator::new(builder.prob, builder.rng, config.expl_cfg.separator_config);

        let sols = exploration_phase(&mut separator, &mut sol_listener, &terminator, &config.expl_cfg);
        let final_explore_sol = sols.last().expect("no solutions found during exploration");

        terminator.new_timeout(COMPRESS_TIMEOUT);
        compression_phase(&mut separator, final_explore_sol, &mut sol_listener, &terminator, &config.cmpr_cfg);
        Ok(())
    }
}

use jagua_rs::io::import::Importer;
use jagua_rs::probs::spp::io::{ext_repr::ExtSPInstance, import_instance};
use rand::{SeedableRng, rngs::Xoshiro256PlusPlus};
use sparrow::config::DEFAULT_SPARROW_CONFIG;
use sparrow::consts::LBF_SAMPLE_CONFIG;
use sparrow::optimizer::lbf::{ConstructionError, LBFBuilder};

fn rectangle_builder(demand: usize, height: f32) -> LBFBuilder {
    let input: ExtSPInstance = serde_json::from_value(serde_json::json!({
        "name": "rectangles", "strip_height": height,
        "items": [{"id": 0, "demand": demand, "allowed_orientations": [0],
            "shape": {"type": "rectangle", "data": {
                "x_min": 0, "y_min": 0, "width": 100, "height": 60
            }}}]
    }))
    .unwrap();
    let importer = Importer::new(DEFAULT_SPARROW_CONFIG.cde_config, None, None, None);
    LBFBuilder::new(
        import_instance(&importer, &input).unwrap(),
        Xoshiro256PlusPlus::seed_from_u64(42),
        LBF_SAMPLE_CONFIG,
    )
}

#[test_case::test_case(1)]
#[test_case::test_case(2)]
fn exact_height_finishes_without_panicking(demand: usize) {
    let error = rectangle_builder(demand, 60.0)
        .construct()
        .err()
        .expect("exact boundary contact must not be accepted");
    assert_eq!(error, ConstructionError::WidthLimitReached { item_id: 0 });
}

#[test]
fn construction_can_expand_and_succeed() {
    let mut builder = rectangle_builder(2, 61.0);
    builder.prob.change_strip_width(101.0);
    let builder = builder.construct().unwrap();
    assert_eq!(builder.prob.layout.placed_items.len(), 2);
    assert!(builder.prob.strip_width() > 200.0);
    assert!(builder.prob.layout.is_feasible());
}

#[test]
fn optimize_propagates_construction_failure_without_reporting_a_solution() {
    use jagua_rs::probs::spp::entities::{SPInstance, SPSolution};
    use sparrow::optimizer::optimize;
    use sparrow::util::listener::{ReportType, SolutionListener};
    use sparrow::util::terminator::BasicTerminator;
    struct RejectReports;
    impl SolutionListener for RejectReports {
        fn report(&mut self, _: ReportType, _: &SPSolution, _: &SPInstance) {
            panic!("failed construction must not report a solution");
        }
    }
    let builder = rectangle_builder(1, 60.0);
    let config = DEFAULT_SPARROW_CONFIG;
    let error = optimize(
        builder.instance,
        builder.rng,
        &mut RejectReports,
        &mut BasicTerminator::new(),
        &config.expl_cfg,
        &config.cmpr_cfg,
        None,
    )
    .err()
    .expect("construction should fail");
    assert_eq!(error, ConstructionError::WidthLimitReached { item_id: 0 });
}

#[test]
fn optimize_accepts_a_supplied_initial_solution() {
    use sparrow::optimizer::optimize;
    use sparrow::util::listener::DummySolListener;
    use sparrow::util::terminator::BasicTerminator;
    let builder = rectangle_builder(2, 61.0).construct().unwrap();
    let solution = builder.prob.save();
    let mut config = DEFAULT_SPARROW_CONFIG;
    config.expl_cfg.time_limit = std::time::Duration::ZERO;
    config.cmpr_cfg.time_limit = std::time::Duration::ZERO;
    let solution = optimize(
        builder.instance,
        builder.rng,
        &mut DummySolListener,
        &mut BasicTerminator::new(),
        &config.expl_cfg,
        &config.cmpr_cfg,
        Some(&solution),
    )
    .unwrap();
    assert_eq!(solution.layout_snapshot.placed_items.len(), 2);
}

#[test]
fn sampler_keeps_single_coordinate_axes() {
    use jagua_rs::geometry::primitives::Rect;
    use sparrow::sample::uniform_sampler::UniformBBoxSampler;
    let builder = rectangle_builder(1, 60.0);
    let item = &builder.instance.items[0].0;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
    for (width, height) in [(100.0, 80.0), (120.0, 60.0), (100.0, 60.0)] {
        let bbox = Rect::try_new(0.0, 0.0, width, height).unwrap();
        let sampler = UniformBBoxSampler::new(bbox, item, bbox).unwrap();
        for _ in 0..16 {
            let (x, y) = sampler.sample(&mut rng).translation();
            assert!((50.0..=width - 50.0).contains(&x));
            assert!((30.0..=height - 30.0).contains(&y));
        }
    }
}

#[test]
fn sampler_rejects_reversed_and_non_finite_ranges() {
    use jagua_rs::geometry::primitives::Rect;
    use sparrow::sample::uniform_sampler::UniformBBoxSampler;
    let builder = rectangle_builder(1, 60.0);
    let item = &builder.instance.items[0].0;
    let bbox = Rect::try_new(0.0, 0.0, 120.0, 80.0).unwrap();
    let too_small = Rect::try_new(0.0, 0.0, 99.0, 59.0).unwrap();
    assert!(UniformBBoxSampler::new(bbox, item, too_small).is_none());
    for x_min in [121.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let invalid = Rect { x_min, ..bbox };
        assert!(UniformBBoxSampler::new(invalid, item, bbox).is_none());
        assert!(UniformBBoxSampler::new(bbox, item, invalid).is_none());
    }
}

#[test]
fn ordinary_sampling_preserves_seeded_half_open_distribution() {
    use jagua_rs::geometry::primitives::Rect;
    use rand::{RngExt, prelude::IndexedRandom};
    use sparrow::sample::uniform_sampler::UniformBBoxSampler;
    let builder = rectangle_builder(1, 60.0);
    let item = &builder.instance.items[0].0;
    let bbox = Rect::try_new(0.0, 0.0, 120.0, 80.0).unwrap();
    let sampler = UniformBBoxSampler::new(bbox, item, bbox).unwrap();
    let mut actual_rng = Xoshiro256PlusPlus::seed_from_u64(42);
    let mut expected_rng = actual_rng.clone();
    for _ in 0..16 {
        [0.0].choose(&mut expected_rng).unwrap();
        let expected = (
            expected_rng.random_range(50.0..70.0),
            expected_rng.random_range(30.0..50.0),
        );
        assert_eq!(sampler.sample(&mut actual_rng).translation(), expected);
    }
}

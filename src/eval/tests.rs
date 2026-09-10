use crate::config::DEFAULT_SPARROW_CONFIG;
use crate::eval::sample_eval::{SampleEval, SampleEvaluator};
use crate::eval::sep_evaluator::SeparationEvaluator;
use crate::quantify::tracker::CollisionTracker;
use jagua_rs::collision_detection::hazards::HazardEntity;
use jagua_rs::geometry::DTransformation;
use jagua_rs::io::import::Importer;
use jagua_rs::probs::spp::entities::{SPPlacement, SPProblem};
use jagua_rs::probs::spp::io::{ext_repr::ExtSPInstance, import_instance};

#[test]
fn unbounded_overflow_is_still_a_candidate() {
    let ext: ExtSPInstance = serde_json::from_value(serde_json::json!({
        "name": "squares", "strip_height": 680,
        "items": [{
            "id": 0, "demand": 2, "allowed_orientations": [0, 90],
            "shape": {
                "type": "simple_polygon",
                "data": [[0, 0], [51, 0], [51, 51], [0, 51]]
            }
        }]
    }))
    .unwrap();
    let importer = Importer::new(DEFAULT_SPARROW_CONFIG.cde_config, None, None, None);
    let instance = import_instance(&importer, &ext).unwrap();
    let mut prob = SPProblem::new(instance.clone());
    prob.change_strip_width(50.9);
    let dt = DTransformation::new(0.0, (0.0, 100.0));
    let pk = prob.place_item(SPPlacement {
        item_id: 0,
        d_transf: dt,
    });
    let other_pk = prob.place_item(SPPlacement {
        item_id: 0,
        d_transf: dt,
    });
    let mut ct = CollisionTracker::new(&prob.layout);
    for weight in [f32::MAX, f32::INFINITY] {
        let idx = ct.pk_idx_map[pk];
        ct.container_collisions[idx].weight = weight;
        let mut loss_evaluator =
            super::collision_loss::CollisionLossEvaluator::new(&prob.layout, &ct, pk);
        let shape = &prob.layout.placed_items[pk].shape;
        loss_evaluator.reload(f32::INFINITY, shape);
        assert!(!loss_evaluator.add(HazardEntity::Exterior, shape));
        assert!(!loss_evaluator.add(
            HazardEntity::from((other_pk, &prob.layout.placed_items[other_pk])),
            shape
        ));
        assert_eq!(loss_evaluator.loss(), f32::INFINITY);
        let mut evaluator = SeparationEvaluator::new(&prob.layout, &instance.items[0].0, pk, &ct);
        for bound in [
            None,
            Some(SampleEval::Invalid),
            Some(SampleEval::Collision {
                loss: f32::INFINITY,
            }),
        ] {
            assert_eq!(
                evaluator.evaluate_sample(dt, bound),
                SampleEval::Collision {
                    loss: f32::INFINITY
                }
            );
        }
        assert_eq!(
            evaluator.evaluate_sample(dt, Some(SampleEval::Collision { loss: 1.0 })),
            SampleEval::Invalid
        );
    }
}

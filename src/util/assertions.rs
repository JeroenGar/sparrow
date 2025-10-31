use crate::eval::specialized_jaguars_pipeline::SpecializedHazardCollector;
use crate::quantify::tracker::CollisionTracker;
use crate::quantify::{quantify_collision_poly_container, quantify_collision_poly_poly};
use float_cmp::{approx_eq, assert_approx_eq};
use itertools::Itertools;
use jagua_rs::util::assertions;
use log::warn;
use std::collections::HashSet;
use jagua_rs::collision_detection::hazards::collector::{BasicHazardCollector, HazardCollector};
use jagua_rs::collision_detection::hazards::HazardEntity;
use jagua_rs::entities::Layout;
use jagua_rs::geometry::primitives::SPolygon;
use jagua_rs::io::svg::SvgDrawOptions;
use jagua_rs::probs::spp::entities::SPProblem;

pub fn tracker_matches_layout(ct: &CollisionTracker, l: &Layout) -> bool {
    assert!(l.placed_items.keys().all(|k| ct.pk_idx_map.contains_key(k)));
    assert!(assertions::layout_qt_matches_fresh_qt(l));

    for (pk1, pi1) in l.placed_items.iter() {
        let mut collector = BasicHazardCollector::new();
        l.cde().collect_poly_collisions(&pi1.shape, &mut collector);
        collector.remove_by_entity(&HazardEntity::from((pk1, pi1)));
        assert_eq!(ct.get_pair_loss(pk1, pk1), 0.0);
        for (pk2, pi2) in l.placed_items.iter().filter(|(k, _)| *k != pk1) {
            let stored_loss = ct.get_pair_loss(pk1, pk2);
            match collector.iter().any(|(_, he)| he == &HazardEntity::from((pk2, pi2))) {
                true => {
                    let calc_loss = quantify_collision_poly_poly(&pi1.shape, &pi2.shape);
                    let calc_loss_r = quantify_collision_poly_poly(&pi2.shape, &pi1.shape);
                    if !approx_eq!(f32,calc_loss,stored_loss,epsilon = 0.10 * stored_loss) && !approx_eq!(f32,calc_loss_r,stored_loss, epsilon = 0.10 * stored_loss) {
                        let mut opp_collector = BasicHazardCollector::new();
                        l.cde().collect_poly_collisions(&pi2.shape, &mut opp_collector);
                        opp_collector.remove_by_entity(&HazardEntity::from((pk2, pi2)));
                        if opp_collector.contains_entity(&((pk1, pi1).into())) {
                            dbg!(&pi1.shape.vertices, &pi2.shape.vertices);
                            dbg!(
                                stored_loss,
                                calc_loss,
                                calc_loss_r,
                                opp_collector.iter().collect_vec(),
                                HazardEntity::from((pk1, pi1)),
                                HazardEntity::from((pk2, pi2))
                            );
                            panic!("tracker error");
                        } else {
                            //detecting collisions is not symmetrical (in edge cases)
                            warn!("inconsistent loss");
                            warn!(
                                "collisions: pi_1 {:?} -> {:?}",
                                HazardEntity::from((pk1, pi1)),
                                collector.iter().collect_vec()
                            );
                            warn!(
                                "opposite collisions: pi_2 {:?} -> {:?}",
                                HazardEntity::from((pk2, pi2)),
                                opp_collector.iter().collect_vec()
                            );

                            warn!(
                                "pi_1: {:?}",
                                pi1.shape
                                    .vertices
                                    .iter()
                                    .map(|p| format!("({},{})", p.0, p.1))
                                    .collect_vec()
                            );
                            warn!(
                                "pi_2: {:?}",
                                pi2.shape
                                    .vertices
                                    .iter()
                                    .map(|p| format!("({},{})", p.0, p.1))
                                    .collect_vec()
                            );
                            panic!("tracker error");
                        }
                    }
                }
                false => {
                    if stored_loss != 0.0 {
                        let calc_loss = quantify_collision_poly_poly(&pi1.shape, &pi2.shape);
                        let mut opp_collector = BasicHazardCollector::new();
                        l.cde().collect_poly_collisions(&pi2.shape, &mut opp_collector);
                        opp_collector.remove_by_entity(&HazardEntity::from((pk2, pi2)));
                        if !opp_collector.contains_entity(&HazardEntity::from((pk1, pi1))) {
                            dbg!(&pi1.shape.vertices, &pi2.shape.vertices);
                            dbg!(
                                stored_loss,
                                calc_loss,
                                opp_collector.iter().collect_vec(),
                                HazardEntity::from((pk1, pi1)),
                                HazardEntity::from((pk2, pi2))
                            );
                            panic!("tracker error");
                        } else {
                            //detecting collisions is not symmetrical (in edge cases)
                            warn!("inconsistent loss");
                            warn!(
                                "collisions: {:?} -> {:?}",
                                HazardEntity::from((pk1, pi1)),
                                collector.iter().collect_vec()
                            );
                            warn!(
                                "opposite collisions: {:?} -> {:?}",
                                HazardEntity::from((pk2, pi2)),
                                opp_collector.iter().collect_vec()
                            );
                        }
                    }
                }
            }
        }
        if collector.contains_entity(&HazardEntity::Exterior) {
            let stored_loss = ct.get_container_loss(pk1);
            let calc_loss = quantify_collision_poly_container(&pi1.shape, l.container.outer_cd.bbox);
            assert_approx_eq!(f32, stored_loss, calc_loss, ulps = 5);
        } else {
            assert_eq!(ct.get_container_loss(pk1), 0.0);
        }
    }

    true
}

pub fn custom_pipeline_matches_jaguars(shape: &SPolygon, det: &SpecializedHazardCollector) -> bool {
    //Standard colllision collection, provided by jagua-rs, for comparison
    let cde = det.layout.cde();
    let base_detector = {
        let pi = &det.layout.placed_items[det.current_pk];
        let pk = det.current_pk;
        let mut coll = BasicHazardCollector::new();
        cde.collect_poly_collisions(shape, &mut coll);
        if coll.contains_entity(&HazardEntity::from((pk, pi))){
            coll.remove_by_entity(&HazardEntity::from((pk, pi)));
        }
        coll
    };

    //make sure these detection maps are equivalent
    let default_set: HashSet<HazardEntity> = base_detector.entities().cloned().collect();
    let custom_set: HashSet<HazardEntity> = det.entities().cloned().collect();

    assert_eq!(default_set, custom_set, "custom cde pipeline does not match jagua-rs! for pk: {:?}", det.current_pk);
    true
}

pub fn strip_width_is_in_check(prob: &SPProblem) -> bool {
    let diameters_of_all_items = prob.instance.items.iter().map(
        |(i,q)| i.shape_cd.diameter * *q as f32
    ).sum::<f32>();
    
    prob.strip_width() < 2.0 * (diameters_of_all_items)
}
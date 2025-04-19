use crate::eval::specialized_jaguars_pipeline::SpecializedHazardDetector;
use crate::quantify::tracker::CollisionTracker;
use crate::quantify::{quantify_collision_poly_bin, quantify_collision_poly_poly};
use crate::util::io::svg_util::SvgDrawOptions;
use float_cmp::{approx_eq, assert_approx_eq};
use itertools::Itertools;
use jagua_rs::util::assertions;
use log::warn;
use std::collections::HashSet;
use jagua_rs::collision_detection::hazards::detector::{BasicHazardDetector, HazardDetector};
use jagua_rs::collision_detection::hazards::HazardEntity;
use jagua_rs::entities::general::Layout;
use jagua_rs::geometry::geo_traits::Shape;
use jagua_rs::geometry::primitives::SPolygon;

pub fn tracker_matches_layout(ct: &CollisionTracker, l: &Layout) -> bool {
    assert!(l.placed_items.keys().all(|k| ct.pk_idx_map.contains_key(k)));
    assert!(assertions::layout_qt_matches_fresh_qt(l));

    for (pk1, pi1) in l.placed_items.iter() {
        let mut detector = BasicHazardDetector::new();
        l.cde().collect_poly_collisions(&pi1.shape, &mut detector);
        detector.remove(&HazardEntity::from((pk1, pi1)));
        assert_eq!(ct.get_pair_loss(pk1, pk1), 0.0);
        for (pk2, pi2) in l.placed_items.iter().filter(|(k, _)| *k != pk1) {
            let stored_loss = ct.get_pair_loss(pk1, pk2);
            match detector.iter().contains(&HazardEntity::from((pk2, pi2))) {
                true => {
                    let calc_loss = quantify_collision_poly_poly(&pi1.shape, &pi2.shape);
                    let calc_loss_r = quantify_collision_poly_poly(&pi2.shape, &pi1.shape);
                    if !approx_eq!(f32,calc_loss,stored_loss,epsilon = 0.10 * stored_loss) && !approx_eq!(f32,calc_loss_r,stored_loss, epsilon = 0.10 * stored_loss) {
                        let mut opp_detector = BasicHazardDetector::new();
                        l.cde().collect_poly_collisions(&pi2.shape, &mut opp_detector);
                        opp_detector.remove(&HazardEntity::from((pk2, pi2)));
                        if opp_detector.contains(&((pk1, pi1).into())) {
                            dbg!(&pi1.shape.vertices, &pi2.shape.vertices);
                            dbg!(
                                stored_loss,
                                calc_loss,
                                calc_loss_r,
                                opp_detector.iter().collect_vec(),
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
                                detector.iter().collect_vec()
                            );
                            warn!(
                                "opposite collisions: pi_2 {:?} -> {:?}",
                                HazardEntity::from((pk2, pi2)),
                                opp_detector.iter().collect_vec()
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

                            {
                                let mut svg_draw_options = SvgDrawOptions::default();
                                svg_draw_options.quadtree = true;
                                panic!("tracker error");
                            }
                        }
                    }
                }
                false => {
                    if stored_loss != 0.0 {
                        let calc_loss = quantify_collision_poly_poly(&pi1.shape, &pi2.shape);
                        let mut opp_detector = BasicHazardDetector::new();
                        l.cde().collect_poly_collisions(&pi2.shape, &mut opp_detector);
                        opp_detector.remove(&HazardEntity::from((pk2, pi2)));
                        if !opp_detector.contains(&HazardEntity::from((pk1, pi1))) {
                            dbg!(&pi1.shape.vertices, &pi2.shape.vertices);
                            dbg!(
                                stored_loss,
                                calc_loss,
                                opp_detector.iter().collect_vec(),
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
                                detector.iter().collect_vec()
                            );
                            warn!(
                                "opposite collisions: {:?} -> {:?}",
                                HazardEntity::from((pk2, pi2)),
                                opp_detector.iter().collect_vec()
                            );
                        }
                    }
                }
            }
        }
        if detector.contains(&HazardEntity::BinExterior) {
            let stored_loss = ct.get_bin_loss(pk1);
            let calc_loss = quantify_collision_poly_bin(&pi1.shape, l.bin.outer_cd.bbox());
            assert_approx_eq!(f32, stored_loss, calc_loss, ulps = 5);
        } else {
            assert_eq!(ct.get_bin_loss(pk1), 0.0);
        }
    }

    true
}

pub fn custom_pipeline_matches_jaguars(shape: &SPolygon, det: &SpecializedHazardDetector) -> bool {
    //Standard colllision collection, provided by jagua-rs, for comparison
    let cde = det.layout.cde();
    let base_detector = {
        let pi = &det.layout.placed_items[det.current_pk];
        let pk = det.current_pk;
        let mut detector = BasicHazardDetector::new();
        cde.collect_poly_collisions(shape, &mut detector);
        detector.remove(&HazardEntity::from((pk, pi)));
        detector
    };

    //make sure these detection maps are equivalent
    let default_set: HashSet<HazardEntity> = base_detector.iter().cloned().collect();
    let custom_set: HashSet<HazardEntity> = det.iter().cloned().collect();

    assert_eq!(default_set, custom_set, "custom cde pipeline does not match jagua-rs!");
    true
}
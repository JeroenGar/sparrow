use crate::eval::specialized_jaguars_pipeline::SpecializedDetectionMap;
use crate::quantify::tracker::CollisionTracker;
use crate::quantify::{quantify_collision_poly_bin, quantify_collision_poly_poly};
use crate::util::io::svg_util::SvgDrawOptions;
use float_cmp::{approx_eq, assert_approx_eq};
use itertools::Itertools;
use jagua_rs::collision_detection::hazard::HazardEntity;
use jagua_rs::collision_detection::hazard_helpers::HazardDetector;
use jagua_rs::entities::layout::Layout;
use jagua_rs::geometry::primitives::simple_polygon::SimplePolygon;
use jagua_rs::util::assertions;
use log::warn;
use std::collections::HashSet;

pub fn tracker_matches_layout(ct: &CollisionTracker, l: &Layout) -> bool {
    assert!(l.placed_items.keys().all(|k| ct.pk_idx_map.contains_key(k)));
    assert!(assertions::layout_qt_matches_fresh_qt(l));

    for (pk1, pi1) in l.placed_items.iter() {
        let collisions = l.cde().collect_poly_collisions(&pi1.shape, &[(pk1, pi1).into()]);
        assert_eq!(ct.get_pair_loss(pk1, pk1), 0.0);
        for (pk2, pi2) in l.placed_items.iter().filter(|(k, _)| *k != pk1) {
            let stored_loss = ct.get_pair_loss(pk1, pk2);
            match collisions.iter().contains(&HazardEntity::from((pk2, pi2))) {
                true => {
                    let calc_loss = quantify_collision_poly_poly(&pi1.shape, &pi2.shape);
                    let calc_loss_r = quantify_collision_poly_poly(&pi2.shape, &pi1.shape);
                    if !approx_eq!(f32,calc_loss,stored_loss,epsilon = 0.10 * stored_loss) && !approx_eq!(f32,calc_loss_r,stored_loss, epsilon = 0.10 * stored_loss) {
                        let opposite_collisions =
                            l.cde().collect_poly_collisions(&pi2.shape, &[(pk2, pi2).into()]);
                        if opposite_collisions.contains(&((pk1, pi1).into())) {
                            dbg!(&pi1.shape.points, &pi2.shape.points);
                            dbg!(
                                stored_loss,
                                calc_loss,
                                calc_loss_r,
                                opposite_collisions,
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
                                collisions
                            );
                            warn!(
                                "opposite collisions: pi_2 {:?} -> {:?}",
                                HazardEntity::from((pk2, pi2)),
                                opposite_collisions
                            );

                            warn!(
                                "pi_1: {:?}",
                                pi1.shape
                                    .points
                                    .iter()
                                    .map(|p| format!("({},{})", p.0, p.1))
                                    .collect_vec()
                            );
                            warn!(
                                "pi_2: {:?}",
                                pi2.shape
                                    .points
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
                        let opposite_collisions = l.cde().collect_poly_collisions(&pi2.shape, &[(pk2, pi2).into()]);
                        if !opposite_collisions.contains(&((pk1, pi1).into())) {
                            dbg!(&pi1.shape.points, &pi2.shape.points);
                            dbg!(
                                stored_loss,
                                calc_loss,
                                opposite_collisions,
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
                                collisions
                            );
                            warn!(
                                "opposite collisions: {:?} -> {:?}",
                                HazardEntity::from((pk2, pi2)),
                                opposite_collisions
                            );
                        }
                    }
                }
            }
        }
        if collisions.contains(&HazardEntity::BinExterior) {
            let stored_loss = ct.get_bin_loss(pk1);
            let calc_loss = quantify_collision_poly_bin(&pi1.shape, l.bin.bbox());
            assert_approx_eq!(f32, stored_loss, calc_loss, ulps = 5);
        } else {
            assert_eq!(ct.get_bin_loss(pk1), 0.0);
        }
    }

    true
}

pub fn custom_pipeline_matches_jaguars(shape: &SimplePolygon, det: &SpecializedDetectionMap) -> bool {
    //Standard colllision collection, provided by jagua-rs, for comparison
    let default_dm = {
        let pi = &det.layout.placed_items[det.current_pk];
        let pk = det.current_pk;
        det.layout.cde().collect_poly_collisions(shape, &[HazardEntity::from((pk, pi))])
    };

    //make sure these detection maps are equivalent
    let default_set: HashSet<HazardEntity> = default_dm.iter().cloned().collect();
    let custom_set: HashSet<HazardEntity> = det.iter().cloned().collect();

    assert_eq!(default_set, custom_set, "custom cde pipeline does not match jagua-rs!");
    true
}
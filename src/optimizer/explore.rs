use crate::config::ExplorationConfig;
use crate::optimizer::separator::{Separator, SeparatorConfig};
use crate::sample::uniform_sampler::convert_sample_to_closest_feasible;
use crate::util::listener::{ReportType, SolutionListener};
use crate::util::terminator::Terminator;
use crate::FMT;
use float_cmp::approx_eq;
use itertools::Itertools;
use jagua_rs::collision_detection::hazards::collector::BasicHazardCollector;
use jagua_rs::collision_detection::hazards::BasicHazardEntity;
use jagua_rs::collision_detection::hazards::filter::NoFilter;
use jagua_rs::entities::{Layout, PItemKey};
use jagua_rs::geometry::geo_traits::CollidesWith;
use jagua_rs::probs::spp::entities::SPSolution;
use jagua_rs::Instant;
use log::{debug, info, warn};
use ordered_float::OrderedFloat;
use rand::prelude::{Distribution, IteratorRandom};
use rand_distr::Normal;
use std::cmp::Reverse;

/// Algorithm 12 from https://doi.org/10.48550/arXiv.2509.13329
pub fn exploration_phase(sep: &mut Separator, sol_listener: &mut impl SolutionListener, term: &impl Terminator, config: &ExplorationConfig) -> Vec<SPSolution> {
    let start = Instant::now();
    let initial_eval_count = sep.total_evals;
    let width_lower_bound = super::strip_width_lower_bound(&sep.prob);
    let mut current_width = sep.prob.strip_width();
    let mut best_width = current_width;

    let mut feasible_sols = vec![sep.prob.save()];

    sol_listener.report(ReportType::ExplFeas, &feasible_sols[0]);
    info!("[EXPL] starting optimization with initial width: {:.3} ({:.3}%)",current_width,sep.prob.density() * 100.0);

    let mut infeas_sol_pool: Vec<(SPSolution, f32)> = vec![];

    while !term.kill() {
        // Attempt to separate the current layout
        let local_best = sep.separate(term, sol_listener);
        let total_loss = local_best.1.get_total_loss();

        if total_loss == 0.0 {
            // If successfully separated
            if current_width < best_width {
                info!("[EXPL] feasible solution found! (width: {:.3}, dens: {:.3}%)",current_width,sep.prob.density() * 100.0);
                best_width = current_width;
                feasible_sols.push(local_best.0.clone());
                sol_listener.report(ReportType::ExplFeas, &local_best.0);
            }
            // Shrink the strip width and clear the infeasible solution pool
            let next_width = current_width * (1.0 - config.shrink_step);
            if next_width < width_lower_bound || next_width >= current_width {
                info!("[EXPL] stopping at strip width lower bound: {:.3}", width_lower_bound);
                break;
            }
            info!("[EXPL] shrinking strip by {}%: {:.3} -> {:.3}", config.shrink_step * 100.0, current_width, next_width);
            if sep.change_strip_width(next_width, None).is_err() {
                break;
            }
            current_width = next_width;
            infeas_sol_pool.clear();
        } else {
            info!("[EXPL] unable to reach feasibility (width: {:.3}, dens: {:.3}%, min loss: {})", current_width, sep.prob.density() * 100.0, FMT().fmt2(total_loss));
            sol_listener.report(ReportType::ExplInfeas, &local_best.0);

            // Separation was not successful add it to the pool of infeasible solutions
            match infeas_sol_pool.binary_search_by(|(_, o)| o.partial_cmp(&total_loss).unwrap()) {
                Ok(idx) | Err(idx) => infeas_sol_pool.insert(idx, (local_best.0.clone(), total_loss)),
            }

            if infeas_sol_pool.len() >= config.max_conseq_failed_attempts.unwrap_or(usize::MAX) {
                info!("[EXPL] max consecutive failed attempts ({}), terminating", infeas_sol_pool.len());
                break;
            }

            // Restore to a random solution from the pool, with better solutions having more chance to be selected
            let selected_sol = {
                // Sample a value in range [0.0, 1.0[ from a normal distribution
                let distribution = Normal::new(0.0, config.solution_pool_distribution_stddev).unwrap();
                let sample = distribution.sample(&mut sep.rng).abs().min(0.999);
                // Map it to an index in the infeasible solution pool (better solutions are at the start of the pool)
                let selected_idx = (sample * infeas_sol_pool.len() as f32) as usize;

                let (selected_sol, loss) = &infeas_sol_pool[selected_idx];
                info!("[EXPL] starting solution {}/{} selected from solution pool (l: {}) to disrupt", selected_idx, infeas_sol_pool.len(), FMT().fmt2(*loss));
                selected_sol
            };

            // Rollback to this solution and disrupt it.
            sep.rollback(selected_sol, None);
            disrupt_solution(sep, config);
        }
    }

    let elapsed_seconds = start.elapsed().as_secs_f64();
    let total_evals = sep.total_evals - initial_eval_count;
    info!("[EXPL] finished, best feasible solution: width: {:.3} ({:.3}%), evals/s: {}, elapsed: {:.3}s",best_width,feasible_sols.last().unwrap().density() * 100.0, FMT().fmt2(total_evals as f64 / elapsed_seconds), elapsed_seconds);

    feasible_sols
}

fn disrupt_solution(sep: &mut Separator, config: &ExplorationConfig) {
    if sep.prob.layout().placed_items().len() < 2 {
        warn!("[DSRP] cannot disrupt solution with less than 2 items");
        return;
    }

    // The general idea is to disrupt a solution by swapping two 'large' items in the layout.
    // 'Large' items are those whose convex hull area falls within a certain top percentile
    // of the total convex hull area of all items in the layout.

    // Step 1: Define what constitutes a 'large' item.

    // Calculate the total convex hull area of all items, considering quantities.
    let total_convex_hull_area: f32 = sep
        .prob
        .instance()
        .items
        .iter()
        .map(|(item, quantity)| item.shape_cd().surrogate().convex_hull_area * (*quantity as f32))
        .sum();

    let cutoff_threshold_area = total_convex_hull_area * config.large_item_ch_area_cutoff_percentile;

    // Sort items by convex hull area in descending order.
    let sorted_items_by_ch_area = sep
        .prob
        .instance()
        .items
        .iter()
        .sorted_by_key(|(item, _)| Reverse(OrderedFloat(item.shape_cd().surrogate().convex_hull_area)))
        .peekable();

    let mut cumulative_ch_area = 0.0;
    let mut ch_area_cutoff = 0.0;

    // Iterate through items, accumulating their convex hull areas until the cumulative sum
    // exceeds the cutoff_threshold_area. The convex hull area of the item that causes
    // this excess becomes the ch_area_cutoff.
    for (item, quantity) in sorted_items_by_ch_area {
        let item_ch_area = item.shape_cd().surrogate().convex_hull_area;
        cumulative_ch_area += item_ch_area * (*quantity as f32);
        if cumulative_ch_area > cutoff_threshold_area {
            ch_area_cutoff = item_ch_area;
            debug!("[DSRP] cutoff ch area: {}, for item idx: {}, bbox: {:?}",ch_area_cutoff, item.idx(), item.shape_cd().bbox());
            break;
        }
    }

    // Step 2: Select two 'large' items and 'swap' them.

    let large_items = sep.prob.layout().placed_items().iter()
        .filter(|(_, pi)| pi.shape().surrogate().convex_hull_area >= ch_area_cutoff);

    //Choose a first item with a large enough convex hull
    let (pk1, pi1) = large_items.clone().choose(&mut sep.rng).expect("[DSRP] failed to choose first item");

    //Choose a second item with a large enough convex hull and different enough from the first.
    //If no such item is found, choose a random one.
    let (pk2, pi2) = large_items.clone()
        .filter(|(_, pi)|
            // Ensure the second item is different from the first
            !approx_eq!(f32, pi.shape().area(),pi1.shape().area(), epsilon = pi1.shape().area() * 0.01) &&
                !approx_eq!(f32, pi.shape().diameter(), pi1.shape().diameter(), epsilon = pi1.shape().diameter() * 0.01)
        )
        .choose(&mut sep.rng)
        .or_else(|| {
            sep.prob.layout().placed_items().iter()
                .filter(|(pk, _)| *pk != pk1) // Ensure the second item is not the same as the first
                .choose(&mut sep.rng)
        }) // As a fallback, choose any item
        .expect("[EXPL] failed to choose second item for disruption");

    // Step 3: Swap the two items' positions in the layout.

    let dt1_old = pi1.d_transf();
    let dt2_old = pi2.d_transf();

    // Make sure the swaps do not violate feasibility (rotation).
    let dt1_new = convert_sample_to_closest_feasible(dt2_old, pi1.item());
    let dt2_new = convert_sample_to_closest_feasible(dt1_old, pi2.item());

    info!("[EXPL] disrupting by swapping two large items (id: {} <-> {})", pi1.item().idx(), pi2.item().idx());

    let pk1 = sep.move_item(pk1, dt1_new);
    let pk2 = sep.move_item(pk2, dt2_new);


    // Step 4: Move all items that are practically contained by one of the swapped items to the "empty space" created by the moved item.
    //         This is particularly important when huge items are swapped with smaller items. 
    //         The huge item will create a large empty space and many of the items which previously 
    //         surrounded the smaller one will be contained by the huge one.
    {
        // transformation to convert the contained items' position (relative to the old and new positions of the swapped items)
        let converting_transformation = dt1_new.compose().inverse()
            .transform(&dt1_old.compose());

        for c1_pk in practically_contained_items(sep.prob.layout(), pk1).into_iter().filter(|c1_pk| *c1_pk != pk2) {
            let c1_pi = &sep.prob.layout().placed_items()[c1_pk];

            let new_dt = c1_pi.d_transf()
                .compose()
                .transform(&converting_transformation)
                .decompose();

            //Ensure the sure the new position is feasible
            let new_feasible_dt = convert_sample_to_closest_feasible(new_dt, c1_pi.item());
            sep.move_item(c1_pk, new_feasible_dt);
        }
    }

    // Do the same for the second item, but using the second transformation
    {
        let converting_transformation = dt2_new.compose().inverse()
            .transform(&dt2_old.compose());

        for c2_pk in practically_contained_items(sep.prob.layout(), pk2).into_iter().filter(|c2_pk| *c2_pk != pk1) {
            let c2_pi = &sep.prob.layout().placed_items()[c2_pk];
            let new_dt = c2_pi.d_transf()
                .compose()
                .transform(&converting_transformation)
                .decompose();

            //make sure the new position is feasible
            let new_feasible_dt = convert_sample_to_closest_feasible(new_dt, c2_pi.item());
            sep.move_item(c2_pk, new_feasible_dt);
        }
    }
}

/// Collects all items which point of inaccessibility (POI) is contained by pk_c's shape.
fn practically_contained_items(layout: &Layout, pk_c: PItemKey) -> Vec<PItemKey> {
    let pi_c = &layout.placed_items()[pk_c];
    // Detect all collisions with the item pk_c's shape.
    let mut collector = BasicHazardCollector::new();
    layout.cde().collect_poly_collisions(pi_c.shape(), &NoFilter, &mut collector);

    // Filter out the items that have their POI contained by pk_c's shape.
    collector.iter()
        .filter_map(|(_,he)| {
            match he {
                BasicHazardEntity::PlacedItem { pk, .. } => Some(*pk),
                _ => None
            }
        })
        .filter(|pk| *pk != pk_c) // Ensure we don't include the item itself
        .filter(|pk| {
            // Check if the POI of the item is contained by pk_c's shape
            let poi = layout.placed_items()[*pk].shape().poi();
            pi_c.shape().collides_with(&poi.center)
        })
        .collect_vec()
}

#[cfg(test)]
mod tests_reflection {
    //! Reflection in the exploration phase.
    //!
    //! A reflected copy is the same `shape_cd` under an improper isometry, so its surrogate keeps
    //! the item's convex hull area. `disrupt_solution` ranks items by that area to pick large
    //! items; if a reflected copy reported a different area, a single-item instance could end with
    //! an empty `large_items` filter and `choose().expect()` would panic.
    use super::*;
    use crate::config::DEFAULT_SPARROW_CONFIG;
    use crate::consts::LBF_SAMPLE_CONFIG;
    use crate::optimizer::lbf::LBFBuilder;
    use crate::optimizer::separator::Separator;
    use jagua_rs::io::ext_repr::{ExtItem, ExtOrientation, ExtRotation, ExtSPolygon, ExtShape};
    use jagua_rs::io::import::Importer;
    use jagua_rs::probs::spp::io::ext_repr::{ExtItem as ExtSPItem, ExtSPInstance};
    use rand::SeedableRng;
    use rand::rngs::Xoshiro256PlusPlus;

    /// A chiral polygon: it is not congruent to its mirror image under any rotation.
    fn chiral_shape() -> Vec<(f32, f32)> {
        vec![
            (0.0, 0.0), (37.3, 0.9), (74.1, 1.7), (110.0, 2.6), (112.4, 19.3), (95.7, 22.1),
            (96.0, 41.9), (61.2, 43.0), (60.9, 28.7), (31.4, 29.9), (30.1, 51.3), (11.7, 52.0),
            (3.2, 26.4),
        ]
    }

    /// A separator over a single chiral item with rotations {0, 180} and the given reflection axes.
    fn separator(reflection_axes: Vec<f32>, demand: u64) -> Separator {
        let ext = ExtSPInstance {
            name: "reflection".into(),
            min_item_separation: 0.0,
            strip_height: 200.0,
            items: vec![ExtSPItem {
                base: ExtItem {
                    id: 0,
                    orientation: ExtOrientation {
                        rotation: ExtRotation::Discrete { angles: vec![0.0, 180.0] },
                        reflection_axes,
                    },
                    shape: ExtShape::SimplePolygon(ExtSPolygon(chiral_shape())),
                    min_quality: None,
                },
                demand,
            }],
        };
        let cfg = DEFAULT_SPARROW_CONFIG;
        let importer = Importer::new(cfg.cde_config, cfg.poly_simpl_tolerance, cfg.narrow_concavity_cutoff_ratio);
        let instance = jagua_rs::probs::spp::io::import_instance(&importer, &ext).expect("import");
        let rng = Xoshiro256PlusPlus::seed_from_u64(1);
        let builder = LBFBuilder::new(instance, rng, LBF_SAMPLE_CONFIG).unwrap().construct().unwrap();
        Separator::new(builder.prob, builder.rng, cfg.expl_cfg.separator_config)
    }

    #[test]
    fn reflected_copies_keep_the_convex_hull_area_and_survive_disruption() {
        let mut sep = separator(vec![0.0], 6);
        let pks: Vec<_> = sep.prob.layout().placed_items().keys().collect();
        for pk in pks {
            let dt = sep.prob.layout().placed_items()[pk].d_transf();
            sep.move_item(pk, dt.with_reflection(true));
        }
        let item_ch = sep.prob.instance().item(0).shape_cd().surrogate().convex_hull_area;
        let mut n_reflected = 0;
        for (_, pi) in sep.prob.layout().placed_items().iter() {
            assert!(pi.d_transf().reflected, "copy should be reflected");
            n_reflected += 1;
            assert_eq!(pi.shape().surrogate().convex_hull_area.to_bits(), item_ch.to_bits(),
                       "a reflected copy reports a different convex hull area than its item");
        }
        assert_eq!(n_reflected, 6);
        for _ in 0..200 {
            disrupt_solution(&mut sep, &DEFAULT_SPARROW_CONFIG.expl_cfg);
        }
        assert_eq!(sep.prob.layout().placed_items().len(), 6, "disruption must not lose items");
    }

    #[test]
    fn disruption_keeps_the_reflection_bit_of_reflectable_items() {
        // Swapping an upright copy with a reflected one swaps their reflections: both stay allowed,
        // so `convert_sample_to_closest_feasible` must keep the bit.
        let mut sep = separator(vec![0.0], 6);
        let pks: Vec<_> = sep.prob.layout().placed_items().keys().collect();
        for (i, pk) in pks.into_iter().enumerate() {
            let dt = sep.prob.layout().placed_items()[pk].d_transf();
            sep.move_item(pk, dt.with_reflection(i % 2 == 0));
        }
        for _ in 0..50 {
            disrupt_solution(&mut sep, &DEFAULT_SPARROW_CONFIG.expl_cfg);
        }
        let n_reflected = sep.prob.layout().placed_items().iter().filter(|(_, pi)| pi.d_transf().reflected).count();
        assert!(n_reflected > 0 && n_reflected < 6, "reflections lost by disruption: {n_reflected}/6 reflected");
    }

    #[test]
    fn reflected_poses_are_sampled_only_when_a_reflection_axis_is_allowed() {
        use crate::sample::uniform_sampler::UniformBBoxSampler;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        for (axes, expected) in [(vec![], false), (vec![0.0], true)] {
            let sep = separator(axes, 2);
            let item = sep.prob.instance().item(0);
            let bbox = sep.prob.layout().container().outer_cd().bbox();
            let sampler = UniformBBoxSampler::new(bbox, item, bbox).expect("sampler");
            let n = (0..2000).filter(|_| sampler.sample(&mut rng).reflected).count();
            assert_eq!(n > 0, expected, "reflection allowed: {expected}, reflected samples: {n}/2000");
            for _ in 0..200 {
                let dt = sampler.sample(&mut rng);
                assert!(item.allowed_orientations().allows(&dt), "sampled pose is not allowed: {dt}");
            }
        }
    }
}

use crate::config::*;
use crate::optimizer::lbf::LBFBuilder;
use crate::optimizer::separator::Separator;
pub use crate::optimizer::terminator::Terminator;
use crate::sample::uniform_sampler::{convert_sample_to_closest_feasible};
use crate::FMT;
use float_cmp::approx_eq;
use itertools::Itertools;
use jagua_rs::collision_detection::hazards::HazardEntity;
use jagua_rs::entities::{Instance, Layout, PItemKey};
use jagua_rs::geometry::geo_traits::CollidesWith;
use jagua_rs::probs::spp::entities::{SPInstance, SPSolution};
use log::{debug, info};
use ordered_float::OrderedFloat;
use rand::prelude::{IteratorRandom, SmallRng};
use rand::{Rng, RngCore, SeedableRng};
use rand_distr::Distribution;
use rand_distr::Normal;
use std::cmp::Reverse;
use std::time::{Duration, Instant};
use slotmap::SecondaryMap;

pub mod lbf;
pub mod separator;
mod worker;
pub mod terminator;

// All high-level heuristic logic
pub fn optimize(instance: SPInstance, mut rng: SmallRng, output_folder_path: String, mut terminator: Terminator, explore_dur: Duration, compress_dur: Duration) -> SPSolution {
    let mut next_rng = || SmallRng::seed_from_u64(rng.next_u64());
    let builder = LBFBuilder::new(instance.clone(), next_rng(), LBF_SAMPLE_CONFIG).construct();

    terminator.set_timeout_from_now(explore_dur);
    let mut expl_separator = Separator::new(builder.instance, builder.prob, next_rng(), output_folder_path.clone(), 0, SEP_CFG_EXPLORE);
    let solutions = exploration_phase(&instance, &mut expl_separator, &terminator);
    let final_explore_sol = solutions.last().unwrap().clone();

    terminator.set_timeout_from_now(compress_dur).reset_ctrlc();
    let mut cmpr_separator = Separator::new(expl_separator.instance, expl_separator.prob, next_rng(), expl_separator.output_svg_folder, expl_separator.svg_counter, SEP_CFG_COMPRESS);
    let cmpr_sol = compression_phase(&instance, &mut cmpr_separator, &final_explore_sol, &terminator);

    cmpr_sol
}

pub fn exploration_phase(instance: &SPInstance, sep: &mut Separator, term: &Terminator) -> Vec<SPSolution> {
    let mut current_width = sep.prob.strip_width();
    let mut best_width = current_width;

    let mut feasible_solutions = vec![sep.prob.save()];

    sep.export_svg(None, "init", false);
    info!("[EXPL] starting optimization with initial width: {:.3} ({:.3}%)",current_width,sep.prob.density() * 100.0);

    let mut solution_pool: Vec<(SPSolution, f32)> = vec![];

    while !term.is_kill() {
        let local_best = sep.separate(&term);
        let total_loss = local_best.1.get_total_loss();

        if total_loss == 0.0 {
            //layout is successfully separated
            if current_width < best_width {
                info!("[EXPL] new best at width: {:.3} ({:.3}%)",current_width,sep.prob.density() * 100.0);
                best_width = current_width;
                feasible_solutions.push(local_best.0.clone());
                sep.export_svg(Some(local_best.0.clone()), "expl_f", false);
            }
            let next_width = current_width * (1.0 - EXPLORE_SHRINK_STEP);
            info!("[EXPL] shrinking width by {}%: {:.3} -> {:.3}", EXPLORE_SHRINK_STEP * 100.0, current_width, next_width);
            sep.change_strip_width(next_width, None);
            current_width = next_width;
            solution_pool.clear();
        } else {
            info!("[EXPL] layout separation unsuccessful, exporting min loss solution");
            sep.export_svg(Some(local_best.0.clone()), "expl_nf", false);

            //layout was not successfully separated, add to local bests
            match solution_pool.binary_search_by(|(_, o)| o.partial_cmp(&total_loss).unwrap()) {
                Ok(idx) | Err(idx) => solution_pool.insert(idx, (local_best.0.clone(), total_loss)),
            }

            //restore to a random solution from the tabu list, better solutions have more chance to be selected
            let selected_sol = {
                //sample a value in range [0.0, 1.0[ from a normal distribution
                let distr = Normal::new(0.0, EXPLORE_SOL_DISTR_STDDEV).unwrap();
                let sample = distr.sample(&mut sep.rng).abs().min(0.999);
                //map it to the range of the solution pool
                let selected_idx = (sample * solution_pool.len() as f32) as usize;

                let (selected_sol, loss) = &solution_pool[selected_idx];
                info!("[EXPL] selected starting solution {}/{} from solution pool (l: {})", selected_idx, solution_pool.len(), FMT().fmt2(*loss));
                selected_sol
            };

            sep.rollback(selected_sol, None);
            disrupt_solution(sep);

            if solution_pool.len() >= 20 {
                solution_pool.drain(15..20);
                solution_pool.drain(0..5);
                info!("[EXPL] solution pool is too large, draining it, keeping the middle solutions (n: {})", solution_pool.len());
            }
        }
    }

    info!("[EXPL] time limit reached, best solution found: {:.3} ({:.3}%)",best_width,feasible_solutions.last().unwrap().density(instance) * 100.0);

    feasible_solutions
}

pub fn compression_phase(instance: &SPInstance, sep: &mut Separator, init: &SPSolution, term: &Terminator) -> SPSolution {
    let mut best = init.clone();
    let start = Instant::now();
    let end = term.timeout.expect("compression running without timeout");
    let step_size = || -> f32 {
        //map the range [COMPRESS_SHRINK_RANGE.0, COMPRESS_SHRINK_RANGE.1] to timeout
        let range = COMPRESS_SHRINK_RANGE.1 - COMPRESS_SHRINK_RANGE.0;
        let elapsed = start.elapsed();
        let remaining = end.duration_since(Instant::now());
        let ratio = elapsed.as_secs_f32() / (elapsed + remaining).as_secs_f32();
        COMPRESS_SHRINK_RANGE.0 + ratio * range
    };
    while !term.is_kill() {
        let step = step_size();
        info!("[CMPR] attempting {:.3}%", step * 100.0);
        match attempt_to_compress(sep, &best, step, &term) {
            Some(compacted_sol) => {
                info!("[CMPR] compressed to {:.3} ({:.3}%)", compacted_sol.strip_width(), compacted_sol.density(instance) * 100.0);
                sep.export_svg(Some(compacted_sol.clone()), "cmpr", false);
                best = compacted_sol;
            }
            None => {}
        }
    }
    info!("[CMPR] finished compression, improved from {:.3}% to {:.3}% (+{:.3}%)", init.density(instance) * 100.0, best.density(instance) * 100.0, (best.density(instance) - init.density(instance)) * 100.0);
    best
}


fn attempt_to_compress(sep: &mut Separator, init: &SPSolution, r_shrink: f32, term: &Terminator) -> Option<SPSolution> {
    //restore to the initial solution and width
    sep.change_strip_width(init.strip_width(), None);
    sep.rollback(&init, None);

    //shrink the container at a random position
    let new_width = init.strip_width() * (1.0 - r_shrink);
    let split_pos = sep.rng.random_range(0.0..sep.prob.strip_width());
    sep.change_strip_width(new_width, Some(split_pos));

    //try to separate layout, if all collisions are eliminated, return the solution
    let (compacted_sol, ot) = sep.separate(term);
    match ot.get_total_loss() == 0.0 {
        true => Some(compacted_sol),
        false => None,
    }
}

// ...existing code...
fn disrupt_solution(sep: &mut Separator) {
    // The general idea is to disrupt a solution by swapping two 'large' items in the layout.
    // 'Large' items are those whose convex hull area falls within a certain top percentile
    // of the total convex hull area of all items in the layout.

    // Step 1: Define what constitutes a 'large' item.

    // Calculate the total convex hull area of all items, considering quantities.
    let total_convex_hull_area: f32 = sep
        .prob
        .instance
        .items
        .iter()
        .map(|(item, quantity)| item.shape_cd.surrogate().convex_hull_area * (*quantity as f32))
        .sum();

    let cutoff_threshold_area = total_convex_hull_area * LARGE_AREA_CH_AREA_CUTOFF_PERCENTILE;

    // Sort items by convex hull area in descending order.
    let sorted_items_by_ch_area = sep
        .prob
        .instance
        .items
        .iter()
        .sorted_by_key(|(item, _)| Reverse(OrderedFloat(item.shape_cd.surrogate().convex_hull_area)))
        .peekable();

    let mut cumulative_ch_area = 0.0;
    let mut ch_area_cutoff = 0.0;

    // Iterate through items, accumulating their convex hull areas until the cumulative sum
    // exceeds the cutoff_threshold_area. The convex hull area of the item that causes
    // this excess becomes the ch_area_cutoff.
    for (item, quantity) in sorted_items_by_ch_area {
        let item_ch_area = item.shape_cd.surrogate().convex_hull_area;
        cumulative_ch_area += item_ch_area * (*quantity as f32);
        if cumulative_ch_area > cutoff_threshold_area {
            ch_area_cutoff = item_ch_area;
            debug!("[DSRP] cutoff ch area: {}, for item id: {}, bbox: {:?}",ch_area_cutoff, item.id, item.shape_cd.bbox);
            break;
        }
    }

    // Step 2: Select two 'large' items and 'swap' them.

    let large_items = sep.prob.layout.placed_items.iter()
        .filter(|(_, pi)| pi.shape.surrogate().convex_hull_area >= ch_area_cutoff);

    //Choose a first item with a large enough convex hull
    let (pk1, pi1) = large_items.clone().choose(&mut sep.rng).expect("[DSRP] failed to choose first item");

    //Choose a second item with a large enough convex hull and different enough from the first.
    //If no such item is found, choose a random one.
    let (pk2, pi2) = large_items.clone()
        .filter(|(_, pi)|
            // Ensure the second item is different from the first
            !approx_eq!(f32, pi.shape.area,pi1.shape.area, epsilon = pi1.shape.area * 0.01) &&
                !approx_eq!(f32, pi.shape.diameter, pi1.shape.diameter, epsilon = pi1.shape.diameter * 0.01)
        )
        .choose(&mut sep.rng)
        .or_else(|| {
            sep.prob.layout.placed_items.iter()
                .filter(|(pk, _)| pk != &pk1) // Ensure the second item is not the same as the first
                .choose(&mut sep.rng)
        }) // As a fallback, choose any item
        .expect("[DSRP] failed to choose second item");

    // Step 3: Swap the two items' positions in the layout.

    let dt1_old = pi1.d_transf;
    let dt2_old = pi2.d_transf;

    // Make sure the swaps do not violate feasibility (rotation).
    let dt1_new = convert_sample_to_closest_feasible(dt2_old, sep.prob.instance.item(pi1.item_id));
    let dt2_new = convert_sample_to_closest_feasible(dt1_old, sep.prob.instance.item(pi2.item_id));

    info!("[EXPL] swapped two large items (ids: {} <-> {})", pi1.item_id, pi2.item_id);

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

        for c1_pk in practically_contained_items(&sep.prob.layout, pk1).into_iter().filter(|c1_pk| *c1_pk != pk2) {
            let c1_pi = &sep.prob.layout.placed_items[c1_pk];

            let new_dt = c1_pi.d_transf
                .compose()
                .transform(&converting_transformation)
                .decompose();

            //Ensure the sure the new position is feasible
            let new_feasible_dt = convert_sample_to_closest_feasible(new_dt, sep.prob.instance.item(c1_pi.item_id));
            sep.move_item(c1_pk, new_feasible_dt);
        }
    }

    // Do the same for the second item, but using the second transformation
    {
        let converting_transformation = dt2_new.compose().inverse()
            .transform(&dt2_old.compose());

        for c2_pk in practically_contained_items(&sep.prob.layout, pk2).into_iter().filter(|c2_pk| *c2_pk != pk1) {
            let c2_pi = &sep.prob.layout.placed_items[c2_pk];
            let new_dt = c2_pi.d_transf
                .compose()
                .transform(&converting_transformation)
                .decompose();

            //make sure the new position is feasible
            let new_feasible_dt = convert_sample_to_closest_feasible(new_dt, sep.prob.instance.item(c2_pi.item_id));
            sep.move_item(c2_pk, new_feasible_dt);
        }
    }
}

/// Collects all items which point of inaccessibility (POI) is contained by pk_c's shape.
fn practically_contained_items(layout: &Layout, pk_c: PItemKey) -> Vec<PItemKey> {
    let pi_c = &layout.placed_items[pk_c];
    // Detect all collisions with the item pk_c's shape.
    let mut collector = SecondaryMap::new();
    layout.cde().collect_poly_collisions(&pi_c.shape, &mut collector);

    // Filter out the items that have their POI contained by pk_c's shape.
    collector.iter()
        .filter_map(|(_,he)| {
            match he {
                HazardEntity::PlacedItem { pk, .. } => Some(*pk),
                _ => None
            }
        })
        .filter(|pk| *pk != pk_c) // Ensure we don't include the item itself
        .filter(|pk| {
            // Check if the POI of the item is contained by pk_c's shape
            let poi = layout.placed_items[*pk].shape.poi;
            pi_c.shape.collides_with(&poi.center)
        })
        .collect_vec()
}

use std::cmp::Reverse;
use float_cmp::approx_eq;
use itertools::Itertools;
use jagua_rs::collision_detection::hazards::HazardEntity;
use jagua_rs::entities::{Instance, Layout, PItemKey};
use jagua_rs::geometry::geo_traits::CollidesWith;
use jagua_rs::probs::spp::entities::{SPInstance, SPSolution};
use log::{debug, info, warn};
use ordered_float::OrderedFloat;
use rand::prelude::{Distribution, IteratorRandom};
use rand_distr::Normal;
use slotmap::SecondaryMap;
use crate::config::ExplorationConfig;
use crate::FMT;
use crate::optimizer::separator::{Separator, SeparatorConfig};
use crate::sample::uniform_sampler::convert_sample_to_closest_feasible;
use crate::util::listener::{ReportType, SolutionListener};
use crate::util::terminator::Terminator;

/// Algorithm 12 from https://doi.org/10.48550/arXiv.2509.13329
pub fn exploration_phase(instance: &SPInstance, sep: &mut Separator, sol_listener: &mut impl SolutionListener,  term: &impl Terminator, config: &ExplorationConfig) -> Vec<SPSolution> {
    let mut current_width = sep.prob.strip_width();
    let mut best_width = current_width;

    let mut feasible_solutions = vec![sep.prob.save()];

    sol_listener.report(ReportType::ExplFeas, &feasible_solutions[0], instance);
    info!("[EXPL] starting optimization with initial width: {:.3} ({:.3}%)",current_width,sep.prob.density() * 100.0);

    let mut solution_pool: Vec<(SPSolution, f32)> = vec![];

    while !term.kill() {
        let local_best = sep.separate(term, sol_listener);
        let total_loss = local_best.1.get_total_loss();

        if total_loss == 0.0 {
            //layout is successfully separated
            if current_width < best_width {
                info!("[EXPL] feasible solution found! (width: {:.3}, dens: {:.3}%)",current_width,sep.prob.density() * 100.0);
                best_width = current_width;
                feasible_solutions.push(local_best.0.clone());
                sol_listener.report(ReportType::ExplFeas, &local_best.0, instance);
            }
            let next_width = current_width * (1.0 - config.shrink_step);
            info!("[EXPL] shrinking strip by {}%: {:.3} -> {:.3}", config.shrink_step * 100.0, current_width, next_width);
            sep.change_strip_width(next_width, None);
            current_width = next_width;
            solution_pool.clear();
        } else {
            info!("[EXPL] unable to reach feasibility (width: {:.3}, dens: {:.3}%, min loss: {:.3})", current_width, sep.prob.density() * 100.0, FMT().fmt2(total_loss));
            sol_listener.report(ReportType::ExplInfeas, &local_best.0, instance);

            //layout was not successfully separated, add to local bests
            match solution_pool.binary_search_by(|(_, o)| o.partial_cmp(&total_loss).unwrap()) {
                Ok(idx) | Err(idx) => solution_pool.insert(idx, (local_best.0.clone(), total_loss)),
            }

            if solution_pool.len() >= config.max_conseq_failed_attempts.unwrap_or(usize::MAX) {
                info!("[EXPL] max consecutive failed attempts ({}), terminating", solution_pool.len());
                break;
            }

            //restore to a random solution from the tabu list, better solutions have more chance to be selected
            let selected_sol = {
                //sample a value in range [0.0, 1.0[ from a normal distribution
                let distr = Normal::new(0.0, config.solution_pool_distribution_stddev).unwrap();
                let sample = distr.sample(&mut sep.rng).abs().min(0.999);
                //map it to the range of the solution pool
                let selected_idx = (sample * solution_pool.len() as f32) as usize;

                let (selected_sol, loss) = &solution_pool[selected_idx];
                info!("[EXPL] starting solution {}/{} selected from solution pool (l: {}) to disrupt", selected_idx, solution_pool.len(), FMT().fmt2(*loss));
                selected_sol
            };

            sep.rollback(selected_sol, None);
            disrupt_solution(sep, config);
        }
    }

    info!("[EXPL] finished, best feasible solution: width: {:.3} ({:.3}%)",best_width,feasible_solutions.last().unwrap().density(instance) * 100.0);

    feasible_solutions
}

fn disrupt_solution(sep: &mut Separator, config: &ExplorationConfig) {
    if sep.prob.layout.placed_items.len() < 2 {
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
        .instance
        .items
        .iter()
        .map(|(item, quantity)| item.shape_cd.surrogate().convex_hull_area * (*quantity as f32))
        .sum();

    let cutoff_threshold_area = total_convex_hull_area * config.large_item_ch_area_cutoff_percentile;

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
    let (pk1, pi1) = large_items.clone()
        .filter(|(pk, _)| sep.ct.get_loss(*pk) > 0.0) // Prefer items that are involved in collisions
        .choose(&mut sep.rng)
        .or_else(|| large_items.clone().choose(&mut sep.rng)) //Fallback: choose any large item
        .expect("[DSRP] failed to choose first item");

    //Choose a second item with a large enough convex hull and different enough from the first.
    //If no such item is found, choose a random one.
    let (pk2, pi2) = large_items.clone()
        .filter(|(pk, _)| *pk != pk1) // Ensure the second item is not the same as the first
        .filter(|(_, pi)| { // Ensure the second item is sufficiently different in shape
            !approx_eq!(f32, pi.shape.area,pi1.shape.area, epsilon = pi1.shape.area * 0.01) &&
                !approx_eq!(f32, pi.shape.diameter, pi1.shape.diameter, epsilon = pi1.shape.diameter * 0.01)
        })
        .choose(&mut sep.rng)
        .or_else(|| { // Fallback: pick any item
            sep.prob.layout.placed_items.iter()
                .filter(|(pk, _)| *pk != pk1)
                .choose(&mut sep.rng)
        }) // As a fallback, choose any item
        .expect("[EXPL] failed to choose second item for disruption");

    // Step 3: Swap the two items' positions in the layout.

    let dt1_old = pi1.d_transf;
    let dt2_old = pi2.d_transf;

    // Make sure the swaps do not violate feasibility (rotation).
    let dt1_new = convert_sample_to_closest_feasible(dt2_old, sep.prob.instance.item(pi1.item_id));
    let dt2_new = convert_sample_to_closest_feasible(dt1_old, sep.prob.instance.item(pi2.item_id));

    info!("[EXPL] disrupting by swapping two large items (id: {} <-> {})", pi1.item_id, pi2.item_id);

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
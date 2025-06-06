use std::cmp::Reverse;
use crate::config::*;
use crate::optimizer::lbf::LBFBuilder;
use crate::optimizer::separator::Separator;
pub use crate::optimizer::terminator::Terminator;
use crate::sample::uniform_sampler::{convert_sample_to_closest_feasible, UniformBBoxSampler};
use crate::FMT;
use float_cmp::approx_eq;
use itertools::Itertools;
use jagua_rs::entities::{Instance, Layout, PItemKey};
use jagua_rs::probs::spp::entities::{SPInstance, SPSolution};
use log::info;
use ordered_float::OrderedFloat;
use rand::prelude::{IteratorRandom, SmallRng};
use rand::{random_range, Rng, RngCore, SeedableRng};
use rand_distr::Distribution;
use rand_distr::Normal;
use std::iter;
use std::time::{Duration, Instant};
use jagua_rs::collision_detection::hazards::detector::{BasicHazardDetector, HazardDetector};
use jagua_rs::collision_detection::hazards::HazardEntity;
use jagua_rs::geometry::geo_traits::CollidesWith;
use jagua_rs::geometry::primitives::Rect;
use rand::distr::Uniform;

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

            //restore and swap two large items
            sep.rollback(selected_sol, None);
            swap_large_pair_of_items(sep);
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

//TODO: polish
fn swap_large_pair_of_items(sep: &mut Separator) {
    //TODO: make a more elaborate way of selecting between significant and non-significant items
    //      to make the disruption more robust across instances

    //sep.export_svg(None, "before_disruption", false);

    let ch_area_sum_percentile = sep.prob.instance.items.iter()
        .map(|(item, q)| item.shape_cd.surrogate().convex_hull_area * (*q as f32))
        .sum::<f32>() * LARGE_AREA_CH_AREA_CUTOFF_PERCENTILE;

    let descending_ch_areas = sep.prob.instance.items.iter()
        .sorted_by_key(|(item, _)| Reverse(OrderedFloat(item.shape_cd.surrogate().convex_hull_area)));

    let mut large_ch_area_cutoff = 0.0;
    let mut ch_area_sum = 0.0;

    for (item, q) in descending_ch_areas {
        let ch_area = item.shape_cd.surrogate().convex_hull_area;
        ch_area_sum += ch_area * (*q as f32);
        if ch_area_sum > ch_area_sum_percentile {
            large_ch_area_cutoff = ch_area;
            info!("[DSRP] cutoff is {}, bbox: {:?}", item.id, item.shape_cd.bbox);
            break;
        }
    }

    let layout = &sep.prob.layout;

    //Choose a first item with a large enough convex hull
    let (pk1, pi1) = layout.placed_items.iter()
        .filter(|(_, pi)| pi.shape.surrogate().convex_hull_area >= large_ch_area_cutoff)
        .choose(&mut sep.rng)
        .unwrap();

    //Choose a second item with a large enough convex hull and different enough from the first.
    //If no such item is found, choose a random one.
    let (pk2, pi2) = layout.placed_items.iter()
        .filter(|(_, pi)| !approx_eq!(f32, pi.shape.area,pi1.shape.area, epsilon = pi1.shape.area * 0.01))
        .filter(|(_, pi)| pi.shape.surrogate().convex_hull_area >= large_ch_area_cutoff)
        .choose(&mut sep.rng)
        .unwrap_or(layout.placed_items.iter()
            .filter(|(pk2, _)| *pk2 != pk1)
            .choose(&mut sep.rng).unwrap());

    let dt1_old = pi1.d_transf;
    let dt2_old = pi2.d_transf;

    let dt1_new = convert_sample_to_closest_feasible(dt2_old, sep.prob.instance.item(pi1.item_id));
    let dt2_new = convert_sample_to_closest_feasible(dt1_old, sep.prob.instance.item(pi2.item_id));

    info!("[EXPL] swapped two large items (ids: {} <-> {})", pi1.item_id, pi2.item_id);

    let pk1 = sep.move_item(pk1, dt1_new);
    let pk2 = sep.move_item(pk2, dt2_new);

    //sep.export_svg(None, "mid_disruption", false);

    {
        let conv_t = dt1_new.compose().inverse()
            .transform(&dt1_old.compose());

        //Move all colliding items to the "empty space" created by the moved item
        for c1_pk in practically_contained_items(&sep.prob.layout, pk1).into_iter().filter(|c1_pk| *c1_pk != pk2) {
            let c1_pi = &sep.prob.layout.placed_items[c1_pk];
            let new_dt = c1_pi.d_transf
                .compose()
                .transform(&conv_t)
                .decompose();

            //make sure the new position is feasible
            let new_feasible_dt = convert_sample_to_closest_feasible(new_dt, sep.prob.instance.item(c1_pi.item_id));
            //let new_feasible_dt = new_dt;
            sep.move_item(c1_pk, new_feasible_dt);
        }
    }

    {
        let conv_t = dt2_new.compose().inverse()
            .transform(&dt2_old.compose());

        //Move all colliding items to the "empty space" created by the moved item
        for c2_pk in practically_contained_items(&sep.prob.layout, pk2).into_iter().filter(|c2_pk| *c2_pk != pk1) {
            let c2_pi = &sep.prob.layout.placed_items[c2_pk];
            let new_dt = c2_pi.d_transf
                .compose()
                .transform(&conv_t)
                .decompose();

            //make sure the new position is feasible
            let new_feasible_dt = convert_sample_to_closest_feasible(new_dt, sep.prob.instance.item(c2_pi.item_id));
            //let new_feasible_dt = new_dt;
            sep.move_item(c2_pk, new_feasible_dt);
        }
    }
    //sep.export_svg(None, "after_disruption", false);
}

fn practically_contained_items(layout: &Layout, pk_c: PItemKey) -> Vec<PItemKey> {
    let new_pi = &layout.placed_items[pk_c];
    let mut hazard_detector = BasicHazardDetector::new();
    layout.cde().collect_poly_collisions(&new_pi.shape, &mut hazard_detector);
    let items = hazard_detector.iter()
        .filter_map(|he| {
            match he {
                HazardEntity::PlacedItem{pk, ..} => {
                   if *pk == pk_c {
                        //Skip the item itself
                        return None;
                    }
                    Some(*pk)
                },
                _ => None
            }
        })
        .filter(|pk| {
            let poi = layout.placed_items[*pk].shape.poi;
            new_pi.shape.collides_with(&poi.center)
        })
        .collect_vec();

    items
}
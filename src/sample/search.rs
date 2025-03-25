use crate::config::UNIQUE_SAMPLE_THRESHOLD;
use crate::eval::sample_eval::{SampleEval, SampleEvaluator};
use crate::sample::best_samples::BestSamples;
use crate::sample::coord_descent::coordinate_descent;
use crate::sample::uniform_sampler::UniformBBoxSampler;
use jagua_rs::entities::item::Item;
use jagua_rs::entities::layout::Layout;
use jagua_rs::entities::placed_item::PItemKey;
use jagua_rs::geometry::d_transformation::DTransformation;
use jagua_rs::geometry::geo_traits::Shape;
use log::debug;
use rand::Rng;

#[derive(Debug, Clone, Copy)]
pub struct SampleConfig {
    pub n_bin_samples: usize,
    pub n_focussed_samples: usize,
    pub n_coord_descents: usize,
}

pub fn search_placement(l: &Layout, item: &Item, ref_pk: Option<PItemKey>, mut evaluator: impl SampleEvaluator, sample_config: SampleConfig, rng: &mut impl Rng) -> (Option<(DTransformation, SampleEval)>, usize) {
    let item_min_dim = f32::min(item.shape.bbox().width(), item.shape.bbox().height());

    let mut best_samples = BestSamples::new(sample_config.n_coord_descents, item_min_dim * UNIQUE_SAMPLE_THRESHOLD);

    let focussed_sampler = match ref_pk {
        Some(ref_pk) => {
            //report the current placement (and eval)
            let dt = l.placed_items[ref_pk].d_transf;
            let eval = evaluator.eval(dt, Some(best_samples.upper_bound()));

            best_samples.report(dt, eval);

            //create a sampler around the current placement
            let pi_bbox = l.placed_items[ref_pk].shape.bbox();
            UniformBBoxSampler::new(pi_bbox, item, l.bin.bbox())
        }
        None => None,
    };

    if let Some(focussed_sampler) = focussed_sampler {
        for _ in 0..sample_config.n_focussed_samples {
            let dt = focussed_sampler.sample(rng);
            let eval = evaluator.eval(dt, Some(best_samples.upper_bound()));
            best_samples.report(dt, eval);
        }
    }

    let bin_sampler = UniformBBoxSampler::new(l.bin.bbox(), item, l.bin.bbox());

    if let Some(bin_sampler) = bin_sampler {
        for _ in 0..sample_config.n_bin_samples {
            let dt = bin_sampler.sample(rng).into();
            let eval = evaluator.eval(dt, Some(best_samples.upper_bound()));
            best_samples.report(dt, eval);
        }
    }

    for start in best_samples.samples.clone() {
        let descended = coordinate_descent(start.clone(), &mut evaluator, item_min_dim, rng);
        best_samples.report(descended.0, descended.1);
    }


    debug!("[S] {} samples evaluated, best: {:?}",evaluator.n_evals(),best_samples.best());
    (best_samples.best(), evaluator.n_evals())
}

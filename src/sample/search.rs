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

pub fn search_placement(l: &Layout, item: &Item, ref_pk: Option<PItemKey>, mut evaluator: impl SampleEvaluator, sample_config: SampleConfig, rng: &mut impl Rng) -> (DTransformation, SampleEval, usize) {
    let item_min_dim = f32::min(item.shape.bbox().width(), item.shape.bbox().height());

    let best_local_sample = match ref_pk {
        Some(ref_pk) => {
            let mut best_samples = BestSamples::new(1, item_min_dim * UNIQUE_SAMPLE_THRESHOLD);
            //report the current placement (and eval)
            let dt = l.placed_items[ref_pk].d_transf;
            let eval = evaluator.eval(dt, Some(best_samples.worst()));

            best_samples.report(dt, eval);

            //create a sampler around the current placement
            let pi_bbox = l.placed_items[ref_pk].shape.bbox();
            let loc_sampler = UniformBBoxSampler::new(pi_bbox, item);

            //sample around the current placement
            for _ in 0..sample_config.n_focussed_samples {
                let dt = loc_sampler.sample(rng);
                let eval = evaluator.eval(dt, Some(best_samples.worst()));
                best_samples.report(dt, eval);
            }
            Some(best_samples.best())
        }
        None => None,
    };

    let best_bin_sample = {
        let bin_sampler = l.bin.bbox()
            .resize_by(-item.shape.poi.radius, -item.shape.poi.radius)
            .map(|bbox| UniformBBoxSampler::new(bbox, item));

        let mut best_samples = BestSamples::new(1, item_min_dim * UNIQUE_SAMPLE_THRESHOLD);

        if let Some(bin_sampler) = bin_sampler {
            for _ in 0..sample_config.n_bin_samples {
                let dt = bin_sampler.sample(rng).into();
                let eval = evaluator.eval(dt, Some(best_samples.worst()));
                best_samples.report(dt, eval);
            }
            Some(best_samples.best())
        } else {
            None
        }
    };

    let cd_best_sample = {
        let mut best_samples = BestSamples::new(1, item_min_dim);
        if let Some(best_local_sample) = best_local_sample {
            let descended = coordinate_descent(best_local_sample, &mut evaluator, item_min_dim, rng);
            best_samples.report(descended.0, descended.1);
        };
        if let Some(best_bin_sample) = best_bin_sample {
            let descended = coordinate_descent(best_bin_sample, &mut evaluator, item_min_dim, rng);
            best_samples.report(descended.0, descended.1);
        }
        best_samples.best()
    };


    debug!("[S] {} samples evaluated, best: {:?}, {}",evaluator.n_evals(),cd_best_sample.1,cd_best_sample.0);

    (cd_best_sample.0, cd_best_sample.1, evaluator.n_evals())
}

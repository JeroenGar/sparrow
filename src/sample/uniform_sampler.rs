use itertools::Itertools;
use jagua_rs::geometry::geo_enums::RotationRange;
use jagua_rs::geometry::geo_traits::{Shape, TransformableFrom};
use rand::prelude::IndexedRandom;
use rand::Rng;
use std::ops::Range;
use jagua_rs::entities::general::Item;
use jagua_rs::geometry::primitives::Rect;
use jagua_rs::geometry::{DTransformation, Transformation};

/// A sampler that creates uniform samples for an item within a bounding box
#[derive(Clone, Debug)]
pub struct UniformBBoxSampler {
    /// The list of possible rotations and their corresponding x and y ranges
    rot_entries: Vec<RotEntry>,
}

#[derive(Clone, Debug)]
struct RotEntry {
    pub r: f32,
    pub x_range: Range<f32>,
    pub y_range: Range<f32>,
}

impl UniformBBoxSampler {
    pub fn new(sample_bbox: Rect, item: &Item, bin_bbox: Rect) -> Option<Self> {
        let rotations = match &item.allowed_rotation {
            RotationRange::None => &vec![0.0],
            RotationRange::Discrete(r) => r,
            RotationRange::Continuous => unimplemented!("Continuous rotation not supported"),
        };

        let mut shape_buffer = item.shape_cd.as_ref().clone();

        let sample_x_range = sample_bbox.x_min..sample_bbox.x_max;
        let sample_y_range = sample_bbox.y_min..sample_bbox.y_max;

        // for each possible rotation, calculate the sample ranges (x and y)
        // where the item resides fully inside the bin and is within the sample bounding box
        let rot_entries = rotations.iter()
            .map(|&r| {
                let r_shape_bbox = shape_buffer.transform_from(item.shape_cd.as_ref(), &Transformation::from_rotation(r)).bbox();

                //narrow the bin range to account for the rotated shape
                let bin_x_range = (bin_bbox.x_min - r_shape_bbox.x_min)..(bin_bbox.x_max - r_shape_bbox.x_max);
                let bin_y_range = (bin_bbox.y_min - r_shape_bbox.y_min)..(bin_bbox.y_max - r_shape_bbox.y_max);

                //intersect with the sample bbox
                let x_range = intersect_range(&bin_x_range, &sample_x_range);
                let y_range = intersect_range(&bin_y_range, &sample_y_range);

                //make sure the ranges are not empty
                if x_range.is_empty() || y_range.is_empty() {
                    None
                } else {
                    Some(RotEntry { r, x_range, y_range })
                }
            }).flatten().collect_vec();

        match rot_entries.is_empty() {
            true => None,
            false => Some(Self { rot_entries }),
        }
    }

    pub fn sample(&self, rng: &mut impl Rng) -> DTransformation {
        // randomly select a rotation
        let r_entry = self.rot_entries.choose(rng).unwrap();

        // sample a random x and y value within the valid range
        let r = r_entry.r;
        let x_sample = rng.random_range(r_entry.x_range.clone());
        let y_sample = rng.random_range(r_entry.y_range.clone());

        DTransformation::new(r, (x_sample, y_sample))
    }
}

fn intersect_range(a: &Range<f32>, b: &Range<f32>) -> Range<f32> {
    let min = f32::max(a.start, b.start);
    let max = f32::min(a.end, b.end);
    min..max
}
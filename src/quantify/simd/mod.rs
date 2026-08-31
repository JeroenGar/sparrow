use crate::consts::OVERLAP_PROXY_EPSILON_DIAM_RATIO;
use crate::quantify::calc_shape_penalty;
use crate::quantify::simd::circles_soa::CirclesSoA;
use crate::quantify::simd::overlap_proxy_simd::poles_overlap_area_proxy_simd_bounded;
use jagua_rs::geometry::primitives::SPolygon;
use std::f32::consts::PI;

pub mod circles_soa;
pub mod overlap_proxy_simd;

/// Quantifies a collision between two simple polygons using SIMD.
/// Mirrors the functionality of `quantify_collision_poly_poly` but leverages SIMD instructions.
#[inline(always)]
pub fn quantify_collision_poly_poly_simd(s1: &SPolygon, s2: &SPolygon, poles2: &CirclesSoA) -> f32 {
    quantify_collision_poly_poly_simd_bounded(s1, s2, poles2, f32::INFINITY).unwrap()
}

#[inline(always)]
pub fn quantify_collision_poly_poly_simd_bounded(
    s1: &SPolygon,
    s2: &SPolygon,
    poles2: &CirclesSoA,
    max_loss: f32,
) -> Option<f32> {
    let epsilon = f32::max(s1.diameter, s2.diameter) * OVERLAP_PROXY_EPSILON_DIAM_RATIO;
    let epsilon_sq = epsilon * epsilon;
    let penalty = calc_shape_penalty(s1, s2);
    let max_unscaled_overlap = {
        let max_sqrt_proxy = max_loss / penalty;
        let overlap = (max_sqrt_proxy * max_sqrt_proxy - epsilon_sq) / PI;
        if overlap < 0.0 {
            return None;
        }
        overlap
    };

    let overlap_proxy = poles_overlap_area_proxy_simd_bounded(
        s1.surrogate(),
        s2.surrogate(),
        epsilon,
        poles2,
        max_unscaled_overlap,
    )? + epsilon_sq;

    Some(overlap_proxy.sqrt() * penalty)
}

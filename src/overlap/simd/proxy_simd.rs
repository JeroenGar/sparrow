use crate::config::OVERLAP_PROXY_EPSILON_DIAM_RATIO;
use crate::overlap::proxy::poles_overlap_area_proxy;
use crate::overlap::simd::circles_soa::CirclesSoA;
use float_cmp::approx_eq;
use jagua_rs::geometry::fail_fast::sp_surrogate::SPSurrogate;
use jagua_rs::geometry::geo_traits::{Distance, Shape};
use jagua_rs::geometry::primitives::circle::Circle;
use jagua_rs::geometry::primitives::point::Point;
use jagua_rs::geometry::primitives::simple_polygon::SimplePolygon;
use std::simd::prelude::{SimdFloat, SimdPartialOrd};
use std::simd::{f32x4, StdFloat};

#[inline(always)]
pub fn eval_overlap_poly_poly_simd(s1: &SimplePolygon, s2: &SimplePolygon, poles2: &CirclesSoA) -> f32 {
    let epsilon = f32::max(s1.diameter(), s2.diameter()) * OVERLAP_PROXY_EPSILON_DIAM_RATIO;

    let overlap_proxy = poles_overlap_area_proxy_simd(&s1.surrogate(), &s2.surrogate(), epsilon, poles2);

    debug_assert!(overlap_proxy.is_normal());

    let penalty = (s1.surrogate().convex_hull_area * s2.surrogate().convex_hull_area).sqrt();

    (overlap_proxy * penalty).sqrt()
}


/// SIMD version of [`poles_overlap_area_proxy`].
/// `p2` should match the poles of `sp2`.
#[inline(always)]
pub fn poles_overlap_area_proxy_simd(sp1: &SPSurrogate, sp2: &SPSurrogate, epsilon: f32, p2: &CirclesSoA) -> f32 {
    //
    let e_4 = f32x4::splat(epsilon);
    let e_sq_4 = f32x4::splat(epsilon * epsilon);
    let two_e_4 = f32x4::splat(2.0 * epsilon);

    let mut total_overlap = 0.0;
    for p1 in sp1.poles.iter() {
        // common values for all chunks
        let r1 = p1.radius;
        let x1_4 = f32x4::splat(p1.center.x());
        let y1_4 = f32x4::splat(p1.center.y());
        let r1_4 = f32x4::splat(r1);

        // process complete chunks of 4 elements with SIMD
        let chunks = p2.x.len() / 4;

        for chunk in 0..chunks {
            let idx = chunk * 4;

            // load the next 4 elements from p2
            let x2 = f32x4::from_slice(&p2.x[idx..idx + 4]);
            let y2 = f32x4::from_slice(&p2.y[idx..idx + 4]);
            let r2 = f32x4::from_slice(&p2.r[idx..idx + 4]);

            // calculate pd
            let dx = x1_4 - x2;
            let dy = y1_4 - y2;

            let pd = r1_4 + r2 - (dx * dx + dy * dy).sqrt();

            // calculate pd_decay
            let pd_mask = pd.simd_ge(e_4);
            let decay_values = e_sq_4 / (-pd + two_e_4);
            let pd_decay = pd_mask.select(pd, decay_values);

            // calculate min radius
            let min_r = r1_4.simd_min(r2);

            total_overlap += (pd_decay * min_r).reduce_sum();
        }

        // process remaining elements (0-3) with scalar operations
        let remaining_idx = chunks * 4;
        for j in remaining_idx..p2.x.len() {
            let p2 = Circle::new(Point(p2.x[j], p2.y[j]), p2.r[j]);

            // Penetration depth between the two poles (circles)
            let pd = (p1.radius + p2.radius) - p1.center.distance(&p2.center);

            let pd_decay = match pd >= epsilon {
                true => pd,
                false => epsilon.powi(2) / (-pd + 2.0 * epsilon),
            };

            total_overlap += pd_decay * f32::min(p1.radius, p2.radius);
        }
    }

    debug_assert!(approx_eq!(f32, total_overlap, poles_overlap_area_proxy(sp1, sp2, epsilon), epsilon = total_overlap * 1e-3), "SIMD and SEQ results do not match: {} vs {}", total_overlap, poles_overlap_area_proxy(sp1, sp2, epsilon));

    debug_assert!(total_overlap.is_normal());
    total_overlap
}



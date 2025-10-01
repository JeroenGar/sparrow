use itertools::izip;
use jagua_rs::geometry::primitives::Circle;

/// Collection of circles, but with a memory layout that's more suitable for SIMD operations:
/// SoA (Structure of Arrays) instead of AoS (Array of Structures).
#[derive(Debug, Clone)]
#[repr(align(32))]
pub struct CirclesSoA {
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub r: Vec<f32>,
}

impl CirclesSoA {
    pub fn new() -> Self {
        Self {
            x: Vec::new(),
            y: Vec::new(),
            r: Vec::new(),
        }
    }
    pub fn load(&mut self, circles: &[Circle]) -> &mut Self {
        self.x.resize(circles.len(), 0.0);
        self.y.resize(circles.len(), 0.0);
        self.r.resize(circles.len(), 0.0);

        //load the circles into the SoA format
        izip!(self.x.iter_mut(), self.y.iter_mut(), self.r.iter_mut())
            .zip(circles.iter())
            .for_each(|((x,y,r),ref_c)| {
                *x = ref_c.center.0;
                *y = ref_c.center.1;
                *r = ref_c.radius;
            });

        self
    }
}

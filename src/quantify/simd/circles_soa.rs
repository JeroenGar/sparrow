use super::SIMD_WIDTH;
use itertools::izip;
use jagua_rs::geometry::primitives::Circle;

/// Structure-of-arrays circle buffer padded with zero-radius circles to [`SIMD_WIDTH`].
#[derive(Debug, Clone)]
#[repr(align(32))]
pub struct CirclesSoA {
    pub(super) x: Vec<f32>,
    pub(super) y: Vec<f32>,
    pub(super) r: Vec<f32>,
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
        let n = circles.len();
        let padded_len = n.next_multiple_of(SIMD_WIDTH);
        self.x.resize(padded_len, 0.0);
        self.y.resize(padded_len, 0.0);
        self.r.resize(padded_len, 0.0);

        //load the circles into the SoA format
        izip!(self.x.iter_mut(), self.y.iter_mut(), self.r.iter_mut())
            .zip(circles.iter())
            .for_each(|((x,y,r),ref_c)| {
                *x = ref_c.center.0;
                *y = ref_c.center.1;
                *r = ref_c.radius;
            });

        // Clear reused padding; resize only initializes newly allocated slots.
        self.x[n..].fill(0.0);
        self.y[n..].fill(0.0);
        self.r[n..].fill(0.0);

        self
    }
}

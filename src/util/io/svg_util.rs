use std::fmt::{Display, Formatter};
use jagua_rs::entities::general::N_QUALITIES;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize, Copy)]
pub struct SvgDrawOptions {
    ///The theme to use for the svg
    #[serde(default)]
    pub theme: SvgLayoutTheme,
    ///Draw the quadtree on top
    #[serde(default)]
    pub quadtree: bool,
    ///Draw the fail fast surrogate on top of each item
    #[serde(default)]
    pub surrogate: bool,
    ///Draw lines between colliding items
    #[serde(default)]
    pub highlight_collisions: bool,
    ///Draw the modified shapes used internally instead of the original ones
    #[serde(default)]
    pub draw_cd_shapes: bool,
    #[serde(default)]
    pub highlight_cd_shapes: bool,
}

impl Default for SvgDrawOptions {
    fn default() -> Self {
        Self {
            theme: SvgLayoutTheme::default(),
            quadtree: false,
            surrogate: true,
            highlight_collisions: true,
            draw_cd_shapes: false,
            highlight_cd_shapes: true,
        }
    }
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize, Copy)]
pub struct SvgLayoutTheme {
    pub stroke_width_multiplier: f32,
    pub bin_fill: Color,
    pub item_fill: Color,
    pub hole_fill: Color,
    pub qz_fill: [Color; N_QUALITIES],
    pub qz_stroke_opac: f32,
    pub collision_highlight_color: Color,
}

impl Default for SvgLayoutTheme {
    fn default() -> Self {
        SvgLayoutTheme::GRAY_THEME
    }
}

impl SvgLayoutTheme{

    #[allow(dead_code)]
    pub(crate) const EARTH_TONES: SvgLayoutTheme = SvgLayoutTheme {
        stroke_width_multiplier: 2.0,
        bin_fill: Color(0xCC, 0x82, 0x4A),
        item_fill: Color(0xFF, 0xC8, 0x79),
        hole_fill: Color(0x2D, 0x2D, 0x2D),
        qz_fill: [
            Color(0x00, 0x00, 0x00), // BLACK
            Color(0xFF, 0x00, 0x00), // RED
            Color(0xFF, 0x5E, 0x00), // ORANGE
            Color(0xFF, 0xA5, 0x00), // LIGHT ORANGE
            Color(0xC7, 0xA9, 0x00), // DARK YELLOW
            Color(0xFF, 0xFF, 0x00), // YELLOW
            Color(0xCB, 0xFF, 0x00), // GREEN
            Color(0xCB, 0xFF, 0x00), // GREEN
            Color(0xCB, 0xFF, 0x00), // GREEN
            Color(0xCB, 0xFF, 0x00), // GREEN
        ],
        qz_stroke_opac: 0.5,
        collision_highlight_color: Color(0x00, 0xFF, 0x00), // LIME
    };

    pub(crate) const GRAY_THEME: SvgLayoutTheme = SvgLayoutTheme {
        stroke_width_multiplier: 2.5,
        bin_fill: Color(0xD3, 0xD3, 0xD3),
        item_fill: Color(0x7A, 0x7A, 0x7A),
        hole_fill: Color(0xFF, 0xFF, 0xFF),
        qz_fill: [
            Color(0x63, 0x63, 0x63), //GRAY
            Color(0x63, 0x63, 0x63), //GRAY
            Color(0x63, 0x63, 0x63), //GRAY
            Color(0x63, 0x63, 0x63), //GRAY
            Color(0x63, 0x63, 0x63), //GRAY
            Color(0x63, 0x63, 0x63), //GRAY
            Color(0x63, 0x63, 0x63), //GRAY
            Color(0x63, 0x63, 0x63), //GRAY
            Color(0x63, 0x63, 0x63), //GRAY
            Color(0x63, 0x63, 0x63), //GRAY
        ],
        qz_stroke_opac: 0.9,
        collision_highlight_color: Color(0xD0, 0x00, 0x00), //LIME
    };
}

pub fn change_brightness(color: Color, fraction: f32) -> Color {
    let Color(r, g, b) = color;

    let r = (r as f32 * fraction) as u8;
    let g = (g as f32 * fraction) as u8;
    let b = (b as f32 * fraction) as u8;
    Color(r, g, b)
}

pub fn blend_colors(color_1: Color, color_2: Color) -> Color {
    //blend color_1 and color_2
    let Color(r_1, g_1, b_1) = color_1;
    let Color(r_2, g_2, b_2) = color_2;

    let r = ((r_1 as f32 * 0.5) + (r_2 as f32 * 0.5)) as u8;
    let g = ((g_1 as f32 * 0.5) + (g_2 as f32 * 0.5)) as u8;
    let b = ((b_1 as f32 * 0.5) + (b_2 as f32 * 0.5)) as u8;

    Color(r, g, b)
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Color(u8, u8, u8);

impl Display for Color {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "#{:02X}{:02X}{:02X}", self.0, self.1, self.2)
    }
}

impl From<String> for Color {
    fn from(mut s: String) -> Self {
        if s.starts_with('#') {
            s.remove(0);
        }
        let r = u8::from_str_radix(&s[0..2], 16).unwrap();
        let g = u8::from_str_radix(&s[2..4], 16).unwrap();
        let b = u8::from_str_radix(&s[4..6], 16).unwrap();
        Color(r, g, b)
    }
}

impl From<&str> for Color {
    fn from(s: &str) -> Self {
        Color::from(s.to_owned())
    }
}

impl Serialize for Color {
    fn serialize<S>(&self, serializer: S) -> Result<<S as Serializer>::Ok, <S as Serializer>::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&*format!("{self}"))
    }
}

impl<'de> Deserialize<'de> for Color {
    fn deserialize<D>(deserializer: D) -> Result<Self, <D as Deserializer<'de>>::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Ok(Color::from(s))
    }
}

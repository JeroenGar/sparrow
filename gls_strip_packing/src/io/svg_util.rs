use std::convert::Into;
use std::fmt::{Display, Formatter};

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use jagua_rs::entities::quality_zone::N_QUALITIES;
use jagua_rs::fsize;

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize, Copy)]
pub struct SvgDrawOptions {
    ///The theme to use for the svg
    #[serde(default)]
    pub theme: SvgLayoutTheme,
    ///Draw the quadtree on top
    #[serde(default)]
    pub quadtree: bool,
    ///Draw the hazard proximity grid on top
    #[serde(default)]
    pub haz_prox_grid: bool,
    ///Draw the fail fast surrogate on top of each item
    #[serde(default)]
    pub surrogate: bool,
    ///Draw lines between overlapping items
    #[serde(default)]
    pub overlap_lines: bool,
}

impl Default for SvgDrawOptions {
    fn default() -> Self {
        Self {
            theme: SvgLayoutTheme::default(),
            quadtree: false,
            haz_prox_grid: false,
            surrogate: false,
            overlap_lines: false,
        }
    }
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize, Copy)]
pub struct SvgLayoutTheme {
    pub stroke_width_multiplier: fsize,
    pub bin_fill: Color,
    pub item_fill: Color,
    pub hole_fill: Color,
    pub qz_fill: [Color; N_QUALITIES],
    pub qz_stroke_opac: fsize,
}

impl Default for SvgLayoutTheme {
    fn default() -> Self {
        Self::gray()
    }
}

impl SvgLayoutTheme {
    pub fn earth_tones() -> Self {
        SvgLayoutTheme {
            stroke_width_multiplier: 2.0,
            bin_fill: "#CC824A".into(),
            item_fill: "#FFC879".into(),
            hole_fill: "#2D2D2D".into(),
            qz_fill: [
                "#000000".into(), //BLACK
                "#FF0000".into(), //RED
                "#FF5E00".into(), //ORANGE
                "#FFA500".into(), //LIGHT ORANGE
                "#C7A900".into(), //DARK YELLOW
                "#FFFF00".into(), //YELLOW
                "#CBFF00".into(), //GREEN
                "#CBFF00".into(), //GREEN
                "#CBFF00".into(), //GREEN
                "#CBFF00".into(), //GREEN
            ],
            qz_stroke_opac: 0.5,
        }
    }

    pub fn gray() -> Self {
        SvgLayoutTheme {
            stroke_width_multiplier: 2.5,
            bin_fill: "#D6D6D6".into(),
            item_fill: "#8F8F8F".into(),
            hole_fill: "#FFFFFF".into(),
            qz_fill: [
                "#636363".into(), //GRAY
                "#636363".into(), //GRAY
                "#636363".into(), //GRAY
                "#636363".into(), //GRAY
                "#636363".into(), //GRAY
                "#636363".into(), //GRAY
                "#636363".into(), //GRAY
                "#636363".into(), //GRAY
                "#636363".into(), //GRAY
                "#636363".into(), //GRAY
            ],
            qz_stroke_opac: 0.9,
        }
    }
}

pub fn change_brightness(color: Color, fraction: fsize) -> Color {
    let Color(r, g, b) = color;

    let r = (r as fsize * fraction) as u8;
    let g = (g as fsize * fraction) as u8;
    let b = (b as fsize * fraction) as u8;
    Color(r, g, b)
}

pub fn blend_colors(color_1: Color, color_2: Color) -> Color {
    //blend color_1 and color_2
    let Color(r_1, g_1, b_1) = color_1;
    let Color(r_2, g_2, b_2) = color_2;

    let r = ((r_1 as fsize * 0.5) + (r_2 as fsize * 0.5)) as u8;
    let g = ((g_1 as fsize * 0.5) + (g_2 as fsize * 0.5)) as u8;
    let b = ((b_1 as fsize * 0.5) + (b_2 as fsize * 0.5)) as u8;

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

use crate::consts::DRAW_OPTIONS;
use crate::util::io;
use crate::util::listener::{ReportType, SolutionListener};
use jagua_rs::io::svg::s_layout_to_svg;
use jagua_rs::probs::spp::entities::{SPInstance, SPSolution};
use log::Level;
use std::path::Path;
pub struct SvgExporter {
    svg_counter: usize,
    /// Path to write the final SVG file to, if provided
    pub final_path: Option<String>,
    /// Directory to write all intermedia solution SVG files to, if provided
    pub intermediate_dir: Option<String>,
    /// Path to write the live SVG file to, if provided
    pub live_path: Option<String>,
}

impl SvgExporter {
    pub fn new(final_path: Option<String>, intermediate_dir: Option<String>, live_path: Option<String>) -> Self {
        // Clean all svg files from the intermediate directory if it is provided
        if let Some(intermediate_dir) = &intermediate_dir
            && let Ok(files_in_dir) = std::fs::read_dir(Path::new(intermediate_dir)) {
                for file in files_in_dir.flatten() {
                    if file.path().extension().unwrap_or_default() == "svg" {
                        std::fs::remove_file(file.path()).unwrap();
                    }
                }
            }
        
        SvgExporter {
            svg_counter: 0,
            final_path,
            intermediate_dir,
            live_path,
        }
    }
}

impl SolutionListener for SvgExporter{
    fn report(&mut self, report_type: ReportType, solution: &SPSolution, instance: &SPInstance) {
        let suffix = match report_type {
            ReportType::CmprFeas => "cmpr",
            ReportType::ExplInfeas => "expl_nf",
            ReportType::ExplFeas => "expl_f",
            ReportType::Final => "final",
            ReportType::ExplImproving => "expl_i"
        };
        let file_name = format!("{}_{:.3}_{}", self.svg_counter, solution.strip_width(), suffix);
        if let Some(live_path) = &self.live_path {
            let svg = s_layout_to_svg(&solution.layout_snapshot, instance, DRAW_OPTIONS, file_name.as_str());
            io::write_svg(&svg, Path::new(live_path), Level::Trace).expect("failed to write live svg");
        }
        if let Some(intermediate_dir) = &self.intermediate_dir && report_type != ReportType::ExplImproving {
            let svg = s_layout_to_svg(&solution.layout_snapshot, instance, DRAW_OPTIONS, file_name.as_str());
            let file_path = &*format!("{intermediate_dir}/{file_name}.svg");
            io::write_svg(&svg, Path::new(file_path), Level::Trace).expect("failed to write intermediate svg");
            self.svg_counter += 1;
        }
        if let Some(final_path) = &self.final_path && report_type == ReportType::Final {
            let stem = Path::new(final_path).file_stem().unwrap();
            let svg = s_layout_to_svg(&solution.layout_snapshot, instance, DRAW_OPTIONS, stem.to_str().unwrap());
            io::write_svg(&svg, Path::new(final_path), Level::Info).expect("failed to write final svg");
        }
    }
}
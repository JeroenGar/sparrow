mod args;
#[cfg(not(feature = "tui"))]
mod cli;
#[cfg(feature = "tui")]
mod tui;

#[cfg(not(feature = "tui"))]
use cli::run;
#[cfg(feature = "tui")]
use tui::run;
// The TUI handles Ctrl-C itself; `ctrlc` is only used by the plain CLI.
#[cfg(feature = "tui")]
use ctrlc as _;

fn main() -> anyhow::Result<()> {
    run()
}

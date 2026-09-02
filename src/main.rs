#[cfg(not(feature = "tui"))]
mod cli;
#[cfg(feature = "tui")]
mod tui;

#[cfg(not(feature = "tui"))]
use cli::run;
#[cfg(feature = "tui")]
use tui::run;

fn main() -> anyhow::Result<()> {
    run()
}

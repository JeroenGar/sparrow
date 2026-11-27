State-of-the-art nesting heuristic for 2D irregular strip packing problems.

Given a set of irregular items and a strip of fixed height, sparrow places all items without overlap
while minimizing the strip's width. The algorithm is described in
["An open-source heuristic to reboot 2D nesting research"](https://doi.org/10.48550/arXiv.2509.13329)
and builds on the [`jagua-rs`](https://crates.io/crates/jagua-rs) collision detection engine.

This crate is the library. The command-line interface, benchmark instances, records and
experiments live in the [GitHub repository](https://github.com/JeroenGar/sparrow).

## Quick start

```toml
[dependencies]
sparrow = "0.3"
```

Instances use the JSON format of `jagua-rs`
([format description](https://github.com/JeroenGar/jagua-rs?tab=readme-ov-file#input)).
`solve` runs the full heuristic:

```rust,no_run
use sparrow::config::SparrowConfig;
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let instance = sparrow::read_instance("swim.json")?;
    let config = SparrowConfig::default().with_time_limit(Duration::from_secs(60));

    let solution = sparrow::solve(&instance, &config)?;
    println!("width: {}, density: {:.2}%", solution.strip_width, solution.density * 100.0);

    sparrow::write_solution("swim_solution.json", &instance, &solution)?;
    std::fs::write("swim_solution.svg", sparrow::to_svg(&instance, &solution)?)?;
    Ok(())
}
```

Every input and output function has a file and an in-memory version:

|                     | File                               | In memory                         |
|---------------------|------------------------------------|-----------------------------------|
| Read an instance    | `read_instance(path)`              | `parse_instance(json)`            |
| Read a solution     | `read_solution(path)`              | `parse_solution(json)`            |
| Write a solution    | `write_solution(path, &i, &s)`     | `to_json(&i, &s)`                 |
| Draw a solution     |                                    | `to_svg(&i, &s)`                  |

A solution file holds the instance together with its solution, the same format the CLI writes.
The SVG is also an exact record of the solution: it holds every item's original shape and the
transformation applied to it.

## Warm starts

`solve_from` continues from an existing solution instead of constructing one,
for example the result of an earlier run:

```rust,no_run
use sparrow::config::SparrowConfig;
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (instance, initial) = sparrow::read_solution("swim_solution.json")?;
    let config = SparrowConfig::default().with_time_limit(Duration::from_secs(60));

    let solution = sparrow::solve_from(&instance, &initial, &config)?;
    sparrow::write_solution("swim_solution.json", &instance, &solution)?;
    Ok(())
}
```

The initial solution should be feasible and place every item.

## Configuration

`SparrowConfig::default()` is the configuration used in the paper. Most users only need:

- `with_time_limit(total)`: splits a time budget over exploration (80%) and compression (20%).
- `with_early_termination()`: lets both phases stop once they stop making progress.
- `rng_seed`: a fixed seed for the random number generator.

The fields of `SparrowConfig` expose everything else, such as the number of worker threads.

sparrow logs its progress through the [`log`](https://crates.io/crates/log) crate;
install any logger to see it.

## More control

`sparrow::optimizer::optimize` is the function behind `solve` and `solve_from`. It works on
imported `jagua-rs` instances and lets you plug in a `SolutionListener` to receive intermediate
solutions (for example `SvgExporter`, which writes them as SVG files), and a `Terminator` to
stop the search early, for example from another thread.

The other modules expose the algorithm's internals for experimentation. They follow the paper
closely, but carry no stability guarantees between versions.

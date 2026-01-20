## Experiments
This `README` contains all required information to reproduce the computational experiments presented in [*"An open-source heuristic to reboot 2D nesting research*"](https://doi.org/10.48550/arXiv.2509.13329).

#### Individual solutions

The final solutions and logs for all experiments that comprise the performance comparison in the paper (Table 2) are stored in the [benchmark_runs](benchmark_runs) folder.
The best solutions ever produced by `sparrow` for every instance (Table 3) are stored in [records](../records) folder.
Each file is both a visual and an exact representation of a solution, more information about the format can be found in the [here](../../README.md#output).

#### Reproducing the comparative experiments
The experiments were performed via GitHub Actions, on a self-hosted runner.
This system was running Ubuntu 20.04 LTS under WSL2 on Windows 11 and was equipped with an AMD Ryzen 9 7950X CPU and 64GB DDR5 at 5600MT/s.
The exact commands to run a benchmark are defined in [single_bench.yml](../../.github/workflows/single_bench.yml).

For every entry in [benchmark_runs](benchmark_runs) the log file contains all the information required for exact reproduction:
```
[BENCH] git commit hash: 4d70ca7f468957a046a74bbb614b896f0ad463e3
[BENCH] system time: 2025-03-28T19:13:40.628237341Z
[BENCH] no seed provided, using: 12552852848582794543
[BENCH] starting bench for swim (13x8 runs across 16 cores, 1200s timelimit)
...
```

Steps to reproduce the benchmark run above:
- Ensure the Rust toolchain (nightly) matches the one that was the most recent at the day of the experiment.
    - `rustup override set nightly-2025-03-28`
- Ensure the repo is checked out at the same commit:
    - `git checkout 4d70ca7f468957a046a74bbb614b896f0ad463e3`).
- Ensure, the seed in [config.rs](../../src/config.rs) is set to the randomly chosen one:
    - `pub const RNG_SEED: Option<usize> = Some(12552852848582794543);`
- Ensure `sparrow` is built and executed exactly as [single_bench.yml](../../.github/workflows/single_bench.yml) defines.
    - ```
      export RUSTFLAGS='-C target-cpu=native'
      cargo run --profile release --features=only_final_svg,simd --bin bench -- data/input/swim.json 1200 100
      ```

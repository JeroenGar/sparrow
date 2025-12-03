## Experiments & Reproducibility
This `README` contains all required information to reproduce the computational experiments presented in [*"An open-source heuristic to reboot 2D nesting research*"](https://doi.org/10.48550/arXiv.2509.13329).

#### Solution files

The best solutions ever produced by `sparrow` for each of the instances are stored in [records](../records) folder.

The final solutions and logs for all experiments that comprise the performance comparison in the paper are stored in the [benchmark_runs](benchmark_runs) folder.

#### Reproduction of comparative experiments
The experiments were all executed via GitHub Actions, on a self-hosted runner equipped with an AMD Ryzen 9 7950X CPU and 64GB DDR5 at 5600MT/s.
This system was running Ubuntu 20.04 LTS under WSL2 on Windows 11.
The exact commands to run a benchmark are defined in [single_bench.yml](../../.github/workflows/single_bench.yml).

For every entry in [benchmark_runs](benchmark_runs) the log file contains all the information required for exact reproduction:
```
[BENCH] git commit hash: 4d70ca7f468957a046a74bbb614b896f0ad463e3
[BENCH] system time: 2025-03-28T19:13:40.628237341Z
[BENCH] no seed provided, using: 12552852848582794543
[BENCH] starting bench for swim (13x8 runs across 16 cores, 1200s timelimit)
...
```

Steps to exactly reproduce this benchmark run:
- Ensure the Rust toolchain (nightly) matches the one that was the most recent at the time of the experiment (28th of March 2025).
- Ensure the repo is checked out at the same commit (same hash).
- In [config.rs](../../src/config.rs), set the seed to the one that was randomly chosen for this particular benchmark run:
    - For example: `pub const RNG_SEED: Option<usize> = Some(12552852848582794543);`
- `sparrow` is built and executed exactly the same as the [single_bench.yml](../../.github/workflows/single_bench.yml) action defines.

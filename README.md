# ECRTS 2026 Artifact: Probabilistic Schedulability Analysis

This artifact accompanies the ECRTS 2026 paper "Probabilistic Schedulability
Analysis for Mixed-Criticality DAG Tasks on Multiprocessors". It contains the
analysis code, DAG-generation wrapper, experiment drivers, and plotting scripts
used for the paper evaluation.

## What to Evaluate

The primary artifact target is the ECRTS 2026 "Results Reproduced" path.

| Paper Result | Artifact Step | Main Scripts |
| --- | --- | --- |
| Figure 3: required cores by CPR bin | Generate DAG pools, run RQ1, plot RQ1 | `05_pre1_generate_dags.py`, `05_1_rq1_eval.py`, `06_ex05_1_boxplot.py` |
| Figure 4: sensitivity to `F_S` | Run RQ2 and plot failure-budget comparisons | `05_2_rq2_eval.py`, `06_ex05_2_boxplot.py` |
| Figure 5: 1s vs 10s optimization | Reuse RQ1 tasksets, run RQ3, plot comparison | `05_3_rq3_eval.py`, `06_ex05_3_compare.py` |
| Development checks only | Small examples and legacy plots | `ex01`-`ex04`, `05_federated_scatter.py`, `06_fed2018_anneal_boxplot.py` |

The main reproduction path is therefore `experiments/ex05/` plus
`experiments/ex06/`.

## Repository Layout

```text
.
|-- README.md
|-- pyproject.toml
|-- uv.lock
|-- src/
|   |-- analyzer/      # Federated and multipath schedulability analyses
|   |-- clustering/    # Probabilistic clustering and simulated annealing
|   |-- common/        # DAG, node, and system models
|   `-- generator/     # RD-Gen wrapper and MC-DAG conversion
|-- experiments/
|   |-- README.md
|   |-- ex05/          # Main RQ1/RQ2/RQ3 drivers
|   `-- ex06/          # Plotting scripts
`-- external/RD-Gen/   # DAG generator submodule
```

Generated outputs are written under `data/`, which is ignored by Git.

## Requirements

Reference environment used for the paper:

- Linux
- Python 3.13
- AMD Ryzen 7 PRO 8840U, 8 cores, 2.9 GHz
- 14 GiB RAM

Evaluator requirements:

- Python 3.13
- `uv`
- Git submodules
- Docker/OCI support is planned but not yet the documented path in this revision.

## Setup

Run from the repository root.

```bash
git submodule update --init --recursive
uv sync --locked
```

Quick functional check:

```bash
./reproduce.sh check
```

## Docker

The Docker image includes Python 3.13, locked Python dependencies, this source
tree, and the RD-Gen submodule contents.

```bash
docker build -t ecrts2026-mc-dag-artifact .
docker run --rm ecrts2026-mc-dag-artifact ./reproduce.sh check
```

For a reduced end-to-end test:

```bash
WORKERS=4 ./reproduce.sh smoke
```

With Docker:

```bash
docker run --rm -e WORKERS=4 ecrts2026-mc-dag-artifact ./reproduce.sh smoke
```

## Reproduce Main Results

First generate the DAG pool:

```bash
uv run python experiments/ex05/05_pre1_generate_dags.py \
  --run-id 05_pre1 \
  --workers 8
```

The same full workflow can be launched as one command:

```bash
WORKERS=8 ./reproduce.sh full
```

### Figure 3 / RQ1

```bash
uv run python experiments/ex05/05_1_rq1_eval.py \
  --pre-run-dir data/results/ex05_pre/05_pre1 \
  --output-dir data/results/ex05_1/ex05_1_final_ECRTS \
  --run-id ex05_1_final_1sec \
  --method auto \
  --axis cp \
  --sets-per-bin 100 \
  --taskset-nodes-max 1000 \
  --fi-mode per-hour \
  --cluster-time-limit 1 \
  --cluster-quiet

uv run python experiments/ex06/06_ex05_1_boxplot.py \
  --input data/results/ex05_1/ex05_1_final_ECRTS/ex05_1_final_1sec__fan-in

uv run python experiments/ex06/06_ex05_1_boxplot.py \
  --input data/results/ex05_1/ex05_1_final_ECRTS/ex05_1_final_1sec__chain
```

### Figure 4 / RQ2

```bash
uv run python experiments/ex05/05_2_rq2_eval.py \
  --pre-run-dir data/results/ex05_pre/05_pre1 \
  --run-id ex05_2_1000 \
  --method mixed \
  --sets-per-bin 1000 \
  --taskset-nodes-max 1000 \
  --fi-mode per-hour \
  --fi-profiles fixed,loguniform \
  --federated-max-cores 1000 \
  --cluster-time-limit 1 \
  --cluster-quiet

uv run python experiments/ex06/06_ex05_2_boxplot.py \
  --input data/results/ex05_2/ex05_2_1000
```

### Figure 5 / RQ3

```bash
uv run python experiments/ex05/05_3_rq3_eval.py \
  --rq1-run-dir data/results/ex05_1/ex05_1_final_ECRTS/ex05_1_final_1sec__chain \
  --run-id ex05_3__chain_10sec \
  --max-sets 1000 \
  --cluster-time-limit 10

uv run python experiments/ex05/05_3_rq3_eval.py \
  --rq1-run-dir data/results/ex05_1/ex05_1_final_ECRTS/ex05_1_final_1sec__fan-in \
  --run-id ex05_3__fan-in_10sec \
  --max-sets 1000 \
  --cluster-time-limit 10

uv run python experiments/ex06/06_ex05_3_compare.py \
  --input-rq1 data/results/ex05_1/ex05_1_final_ECRTS/ex05_1_final_1sec__chain \
  --input-rq3 data/results/ex05_3/ex05_3__chain_10sec \
  --input-rq1-right data/results/ex05_1/ex05_1_final_ECRTS/ex05_1_final_1sec__fan-in \
  --input-rq3-right data/results/ex05_3/ex05_3__fan-in_10sec \
  --heatmap
```

## Output Files

Each run records:

- `*_meta.json`: command parameters and run metadata.
- `*_set_seeds.txt`: deterministic seeds.
- `tasksets/taskset_XXXX.json`: generated tasksets.
- `*_results.csv`: per-taskset results.
- `plots/`: generated figures in PNG/PDF form.

## Notes for Evaluators

- Full reproduction can take hours because it regenerates DAG pools and runs
  simulated annealing per taskset.
- For a fast pipeline check, reduce `--sets-per-bin`, `--taskset-nodes-max`, and
  `--cluster-time-limit`.
- Existing results are skipped by default. Use `--no-skip-existing` to force a
  rerun.
- Additional script details are in `experiments/README.md` and
  `experiments/ex05/README.md`.

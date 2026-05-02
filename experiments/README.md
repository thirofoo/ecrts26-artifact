# Experiments

Run all commands from the repository root so relative paths resolve correctly.
Use `uv run python ...` for all scripts to ensure the locked dependencies are
active.

The main ECRTS artifact path is `ex05` plus the plotting scripts in `ex06`.
Earlier experiments (`ex01` through `ex04`) are development and demonstration
scripts kept for traceability.

## Common Prerequisites

- RD-Gen must be available at `external/RD-Gen`.
- Python dependencies must be installed with `uv sync --locked`.
- Generated data is written under `data/`, which is ignored by Git.

## Main Scripts

| Script | Purpose | Primary Outputs |
| --- | --- | --- |
| `experiments/ex05/05_pre1_generate_dags.py` | Generate a fan-in/chain DAG pool and CPR rosters. | `data/results/ex05_pre/<run_id>/` |
| `experiments/ex05/05_pre3_roster_stats.py` | Summarize the generated DAG pool. | `<pre_run_dir>/plots/` |
| `experiments/ex05/05_1_rq1_eval.py` | RQ1: compare required cores across CPR, ratio, and tightness axes. | `data/results/ex05_1/<run_id>/` |
| `experiments/ex05/05_2_rq2_eval.py` | RQ2: sweep system-level failure budgets. | `data/results/ex05_2/<run_id>/` |
| `experiments/ex05/05_3_rq3_eval.py` | RQ3: re-evaluate existing RQ1 tasksets with a longer clustering time budget. | `data/results/ex05_3/<run_id>/` |
| `experiments/ex06/06_ex05_1_boxplot.py` | Plot RQ1 results. | `<rq1_run_dir>/plots/` |
| `experiments/ex06/06_ex05_2_boxplot.py` | Plot RQ2 results. | `<rq2_run_dir>/plots/` |
| `experiments/ex06/06_ex05_3_compare.py` | Plot RQ3 comparisons. | `<rq3_run_dir>/plots/` |

Use each script's `--help` output for the complete option list.

## DAG Pool Generation

```bash
uv run python experiments/ex05/05_pre1_generate_dags.py \
  --run-id 05_pre1 \
  --workers 8
```

Important defaults:

- `--rdgen-method both`: generate both fan-in and chain DAGs.
- `--class-cap 100`: keep up to 100 DAGs per CPR bin and per method.
- CPR bins `[0.1, 0.2)` through `[0.6, 0.7)` are retained.
- Node count is filtered to `[20, 100]`.
- `C^HI` is sampled from `[50, 500]`.

Outputs:

- `dag_tasks/`: generated XML DAGs.
- `rosters/dag_catalog.csv`: full DAG catalog.
- `rosters/dag_roster_cpr_<bin>.csv`: combined rosters.
- `rosters/dag_roster_cpr_<bin>__fan-in.csv` and `__chain.csv`: method-specific rosters.
- `ex05_pre_meta.json`: generation metadata.

## RQ1

```bash
uv run python experiments/ex05/05_1_rq1_eval.py \
  --pre-run-dir data/results/ex05_pre/05_pre1 \
  --output-dir data/results/ex05_1 \
  --run-id rq1 \
  --method auto \
  --axis cp \
  --sets-per-bin 100 \
  --taskset-nodes-max 1000 \
  --fi-mode per-hour \
  --cluster-time-limit 1 \
  --cluster-quiet
```

`--method auto` runs fan-in and chain as separate run directories:

- `data/results/ex05_1/rq1__fan-in/`
- `data/results/ex05_1/rq1__chain/`

Plot each directory:

```bash
uv run python experiments/ex06/06_ex05_1_boxplot.py \
  --input data/results/ex05_1/rq1__fan-in

uv run python experiments/ex06/06_ex05_1_boxplot.py \
  --input data/results/ex05_1/rq1__chain
```

## RQ2

```bash
uv run python experiments/ex05/05_2_rq2_eval.py \
  --pre-run-dir data/results/ex05_pre/05_pre1 \
  --run-id rq2 \
  --method mixed \
  --sets-per-bin 1000 \
  --taskset-nodes-max 1000 \
  --fi-mode per-hour \
  --fi-profiles fixed,loguniform \
  --federated-max-cores 1000 \
  --cluster-time-limit 1 \
  --cluster-quiet
```

Plot the result:

```bash
uv run python experiments/ex06/06_ex05_2_boxplot.py \
  --input data/results/ex05_2/rq2
```

Key defaults:

- `--fs-values 1e-7,1e-8,1e-9`
- `--fi 1e-6`
- `--fi-mode per-hour`
- `--fi-profiles fixed,loguniform`

## RQ3

```bash
uv run python experiments/ex05/05_3_rq3_eval.py \
  --rq1-run-dir data/results/ex05_1/rq1__chain \
  --run-id rq3__chain \
  --max-sets 1000 \
  --cluster-time-limit 10

uv run python experiments/ex05/05_3_rq3_eval.py \
  --rq1-run-dir data/results/ex05_1/rq1__fan-in \
  --run-id rq3__fan-in \
  --max-sets 1000 \
  --cluster-time-limit 10
```

Plot the side-by-side comparison:

```bash
uv run python experiments/ex06/06_ex05_3_compare.py \
  --input-rq1 data/results/ex05_1/rq1__chain \
  --input-rq3 data/results/ex05_3/rq3__chain \
  --input-rq1-right data/results/ex05_1/rq1__fan-in \
  --input-rq3-right data/results/ex05_3/rq3__fan-in \
  --heatmap
```

## Development Scripts

- `experiments/ex01/01_generate_data.py`: small MC-DAG generation demo.
- `experiments/ex02/02_clustering.py`: clustering demo on one DAG.
- `experiments/ex03/03_mcfq.py`: random DAG generation and Graham/multipath comparison.
- `experiments/ex04/04_federated_vs_clustering.py`: early federated-vs-clustering comparison.
- `experiments/ex05/05_federated_scatter.py`: earlier large-scale scatter evaluation retained for compatibility.
- `experiments/ex06/06_fed2018_anneal_boxplot.py`: plotting script for the older `05_federated_scatter.py` output.

These scripts are not the primary artifact reproduction path for the ECRTS 2026
paper figures.

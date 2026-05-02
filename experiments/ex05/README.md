# ex05 Main Experiment Defaults

This directory contains the main evaluation and preprocessing scripts for the
ECRTS 2026 artifact. The root `README.md` gives the recommended reproduction
workflow; this file records the important defaults used by the ex05 scripts.

## `05_pre1_generate_dags.py`

Generates the DAG pool used by RQ1, RQ2, and RQ3.

Default behavior:

- CPR rosters cover `[0.1, 0.2)` through `[0.6, 0.7)`.
- `--class-cap 100`: up to 100 DAGs per CPR bin and per method.
- `--rdgen-method both`: generate both fan-in and chain DAGs.
- `--chain-extra-edges-min 0 --chain-extra-edges-max 0`: no extra random chain edges.
- `--nodes-min 20 --nodes-max 100`: node-count filter.
- `--chi-min 50 --chi-max 500`: per-node `C^HI` range.

Fan-in defaults:

- `--edge-density-min 1.2 --edge-density-max 4.0`
- `--degree-mode random`
- `--degree-min 1 --degree-max 2 --degree-step 1`
- `--degree-weights 1,1` only matters with `--degree-mode weighted`.

Chain defaults:

- `--chain-count-min 1 --chain-count-max 1 --chain-count-step 1`
- `--chain-main-min 1 --chain-main-max 30 --chain-main-step 1`
- `--chain-main-sampling log`
- `--chain-sub-min 0 --chain-sub-max 10 --chain-sub-step 1`
- `--chain-source-nodes 1 --chain-sink-nodes 1`
- Internal fixed settings: `chain_merge_middle=false`, `chain_merge_sink=true`.
- `--chain-extra-edges-min 0 --chain-extra-edges-max 0`

Outputs:

- Combined rosters: `dag_roster_cpr_<bin>.csv`
- Method-specific rosters:
  `dag_roster_cpr_<bin>__fan-in.csv`,
  `dag_roster_cpr_<bin>__chain.csv`
- Full catalog: `dag_catalog.csv`
- Generated DAG XML files under `dag_tasks/`

Example:

```bash
uv run python experiments/ex05/05_pre1_generate_dags.py --workers 8
```

## `05_pre3_roster_stats.py`

Summarizes the generated DAG pool.

Default behavior:

- Produces combined summaries by default.
- Use `--split-by-method` to emit separate fan-in and chain plots.

Example:

```bash
uv run python experiments/ex05/05_pre3_roster_stats.py \
  --run-dir data/results/ex05_pre/05_pre1
```

Outputs under `<run-dir>/plots`:

- `roster_summary_by_cpr.csv`
- `roster_bin_counts.png` and `.pdf`
- `roster_boxplot_<metric>.png` and `.pdf`
- `roster_scatter_cpr_panels.png` and `.pdf`
- `roster_scatter_cpr_vs_nodes_by_<category>.png` and `.pdf`
- `roster_boxplot_utilization_lo_hi.png` and `.pdf`

## `05_1_rq1_eval.py`

Evaluates RQ1 on the ex05_pre DAG pool.

Important defaults:

- `--method auto`: run fan-in and chain separately.
- `--axis cp`: sweep CPR bins unless another axis is selected.
- `--sets-per-bin 50`: script default; the paper reproduction command uses 100.
- `--tasks 20`: task count target unless `--taskset-nodes-max` is set.
- `--cluster-score both`: evaluate both `cores+critical` and `critical` objectives
  on the same taskset and store both result columns.
- `--cluster-time-limit 10`: script default; the Figure 3 reproduction command
  passes `--cluster-time-limit 1`.
- `--fi-mode per-hour`

Output directories with `--method auto`:

- `data/results/ex05_1/<run_id>__fan-in/`
- `data/results/ex05_1/<run_id>__chain/`

Primary output files:

- `tasksets/taskset_XXXX.json`
- `ex05_1_plan.csv`
- `ex05_1_results.csv`
- `ex05_1_meta.json`
- `ex05_1_set_seeds.txt`

Example:

```bash
uv run python experiments/ex05/05_1_rq1_eval.py \
  --pre-run-dir data/results/ex05_pre/05_pre1 \
  --axis cp \
  --sets-per-bin 100 \
  --taskset-nodes-max 1000 \
  --fi-mode per-hour \
  --cluster-time-limit 1
```

## `05_2_rq2_eval.py`

Evaluates sensitivity to the system-level failure budget.

Important defaults:

- `--sets-per-bin 100`
- `--taskset-nodes-max 1000`
- `--fs-values 1e-7,1e-8,1e-9`
- `--fi 1e-6`
- `--fi-profiles fixed,loguniform`
- `--fi-mode per-hour`
- `--federated-max-cores 1000` in the final RQ2 run.
- `--cluster-time-limit 1`
- `--cluster-score both`

Use `--method mixed` for the mixed fan-in/chain RQ2 workflow.

Example:

```bash
uv run python experiments/ex05/05_2_rq2_eval.py \
  --pre-run-dir data/results/ex05_pre/05_pre1 \
  --method mixed \
  --sets-per-bin 1000 \
  --taskset-nodes-max 1000 \
  --fi-mode per-hour \
  --fi-profiles fixed,loguniform \
  --federated-max-cores 1000
```

## `05_3_rq3_eval.py`

Re-evaluates existing RQ1 tasksets with a longer clustering time budget.

Important defaults:

- Requires `--rq1-run-dir`.
- Reads tasksets and metadata from the RQ1 run directory.
- `--cluster-score from-rq1`: reuse the RQ1 score setting.
- `--cluster-time-limit 10`
- `--max-sets 100`: script default; the final Figure 5 workflow passes 1000
  and evaluates the selected successful RQ1 tasksets.

Example:

```bash
uv run python experiments/ex05/05_3_rq3_eval.py \
  --rq1-run-dir data/results/ex05_1/rq1__chain \
  --run-id rq3__chain \
  --max-sets 1000 \
  --cluster-time-limit 10
```

## Plotting Scripts

RQ1:

```bash
uv run python experiments/ex06/06_ex05_1_boxplot.py \
  --input data/results/ex05_1/ex05_1__chain
```

RQ2:

```bash
uv run python experiments/ex06/06_ex05_2_boxplot.py \
  --input data/results/ex05_2/ex05_2
```

RQ3:

```bash
uv run python experiments/ex06/06_ex05_3_compare.py \
  --input-rq1 data/results/ex05_1/ex05_1__chain \
  --input-rq3 data/results/ex05_3/ex05_3__chain \
  --heatmap
```

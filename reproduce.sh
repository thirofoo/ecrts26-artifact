#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-check}"
WORKERS="${WORKERS:-1}"

usage() {
  cat <<'USAGE'
Usage: ./reproduce.sh [check|smoke|full]

Modes:
  check   Compile sources and show help for the main scripts.
  smoke   Run a small end-to-end experiment with reduced parameters.
  full    Run the paper-scale Figure 3, Figure 4, and Figure 5 workflow.

Environment:
  WORKERS  Parallel DAG-generation workers (default: 1).
USAGE
}

check() {
  uv run python -m compileall src experiments
  uv run python experiments/ex05/05_pre1_generate_dags.py --help >/dev/null
  uv run python experiments/ex05/05_1_rq1_eval.py --help >/dev/null
  uv run python experiments/ex05/05_2_rq2_eval.py --help >/dev/null
  uv run python experiments/ex05/05_3_rq3_eval.py --help >/dev/null
}

smoke() {
  uv run python experiments/ex05/05_pre1_generate_dags.py \
    --run-id smoke_pre \
    --class-cap 1 \
    --workers "${WORKERS}"

  uv run python experiments/ex05/05_1_rq1_eval.py \
    --pre-run-dir data/results/ex05_pre/smoke_pre \
    --run-id smoke_rq1 \
    --method mixed \
    --axis cp \
    --sets-per-bin 1 \
    --taskset-nodes-max 200 \
    --fi-mode per-hour \
    --cluster-time-limit 0.1 \
    --cluster-quiet

  uv run python experiments/ex06/06_ex05_1_boxplot.py \
    --input data/results/ex05_1/smoke_rq1

  uv run python experiments/ex05/05_2_rq2_eval.py \
    --pre-run-dir data/results/ex05_pre/smoke_pre \
    --run-id smoke_rq2 \
    --method mixed \
    --sets-per-bin 2 \
    --taskset-nodes-max 200 \
    --fi-mode per-hour \
    --fi-profiles fixed,loguniform \
    --federated-max-cores 1000 \
    --cluster-time-limit 0.1 \
    --cluster-quiet

  uv run python experiments/ex06/06_ex05_2_boxplot.py \
    --input data/results/ex05_2/smoke_rq2

  uv run python experiments/ex05/05_3_rq3_eval.py \
    --rq1-run-dir data/results/ex05_1/smoke_rq1 \
    --run-id smoke_rq3 \
    --max-sets 2 \
    --cluster-time-limit 0.2

  uv run python experiments/ex06/06_ex05_3_compare.py \
    --input-rq1 data/results/ex05_1/smoke_rq1 \
    --input-rq3 data/results/ex05_3/smoke_rq3
}

full() {
  local rq1_output_dir="data/results/ex05_1/ex05_1_final_ECRTS"
  local rq1_run_id="ex05_1_final_1sec"
  local rq1_chain_dir="${rq1_output_dir}/${rq1_run_id}__chain"
  local rq1_fanin_dir="${rq1_output_dir}/${rq1_run_id}__fan-in"
  local rq2_run_dir="data/results/ex05_2/ex05_2_1000"
  local rq3_chain_dir="data/results/ex05_3/ex05_3__chain_10sec"
  local rq3_fanin_dir="data/results/ex05_3/ex05_3__fan-in_10sec"

  uv run python experiments/ex05/05_pre1_generate_dags.py \
    --run-id 05_pre1 \
    --workers "${WORKERS}"

  uv run python experiments/ex05/05_1_rq1_eval.py \
    --pre-run-dir data/results/ex05_pre/05_pre1 \
    --output-dir "${rq1_output_dir}" \
    --run-id "${rq1_run_id}" \
    --method auto \
    --axis cp \
    --sets-per-bin 100 \
    --taskset-nodes-max 1000 \
    --fi-mode per-hour \
    --cluster-time-limit 1 \
    --cluster-quiet

  uv run python experiments/ex06/06_ex05_1_boxplot.py \
    --input "${rq1_fanin_dir}"

  uv run python experiments/ex06/06_ex05_1_boxplot.py \
    --input "${rq1_chain_dir}"

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
    --input "${rq2_run_dir}"

  uv run python experiments/ex05/05_3_rq3_eval.py \
    --rq1-run-dir "${rq1_chain_dir}" \
    --run-id ex05_3__chain_10sec \
    --max-sets 1000 \
    --cluster-time-limit 10

  uv run python experiments/ex05/05_3_rq3_eval.py \
    --rq1-run-dir "${rq1_fanin_dir}" \
    --run-id ex05_3__fan-in_10sec \
    --max-sets 1000 \
    --cluster-time-limit 10

  uv run python experiments/ex06/06_ex05_3_compare.py \
    --input-rq1 "${rq1_chain_dir}" \
    --input-rq3 "${rq3_chain_dir}" \
    --input-rq1-right "${rq1_fanin_dir}" \
    --input-rq3-right "${rq3_fanin_dir}" \
    --heatmap
}

case "${MODE}" in
  check)
    check
    ;;
  smoke)
    smoke
    ;;
  full)
    full
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

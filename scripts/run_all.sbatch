#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH --account=ag_gipp
#SBATCH --partition=scc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -G A100:1
#SBATCH --constraint=inet
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Self-contained unattended run: build the env ON this compute node, start vLLM,
# run the baseline against it, then shut vLLM down and exit (freeing the GPU).
# A trap kills vLLM even on error, timeout, or scancel. Nothing runs on the
# login node, so you can leave it unattended.
#
# Why build the venv here (not on the login node): the scc-gpu nodes are AMD
# (zen3) with no AVX-512, while the login node is Intel with AVX-512. A venv /
# vLLM built on the login node crashes here with "illegal instruction". Building
# on the compute node matches the arch, so plain pip vLLM works -- no container.
#
#   sbatch scripts/run_all.sbatch                       # 50 puzzles (quick check)
#   sbatch --export=ALL,LIMIT=0 scripts/run_all.sbatch  # whole test split
#   sbatch --time=24:00:00 --export=ALL,LIMIT=0 scripts/run_all.sbatch
#   REBUILD=1 sbatch --export=ALL scripts/run_all.sbatch   # force venv rebuild

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results
echo "[$(date +%T)] job on $HOSTNAME"

# ---- knobs ----
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
SERVED_NAME="${SERVED_NAME:-qwen}"
PORT="${PORT:-8000}"
SPLIT="${SPLIT:-test}"
CONCURRENCY="${CONCURRENCY:-32}"
LIMIT="${LIMIT:-50}"                 # 0 = whole split
GPU_UTIL="${GPU_UTIL:-0.90}"
SERVE_TIMEOUT="${SERVE_TIMEOUT:-1800}"   # secs for build + model download + load

PROJECT="${PROJECT:-/projects/scc/UGOE/UXEI/UMIN/ag_gipp/dir.project}"
# User-owned cache/venv under the project root (ceph, effectively unlimited).
# Namespaced by $USER because the shared $PROJECT/hf_cache belongs to a labmate
# and isn't group-writable.
export HF_HOME="${HF_HOME:-$PROJECT/$USER/hf_cache}"
VENV="${VENV:-$PROJECT/.venv_sparc}"             # persisted, reused across jobs
mkdir -p "$HF_HOME"

# conda-free
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_PYTHON_EXE 2>/dev/null || true

module purge
module load gcc/13.2.0
module load python/3.11.9

# ---- build the venv on this node (first run only, then reused) ----
if [[ "${REBUILD:-0}" == "1" || ! -x "$VENV/bin/vllm" ]]; then
    echo "[$(date +%T)] building venv at $VENV"
    rm -rf "$VENV"
    python3 -m venv --without-pip "$VENV"
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    python -m pip --version >/dev/null 2>&1 || {
        curl -sSf https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
        python /tmp/get-pip.py
    }
    pip install --upgrade pip
    pip install vllm                                   # brings a matching torch
    pip install -e .                                   # memory_agents + driver deps
    pip install -r requirements.txt                    # Spatial-Gym (the gym env)
else
    echo "[$(date +%T)] reusing venv at $VENV (REBUILD=1 to force)"
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi

# ---- start the server in the background ----
echo "[$(date +%T)] starting vLLM: $MODEL"
vllm serve "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --host 0.0.0.0 --port "$PORT" \
    --tensor-parallel-size "${SLURM_GPUS_ON_NODE:-1}" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --enable-prefix-caching \
    --max-num-seqs 64 \
    --trust-remote-code \
    --seed 0 &
VLLM_PID=$!

cleanup() {
    echo "[$(date +%T)] cleanup: stopping vLLM"
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ---- wait until healthy (or give up) ----
echo "[$(date +%T)] waiting for vLLM (timeout ${SERVE_TIMEOUT}s)..."
waited=0
until curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[run_all] vLLM died before becoming healthy; see the log above." >&2
        exit 1
    fi
    sleep 5; waited=$((waited + 5))
    if (( waited >= SERVE_TIMEOUT )); then
        echo "[run_all] timed out waiting for vLLM." >&2
        exit 1
    fi
done
echo "[$(date +%T)] vLLM healthy."

# ---- point the driver at the local server and run ----
export LLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
export LLM_MODEL="$SERVED_NAME"

echo "[$(date +%T)] smoke check..."
python scripts/smoke_test.py --n 2

echo "[$(date +%T)] baseline: split=$SPLIT concurrency=$CONCURRENCY limit=$LIMIT"
python scripts/run_baseline.py \
    --split "$SPLIT" --concurrency "$CONCURRENCY" --limit "$LIMIT"

echo "[$(date +%F\ %T)] done; cleanup trap will stop vLLM and free the GPU."

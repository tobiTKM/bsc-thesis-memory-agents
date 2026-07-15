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
module load uv    # uv brings a portable standalone python; the arch-specific
                  # module python (a zen2 build) crashes with "illegal
                  # instruction" when creating the venv on some nodes.

# ---- build the venv on this node (first run only, then reused) ----
if [[ "${REBUILD:-0}" == "1" || ! -x "$VENV/bin/vllm" ]]; then
    echo "[$(date +%T)] building venv at $VENV"
    rm -rf "$VENV"
    uv venv --managed-python --python 3.11 "$VENV"
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    # Pin CUDA build to cu121: the scc-gpu driver is 12.8, so a newer torch
    # (cu129+, pulled by the latest vLLM) fails with "driver too old". cu121 is
    # backward-compatible with a 12.8 driver. torch-then-vLLM matches the proven
    # recipe; vLLM 0.7.3 needs torch==2.5.1 and supports Qwen2.5.
    uv pip install "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1" \
        --index-url https://download.pytorch.org/whl/cu121
    uv pip install "vllm==0.7.3"
    uv pip install -e .                                # memory_agents + driver deps
    uv pip install -r requirements.txt                # Spatial-Gym (the gym env)
    # vLLM 0.7.3 only sets transformers>=4.48.2, so uv otherwise grabs the latest
    # (5.x), whose Qwen2Tokenizer dropped all_special_tokens_extended. Pin a 4.49
    # last so nothing re-bumps it.
    uv pip install "transformers==4.49.0"
else
    echo "[$(date +%T)] reusing venv at $VENV (REBUILD=1 to force)"
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi

# -C inet sets http_proxy on the node for internet access, but that proxy also
# hijacks localhost requests -- so the health curl and the Python client must
# bypass it for 127.0.0.1, or they can't reach our own vLLM server.
export no_proxy="localhost,127.0.0.1,0.0.0.0,::1"
export NO_PROXY="$no_proxy"

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
until curl -sf --noproxy '*' "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; do
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

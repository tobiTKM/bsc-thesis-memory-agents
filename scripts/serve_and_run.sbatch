#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH --account=ag_gipp
#SBATCH --partition=scc-gpu
#SBATCH --nodes=1                    # keep all GPUs on ONE node (needed for tensor-parallel)
#SBATCH --ntasks=1
#SBATCH -G A100:2                    # = TP below; override on the command line with -G A100:<n>
#SBATCH --constraint=inet&80gb       # inet: no internet on the node otherwise; 80gb: ggpu135/136 are 40 GB cards, too small for a 32B
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Unattended run: host Qwen3-32B with vLLM from the shared ag_gipp Apptainer image,
# wait until it is healthy, run the SPaRC baseline against it, tear the server down.
# The trap kills vLLM on error, timeout or scancel, so the GPUs are always freed.
#
#   sbatch scripts/serve_and_run.sbatch                          # 50 puzzles (quick check)
#   sbatch --export=ALL,LIMIT=0 scripts/serve_and_run.sbatch     # whole test split
#   sbatch -G A100:4 --export=ALL,TP=4,LIMIT=0 scripts/serve_and_run.sbatch
#   sbatch --export=ALL,MODEL=Qwen/Qwen2.5-3B-Instruct,TP=1 -G A100:1 scripts/serve_and_run.sbatch
#
# Reflexion (three jobs, linked by the pass1 file it prints). The control arm is not
# optional: only failures get retried, so pass2's "lift" is >= 0 by construction, and
# a second sample at TEMPERATURE=0.6 converts some failures with no memory at all.
# Reflexion's real effect is pass2's retried_accuracy minus control's.
#   sbatch -J reflexion --export=ALL,DRIVER=reflexion,PHASE=pass1,LIMIT=0 scripts/serve_and_run.sbatch
#   sbatch -J reflexion --export=ALL,DRIVER=reflexion,PHASE=pass2,PASS1=results/json/reflexion_pass1_test_<stamp>.json scripts/serve_and_run.sbatch
#   sbatch -J reflexion --export=ALL,DRIVER=reflexion,PHASE=control,PASS1=results/json/reflexion_pass1_test_<stamp>.json scripts/serve_and_run.sbatch
#
# Some nodes SIGILL the Apptainer launcher (see PREFLIGHT below). The job detects that
# in ~30 s, appends the node to logs/bad_nodes.txt and resubmits itself excluding every
# node on that list, up to MAX_ATTEMPTS (8) times. AUTO_RESUBMIT=0 turns that off.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results
echo "[$(date +%T)] job $SLURM_JOB_ID on $(hostname -s)"

# ---- knobs ------------------------------------------------------------------
MODEL="${MODEL:-Qwen/Qwen3-32B}"
SERVED_NAME="${SERVED_NAME:-qwen}"        # what the client asks for; keeps result files stable
TP="${TP:-${SLURM_GPUS_ON_NODE:-2}}"      # tensor-parallel size = GPUs requested
GPU_UTIL="${GPU_UTIL:-0.90}"
MAXLEN="${MAXLEN:-32768}"                 # Qwen3 native ctx; must exceed prompt + MAX_TOKENS
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
REASONING_PARSER="${REASONING_PARSER-qwen3}"    # strips <think>...</think> out of .content.
                                                # NOTE: `-` not `:-` on purpose, so REASONING_PARSER=
                                                # (explicit empty) truly DISABLES it; only *unset* defaults
                                                # to qwen3. With `:-`, an empty value silently re-defaulted.
SERVE_TIMEOUT="${SERVE_TIMEOUT:-2700}"    # 45 min: 62 GB of weights off ceph is slow

SPLIT="${SPLIT:-test}"
LIMIT="${LIMIT:-50}"                      # 0 = whole split
CONCURRENCY="${CONCURRENCY:-32}"
# Qwen3 in thinking mode: greedy decoding loops, and the answer only comes after a
# long think block -> the official sampling params + a big token budget.
# 4096 was too tight in the 2026-07-26 run: successful steps averaged ~2.2k completion
# tokens, and 66% of episodes died on unparseable replies (a reply that hits the cap
# mid-<think> arrives with empty .content once the reasoning parser strips it).
# Prompts run ~2k tokens, so 12k still leaves plenty of room under MAXLEN=32768.
MAX_TOKENS="${MAX_TOKENS:-12288}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
SMOKE="${SMOKE:-1}"                       # 0 skips the 2-prompt smoke check

# Which driver runs against the served model. `baseline` is the memoryless run;
# `reflexion` is the two-phase memory method (see scripts/run_reflexion.py). New
# memory methods slot in here as extra DRIVER cases -- the serve block above is
# method-agnostic, so nothing else changes.
DRIVER="${DRIVER:-baseline}"              # baseline | reflexion
PHASE="${PHASE:-pass1}"                   # reflexion: pass1 | pass2 | control
PASS1="${PASS1:-}"                         # reflexion pass2/control: results/json/reflexion_pass1_*.json
REFLECT_MAX_TOKENS="${REFLECT_MAX_TOKENS:-$MAX_TOKENS}"   # self-reflection budget (pass1)
SEED="${SEED:-0}"                         # trial-1 / baseline decoding seed
# Seed for the pass2 and control retries. MUST differ from SEED: same prompt + same
# seed makes vLLM reproduce trial-1's reply, so the retry would replay rather than
# resample and the control arm would measure nothing.
RETRY_SEED="${RETRY_SEED:-1}"

# ---- modules: re-resolve Lmod for THIS node's CPU ---------------------------
# This is what killed six earlier jobs. `sbatch --export=ALL` (the default) carries
# the LOGIN node's Lmod state into the job -- MODULEPATH and LMOD_MODULEPATH_ROOT
# still point at the Intel `cascadelake_opx_rocky8` tree. `module load apptainer`
# then resolves the `linux-rocky8-skylake_avx512` build, which contains 276 AVX-512
# instructions; the first `vmovdqu64` it executes on the Zen3 EPYC 7513 compute
# nodes (no AVX-512) is an instant SIGILL, ~30 s in, before vLLM starts. Dropping
# the inherited Lmod state and re-sourcing the site profile makes Lmod resolve
# against this node's architecture instead (`linux-rocky8-zen`, 0 AVX-512 insns).
module purge >/dev/null 2>&1 || true
unset MODULEPATH LMOD_MODULEPATH_ROOT LMOD_RC LOADEDMODULES _LMFILES_ \
      __LMOD_REF_COUNT_PATH __LMOD_REF_COUNT_MODULEPATH 2>/dev/null || true
set +eu   # the site profile is not written for `set -u`
source /etc/profile.d/zzz_unified_shell_profile.sh >/dev/null 2>&1 || true
set -eu
# conda also rides in on --export=ALL and shadows python; drop it too.
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_PYTHON_EXE LD_PRELOAD 2>/dev/null || true
PATH="$(tr ':' '\n' <<<"$PATH" | grep -v -e conda -e '\.vscode-server' | paste -sd:)"
export PATH
module load apptainer/1.4.3 uv/0.9.28

# Belt and braces: if Lmod still handed us an AVX-512 build on a CPU that can't run
# one, swap in a build compiled for this node before we get a SIGILL.
APPTAINER_BIN="$(command -v apptainer || true)"
echo "[$(date +%T)] apptainer: ${APPTAINER_BIN:-NOT FOUND}"
if [[ "$APPTAINER_BIN" == *avx512* ]] && ! grep -qm1 avx512 /proc/cpuinfo; then
    alt="$(ls -d /sw/rev/*/*/linux-rocky8-zen/gcc-*/apptainer-1.4.3-*/bin 2>/dev/null | head -1)"
    if [[ -n "$alt" ]]; then
        echo "[$(date +%T)] AVX-512 apptainer on a non-AVX-512 CPU; using $alt instead"
        PATH="$alt:$PATH"; export PATH
    fi
fi

# ---- paths (AFTER the profile, which exports its own $PROJECT) --------------
# The site profile sets PROJECT=/projects/scc/.../dir.project (the symlink). It
# silently clobbered our resolved value once already, which left the model path
# un-rewritten and made vLLM treat it as a HuggingFace repo id. Hence AGPROJ, set
# after the profile has had its say, and never a name the site owns.
AGPROJ="$(readlink -f "${AGPROJ:-/projects/scc/UGOE/UXEI/UMIN/ag_gipp/dir.project}")"
PROJ_IN="/proj"                           # where $AGPROJ is mounted INSIDE the container
SIF="${SIF:-$AGPROJ/vllm-container/vllm-25.1.sif}"    # vLLM 0.25.1, cu129
HUB="${HUB:-$AGPROJ/hf/hub}"              # group-shared weights (Qwen3-32B lives here)
# Per-user, not per-group. `uv pip install -e .` writes this repo's *absolute*
# path into the venv, so a venv shared across ag_gipp would resolve
# `import memory_agents` to whoever submitted last -- silently running someone
# else's checkout. REBUILD=1 also rm -rf's this path, which would delete the
# interpreter out from under another member's running job.
VENV="${VENV:-$AGPROJ/$USER/.venv_client}"  # host-side driver env: openai+datasets+gym, NO torch
export HF_HOME="${HF_HOME:-$AGPROJ/$USER/hf_cache}"   # client side: SPaRC dataset
# Torch/Triton/Inductor compile caches go to the node's own /tmp, not to ceph: an
# earlier attempt died because torch's cache_dir() does makedirs() on import and the
# ceph path was not writable from inside the container ("Read-only file system:
# '/mnt/ceph-hdd'"). Node-local /tmp is always writable, is on SSD, and these caches
# are disposable -- only the weights genuinely need the shared filesystem.
CACHE_IN="/tmp/vllm-cache-$SLURM_JOB_ID"  # apptainer bind-mounts the host /tmp in
mkdir -p "$HF_HOME" "$CACHE_IN"

[[ -r "$SIF" ]] || { echo "no such image: $SIF" >&2; exit 1; }
echo "[$(date +%T)] AGPROJ=$AGPROJ -> $PROJ_IN (in container)"

# ---- PREFLIGHT: prove the container actually launches ------------------------
# The AVX-512 mismatch above is fixed, but this stays as the cheap general guard:
# any launcher failure (bad image, missing /dev/nvidia, arch mismatch we didn't
# foresee) shows up in two seconds here rather than after the model has loaded.
# NOTE: the auto-resubmit blacklists the *node*, which was the wrong model of the
# AVX-512 bug -- those nodes were fine. Keep an eye on logs/bad_nodes.txt: if a
# failure is environmental rather than node-local, excluding nodes will not help.
PREFLIGHT_LOG="logs/preflight-$SLURM_JOB_ID.log"
# Check what we actually depend on: the container starts, the bind landed, and the
# weights are visible through it -- not just that the launcher can exit 0.
if ! apptainer exec --nv --bind "$AGPROJ:$PROJ_IN" "$SIF" \
        test -d "$PROJ_IN/hf/hub" > "$PREFLIGHT_LOG" 2>&1; then
    node="$(hostname -s)"
    echo "[$(date +%T)] PREFLIGHT FAILED on $node -- first lines of $PREFLIGHT_LOG:" >&2
    head -5 "$PREFLIGHT_LOG" >&2
    grep -qxF "$node" logs/bad_nodes.txt 2>/dev/null || echo "$node" >> logs/bad_nodes.txt

    ATTEMPT="${ATTEMPT:-1}"
    if [[ "${AUTO_RESUBMIT:-1}" == "1" && "$ATTEMPT" -lt "${MAX_ATTEMPTS:-8}" ]]; then
        # Rebuild the original request from the job's own allocation, since the
        # -G/-t/-J overrides live on the command line, not in the header above.
        excl="$(paste -sd, logs/bad_nodes.txt)"
        tl="$(squeue -h -j "$SLURM_JOB_ID" -o %l | tr -d ' ')"
        echo "[$(date +%T)] attempt $ATTEMPT failed; resubmitting excluding: $excl" >&2
        sbatch --exclude="$excl" \
               -J "$SLURM_JOB_NAME" \
               -G "A100:${SLURM_GPUS_ON_NODE:-2}" \
               --cpus-per-task="${SLURM_CPUS_PER_TASK:-32}" \
               --mem="${SLURM_MEM_PER_NODE:-192000}M" \
               -t "${tl:-12:00:00}" \
               --export=ALL,ATTEMPT=$((ATTEMPT + 1)) \
               "${SCRIPT:-scripts/serve_and_run.sbatch}" >&2
    else
        echo "resubmit with: sbatch --exclude=\$(paste -sd, logs/bad_nodes.txt) \
scripts/serve_and_run.sbatch" >&2
    fi
    exit 1
fi
rm -f "$PREFLIGHT_LOG"

# ---- client venv (host side, built once and reused) -------------------------
if [[ "${REBUILD:-0}" == "1" || ! -x "$VENV/bin/python" ]]; then
    echo "[$(date +%T)] building client venv at $VENV"
    rm -rf "$VENV"
    mkdir -p "$(dirname "$VENV")"                    # $AGPROJ/$USER may not exist yet
    uv venv --managed-python --python 3.11 "$VENV"   # module python is a zen2 build
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    uv pip install -e . -r requirements.txt
else
    echo "[$(date +%T)] reusing client venv at $VENV (REBUILD=1 to force)"
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi

# -C inet sets an http proxy on the node, and that proxy also hijacks requests to
# localhost -- so the health curl and the driver must bypass it for 127.0.0.1.
export no_proxy="localhost,127.0.0.1,0.0.0.0,::1"
export NO_PROXY="$no_proxy"

# ---- resolve the model to a local snapshot when we have one -----------------
# Serving the path directly skips the hub machinery (and its lock writes) in the
# shared cache; an uncached repo id still downloads normally via the proxy.
# Paths handed to the server must be rewritten to where the bind lands ($PROJ_IN).
in_container() { echo "${1/#$AGPROJ/$PROJ_IN}"; }

MODEL_ARG="$MODEL"
HUB_IN="$(in_container "$HUB")"
OFFLINE=1
if [[ ! -d "$MODEL" ]]; then
    snap=$(ls -d "$HUB/models--${MODEL//\//--}"/snapshots/*/ 2>/dev/null | head -1 || true)
    if [[ -n "$snap" ]]; then
        MODEL_ARG="$(in_container "${snap%/}")"
        echo "[$(date +%T)] using cached weights: $MODEL_ARG (host: ${snap%/})"
    else
        echo "[$(date +%T)] '$MODEL' not in $HUB -- will download into $HF_HOME"
        HUB_IN="$(in_container "$HF_HOME")/hub"
        OFFLINE=0
    fi
else
    MODEL_ARG="$(in_container "$MODEL")"
fi

# ---- (a) start vLLM in the container, in the background ---------------------
PORT="${PORT:-$((10000 + SLURM_JOB_ID % 50000))}"
SERVER_LOG="logs/server-$SLURM_JOB_ID.log"
echo "[$(date +%T)] serving '$MODEL' from $(basename "$SIF") on :$PORT (TP=$TP) -> $SERVER_LOG"

apptainer exec --nv \
    --bind "$AGPROJ:$PROJ_IN" \
    --env HF_HUB_CACHE="$HUB_IN" \
    --env HF_HUB_OFFLINE="$OFFLINE" \
    --env HF_HOME="$CACHE_IN/hf" \
    --env VLLM_CACHE_ROOT="$CACHE_IN/vllm" \
    --env XDG_CACHE_HOME="$CACHE_IN" \
    --env TRITON_CACHE_DIR="$CACHE_IN/triton" \
    --env TORCHINDUCTOR_CACHE_DIR="$CACHE_IN/inductor" \
    --env OUTLINES_CACHE_DIR="$CACHE_IN/outlines" \
    "$SIF" \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_ARG" \
        --served-model-name "$SERVED_NAME" \
        --host 127.0.0.1 --port "$PORT" \
        --tensor-parallel-size "$TP" \
        --gpu-memory-utilization "$GPU_UTIL" \
        --max-model-len "$MAXLEN" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --enable-prefix-caching \
        --trust-remote-code \
        --seed 0 \
        ${REASONING_PARSER:+--reasoning-parser "$REASONING_PARSER"} \
    > "$SERVER_LOG" 2>&1 &
VLLM_PID=$!

cleanup() {
    echo "[$(date +%T)] cleanup: stopping vLLM ($VLLM_PID)"
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    # The node's /tmp survives the job, and the inductor/triton caches are GBs per
    # run keyed by job id, so nothing ever overwrites them. Left alone they fill a
    # node's disk and the *next* job there dies in torch.compile with ENOSPC.
    rm -rf "$CACHE_IN"
}
trap cleanup EXIT INT TERM

# ---- (b) wait until healthy -------------------------------------------------
echo "[$(date +%T)] waiting for vLLM (timeout ${SERVE_TIMEOUT}s)..."
waited=0
until curl -sf --noproxy '*' "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[$(date +%T)] vLLM died before becoming healthy -- tail of $SERVER_LOG:" >&2
        tail -40 "$SERVER_LOG" >&2
        exit 1
    fi
    sleep 10; waited=$((waited + 10))
    if (( waited >= SERVE_TIMEOUT )); then
        echo "[$(date +%T)] timed out waiting for vLLM; tail of $SERVER_LOG:" >&2
        tail -40 "$SERVER_LOG" >&2
        exit 1
    fi
done
echo "[$(date +%T)] vLLM healthy after ~${waited}s."

# ---- (c) run the driver against it ------------------------------------------
export LLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
export LLM_MODEL="$SERVED_NAME"
export LLM_MODEL_ID="$MODEL"   # real checkpoint behind the alias, recorded in results

if [[ "$SMOKE" == "1" ]]; then
    echo "[$(date +%T)] smoke check..."
    python -u scripts/smoke_test.py --n 2 \
        --max-tokens "$MAX_TOKENS" --temperature "$TEMPERATURE" \
        --top-p "$TOP_P" ${TOP_K:+--top-k "$TOP_K"}
fi

# Sampling/budget flags shared by every driver; --limit is added per-driver since
# reflexion pass2 runs over the pass1 episode set, not a fresh --limit slice.
COMMON_ARGS=(--split "$SPLIT" --concurrency "$CONCURRENCY" \
    --max-tokens "$MAX_TOKENS" --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" --seed "$SEED" ${TOP_K:+--top-k "$TOP_K"})

case "$DRIVER" in
  baseline)
    echo "[$(date +%T)] baseline: split=$SPLIT concurrency=$CONCURRENCY limit=$LIMIT"
    python -u scripts/run_baseline.py "${COMMON_ARGS[@]}" --limit "$LIMIT"
    ;;
  reflexion)
    if [[ "$PHASE" == "pass1" ]]; then
        echo "[$(date +%T)] reflexion pass1 (trial-1 + reflect): split=$SPLIT limit=$LIMIT"
        python -u scripts/run_reflexion.py --phase pass1 "${COMMON_ARGS[@]}" \
            --limit "$LIMIT" --reflect-max-tokens "$REFLECT_MAX_TOKENS"
    else
        [[ -n "$PASS1" ]] || { echo "PHASE=$PHASE needs PASS1=<reflexion_pass1_*.json>" >&2; exit 1; }
        echo "[$(date +%T)] reflexion $PHASE (retry at seed $RETRY_SEED): PASS1=$PASS1"
        python -u scripts/run_reflexion.py --phase "$PHASE" --pass1 "$PASS1" \
            --retry-seed "$RETRY_SEED" "${COMMON_ARGS[@]}"
    fi
    ;;
  *)
    echo "unknown DRIVER=$DRIVER (expected: baseline | reflexion)" >&2; exit 1 ;;
esac

echo "[$(date +%T)] done; the cleanup trap stops vLLM and frees the GPUs."

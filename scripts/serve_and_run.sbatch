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
# Serve a model with vLLM from the ag_gipp Apptainer image, run one driver against it,
# then stop the server. The trap frees the GPUs on error, timeout or scancel.
#
# Baseline:
#   sbatch scripts/serve_and_run.sbatch                          # 50 puzzles
#   sbatch --export=ALL,LIMIT=0 scripts/serve_and_run.sbatch     # whole test split
#   sbatch --export=ALL,MODEL=Qwen/Qwen2.5-3B-Instruct,TP=1 -G A100:1 scripts/serve_and_run.sbatch
#
# Reflexion: one root job (trial 1, shared by both arms), then one job per arm. The
# control arm is required: re-attempting only unsolved puzzles raises accuracy on its own.
#   sbatch -J refl-root --export=ALL,DRIVER=reflexion,PHASE=root,LIMIT=0 scripts/serve_and_run.sbatch
#   sbatch -J refl-mem  --export=ALL,DRIVER=reflexion,PHASE=arm,ARM=memory,TRIALS=10,ROOT=results/json/reflexion_root_test_<stamp>.json scripts/serve_and_run.sbatch
#   sbatch -J refl-ctl  --export=ALL,DRIVER=reflexion,PHASE=arm,ARM=control,TRIALS=10,ROOT=results/json/reflexion_root_test_<stamp>.json scripts/serve_and_run.sbatch
# Ten trials on the full test split exceed 12 h; arms checkpoint per trial, so resubmit with RESUME:
#   sbatch -J refl-mem --export=ALL,DRIVER=reflexion,PHASE=arm,TRIALS=10,RESUME=results/json/reflexion_memory_test_<stamp>.json scripts/serve_and_run.sbatch
#   python scripts/run.py reflexion --phase compare --memory <memory.json> --control <control.json>   # login node
#
# Memp: build on train, distill a bank, embed, eval. The embed job serves the embedding
# model, so it is a separate job; it can run in parallel with build/distill.
#   sbatch -J memp-build --export=ALL,DRIVER=memp,PHASE=build,SPLIT=train,LIMIT=0 scripts/serve_and_run.sbatch
#   sbatch -J memp-dist  --export=ALL,DRIVER=memp,PHASE=distill,BUILD=results/json/memp_build_train_<stamp>.json scripts/serve_and_run.sbatch
#   sbatch -J memp-embed -G A100:1 -t 01:00:00 --export=ALL,DRIVER=memp,PHASE=embed,TP=1 scripts/serve_and_run.sbatch
#   sbatch -J memp-ret   --export=ALL,DRIVER=memp,PHASE=eval,ARM=retrieved,LIMIT=0,BANK=<bank.json>,EMBEDDINGS=<emb.npz> scripts/serve_and_run.sbatch
#   sbatch -J memp-rnd   --export=ALL,DRIVER=memp,PHASE=eval,ARM=random,LIMIT=0,BANK=<bank.json>,EMBEDDINGS=<emb.npz> scripts/serve_and_run.sbatch
# Fixed OUT= paths let later jobs be submitted up front with --dependency=afterok:<jobid>.
# Build and eval checkpoint every CHUNK puzzles; resubmit with RESUME=<file> to continue.
#   python scripts/run.py memp --phase compare --baseline <baseline.json> --retrieved <ret.json> --random <rnd.json>
#
# MemoryBank: reuses Memp's build file and embeddings; RULES_PREAMBLE is frozen into the bank.
#   sbatch -J mb-dist --export=ALL,DRIVER=memorybank,PHASE=distill,RULES_PREAMBLE=on,BUILD=results/json/memp_build_train_<stamp>.json scripts/serve_and_run.sbatch
#   sbatch -J mb-ret  --export=ALL,DRIVER=memorybank,PHASE=eval,ARM=retrieved,LIMIT=0,BANK=<memorybank_bank_*.json>,EMBEDDINGS=<emb.npz> scripts/serve_and_run.sbatch
#   sbatch -J mb-rnd  --export=ALL,DRIVER=memorybank,PHASE=eval,ARM=random,LIMIT=0,BANK=<memorybank_bank_*.json>,EMBEDDINGS=<emb.npz> scripts/serve_and_run.sbatch
#   python scripts/run.py memorybank --phase compare --baseline <root.json> --retrieved <ret.json> --random <rnd.json> --memp <memp_eval_retrieved_*.json>
#
# Contrastive: DOs from successes, DON'Ts from failures. CO_GRANULARITY=hindsight is the
# control arm; run it alongside the method arm.
#   sbatch -J co-dist --export=ALL,DRIVER=contrastive,PHASE=distill,BUILD=<memp_build_train_*.json> scripts/serve_and_run.sbatch
#   sbatch -J co-avo  --export=ALL,DRIVER=contrastive,PHASE=eval,ARM=retrieved,CO_GRANULARITY=avoid,LIMIT=0,BANK=<contrastive_bank_*.json>,EMBEDDINGS=<emb.npz> scripts/serve_and_run.sbatch
#   sbatch -J co-hin  --export=ALL,DRIVER=contrastive,PHASE=eval,ARM=retrieved,CO_GRANULARITY=hindsight,LIMIT=0,BANK=<bank>,EMBEDDINGS=<emb.npz> scripts/serve_and_run.sbatch
#   sbatch -J co-pos  --export=ALL,DRIVER=contrastive,PHASE=eval,ARM=retrieved,CO_GRANULARITY=positive-only,LIMIT=0,BANK=<bank>,EMBEDDINGS=<emb.npz> scripts/serve_and_run.sbatch
#   sbatch -J co-bal  --export=ALL,DRIVER=contrastive,PHASE=eval,CO_ARM=balanced,MIN_POSITIVE=1,CO_GRANULARITY=avoid,LIMIT=0,BANK=<bank>,EMBEDDINGS=<emb.npz> scripts/serve_and_run.sbatch
#   sbatch -J co-rnd  --export=ALL,DRIVER=contrastive,PHASE=eval,ARM=random,CO_GRANULARITY=avoid,LIMIT=0,BANK=<bank>,EMBEDDINGS=<emb.npz> scripts/serve_and_run.sbatch
#   python scripts/run.py contrastive --phase compare --baseline <root.json> --retrieved <avoid.json> --hindsight <hin.json> --positive <pos.json> --random <rnd.json>
#
# StepGame: add BENCHMARK=stepgame to any job that does not start from a state file
# (the splits are frozen files, so LIMIT is only for smoke tests). In dependency order:
#   sbatch -J sg-base  --export=ALL,BENCHMARK=stepgame,DRIVER=baseline,SPLIT=test scripts/serve_and_run.sbatch
#   sbatch -J sg-refl  --export=ALL,BENCHMARK=stepgame,DRIVER=reflexion,PHASE=root,SPLIT=test scripts/serve_and_run.sbatch
#   sbatch -J sg-mem   --export=ALL,DRIVER=reflexion,PHASE=arm,ARM=memory,TRIALS=10,ROOT=<reflexion_root_stepgame_*.json> scripts/serve_and_run.sbatch
#   sbatch -J sg-build --export=ALL,BENCHMARK=stepgame,DRIVER=memp,PHASE=build,SPLIT=train scripts/serve_and_run.sbatch
#   sbatch -J sg-dist  --export=ALL,BENCHMARK=stepgame,DRIVER=memp,PHASE=distill,BUILD=<memp_build_stepgame_train_*.json> scripts/serve_and_run.sbatch
#   sbatch -J sg-emb   -G A100:1 -t 01:00:00 --export=ALL,BENCHMARK=stepgame,DRIVER=memp,PHASE=embed,TP=1 scripts/serve_and_run.sbatch
# PHASE=upgrade is not a job (no GPU, and no branch below); run it on the login node
# between distill and eval:
#   python scripts/run.py memp --phase upgrade --benchmark stepgame \
#       --bank results/json/memp_bank_stepgame_<tag>.json
#   sbatch -J sg-ret   --export=ALL,BENCHMARK=stepgame,DRIVER=memp,PHASE=eval,ARM=retrieved,BANK=<bank_traj.json>,EMBEDDINGS=<emb.npz> scripts/serve_and_run.sbatch
#   sbatch -J sg-fix   --export=ALL,BENCHMARK=stepgame,DRIVER=memp,PHASE=eval,ARM=fixed,BANK=<bank_traj.json>,EMBEDDINGS=<emb.npz> scripts/serve_and_run.sbatch
#   sbatch -J sg-mbd   --export=ALL,BENCHMARK=stepgame,DRIVER=memorybank,PHASE=distill,RULES_PREAMBLE=on,BUILD=<memp_build_stepgame_train_*.json> scripts/serve_and_run.sbatch
#   sbatch -J sg-mbr   --export=ALL,BENCHMARK=stepgame,DRIVER=memorybank,PHASE=eval,ARM=retrieved,BANK=<memorybank_bank_stepgame_*.json>,EMBEDDINGS=<emb.npz> scripts/serve_and_run.sbatch
#
# If the container fails to launch, the node is added to logs/bad_nodes.txt and the job
# resubmits itself excluding those nodes, up to MAX_ATTEMPTS (8). AUTO_RESUBMIT=0 disables this.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results
echo "[$(date +%T)] job $SLURM_JOB_ID on $(hostname -s)"

# ---- knobs ------------------------------------------------------------------
MODEL="${MODEL:-Qwen/Qwen3-32B}"
SERVED_NAME="${SERVED_NAME:-$(basename "$MODEL")}"
TP="${TP:-${SLURM_GPUS_ON_NODE:-2}}"      # tensor-parallel size = GPUs requested
GPU_UTIL="${GPU_UTIL:-0.90}"
MAXLEN="${MAXLEN:-32768}"                 # must exceed prompt + MAX_TOKENS
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
# `-` not `:-`: an explicit empty REASONING_PARSER= disables it; only unset defaults to qwen3.
REASONING_PARSER="${REASONING_PARSER-qwen3}"    # strips <think>...</think> out of .content.
SERVE_TIMEOUT="${SERVE_TIMEOUT:-2700}"    # seconds; loading weights off ceph is slow

SPLIT="${SPLIT:-test}"
LIMIT="${LIMIT:-50}"                      # 0 = whole split
CONCURRENCY="${CONCURRENCY:-32}"
# Qwen3 thinking mode: greedy decoding loops, and a reply cut off mid-<think> arrives with
# empty .content, so use the official sampling params and a large token budget.
MAX_TOKENS="${MAX_TOKENS:-12288}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
SMOKE="${SMOKE:-1}"                       # 0 skips the 2-prompt smoke check

DRIVER="${DRIVER:-baseline}"              # baseline | probe | reflexion | memp | memorybank | contrastive

# ---- Benchmark selection (all drivers) --------------------------------------
# Empty split knobs pick the standard k-stratified pair; a *_byhops_* STEPGAME_TEST
# selects the hop-stratified split. KEY_VARIANT must match between embed and eval jobs.
BENCHMARK="${BENCHMARK:-sparc}"           # sparc | stepgame
STEPGAME_TRAIN="${STEPGAME_TRAIN:-}"      # split JSONL; empty = the standard one
STEPGAME_TEST="${STEPGAME_TEST:-}"        # split JSONL; empty = the standard one
KEY_VARIANT="${KEY_VARIANT:-structural}"  # structural | lexical
# `upstream` is a 10-shot calibration prompt (its exemplars are injected memory), so
# never use it as a memoryless baseline or control arm.
PROMPT_PROTOCOL="${PROMPT_PROTOCOL:-ours}"  # ours | upstream-header | upstream
PHASE="${PHASE:-root}"                    # reflexion: root | arm    memp: build | distill | embed | eval    memorybank: distill | eval    contrastive: distill | eval
ARM="${ARM:-}"                            # reflexion arm: memory | control    memp/memorybank eval: retrieved | random
ROOT="${ROOT:-}"                          # start an arm from this root file
RESUME="${RESUME:-}"                      # ...or continue this arm file where it stopped
TRIALS="${TRIALS:-10}"                    # total trials including trial 1
MEMORY_CAP="${MEMORY_CAP:-3}"             # reflections kept per puzzle
# Unset = an auto-named, timestamped file. A fixed path keeps runs of several MODELs
# apart and lets dependent jobs name their ROOT before it exists.
OUT="${OUT:-}"
REFLECT_MAX_TOKENS="${REFLECT_MAX_TOKENS:-$MAX_TOKENS}"   # self-reflection budget (memory arm)
# Trial t decodes at SEED + t - 1 in both arms: vLLM replays a reply for the same prompt
# and seed, so a fixed seed would make the control arm repeat trial 1.
SEED="${SEED:-0}"

# ---- Memp knobs (DRIVER=memp; ignored otherwise) ----------------------------
BUILD="${BUILD:-}"                        # PHASE=distill: the memp_build_*.json to distill from
BANK="${BANK:-}"                          # PHASE=eval/upgrade: the memp_bank_*.json
EMBEDDINGS="${EMBEDDINGS:-}"              # PHASE=eval: the memp_embeddings_*.npz key/query vectors
TOP_K_MEMORIES="${TOP_K_MEMORIES:-3}"     # memories injected per test puzzle
GRANULARITY="${GRANULARITY:-script}"      # script | trajectory | proceduralization
PROVENANCE="${PROVENANCE:-all}"           # all | agent | gold -- 'agent' = successes-only build
UPDATE="${UPDATE:-none}"                  # none | vanilla | validation | adjustment
GROUPS="${GROUPS:-5}"                     # test-set groups between memory refreshes
KEYWORDS="${KEYWORDS:-}"                  # memp_keywords_*.json for ARM=avefact
CHUNK="${CHUNK:-100}"                     # puzzles per checkpoint (build/eval) / calls per checkpoint (distill)
DISTILL_MAX_TOKENS="${DISTILL_MAX_TOKENS:-$MAX_TOKENS}"
EMBED_MODEL="${EMBED_MODEL:-Qwen/Qwen3-Embedding-0.6B}"   # served by PHASE=embed

# ---- MemoryBank knobs (DRIVER=memorybank; ignored otherwise) -----------------
# Shares BUILD / BANK / EMBEDDINGS / TOP_K_MEMORIES / PROVENANCE / CHUNK /
# DISTILL_MAX_TOKENS with memp above.
MB_GRANULARITY="${MB_GRANULARITY:-case-record}"   # case-record | case
RULES_PREAMBLE="${RULES_PREAMBLE:-on}"    # on | off; frozen into the bank
# Separate from LIMIT so a bank is never silently truncated. 0 = the whole build file.
DISTILL_LIMIT="${DISTILL_LIMIT:-0}"

# ---- Contrastive knobs (DRIVER=contrastive; ignored otherwise) ---------------
# avoid | hindsight | positive-only | avoid-only. Separate from GRANULARITY so a wrong
# value fails the driver's choices check. `hindsight` is the control arm.
CO_GRANULARITY="${CO_GRANULARITY:-avoid}"
# retrieved | balanced | random | random-balanced. `balanced` guarantees MIN_POSITIVE
# DO entries per block.
CO_ARM="${CO_ARM:-}"                      # defaults to ARM when unset
MIN_POSITIVE="${MIN_POSITIVE:-1}"
# on = append the concrete winning/losing line to each item. Needs a bank run through
# `--phase exemplars` (CPU, login node).
CO_EXEMPLARS="${CO_EXEMPLARS:-off}"
# mixed = the two-sign header on every arm; matched = a header per arm.
CO_FRAMING="${CO_FRAMING:-mixed}"
# Extra args appended to the vLLM serve command; PHASE=embed sets a default below.
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"

# PHASE=embed serves the embedding model instead of the chat model. A pooling server has
# no <think> to strip and cannot take the chat smoke test. `--runner pooling` needs
# vLLM >= 0.25; older images need EXTRA_VLLM_ARGS="--task embed".
if [[ "$DRIVER" == "memp" && ( "$PHASE" == "embed" || "$PHASE" == "embed-keywords" ) ]]; then
    MODEL="$EMBED_MODEL"
    REASONING_PARSER=""
    SMOKE=0
    EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:---runner pooling}"
fi

# ---- modules: re-resolve Lmod for THIS node's CPU ---------------------------
# --export=ALL carries the login node's Lmod state, which resolves an AVX-512 apptainer
# build that SIGILLs on the Zen3 compute nodes. Purge it and re-source the site profile.
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

# Fallback: if Lmod still picked an AVX-512 build on a non-AVX-512 CPU, use the zen build.
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
# Not $PROJECT: the site profile overwrites it with the unresolved symlink.
AGPROJ="$(readlink -f "${AGPROJ:-/projects/scc/UGOE/UXEI/UMIN/ag_gipp/dir.project}")"
PROJ_IN="/proj"                           # where $AGPROJ is mounted INSIDE the container
SIF="${SIF:-$AGPROJ/vllm-container/vllm-25.1.sif}"    # vLLM 0.25.1, cu129
HUB="${HUB:-$AGPROJ/hf/hub}"              # group-shared weights
# Per-user: the editable install records this checkout's absolute path, and REBUILD=1
# deletes the venv, so a shared one would break other members' jobs.
VENV="${VENV:-$AGPROJ/$USER/.venv_client}"  # host-side driver env: openai+datasets+gym, NO torch
export HF_HOME="${HF_HOME:-$AGPROJ/$USER/hf_cache}"   # client side: datasets
# Compile caches go to node-local /tmp: the ceph path is read-only inside the container.
CACHE_IN="/tmp/vllm-cache-$SLURM_JOB_ID"  # apptainer bind-mounts the host /tmp in
mkdir -p "$HF_HOME" "$CACHE_IN" "$CACHE_IN/tmp" "$CACHE_IN/tiktoken"

[[ -r "$SIF" ]] || { echo "no such image: $SIF" >&2; exit 1; }
echo "[$(date +%T)] AGPROJ=$AGPROJ -> $PROJ_IN (in container)"

# ---- PREFLIGHT: prove the container actually launches ------------------------
# Fails in seconds, before the model loads, if the container cannot start or see the
# weights. Resubmitting only excludes nodes, so it will not help an environmental failure.
PREFLIGHT_LOG="logs/preflight-$SLURM_JOB_ID.log"
if ! apptainer exec --nv --bind "$AGPROJ:$PROJ_IN" "$SIF" \
        test -d "$PROJ_IN/hf/hub" > "$PREFLIGHT_LOG" 2>&1; then
    node="$(hostname -s)"
    echo "[$(date +%T)] PREFLIGHT FAILED on $node -- first lines of $PREFLIGHT_LOG:" >&2
    head -5 "$PREFLIGHT_LOG" >&2
    grep -qxF "$node" logs/bad_nodes.txt 2>/dev/null || echo "$node" >> logs/bad_nodes.txt

    ATTEMPT="${ATTEMPT:-1}"
    if [[ "${AUTO_RESUBMIT:-1}" == "1" && "$ATTEMPT" -lt "${MAX_ATTEMPTS:-8}" ]]; then
        # Command-line -G/-t/-J overrides are not in the header; rebuild them from the allocation.
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
    mkdir -p "$(dirname "$VENV")"
    uv venv --managed-python --python 3.11 "$VENV"   # module python is a zen2 build
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    uv pip install -e .
else
    echo "[$(date +%T)] reusing client venv at $VENV (REBUILD=1 to force)"
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi

# The -C inet proxy also intercepts localhost; bypass it for the health check and driver.
export no_proxy="localhost,127.0.0.1,0.0.0.0,::1"
export NO_PROXY="$no_proxy"

# ---- resolve the model to a local snapshot when we have one -----------------
# Serving a cached snapshot by path avoids hub lock writes in the shared cache.
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
    --env TMPDIR="$CACHE_IN/tmp" \
    --env TIKTOKEN_RS_CACHE_DIR="$CACHE_IN/tiktoken" \
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
        ${EXTRA_VLLM_ARGS:+$EXTRA_VLLM_ARGS} \
    > "$SERVER_LOG" 2>&1 &
VLLM_PID=$!

cleanup() {
    echo "[$(date +%T)] cleanup: stopping vLLM ($VLLM_PID)"
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    # Node /tmp outlives the job; leftover compile caches fill the disk (ENOSPC for the next job).
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

# --limit is added per driver. --resume runs pass neither COMMON_ARGS nor BENCH_ARGS:
# benchmark, splits and sampling are frozen in the state file.
BENCH_ARGS=(--benchmark "$BENCHMARK")
if [[ "$BENCHMARK" == "stepgame" ]]; then
    BENCH_ARGS+=(--key-variant "$KEY_VARIANT")
    BENCH_ARGS+=(--prompt-protocol "$PROMPT_PROTOCOL")
    [[ -n "$STEPGAME_TRAIN" ]] && BENCH_ARGS+=(--stepgame-train "$STEPGAME_TRAIN")
    [[ -n "$STEPGAME_TEST"  ]] && BENCH_ARGS+=(--stepgame-test  "$STEPGAME_TEST")
fi

COMMON_ARGS=(--split "$SPLIT" --concurrency "$CONCURRENCY" \
    --max-tokens "$MAX_TOKENS" --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" --seed "$SEED" ${TOP_K:+--top-k "$TOP_K"} \
    "${BENCH_ARGS[@]}")

case "$DRIVER" in
  baseline)
    echo "[$(date +%T)] baseline: split=$SPLIT concurrency=$CONCURRENCY limit=$LIMIT"
    python -u scripts/run.py baseline "${COMMON_ARGS[@]}" --limit "$LIMIT" \
        ${OUT:+--out "$OUT"}
    ;;
  probe)
    # Adversarial probe on the StepGame baseline (scripts/probe_stepgame.py), outside the
    # main runs. PER_K sets the sample size per file-k.
    echo "[$(date +%T)] probe: per_k=${PER_K:-20} arms=${ARMS:-full,bare,renamed,scrambled}"
    python -u scripts/probe_stepgame.py "${COMMON_ARGS[@]}" \
        --per-k "${PER_K:-20}" --arms "${ARMS:-full,bare,renamed,scrambled}" \
        ${OUT:+--out "$OUT"}
    ;;
  reflexion)
    case "$PHASE" in
      root)
        echo "[$(date +%T)] reflexion root (trial 1, shared by both arms): split=$SPLIT limit=$LIMIT"
        python -u scripts/run.py reflexion --phase root "${COMMON_ARGS[@]}" \
            --limit "$LIMIT" --trials "$TRIALS" --memory-cap "$MEMORY_CAP" \
            ${OUT:+--out "$OUT"}
        ;;
      arm)
        # The run's configuration is frozen in the root/arm file; arm jobs pass only per-job settings.
        if [[ -n "$RESUME" ]]; then
            echo "[$(date +%T)] reflexion arm: resuming $RESUME up to $TRIALS trials"
            python -u scripts/run.py reflexion --phase arm --resume "$RESUME" \
                --trials "$TRIALS" --concurrency "$CONCURRENCY" \
                --reflect-max-tokens "$REFLECT_MAX_TOKENS"
        else
            [[ -n "$ROOT" ]] || { echo "PHASE=arm needs ROOT=<reflexion_root_*.json> or RESUME=<arm file>" >&2; exit 1; }
            [[ -n "$ARM"  ]] || { echo "PHASE=arm needs ARM=memory or ARM=control" >&2; exit 1; }
            echo "[$(date +%T)] reflexion arm=$ARM: $TRIALS trials from $ROOT"
            python -u scripts/run.py reflexion --phase arm --arm "$ARM" --root "$ROOT" \
                --trials "$TRIALS" --concurrency "$CONCURRENCY" \
                --reflect-max-tokens "$REFLECT_MAX_TOKENS" ${OUT:+--out "$OUT"}
        fi
        ;;
      *)
        echo "unknown PHASE=$PHASE for DRIVER=reflexion (expected: root | arm)" >&2; exit 1 ;;
    esac
    ;;
  memp)
    case "$PHASE" in
      build)
        if [[ -n "$RESUME" ]]; then
            echo "[$(date +%T)] memp build: resuming $RESUME"
            python -u scripts/run.py memp --phase build --resume "$RESUME" \
                --concurrency "$CONCURRENCY" --chunk "$CHUNK"
        else
            echo "[$(date +%T)] memp build: memoryless pass over train, limit=$LIMIT"
            # --split train comes after COMMON_ARGS so it wins.
            python -u scripts/run.py memp --phase build "${COMMON_ARGS[@]}" \
                --split train --limit "$LIMIT" --chunk "$CHUNK" ${OUT:+--out "$OUT"}
        fi
        ;;
      distill)
        if [[ -n "$RESUME" ]]; then
            echo "[$(date +%T)] memp distill: resuming $RESUME"
            python -u scripts/run.py memp --phase distill --resume "$RESUME" \
                --concurrency "$CONCURRENCY" --chunk "$CHUNK"
        else
            [[ -n "$BUILD" ]] || { echo "PHASE=distill needs BUILD=<memp_build_*.json> or RESUME=<bank file>" >&2; exit 1; }
            echo "[$(date +%T)] memp distill: $BUILD -> bank"
            python -u scripts/run.py memp --phase distill "${COMMON_ARGS[@]}" \
                --build "$BUILD" --distill-max-tokens "$DISTILL_MAX_TOKENS" \
                --chunk "$CHUNK" ${OUT:+--out "$OUT"}
        fi
        ;;
      embed)
        # Always the full dataset: a partial .npz would leave later evals with missing vectors.
        echo "[$(date +%T)] memp embed: all train keys + test queries via $MODEL"
        python -u scripts/run.py memp --phase embed "${BENCH_ARGS[@]}" \
            ${OUT:+--out "$OUT"}
        ;;
      keywords)
        # AveFact key construction: one short LLM call per puzzle, resumable.
        echo "[$(date +%T)] memp keywords: extracting AveFact keys"
        python -u scripts/run.py memp --phase keywords "${COMMON_ARGS[@]}" \
            --limit "$LIMIT" --chunk "$CHUNK" \
            --distill-max-tokens "$DISTILL_MAX_TOKENS" ${OUT:+--out "$OUT"}
        ;;
      embed-keywords)
        [[ -n "$KEYWORDS" ]] || { echo "PHASE=embed-keywords needs KEYWORDS=<memp_keywords_*.json>" >&2; exit 1; }
        echo "[$(date +%T)] memp embed-keywords: $KEYWORDS via $MODEL"
        python -u scripts/run.py memp --phase embed-keywords --keywords "$KEYWORDS" \
            ${OUT:+--out "$OUT"}
        ;;
      eval)
        if [[ -n "$RESUME" ]]; then
            echo "[$(date +%T)] memp eval: resuming $RESUME"
            python -u scripts/run.py memp --phase eval --resume "$RESUME" \
                --concurrency "$CONCURRENCY" --chunk "$CHUNK"
        else
            [[ -n "$BANK" && -n "$EMBEDDINGS" ]] || { echo "PHASE=eval needs BANK=<memp_bank_*.json> and EMBEDDINGS=<memp_embeddings_*.npz>" >&2; exit 1; }
            [[ -n "$ARM" ]] || { echo "PHASE=eval needs ARM=retrieved or ARM=random" >&2; exit 1; }
            echo "[$(date +%T)] memp eval: arm=$ARM k=$TOP_K_MEMORIES limit=$LIMIT " \
                 "granularity=$GRANULARITY provenance=$PROVENANCE update=$UPDATE"
            python -u scripts/run.py memp --phase eval "${COMMON_ARGS[@]}" \
                --limit "$LIMIT" --chunk "$CHUNK" --arm "$ARM" --bank "$BANK" \
                --embeddings "$EMBEDDINGS" --top-k-memories "$TOP_K_MEMORIES" \
                --granularity "$GRANULARITY" --provenance "$PROVENANCE" \
                --update "$UPDATE" --groups "$GROUPS" \
                --distill-max-tokens "$DISTILL_MAX_TOKENS" \
                ${KEYWORDS:+--keywords "$KEYWORDS"} ${OUT:+--out "$OUT"}
        fi
        ;;
      *)
        echo "unknown PHASE=$PHASE for DRIVER=memp (expected: build | distill | embed | eval)" >&2; exit 1 ;;
    esac
    ;;
  memorybank)
    # The build pass and embeddings are Memp's, so there are only two phases here.
    case "$PHASE" in
      distill)
        if [[ -n "$RESUME" ]]; then
            echo "[$(date +%T)] memorybank distill: resuming $RESUME"
            python -u scripts/run.py memorybank --phase distill --resume "$RESUME" \
                --concurrency "$CONCURRENCY" --chunk "$CHUNK"
        else
            [[ -n "$BUILD" ]] || { echo "PHASE=distill needs BUILD=<memp_build_*.json> or RESUME=<bank file>" >&2; exit 1; }
            echo "[$(date +%T)] memorybank distill: $BUILD -> case bank (rules_preamble=$RULES_PREAMBLE)"
            python -u scripts/run.py memorybank --phase distill "${COMMON_ARGS[@]}" \
                --build "$BUILD" --rules-preamble "$RULES_PREAMBLE" \
                --distill-max-tokens "$DISTILL_MAX_TOKENS" \
                --chunk "$CHUNK" --limit "$DISTILL_LIMIT" ${OUT:+--out "$OUT"}
        fi
        ;;
      eval)
        if [[ -n "$RESUME" ]]; then
            echo "[$(date +%T)] memorybank eval: resuming $RESUME"
            python -u scripts/run.py memorybank --phase eval --resume "$RESUME" \
                --concurrency "$CONCURRENCY" --chunk "$CHUNK"
        else
            [[ -n "$BANK" && -n "$EMBEDDINGS" ]] || { echo "PHASE=eval needs BANK=<memorybank_bank_*.json> and EMBEDDINGS=<memp_embeddings_*.npz>" >&2; exit 1; }
            [[ -n "$ARM" ]] || { echo "PHASE=eval needs ARM=retrieved or ARM=random" >&2; exit 1; }
            echo "[$(date +%T)] memorybank eval: arm=$ARM k=$TOP_K_MEMORIES limit=$LIMIT " \
                 "granularity=$MB_GRANULARITY provenance=$PROVENANCE"
            python -u scripts/run.py memorybank --phase eval "${COMMON_ARGS[@]}" \
                --limit "$LIMIT" --chunk "$CHUNK" --arm "$ARM" --bank "$BANK" \
                --embeddings "$EMBEDDINGS" --top-k-memories "$TOP_K_MEMORIES" \
                --granularity "$MB_GRANULARITY" --provenance "$PROVENANCE" \
                ${OUT:+--out "$OUT"}
        fi
        ;;
      *)
        echo "unknown PHASE=$PHASE for DRIVER=memorybank (expected: distill | eval)" >&2; exit 1 ;;
    esac
    ;;
  contrastive)
    # The build pass and embeddings are Memp's. distill writes the strategy, avoid and
    # hindsight texts in one pass, so the method arm and its control share one bank.
    case "$PHASE" in
      distill)
        if [[ -n "$RESUME" ]]; then
            echo "[$(date +%T)] contrastive distill: resuming $RESUME"
            python -u scripts/run.py contrastive --phase distill --resume "$RESUME" \
                --concurrency "$CONCURRENCY" --chunk "$CHUNK"
        else
            [[ -n "$BUILD" ]] || { echo "PHASE=distill needs BUILD=<memp_build_*.json> or RESUME=<bank file>" >&2; exit 1; }
            echo "[$(date +%T)] contrastive distill: $BUILD -> DOs from successes, DON'Ts from failures"
            python -u scripts/run.py contrastive --phase distill "${COMMON_ARGS[@]}" \
                --build "$BUILD" --distill-max-tokens "$DISTILL_MAX_TOKENS" \
                --chunk "$CHUNK" --limit "$DISTILL_LIMIT" ${OUT:+--out "$OUT"}
        fi
        ;;
      eval)
        if [[ -n "$RESUME" ]]; then
            echo "[$(date +%T)] contrastive eval: resuming $RESUME"
            python -u scripts/run.py contrastive --phase eval --resume "$RESUME" \
                --concurrency "$CONCURRENCY" --chunk "$CHUNK"
        else
            [[ -n "$BANK" && -n "$EMBEDDINGS" ]] || { echo "PHASE=eval needs BANK=<contrastive_bank_*.json> and EMBEDDINGS=<memp_embeddings_*.npz>" >&2; exit 1; }
            # CO_ARM adds balanced and random-balanced; ARM still works for retrieved and random.
            EFFECTIVE_ARM="${CO_ARM:-$ARM}"
            [[ -n "$EFFECTIVE_ARM" ]] || { echo "PHASE=eval needs ARM or CO_ARM (retrieved | balanced | random | random-balanced)" >&2; exit 1; }
            echo "[$(date +%T)] contrastive eval: arm=$EFFECTIVE_ARM k=$TOP_K_MEMORIES limit=$LIMIT " \
                 "granularity=$CO_GRANULARITY min_positive=$MIN_POSITIVE exemplars=$CO_EXEMPLARS" \
                 "framing=$CO_FRAMING provenance=$PROVENANCE"
            python -u scripts/run.py contrastive --phase eval "${COMMON_ARGS[@]}" \
                --limit "$LIMIT" --chunk "$CHUNK" --arm "$EFFECTIVE_ARM" --bank "$BANK" \
                --embeddings "$EMBEDDINGS" --top-k-memories "$TOP_K_MEMORIES" \
                --granularity "$CO_GRANULARITY" --min-positive "$MIN_POSITIVE" \
                --exemplars "$CO_EXEMPLARS" --framing "$CO_FRAMING" \
                --provenance "$PROVENANCE" ${OUT:+--out "$OUT"}
        fi
        ;;
      *)
        echo "unknown PHASE=$PHASE for DRIVER=contrastive (expected: distill | eval; --phase exemplars takes no GPU, run it on the login node)" >&2; exit 1 ;;
    esac
    ;;
  *)
    echo "unknown DRIVER=$DRIVER (expected: baseline | reflexion | memp | memorybank | contrastive)" >&2; exit 1 ;;
esac

echo "[$(date +%T)] done; the cleanup trap stops vLLM and frees the GPUs."

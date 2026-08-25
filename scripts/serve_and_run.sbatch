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
# Reflexion, run as the paper does it: up to TRIALS attempts per puzzle, reflecting on
# each failure, with a control arm over the SAME number of trials. The control is not
# optional -- only unsolved puzzles are re-attempted and solved ones are carried
# forward, so cumulative accuracy rises with the trial index even with no memory at
# all, and at TEMPERATURE=0.6 ten resamples convert failures on their own. Reflexion's
# effect is the memory arm minus the control arm at the same trial index.
#
# One job runs trial 1 (shared by both arms), then one job per arm:
#   sbatch -J refl-root --export=ALL,DRIVER=reflexion,PHASE=root,LIMIT=0 scripts/serve_and_run.sbatch
#   sbatch -J refl-mem  --export=ALL,DRIVER=reflexion,PHASE=arm,ARM=memory,TRIALS=10,ROOT=results/json/reflexion_root_test_<stamp>.json scripts/serve_and_run.sbatch
#   sbatch -J refl-ctl  --export=ALL,DRIVER=reflexion,PHASE=arm,ARM=control,TRIALS=10,ROOT=results/json/reflexion_root_test_<stamp>.json scripts/serve_and_run.sbatch
#
# A trial over the 500-puzzle test split costs ~2 h, so 10 trials do NOT fit in this
# job's 12 h limit. Each arm checkpoints after every trial; resubmit with RESUME to
# continue the same file where it stopped (no trial is repeated):
#   sbatch -J refl-mem --export=ALL,DRIVER=reflexion,PHASE=arm,TRIALS=10,RESUME=results/json/reflexion_memory_test_<stamp>.json scripts/serve_and_run.sbatch
#
# Then, on the login node (no GPU needed):
#   python scripts/run_reflexion.py --phase compare --memory <memory.json> --control <control.json>
#
# Memp (procedural memory): build a bank on the TRAIN split, freeze it, retrieve at
# test time. Four GPU jobs -- the embed job serves a different (embedding) model, so
# it cannot share a job with the chat phases; it depends only on the dataset and can
# run in parallel with build/distill. See scripts/run_memp.py for the full story.
#   sbatch -J memp-build --export=ALL,DRIVER=memp,PHASE=build,SPLIT=train,LIMIT=0 scripts/serve_and_run.sbatch
#   sbatch -J memp-dist  --export=ALL,DRIVER=memp,PHASE=distill,BUILD=results/json/memp_build_train_<stamp>.json scripts/serve_and_run.sbatch
#   sbatch -J memp-embed -G A100:1 -t 01:00:00 --export=ALL,DRIVER=memp,PHASE=embed,TP=1 scripts/serve_and_run.sbatch
#   sbatch -J memp-ret   --export=ALL,DRIVER=memp,PHASE=eval,ARM=retrieved,LIMIT=0,BANK=<bank.json>,EMBEDDINGS=<emb.npz> scripts/serve_and_run.sbatch
#   sbatch -J memp-rnd   --export=ALL,DRIVER=memp,PHASE=eval,ARM=random,LIMIT=0,BANK=<bank.json>,EMBEDDINGS=<emb.npz> scripts/serve_and_run.sbatch
# Set OUT= on the build/distill jobs to fixed paths and the later jobs can be submitted
# up front with --dependency=afterok:<jobid> (same reasoning as the reflexion OUT note
# below). Build and eval checkpoint every CHUNK puzzles; resubmit with RESUME=<file> to
# continue. Then, on the login node:
#   python scripts/run_memp.py --phase compare --baseline <baseline.json> --retrieved <ret.json> --random <rnd.json>
#
# MemoryBank (declarative memory): the same cross-task shape as Memp, but the memories
# are past CASES rather than procedures. Two GPU phases only -- the train build pass and
# the embeddings are Memp's, reused verbatim. RULES_PREAMBLE=on|off selects the prompt
# ablation and is frozen into the bank it builds. See scripts/run_memorybank.py.
#   sbatch -J mb-dist --export=ALL,DRIVER=memorybank,PHASE=distill,RULES_PREAMBLE=on,BUILD=results/json/memp_build_train_<stamp>.json scripts/serve_and_run.sbatch
#   sbatch -J mb-ret  --export=ALL,DRIVER=memorybank,PHASE=eval,ARM=retrieved,LIMIT=0,BANK=<memorybank_bank_*.json>,EMBEDDINGS=<emb.npz> scripts/serve_and_run.sbatch
#   sbatch -J mb-rnd  --export=ALL,DRIVER=memorybank,PHASE=eval,ARM=random,LIMIT=0,BANK=<memorybank_bank_*.json>,EMBEDDINGS=<emb.npz> scripts/serve_and_run.sbatch
#   python scripts/run_memorybank.py --phase compare --baseline <root.json> --retrieved <ret.json> --random <rnd.json> --memp <memp_eval_retrieved_*.json>
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
# The alias the client asks for (vLLM --served-model-name). Derived from MODEL so it
# describes what is actually being served: it used to be a flat "qwen" for every
# checkpoint, which was harmless while everything WAS a Qwen, but once gpt-oss and
# Qwen3-32B ran side by side the results read "served as qwen" for all of them.
# Results still record the full checkpoint separately as model_id.
SERVED_NAME="${SERVED_NAME:-$(basename "$MODEL")}"
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
DRIVER="${DRIVER:-baseline}"              # baseline | reflexion | memp | memorybank
PHASE="${PHASE:-root}"                    # reflexion: root | arm    memp: build | distill | embed | eval    memorybank: distill | eval
ARM="${ARM:-}"                            # reflexion arm: memory | control    memp/memorybank eval: retrieved | random
ROOT="${ROOT:-}"                          # start an arm from this root file
RESUME="${RESUME:-}"                      # ...or continue this arm file where it stopped
TRIALS="${TRIALS:-10}"                    # total trials INCLUDING trial 1 (the paper runs ~10)
MEMORY_CAP="${MEMORY_CAP:-3}"             # reflections kept per puzzle (the paper's Omega)
# Explicit output path. Unset = an auto-named, timestamped file. Set it when running
# several MODELs: the auto names carry only split+timestamp, so three models' root
# files are told apart only by when they happened to run. A fixed name is also what
# lets the arm jobs be submitted up front with --dependency, since they must name
# their ROOT before the root job has produced it.
OUT="${OUT:-}"
REFLECT_MAX_TOKENS="${REFLECT_MAX_TOKENS:-$MAX_TOKENS}"   # self-reflection budget (memory arm)
# Seed for trial 1. Trial t decodes at SEED + t - 1, in BOTH arms: distinct per trial
# (with the same prompt and seed vLLM replays its previous reply, so a fixed seed would
# make the control arm repeat trial 1 instead of resampling) and identical across arms
# (so the two differ only in whether the reflection block is injected).
SEED="${SEED:-0}"

# ---- Memp knobs (DRIVER=memp; ignored otherwise) ----------------------------
BUILD="${BUILD:-}"                        # PHASE=distill: the memp_build_*.json to distill from
BANK="${BANK:-}"                          # PHASE=eval/upgrade: the memp_bank_*.json
EMBEDDINGS="${EMBEDDINGS:-}"              # PHASE=eval: the memp_embeddings_*.npz key/query vectors
TOP_K_MEMORIES="${TOP_K_MEMORIES:-3}"     # memories injected per test puzzle (paper's optimum ~10)
# Paper-replication knobs (all default to the original frozen-bank behaviour).
GRANULARITY="${GRANULARITY:-script}"      # script | trajectory | proceduralization  (Table 1)
PROVENANCE="${PROVENANCE:-all}"           # all | agent | gold -- 'agent' = successes-only build
UPDATE="${UPDATE:-none}"                  # none | vanilla | validation | adjustment (section 4.3)
GROUPS="${GROUPS:-5}"                     # test-set groups between memory refreshes
KEYWORDS="${KEYWORDS:-}"                  # memp_keywords_*.json for ARM=avefact
CHUNK="${CHUNK:-100}"                     # puzzles per checkpoint (build/eval) / calls per checkpoint (distill)
DISTILL_MAX_TOKENS="${DISTILL_MAX_TOKENS:-$MAX_TOKENS}"   # distillation budget
EMBED_MODEL="${EMBED_MODEL:-Qwen/Qwen3-Embedding-0.6B}"   # served by PHASE=embed (cached in the group hub)

# ---- MemoryBank knobs (DRIVER=memorybank; ignored otherwise) -----------------
# Shares BUILD / BANK / EMBEDDINGS / TOP_K_MEMORIES / PROVENANCE / CHUNK /
# DISTILL_MAX_TOKENS with memp above -- both are cross-task banks over the same
# build pass and the same embeddings, which is exactly the point of the design.
MB_GRANULARITY="${MB_GRANULARITY:-case-record}"   # case-record | case (its own scale, so not GRANULARITY)
RULES_PREAMBLE="${RULES_PREAMBLE:-on}"    # on | off -- the rulebook ablation, frozen into the bank
# Deliberately NOT the shared LIMIT (which defaults to 50 and means "eval split
# size"): a bank silently truncated to 50 of 500 entries would still run, still
# retrieve, and quietly measure the wrong thing. 0 = the whole build file.
DISTILL_LIMIT="${DISTILL_LIMIT:-0}"       # smoke-test knob for PHASE=distill
# Extra args spliced verbatim into the vLLM serve command. Empty by default, so
# existing drivers are untouched; PHASE=embed defaults it below.
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"

# PHASE=embed serves the EMBEDding model instead of the chat model: vLLM runs one
# model per process, so embeddings get their own small job (1 GPU, minutes) and the
# eval job later needs only the chat model plus the .npz this one writes.
# REASONING_PARSER= (explicit empty -- see the `-` vs `:-` note above) because an
# embedding model has no <think> to strip, and SMOKE=0 because the chat smoke test
# cannot run against a pooling server (the driver does its own embed smoke instead).
# `--runner pooling` is verified against vllm-25.1.sif (vLLM 0.25.1: EngineArgs has
# `runner`, not the older `task`); an older image would need EXTRA_VLLM_ARGS="--task embed".
if [[ "$DRIVER" == "memp" && ( "$PHASE" == "embed" || "$PHASE" == "embed-keywords" ) ]]; then
    MODEL="$EMBED_MODEL"
    REASONING_PARSER=""
    SMOKE=0
    EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:---runner pooling}"
fi

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
mkdir -p "$HF_HOME" "$CACHE_IN" "$CACHE_IN/tmp" "$CACHE_IN/tiktoken"

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
# a reflexion arm runs over the puzzle set stored in its state file, not a fresh --limit slice.
COMMON_ARGS=(--split "$SPLIT" --concurrency "$CONCURRENCY" \
    --max-tokens "$MAX_TOKENS" --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" --seed "$SEED" ${TOP_K:+--top-k "$TOP_K"})

case "$DRIVER" in
  baseline)
    echo "[$(date +%T)] baseline: split=$SPLIT concurrency=$CONCURRENCY limit=$LIMIT"
    python -u scripts/run_baseline.py "${COMMON_ARGS[@]}" --limit "$LIMIT"
    ;;
  reflexion)
    case "$PHASE" in
      root)
        echo "[$(date +%T)] reflexion root (trial 1, shared by both arms): split=$SPLIT limit=$LIMIT"
        python -u scripts/run_reflexion.py --phase root "${COMMON_ARGS[@]}" \
            --limit "$LIMIT" --trials "$TRIALS" --memory-cap "$MEMORY_CAP" \
            ${OUT:+--out "$OUT"}
        ;;
      arm)
        # Everything that defines the run (split, sampling, base seed, memory cap) is
        # frozen in the root/arm file, so an arm job passes only what may legitimately
        # differ between jobs. That is what lets a resume extend a curve rather than
        # silently start a differently-configured one halfway through.
        if [[ -n "$RESUME" ]]; then
            echo "[$(date +%T)] reflexion arm: resuming $RESUME up to $TRIALS trials"
            python -u scripts/run_reflexion.py --phase arm --resume "$RESUME" \
                --trials "$TRIALS" --concurrency "$CONCURRENCY" \
                --reflect-max-tokens "$REFLECT_MAX_TOKENS"
        else
            [[ -n "$ROOT" ]] || { echo "PHASE=arm needs ROOT=<reflexion_root_*.json> or RESUME=<arm file>" >&2; exit 1; }
            [[ -n "$ARM"  ]] || { echo "PHASE=arm needs ARM=memory or ARM=control" >&2; exit 1; }
            echo "[$(date +%T)] reflexion arm=$ARM: $TRIALS trials from $ROOT"
            python -u scripts/run_reflexion.py --phase arm --arm "$ARM" --root "$ROOT" \
                --trials "$TRIALS" --concurrency "$CONCURRENCY" \
                --reflect-max-tokens "$REFLECT_MAX_TOKENS" ${OUT:+--out "$OUT"}
        fi
        ;;
      *)
        echo "unknown PHASE=$PHASE for DRIVER=reflexion (expected: root | arm)" >&2; exit 1 ;;
    esac
    ;;
  memp)
    # Same frozen-config discipline as the reflexion arm: a RESUME job passes only
    # what may legitimately differ between jobs (concurrency, chunk); everything
    # that defines the run comes from the state file it is continuing.
    case "$PHASE" in
      build)
        if [[ -n "$RESUME" ]]; then
            echo "[$(date +%T)] memp build: resuming $RESUME"
            python -u scripts/run_memp.py --phase build --resume "$RESUME" \
                --concurrency "$CONCURRENCY" --chunk "$CHUNK"
        else
            echo "[$(date +%T)] memp build: memoryless pass over train, limit=$LIMIT"
            # --split train appended AFTER the common args so it wins: the bank must
            # be built from puzzles the eval never sees (the driver enforces it too).
            python -u scripts/run_memp.py --phase build "${COMMON_ARGS[@]}" \
                --split train --limit "$LIMIT" --chunk "$CHUNK" ${OUT:+--out "$OUT"}
        fi
        ;;
      distill)
        if [[ -n "$RESUME" ]]; then
            echo "[$(date +%T)] memp distill: resuming $RESUME"
            python -u scripts/run_memp.py --phase distill --resume "$RESUME" \
                --concurrency "$CONCURRENCY" --chunk "$CHUNK"
        else
            [[ -n "$BUILD" ]] || { echo "PHASE=distill needs BUILD=<memp_build_*.json> or RESUME=<bank file>" >&2; exit 1; }
            echo "[$(date +%T)] memp distill: $BUILD -> bank"
            python -u scripts/run_memp.py --phase distill "${COMMON_ARGS[@]}" \
                --build "$BUILD" --distill-max-tokens "$DISTILL_MAX_TOKENS" \
                --chunk "$CHUNK" ${OUT:+--out "$OUT"}
        fi
        ;;
      embed)
        # Always the full dataset (minutes of work): a limited .npz would poison
        # every later eval with missing vectors.
        echo "[$(date +%T)] memp embed: all train keys + test queries via $MODEL"
        python -u scripts/run_memp.py --phase embed ${OUT:+--out "$OUT"}
        ;;
      keywords)
        # AveFact key construction: one short LLM call per puzzle, resumable.
        echo "[$(date +%T)] memp keywords: extracting AveFact keys"
        python -u scripts/run_memp.py --phase keywords "${COMMON_ARGS[@]}" \
            --limit "$LIMIT" --chunk "$CHUNK" \
            --distill-max-tokens "$DISTILL_MAX_TOKENS" ${OUT:+--out "$OUT"}
        ;;
      embed-keywords)
        [[ -n "$KEYWORDS" ]] || { echo "PHASE=embed-keywords needs KEYWORDS=<memp_keywords_*.json>" >&2; exit 1; }
        echo "[$(date +%T)] memp embed-keywords: $KEYWORDS via $MODEL"
        python -u scripts/run_memp.py --phase embed-keywords --keywords "$KEYWORDS" \
            ${OUT:+--out "$OUT"}
        ;;
      eval)
        if [[ -n "$RESUME" ]]; then
            echo "[$(date +%T)] memp eval: resuming $RESUME"
            python -u scripts/run_memp.py --phase eval --resume "$RESUME" \
                --concurrency "$CONCURRENCY" --chunk "$CHUNK"
        else
            [[ -n "$BANK" && -n "$EMBEDDINGS" ]] || { echo "PHASE=eval needs BANK=<memp_bank_*.json> and EMBEDDINGS=<memp_embeddings_*.npz>" >&2; exit 1; }
            [[ -n "$ARM" ]] || { echo "PHASE=eval needs ARM=retrieved or ARM=random" >&2; exit 1; }
            echo "[$(date +%T)] memp eval: arm=$ARM k=$TOP_K_MEMORIES limit=$LIMIT " \
                 "granularity=$GRANULARITY provenance=$PROVENANCE update=$UPDATE"
            python -u scripts/run_memp.py --phase eval "${COMMON_ARGS[@]}" \
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
    # Two phases only: the train build pass and the embeddings are Memp's, reused
    # verbatim (both methods key on the identical puzzle_key_text), so there is
    # nothing here to re-run for them.
    case "$PHASE" in
      distill)
        if [[ -n "$RESUME" ]]; then
            echo "[$(date +%T)] memorybank distill: resuming $RESUME"
            python -u scripts/run_memorybank.py --phase distill --resume "$RESUME" \
                --concurrency "$CONCURRENCY" --chunk "$CHUNK"
        else
            [[ -n "$BUILD" ]] || { echo "PHASE=distill needs BUILD=<memp_build_*.json> or RESUME=<bank file>" >&2; exit 1; }
            echo "[$(date +%T)] memorybank distill: $BUILD -> case bank (rules_preamble=$RULES_PREAMBLE)"
            python -u scripts/run_memorybank.py --phase distill "${COMMON_ARGS[@]}" \
                --build "$BUILD" --rules-preamble "$RULES_PREAMBLE" \
                --distill-max-tokens "$DISTILL_MAX_TOKENS" \
                --chunk "$CHUNK" --limit "$DISTILL_LIMIT" ${OUT:+--out "$OUT"}
        fi
        ;;
      eval)
        if [[ -n "$RESUME" ]]; then
            echo "[$(date +%T)] memorybank eval: resuming $RESUME"
            python -u scripts/run_memorybank.py --phase eval --resume "$RESUME" \
                --concurrency "$CONCURRENCY" --chunk "$CHUNK"
        else
            [[ -n "$BANK" && -n "$EMBEDDINGS" ]] || { echo "PHASE=eval needs BANK=<memorybank_bank_*.json> and EMBEDDINGS=<memp_embeddings_*.npz>" >&2; exit 1; }
            [[ -n "$ARM" ]] || { echo "PHASE=eval needs ARM=retrieved or ARM=random" >&2; exit 1; }
            echo "[$(date +%T)] memorybank eval: arm=$ARM k=$TOP_K_MEMORIES limit=$LIMIT " \
                 "granularity=$MB_GRANULARITY provenance=$PROVENANCE"
            python -u scripts/run_memorybank.py --phase eval "${COMMON_ARGS[@]}" \
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
  *)
    echo "unknown DRIVER=$DRIVER (expected: baseline | reflexion | memp | memorybank)" >&2; exit 1 ;;
esac

echo "[$(date +%T)] done; the cleanup trap stops vLLM and frees the GPUs."

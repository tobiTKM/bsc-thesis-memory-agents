#!/bin/bash
# The contrastive campaign: DOs distilled from the actor's successes, DON'Ts from
# its failures, on SPaRC and StepGame. See docs/contrastive.md.
#
# Mirrors run_stepgame_campaign.sh and run_memorybank_campaign.sh because it mirrors
# their experiment -- same actors, same seed and sampling, same paired exact-McNemar
# contrasts -- so one BENCH knob serves both benchmarks rather than two scripts that
# drift apart.
#
#   BENCH=sparc    bash scripts/run_contrastive_campaign.sh distill   # 1 job
#   BENCH=sparc    bash scripts/run_contrastive_campaign.sh exemplars # NO GPU (login node)
#   BENCH=sparc    bash scripts/run_contrastive_campaign.sh wave2     # 4 jobs
#   BENCH=sparc    bash scripts/run_contrastive_campaign.sh standard  # the reported config
#   BENCH=sparc    bash scripts/run_contrastive_campaign.sh eval      # 5 jobs
#   BENCH=stepgame bash scripts/run_contrastive_campaign.sh distill   # 1 job
#   BENCH=stepgame bash scripts/run_contrastive_campaign.sh eval      # 5 jobs
#                  bash scripts/run_contrastive_campaign.sh compare   # NO GPU
#   bash scripts/run_contrastive_campaign.sh status                   # what exists
#
# EXCLUDE="ggpu195" keeps jobs off named nodes; ONLY="hbo aex" restricts the wave2
# stage to those arms (a partial retry). DRYRUN=1 prints instead of submitting.
#
# DRYRUN=1 prints the submit lines instead of submitting. Use it after any edit:
# every line names files that must match across stages, and a typo in an EMBEDDINGS
# or BANK path stays invisible until the numbers disagree.
#
# MODELS="qwen32b" restricts a stage. The defaults are deliberate, not laziness:
#   SPaRC    -> gptoss120b. The only actor on which ANY memory method helps
#               (README headline 1); the other two are pre-registered as
#               uninterpretable here (docs/contrastive.md 6.5).
#   StepGame -> qwen32b. gpt-oss is at ceiling and was DROPPED (docs/stepgame.md 6
#               risk 1); qwen06b sits at the chance floor at depth. qwen32b is also
#               the one bank whose polarity runs the right way (64.6% positive).
#
# DEP_DISTILL_<TAG>=<jobid> chains eval off an in-flight distill instead of waiting
# for the bank to land, exactly as run_stepgame_campaign.sh's DEP_BUILD_<TAG> does.
set -euo pipefail
cd "$(dirname "$0")"/..

STAGE="${1:-}"
BENCH="${BENCH:-sparc}"

# The ag_gipp PROJECT store sits near its 7M-inode quota, which makes load_dataset's
# lock file creation fail (see rerun_memp_v2.sh for the diagnosis). SPaRC loads a HF
# dataset on every phase, so this is load-bearing there and belt-and-braces here.
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/mnt/vast-standard/home/t.mark/u21167/hf_datasets}"
[[ -d "$HF_DATASETS_CACHE" ]] || { echo "no dataset cache at $HF_DATASETS_CACHE" >&2; exit 1; }
ENVX="HF_DATASETS_CACHE=$HF_DATASETS_CACHE"

case "$BENCH" in
  sparc)
    BENCHX="BENCHMARK=sparc"
    PTAG=""
    BM=""                                   # filename infix: SPaRC files carry none
    EMB="results/json/memp_embeddings_qwen3emb06b.npz"
    BUILD_FMT="results/json/memp_build_train_%s.json"
    DEFAULT_MODELS="gptoss120b"
    ;;
  stepgame)
    SPLITS=data/stepgame/splits
    # Identical to run_stepgame_campaign.sh's, and they MUST stay identical: the
    # bank is built from that campaign's build file, and an eval scoring a different
    # test split than the arms it is compared against would still look internally
    # consistent while being unpaired.
    TEST="${TEST:-$SPLITS/stepgame_clean_test_k1-10_n1000.jsonl}"
    TRAIN="${TRAIN:-$SPLITS/stepgame_clean_train_byhops_k1-5_n500.jsonl}"
    PROMPT_PROTOCOL="${PROMPT_PROTOCOL:-upstream-header}"
    BENCHX="BENCHMARK=stepgame,STEPGAME_TRAIN=$TRAIN,STEPGAME_TEST=$TEST,PROMPT_PROTOCOL=$PROMPT_PROTOCOL"
    PTAG=""; [[ "$PROMPT_PROTOCOL" == "ours" ]] || PTAG="_$PROMPT_PROTOCOL"
    BM="stepgame_"
    BENCH_FLAGS="--stepgame-train $TRAIN --stepgame-test $TEST --prompt-protocol $PROMPT_PROTOCOL"
    EMB="results/json/memp_embeddings_stepgame.npz"
    BUILD_FMT="results/json/memp_build_stepgame_train_%s${PTAG}.json"
    DEFAULT_MODELS="qwen32b"
    ;;
  *) echo "BENCH must be sparc or stepgame (got '$BENCH')" >&2; exit 1 ;;
esac
KEYX="KEY_VARIANT=${KEY_VARIANT:-structural}"

# tag|model|TP|gpus|extra serve env -- rerun_memp_v2.sh's, verbatim.
ALL_MODELS=(
  "qwen06b|Qwen/Qwen3-0.6B|1|A100:1|"
  "qwen32b|Qwen/Qwen3-32B|2|A100:2|"
  "gptoss120b|openai/gpt-oss-120b|4|A100:4|REASONING_PARSER=openai_gptoss"
)
keep=()
for spec in "${ALL_MODELS[@]}"; do
  for want in ${MODELS:-$DEFAULT_MODELS}; do
    [[ "${spec%%|*}" == "$want" ]] && keep+=("$spec")
  done
done
[[ ${#keep[@]} -gt 0 ]] || { echo "MODELS matched no known tag" >&2; exit 1; }
ALL_MODELS=("${keep[@]}")

# Per-stage walltimes, sized from what the completed SPaRC campaign actually took
# rather than from its limits -- an oversized limit blocks backfill on a saturated
# partition, and every phase checkpoints so an overrun resumes with RESUME=<file>.
# Observed on Qwen3-32B/SPaRC: memp distill 0:15-0:24, memorybank distill 0:34-0:37,
# evals 9:01-12:52. This method's distill is ~2x Memp's (a strategy for every solved
# train task AND both an avoid and a hindsight text for every failed one -- 959 calls
# on qwen32b against Memp's 500) and its block is ~800 tokens, between Memp's script
# and MemoryBank's ~1.8k case-record.
t_of() {   # t_of <stage> <tag>
  case "$BENCH:$1:$2" in
    sparc:distill:qwen06b)       echo 03:00:00 ;;
    sparc:distill:qwen32b)       echo 03:00:00 ;;
    sparc:distill:gptoss120b)    echo 05:00:00 ;;
    sparc:eval:qwen06b)          echo 03:00:00 ;;
    sparc:eval:qwen32b)          echo 14:00:00 ;;
    sparc:eval:gptoss120b)       echo 06:00:00 ;;
    stepgame:distill:qwen06b)    echo 02:00:00 ;;
    stepgame:distill:qwen32b)    echo 04:00:00 ;;
    stepgame:distill:gptoss120b) echo 03:00:00 ;;
    stepgame:eval:qwen06b)       echo 03:00:00 ;;
    stepgame:eval:qwen32b)       echo 08:00:00 ;;
    stepgame:eval:gptoss120b)    echo 05:00:00 ;;
    *) echo "no walltime for $BENCH stage '$1' tag '$2'" >&2; exit 1 ;;
  esac
}

bank_of() { printf 'results/json/contrastive_bank_%s%s%s%s.json' "$BM" "$1" "$PTAG" "${2:-}"; }

# EXCLUDE="ggpu195 ggpu142" keeps a job off named nodes. A node can be `draining`
# with a failed health check (NHC: check_fs_mount) and still accept jobs, which kills
# them in ~20 s with no log file at all -- that is what took 15949587/88 on
# 2026-09-16. `sinfo -p scc-gpu -t drain` names the current offenders.
EXCL=""
[[ -n "${EXCLUDE:-}" ]] && EXCL="--exclude=$(echo "$EXCLUDE" | tr ' ' ',')"

# Job names carry the benchmark: both benchmarks run the same arms on the same actor
# tags, so without this `squeue` shows two identical `co-sret-qwen06b` rows that write
# to entirely different files.
JPFX="co"; [[ "$BENCH" == "stepgame" ]] && JPFX="cosg"

sub() {   # sub <jobname-suffix> <gpus> <time> <dep-or-empty> <exports...>
  local name="$JPFX-$1" gpus=$2 tlimit=$3 dep=$4; shift 4
  if [[ "${DRYRUN:-0}" == "1" ]]; then
    printf 'sbatch -J %s -G %s -t %s %s%s--export=ALL,%s scripts/serve_and_run.sbatch\n' \
           "$name" "$gpus" "$tlimit" "${dep:+--dependency=afterok:$dep }" \
           "${EXCL:+$EXCL }" "$*"
  else
    sbatch -J "$name" -G "$gpus" -t "$tlimit" ${dep:+--dependency=afterok:$dep} \
      $EXCL --export=ALL,"$*" scripts/serve_and_run.sbatch
  fi
}

# granularity|arm|jobname-infix. The five core arms of docs/contrastive.md section 3.
#
# `hindsight` is the CONTROL, not an extra: it renders the same bank's failed entries
# as a gold-derived procedure (Memp's policy, our renderer), so avoid-vs-hindsight is
# matched on bank, entry count, retriever, k, positive half AND tokens to within 2%.
# Dropping it would leave the method with nothing to be tested against.
#
# `balanced` is not a retrieval nicety either. A bank's DO/DON'T mix IS the actor's
# train accuracy -- 71/429 on gpt-oss SPaRC -- so a plain top-3 injects ~0.4 DO
# entries per item and most items see no worked example at all, which is the one
# ingredient this campaign found active. Without this arm a null on `avoid` cannot
# be read (docs/contrastive.md section 2).
ARMS=(
  "avoid|retrieved|avo"
  "hindsight|retrieved|hin"
  "positive-only|retrieved|pos"
  "avoid|balanced|bal"
  "avoid|random|rnd"
)

# Wave 2 (2026-09-15). docs/contrastive.md 6.6 left ONE thing unexplained: the
# `hindsight` control landed 10 puzzles below Memp's script arm (80 vs 90, p=0.289)
# on matched text, matched bank and matched delivery. Two renderer differences remain,
# both introduced here, and this stage is the 2x2 that separates them:
#
#   exemplars  Memp's `script` attaches a verified worked example to every entry and
#              this method attached none. The campaign's most robust finding is that
#              a correct worked example is the active ingredient, so this is the
#              leading suspect.
#   framing    `hindsight` renders every item as a DO but inherited the two-sign
#              header, which promises failures its block never contains -- i.e. the
#              control was handicapped relative to the method it controls for.
#
# The banks are NOT rebuilt: `--phase exemplars` attaches the artifacts in code, so
# every strategy/avoid/hindsight text stays byte-identical to the arms already run
# and the only thing that moves is the artifact. Verified in the stage below.
# gran|arm|exemplars|framing|infix
WAVE2=(
  "hindsight|retrieved|on|mixed|hex"     # exemplar alone
  "hindsight|retrieved|off|matched|hmf"  # header alone
  "hindsight|retrieved|on|matched|hbo"   # the control at its best
  "avoid|retrieved|on|mixed|aex"         # the method at its best (its header is already correct)
)

case "$STAGE" in
  distill)
    for spec in "${ALL_MODELS[@]}"; do
      IFS='|' read -r tag model tp gpus extra <<< "$spec"
      build=$(printf "$BUILD_FMT" "$tag")
      [[ -f "$build" ]] || { echo "skipping $tag: no $build" >&2; continue; }
      sub "dist-$tag" "$gpus" "$(t_of distill "$tag")" "" \
        "$ENVX,$BENCHX,$KEYX,DRIVER=contrastive,PHASE=distill,BUILD=$build,MODEL=$model,TP=$tp,${extra:+$extra,}OUT=$(bank_of "$tag")"
    done
    ;;

  eval)
    [[ -f "$EMB" ]] || { echo "no $EMB (run the memp embed phase first)" >&2; exit 1; }
    for spec in "${ALL_MODELS[@]}"; do
      IFS='|' read -r tag model tp gpus extra <<< "$spec"
      bank=$(bank_of "$tag")
      depvar="DEP_DISTILL_$(echo "$tag" | tr '[:lower:]-' '[:upper:]_')"
      dep="${!depvar:-}"
      [[ -f "$bank" || -n "$dep" ]] || {
        echo "skipping $tag: no $bank and no $depvar=<jobid>" >&2; continue; }
      for a in "${ARMS[@]}"; do
        IFS='|' read -r gran arm infix <<< "$a"
        out="results/json/contrastive_eval_${BM}${gran}_${arm}_test_${tag}${PTAG}.json"
        sub "$infix-$tag" "$gpus" "$(t_of eval "$tag")" "$dep" \
          "$ENVX,$BENCHX,$KEYX,DRIVER=contrastive,PHASE=eval,CO_ARM=$arm,CO_GRANULARITY=$gran,MIN_POSITIVE=${MIN_POSITIVE:-1},LIMIT=0,BANK=$bank,EMBEDDINGS=$EMB,MODEL=$model,TP=$tp,${extra:+$extra,}OUT=$out"
      done
    done
    ;;

  exemplars)
    # CPU, no model, no GPU -- run on the login node.
    for spec in "${ALL_MODELS[@]}"; do
      IFS='|' read -r tag model tp gpus extra <<< "$spec"
      bank=$(bank_of "$tag"); build=$(printf "$BUILD_FMT" "$tag")
      [[ -f "$bank" ]] || { echo "skipping $tag: no $bank" >&2; continue; }
      python scripts/run_contrastive.py --phase exemplars --benchmark "$BENCH" \
        ${BENCH_FLAGS:-} --bank "$bank" --build "$build" \
        --out "$(bank_of "$tag" _ex)"
    done
    ;;

  wave2)
    for spec in "${ALL_MODELS[@]}"; do
      IFS='|' read -r tag model tp gpus extra <<< "$spec"
      bank=$(bank_of "$tag" _ex)
      [[ -f "$bank" ]] || { echo "skipping $tag: no $bank (run the exemplars stage)" >&2; continue; }
      for a in "${WAVE2[@]}"; do
        IFS='|' read -r gran arm ex fr infix <<< "$a"
        # ONLY="hbo aex" re-runs just those arms, for a partial retry after a node
        # failure without re-burning the ones that already landed.
        if [[ -n "${ONLY:-}" ]] && [[ " ${ONLY} " != *" $infix "* ]]; then continue; fi
        sfx=""; [[ "$ex" == "on" ]] && sfx="${sfx}_ex"
        [[ "$fr" == "matched" ]] && sfx="${sfx}_mf"
        out="results/json/contrastive_eval_${BM}${gran}_${arm}_test_${tag}${PTAG}${sfx}.json"
        sub "$infix-$tag" "$gpus" "$(t_of eval "$tag")" "" \
          "$ENVX,$BENCHX,$KEYX,DRIVER=contrastive,PHASE=eval,CO_ARM=$arm,CO_GRANULARITY=$gran,CO_EXEMPLARS=$ex,CO_FRAMING=$fr,MIN_POSITIVE=${MIN_POSITIVE:-1},LIMIT=0,BANK=$bank,EMBEDDINGS=$EMB,MODEL=$model,TP=$tp,${extra:+$extra,}OUT=$out"
      done
    done
    ;;

  standard)
    # The method's standard configuration, fixed 2026-09-21: `avoid` with the worked
    # exemplar attached (docs/contrastive.md 6.6 -- omitting it cost ~20 puzzles on
    # gpt-oss). Runs the retrieval arms the wave-1 set had, now all at that config, so
    # the relevance and polarity ablations test the arm the thesis actually reports
    # rather than a superseded one. `positive-only` comes along because it is the
    # "what do the failed entries contribute at all" contrast and has to match.
    #
    # `hindsight` is NOT re-run here: it is already on disk at its own best config, and
    # framing=matched is a no-op for every arm except hindsight, so these all carry the
    # plain `_ex` suffix.
    #
    # ONLY="bal rnd" restricts it, as in wave2.
    for spec in "${ALL_MODELS[@]}"; do
      IFS='|' read -r tag model tp gpus extra <<< "$spec"
      # Banks distilled from 2026-09-21 carry the artifacts already; older ones were
      # retrofitted by `--phase exemplars` into a sibling `_ex` file.
      bank=$(bank_of "$tag" _ex); [[ -f "$bank" ]] || bank=$(bank_of "$tag")
      depvar="DEP_DISTILL_$(echo "$tag" | tr '[:lower:]-' '[:upper:]_')"; dep="${!depvar:-}"
      [[ -f "$bank" || -n "$dep" ]] || { echo "skipping $tag: no bank and no $depvar=<jobid>" >&2; continue; }
      for a in "avoid|retrieved|ret" "avoid|balanced|bal" "avoid|random|rnd" \
               "positive-only|retrieved|pos" "hindsight|retrieved|hin"; do
        IFS='|' read -r gran arm infix <<< "$a"
        if [[ -n "${ONLY:-}" ]] && [[ " ${ONLY} " != *" $infix "* ]]; then continue; fi
        fr=mixed; sfx=_ex
        # framing=matched is a no-op for every arm except `hindsight`, whose items are
        # all DO and whose two-sign header would otherwise promise failures it has none
        # of (docs/contrastive.md 6.6).
        [[ "$gran" == "hindsight" ]] && { fr=matched; sfx=_ex_mf; }
        out="results/json/contrastive_eval_${BM}${gran}_${arm}_test_${tag}${PTAG}${sfx}.json"
        [[ -f "$out" && "${FORCE:-0}" != "1" ]] && { echo "skipping $infix-$tag: $out exists (FORCE=1 to redo)" >&2; continue; }
        sub "s${infix}-$tag" "$gpus" "$(t_of eval "$tag")" "$dep" \
          "$ENVX,$BENCHX,$KEYX,DRIVER=contrastive,PHASE=eval,CO_ARM=$arm,CO_GRANULARITY=$gran,CO_EXEMPLARS=on,CO_FRAMING=$fr,MIN_POSITIVE=${MIN_POSITIVE:-1},LIMIT=0,BANK=$bank,EMBEDDINGS=$EMB,MODEL=$model,TP=$tp,${extra:+$extra,}OUT=$out"
      done
    done
    ;;

  balanced-fix)
    # Re-run of the `balanced` arm after the 2026-09-16 ordering fix. The first run
    # re-sorted the block by entry id while `retrieved` used similarity order, so on
    # every item whose plain top-k already held a DO the two arms injected the SAME
    # entries in a DIFFERENT order -- and 7 of that arm's 10-puzzle margin sat on
    # those items. The output name carries `_v2` so the confounded run stays on disk
    # rather than being overwritten by the thing that supersedes it.
    for spec in "${ALL_MODELS[@]}"; do
      IFS='|' read -r tag model tp gpus extra <<< "$spec"
      bank=$(bank_of "$tag")
      [[ -f "$bank" ]] || { echo "skipping $tag: no $bank" >&2; continue; }
      out="results/json/contrastive_eval_${BM}avoid_balanced_test_${tag}${PTAG}_v2.json"
      sub "bal2-$tag" "$gpus" "$(t_of eval "$tag")" "" \
        "$ENVX,$BENCHX,$KEYX,DRIVER=contrastive,PHASE=eval,CO_ARM=balanced,CO_GRANULARITY=avoid,MIN_POSITIVE=${MIN_POSITIVE:-1},LIMIT=0,BANK=$bank,EMBEDDINGS=$EMB,MODEL=$model,TP=$tp,${extra:+$extra,}OUT=$out"
    done
    ;;

  compare)
    for spec in "${ALL_MODELS[@]}"; do
      IFS='|' read -r tag _m _tp _g _e <<< "$spec"
      e() { printf 'results/json/contrastive_eval_%s%s_%s_test_%s%s.json' \
                   "$BM" "$1" "$2" "$tag" "$PTAG"; }
      avoid=$(e avoid retrieved)
      [[ -f "$avoid" ]] || { echo "skipping $tag: no $avoid" >&2; continue; }
      if [[ "$BENCH" == "sparc" ]]; then
        base="results/json/reflexion_root_${tag}_test.json"
        memp="results/json/memp_eval_retrieved_test_${tag}_v2.json"
      else
        base="results/json/reflexion_root_stepgame_${tag}_test${PTAG}.json"
        memp="results/json/memp_eval_stepgame_retrieved_test_${tag}${PTAG}.json"
      fi
      [[ -f "$base" ]] || { echo "skipping $tag: no baseline $base" >&2; continue; }
      echo "=== contrastive $BENCH $tag ==="
      python scripts/run_contrastive.py --phase compare \
        --baseline "$base" --retrieved "$avoid" \
        $([[ -f "$(e hindsight retrieved)" ]]     && echo --hindsight "$(e hindsight retrieved)") \
        $([[ -f "$(e positive-only retrieved)" ]] && echo --positive  "$(e positive-only retrieved)") \
        $([[ -f "$(e avoid random)" ]]            && echo --random    "$(e avoid random)") \
        $([[ -f "$memp" ]]                        && echo --memp      "$memp") \
        --out "results/txt/contrastive_vs_baseline_${BM}${tag}${PTAG}.txt"
      # The balanced arm is a separate comparison: it differs from `avoid` in the
      # retrieval policy, not in the content, so it is not one of the content arms.
      if [[ -f "$(e avoid balanced)" ]]; then
        echo "=== contrastive $BENCH $tag: balanced retrieval ==="
        python scripts/run_contrastive.py --phase compare \
          --baseline "$base" --retrieved "$(e avoid balanced)" \
          $([[ -f "$(e hindsight retrieved)" ]] && echo --hindsight "$(e hindsight retrieved)") \
          --out "results/txt/contrastive_balanced_${BM}${tag}${PTAG}.txt"
      fi
    done
    ;;

  status)
    printf '%-14s %-22s %s\n' BENCH ARTEFACT STATE
    for b in sparc stepgame; do
      bm=""; pt=""
      [[ "$b" == stepgame ]] && { bm="stepgame_"; pt="_upstream-header"; }
      for spec in "${ALL_MODELS[@]}"; do
        IFS='|' read -r tag _r <<< "$spec"
        f="results/json/contrastive_bank_${bm}${tag}${pt}.json"
        [[ -f "$f" ]] && printf '%-14s %-22s %s\n' "$b" "bank $tag" "$(du -h "$f" | cut -f1)"
      done
      for f in results/json/contrastive_eval_${bm}*.json; do
        [[ -e "$f" ]] || continue
        n=$(python3 -c "import json,sys;d=json.load(open(sys.argv[1]));s=d.get('summary',{});print(f\"{s.get('success','?')}/{s.get('n','?')} solved\" if s else f\"{len(d.get('puzzles',{}))} done (running)\")" "$f" 2>/dev/null || echo '?')
        printf '%-14s %-22s %s\n' "$b" "$(basename "$f" .json | sed "s/contrastive_eval_${bm}//;s/_test_/ /")" "$n"
      done
    done
    ;;

  *)
    sed -n '2,32p' "$0"; exit 1 ;;
esac

# Reflexion on SPaRC — setup and results

Reflexion (Shinn et al., 2023) applied to the SPaRC spatial-reasoning benchmark, over
three models, with a trial-matched control arm. Source data: `results/json/reflexion_*`,
summarised in `results/txt/reflexion/reflexion_all_models.txt`.

**Headline:** Reflexion produces a small benefit on the strongest model only, and that
benefit arrives entirely in the *first* reflection round. Trials 3–10 are
indistinguishable from plain resampling on every model tested.

---

## 1. Setup

### The method as implemented

Reflexion's loop (Algorithm 1 in the paper): attempt a task; while it is unsolved and
trials remain, turn the failed trajectory into a short verbal **reflection**, append it
to that task's episodic memory, and re-attempt the **same** task with that memory in
context. Faithful to the paper on every point that matters:

| Paper | Here |
|---|---|
| Actor | Unchanged SPaRC baseline agent; the memory block is the only difference |
| Evaluator | The environment itself (ground-truth `outcome_reward == 1`) — the paper's external-feedback variant, stronger than its self-evaluated AlfWorld setup |
| Self-reflection | One LLM call per failed episode: full move trajectory, the final grid, per-rule pass/fail, two few-shot examples, **plus the plans already tried** |
| Memory scope | Per-puzzle. A lesson about puzzle X is only ever shown when re-attempting X |
| Memory bound Ω | Sliding window of the **last 3** reflections (the paper's "usually 1–3") |
| Skip solved | Solved puzzles are never re-attempted (the reference implementation's `skip` flag) |
| Trials | **10**, matching the paper's decision-making experiments |

### Experimental design

Each model contributes three runs: one shared **root** (trial 1) and two arms.

```
root (trial 1, memoryless)
  ├── memory arm   trials 2-10, reflections injected
  └── control arm  trials 2-10, NO memory, no reflections generated
```

Three properties make the comparison readable:

1. **Shared trial 1.** Both arms resume from the *same* root file, so they begin from
   identical episodes rather than two independent samples. Verified: both arms solve an
   identical trial-1 puzzle set.
2. **Matched per-trial seeds.** Trial *t* decodes at `base_seed + t - 1` in both arms.
   Distinct per trial is load-bearing — the control sends a byte-identical prompt every
   trial, so a fixed seed would replay trial 1 rather than resample. Identical across
   arms leaves the injected block as the only difference.
3. **The control mirrors the trial count.** Only unsolved puzzles are re-attempted and
   solved ones are carried forward, so **cumulative accuracy can only rise** — an
   upward curve is guaranteed with or without memory.

> **Why the control is not optional.** gpt-oss-120b climbs 14.0% → 35.8% across ten
> trials, a 2.6× gain that looks decisive in isolation. Its control reaches 31.8% doing
> nothing but resampling. Roughly **80% of the apparent gain is retry, not memory.**
> Without the control arm all three models would have looked like clear successes.

### Configuration

| | |
|---|---|
| Benchmark | SPaRC, `test` split, all 500 puzzles |
| Models | gpt-oss-120b (TP=4), Qwen3-32B (TP=4), Qwen3-0.6B (TP=1) |
| Serving | vLLM in Apptainer, 1× A100 node, concurrency 64 |
| Sampling | temp 0.6, top_p 0.95, top_k 20, max_tokens 12288 |
| Seeds | trial *t* = `t - 1`, identical across arms |
| Memory window | 3 reflections per puzzle |
| Reliability | **0 crashed episodes** across all 1,500 root episodes and both arms |

---

## 2. Results

### Headline: memory vs control at trial 10

| Model | Trial 1 | Memory | Control | Delta |
|---|---|---|---|---|
| gpt-oss-120b | 14.0% | **35.8%** | 31.8% | **+4.0pp** |
| Qwen3-32B | 11.0% | 25.6% | 26.6% | −1.0pp |
| Qwen3-0.6B | 2.8% | 11.6% | 10.0% | +1.6pp |

### Baseline vs control vs memory, at trials 1 / 2 / 10

The three reference points in one table. Trial 1 is the shared memoryless baseline and
is identical for both arms by construction; trial 2 is after a single reflection; trial
10 is the end state. Accuracy over all 500 puzzles, solved counts in brackets.

| Model | Baseline (T1) | Control @T2 | Memory @T2 | Delta @T2 | Control @T10 | Memory @T10 | Delta @T10 |
|---|---|---|---|---|---|---|---|
| gpt-oss-120b | 14.0% (70) | 19.6% (98) | 24.0% (120) | **+4.4pp** | 31.8% (159) | 35.8% (179) | **+4.0pp** |
| Qwen3-32B | 11.0% (55) | 15.4% (77) | 15.8% (79) | +0.4pp | 26.6% (133) | 25.6% (128) | −1.0pp |
| Qwen3-0.6B | 2.8% (14) | 5.2% (26) | 5.8% (29) | +0.6pp | 10.0% (50) | 11.6% (58) | +1.6pp |

Read the two delta columns against each other: on gpt-oss the advantage is **+4.4pp
after one reflection and +4.0pp after nine more**. Eight additional trials did not widen
it. The same table also shows how much the control alone achieves — 14.0% → 31.8% on
gpt-oss, purely by retrying.

Decomposing where each arm's puzzles come from makes the point arithmetically:

| Model | Trial 1 | Gained at T2 (mem / ctl) | Gained over T3–10 (mem / ctl) | Final (mem / ctl) |
|---|---|---|---|---|
| gpt-oss-120b | 70 | **+50 / +28** | +59 / +61 | 179 / 159 |
| Qwen3-32B | 55 | +24 / +22 | +49 / +56 | 128 / 133 |
| Qwen3-0.6B | 14 | +15 / +12 | +29 / +24 | 58 / 50 |

gpt-oss's entire 20-puzzle advantage is bought at trial 2 (+50 vs +28); across trials
3–10 the memory arm actually gains **fewer** puzzles than its control (59 vs 61). The
final margin is the trial-2 margin, carried forward.

### The effect is entirely in the first reflection

Per-trial conversion rate on the puzzles each arm actually re-attempted:

| Model | Trial 2 memory | Trial 2 control | Trials 3–10 memory | Trials 3–10 control |
|---|---|---|---|---|
| gpt-oss-120b | **11.6%** | 6.5% | 2.12% | 2.07% |
| Qwen3-32B | 5.4% | 4.9% | 1.56% | 1.79% |
| Qwen3-0.6B | 3.1% | 2.5% | 0.79% | 0.65% |

At trial 2, gpt-oss nearly **doubles** the control's conversion rate. From trial 3 on,
the arms are indistinguishable on every model. gpt-oss's delta is +0.044 at trial 2 and
+0.040 at trial 10 — the advantage is won once and then merely held.

**This is not a memory-staleness artifact.** A fresh reflection is generated every trial
for every unsolved puzzle (430, 380, 365, … , 323 on gpt-oss), with zero empty and zero
failed generations. The reflections are also textually new each time: consecutive
reflections for the same puzzle share only **0.087** similarity, with no near-duplicates
across 745 pairs.

They are new *wording* for the same *insight*. Strategy-keyword overlap between
consecutive reflections is **0.554**, and inspection shows the pattern directly:

> "let the trail **cut off** the lower-left pocket before verifying a reachable dot remained"
> "let the trail **seal off** a section before guaranteeing every unvisited mandatory cell…"
> "let the trail **close off** a narrow passage before guaranteeing the remaining cells…"

The model re-derives one correct lesson — *don't enclose a region before checking
reachability* — and restates it every trial. The first statement is informative; the
rest add no information and so convert nothing the first did not.

**Reflection quality is not the bottleneck; reflection *diversity* is.**

### Full trial curves

Cumulative accuracy (solved at least once by trial *t*) for both arms, with the puzzles
each arm newly converted in that trial. Trial 2 — the only trial where the arms
genuinely diverge — is bolded. Both curves are monotonic by construction.

**gpt-oss-120b** (n=500)

| Trial | Memory | Control | Delta | New (mem) | New (ctl) |
|---|---|---|---|---|---|
| 1 | 0.140 | 0.140 | +0.000 | 70 | 70 |
| 2 | 0.240 | 0.196 | **+0.044** | **50** | **28** |
| 3 | 0.270 | 0.226 | +0.044 | 15 | 15 |
| 4 | 0.288 | 0.246 | +0.042 | 9 | 10 |
| 5 | 0.298 | 0.268 | +0.030 | 5 | 11 |
| 6 | 0.312 | 0.282 | +0.030 | 7 | 7 |
| 7 | 0.334 | 0.288 | +0.046 | 11 | 3 |
| 8 | 0.344 | 0.298 | +0.046 | 5 | 5 |
| 9 | 0.354 | 0.312 | +0.042 | 5 | 7 |
| 10 | 0.358 | 0.318 | +0.040 | 2 | 3 |

The delta is established at trial 2 (+0.044) and ends at +0.040 — it never grows. The
`New` columns show why: 50 vs 28 at trial 2, then near-parity for the rest of the run.

**Qwen3-32B** (n=500)

| Trial | Memory | Control | Delta | New (mem) | New (ctl) |
|---|---|---|---|---|---|
| 1 | 0.110 | 0.110 | +0.000 | 55 | 55 |
| 2 | 0.158 | 0.154 | **+0.004** | **24** | **22** |
| 3 | 0.184 | 0.178 | +0.006 | 13 | 12 |
| 4 | 0.202 | 0.200 | +0.002 | 9 | 11 |
| 5 | 0.216 | 0.214 | +0.002 | 7 | 7 |
| 6 | 0.224 | 0.240 | −0.016 | 4 | 13 |
| 7 | 0.232 | 0.250 | −0.018 | 4 | 5 |
| 8 | 0.240 | 0.260 | −0.020 | 4 | 5 |
| 9 | 0.252 | 0.264 | −0.012 | 6 | 2 |
| 10 | 0.256 | 0.266 | −0.010 | 2 | 1 |

Never meaningfully positive, and negative from trial 6 onward. The crossover is a single
trial: the control converted 13 puzzles at trial 6 against the memory arm's 4. With one
run per arm that is as consistent with a lucky control trial as with genuine
interference — a distinction only repeated seeds could settle.

**Qwen3-0.6B** (n=500)

| Trial | Memory | Control | Delta | New (mem) | New (ctl) |
|---|---|---|---|---|---|
| 1 | 0.028 | 0.028 | +0.000 | 14 | 14 |
| 2 | 0.058 | 0.052 | **+0.006** | **15** | **12** |
| 3 | 0.064 | 0.060 | +0.004 | 3 | 4 |
| 4 | 0.072 | 0.068 | +0.004 | 4 | 4 |
| 5 | 0.092 | 0.076 | +0.016 | 10 | 4 |
| 6 | 0.096 | 0.082 | +0.014 | 2 | 3 |
| 7 | 0.104 | 0.084 | +0.020 | 4 | 1 |
| 8 | 0.106 | 0.088 | +0.018 | 1 | 2 |
| 9 | 0.112 | 0.094 | +0.018 | 3 | 3 |
| 10 | 0.116 | 0.100 | +0.016 | 2 | 3 |

Consistently positive but on tiny counts — the whole +1.6pp is 8 puzzles, and the delta
moves by single conversions (the jump at trial 5 is 10 vs 4).

### Results per difficulty

SPaRC difficulty 1–5, n = 86 / 118 / 121 / 86 / 89.

**gpt-oss-120b** — where the effect lives:

| Difficulty | n | Trial 1 | Control | Memory | Delta |
|---|---|---|---|---|---|
| 1 | 86 | 38.4% | 75.6% | **82.6%** | +7.0pp |
| 2 | 118 | 17.8% | 44.9% | **48.3%** | +3.4pp |
| 3 | 121 | 9.9% | 24.8% | **33.1%** | **+8.3pp** |
| 4 | 86 | 3.5% | 10.5% | 10.5% | 0.0 |
| 5 | 89 | 1.1% | 2.2% | 2.2% | 0.0 |

The entire gain sits at **difficulties 1–3** (+20 puzzles). At difficulty 4–5 the arms
are *identical* — 9 and 2 puzzles each. Reflexion helps where the model is already
partly competent and does nothing where it is not: a verbal hint cannot substitute for
absent capability.

**Qwen3-32B** — memory is net negative, and the loss is concentrated at the extremes:

| Difficulty | n | Trial 1 | Control | Memory | Delta |
|---|---|---|---|---|---|
| 1 | 86 | 32.6% | **67.4%** | 60.5% | −6.9pp |
| 2 | 118 | 13.6% | 35.6% | **37.3%** | +1.7pp |
| 3 | 121 | 6.6% | 20.7% | **23.1%** | +2.4pp |
| 4 | 86 | 3.5% | **7.0%** | 4.7% | −2.3pp |
| 5 | 89 | 0.0% | **2.2%** | 0.0% | −2.2pp |

**Qwen3-0.6B** — favourable but tiny in absolute terms (58 vs 50 puzzles):

| Difficulty | n | Trial 1 | Control | Memory | Delta |
|---|---|---|---|---|---|
| 1 | 86 | 4.7% | 24.4% | **30.2%** | +5.8pp |
| 2 | 118 | 5.9% | 12.7% | **14.4%** | +1.7pp |
| 3 | 121 | 2.5% | **9.9%** | 9.1% | −0.8pp |
| 4 | 86 | 0.0% | 2.3% | **3.5%** | +1.2pp |
| 5 | 89 | 0.0% | 0.0% | **1.1%** | +1.1pp |

### Per-difficulty at trial 2 — where the effect is created

The tables above are the *end state*. Since the whole effect is produced at trial 2, the
more diagnostic view is per-difficulty at that trial, with the final delta alongside for
contrast. (Delta @T10 in italics repeats the tables above.)

**gpt-oss-120b**

| Difficulty | n | Trial 1 | Control @T2 | Memory @T2 | Delta @T2 | *(Delta @T10)* |
|---|---|---|---|---|---|---|
| 1 | 86 | 38.4% | 53.5% | 60.5% | **+7.0pp** | *+7.0pp* |
| 2 | 118 | 17.8% | 23.7% | 34.7% | **+11.0pp** | *+3.4pp* |
| 3 | 121 | 9.9% | 14.9% | 17.4% | +2.5pp | *+8.3pp* |
| 4 | 86 | 3.5% | 5.8% | 5.8% | 0.0 | *0.0* |
| 5 | 89 | 1.1% | 1.1% | 1.1% | 0.0 | *0.0* |

**Qwen3-32B**

| Difficulty | n | Trial 1 | Control @T2 | Memory @T2 | Delta @T2 | *(Delta @T10)* |
|---|---|---|---|---|---|---|
| 1 | 86 | 32.6% | 41.9% | 41.9% | 0.0 | *−7.0pp* |
| 2 | 118 | 13.6% | 19.5% | 22.9% | +3.4pp | *+1.7pp* |
| 3 | 121 | 6.6% | 10.7% | 10.7% | 0.0 | *+2.5pp* |
| 4 | 86 | 3.5% | 5.8% | 3.5% | −2.3pp | *−2.3pp* |
| 5 | 89 | 0.0% | 0.0% | 0.0% | 0.0 | *−2.2pp* |

**Qwen3-0.6B**

| Difficulty | n | Trial 1 | Control @T2 | Memory @T2 | Delta @T2 | *(Delta @T10)* |
|---|---|---|---|---|---|---|
| 1 | 86 | 4.7% | 14.0% | 12.8% | −1.2pp | *+5.8pp* |
| 2 | 118 | 5.9% | 7.6% | 11.9% | **+4.2pp** | *+1.7pp* |
| 3 | 121 | 2.5% | 3.3% | 2.5% | −0.8pp | *−0.8pp* |
| 4 | 86 | 0.0% | 1.2% | 1.2% | 0.0 | *+1.2pp* |
| 5 | 89 | 0.0% | 0.0% | 0.0% | 0.0 | *+1.1pp* |

Three things this view shows that the trial-10 tables do not:

**Difficulty 2 is where the first reflection pays off, on all three models** — the only
band positive for every model at trial 2 (+11.0, +3.4, +4.2pp). On gpt-oss that is 41
solved against the control's 28. This is the single most consistent cross-model signal
in the experiment.

**Difficulty 4–5 is dead on arrival.** Every model shows a delta of exactly 0.0 at both
levels at trial 2. The memory arm does not convert a single hard puzzle the control
misses at the point where reflection is most informative. Whatever Reflexion does, it
does not extend the difficulty frontier — it converts puzzles already near the model's
reach.

**gpt-oss's difficulty-3 gain accrues *later*, not at trial 2** (+2.5pp at T2 → +8.3pp
at T10). This is the one place where the extra trials appear to buy something the first
reflection did not, and it is the reason gpt-oss's overall delta holds up rather than
decaying. Note it rests on 40 vs 30 puzzles, so it is suggestive rather than established.

It also qualifies the 32B's negative result: at trial 2 the 32B is neutral-to-positive
everywhere except difficulty 4 (0.0, +3.4, 0.0, −2.3, 0.0). Its −7.0pp at difficulty 1
develops over later trials, which is consistent with the single-trial crossover noted
above rather than with memory being harmful from the outset.

### Failure modes

At trial 1, consistently across all three models, **dead-ends dominate**:

| Model | Dead-end | Wrong solution | Solved | Format failure |
|---|---|---|---|---|
| gpt-oss-120b | 227 (45%) | 203 (41%) | 70 | **0** |
| Qwen3-32B | 235 (47%) | 210 (42%) | 55 | 0 |
| Qwen3-0.6B | 303 (61%) | 176 (35%) | 14 | 7 |

Models lose the *path* (no legal move left) more often than they lose the *rules*
(reached the exit with a constraint violated).

#### How the failure mix moves under memory

Every puzzle classified by its state at each checkpoint — solved, or the failure mode of
its most recent attempt. Columns sum to 500. A **negative** delta on a failure row means
the memory arm has *fewer* of that failure, which is the good direction.

**gpt-oss-120b**

| Outcome | Baseline (T1) | Control @T2 | Memory @T2 | Delta @T2 | Control @T10 | Memory @T10 | Delta @T10 |
|---|---|---|---|---|---|---|---|
| solved | 70 | 98 | 120 | **+22** | 159 | 179 | **+20** |
| dead-end | 227 | 210 | 211 | +1 | 179 | 205 | **+26** |
| wrong-solution | 203 | 192 | 169 | **−23** | 162 | 115 | **−47** |
| format-failure | 0 | 0 | 0 | +0 | 0 | 1 | +1 |

**Qwen3-32B**

| Outcome | Baseline (T1) | Control @T2 | Memory @T2 | Delta @T2 | Control @T10 | Memory @T10 | Delta @T10 |
|---|---|---|---|---|---|---|---|
| solved | 55 | 77 | 79 | +2 | 133 | 128 | −5 |
| dead-end | 235 | 224 | 250 | **+26** | 200 | 260 | **+60** |
| wrong-solution | 210 | 199 | 171 | **−28** | 167 | 112 | **−55** |

**Qwen3-0.6B**

| Outcome | Baseline (T1) | Control @T2 | Memory @T2 | Delta @T2 | Control @T10 | Memory @T10 | Delta @T10 |
|---|---|---|---|---|---|---|---|
| solved | 14 | 26 | 29 | +3 | 50 | 58 | +8 |
| dead-end | 303 | 281 | 281 | +0 | 292 | 283 | −9 |
| wrong-solution | 176 | 186 | 179 | −7 | 146 | 154 | +8 |
| format-failure | 7 | 7 | 11 | +4 | 12 | 5 | −7 |

#### What memory actually changes

**Reflexion does not reduce the dominant failure — it trades one failure for another.**
On both larger models the memory arm ends with far *fewer* wrong solutions and
substantially *more* dead-ends:

| Model @T10 | Wrong solution | Dead-end | Solved |
|---|---|---|---|
| gpt-oss-120b | **−47** | +26 | **+20** |
| Qwen3-32B | **−55** | +60 | −5 |

The reflections warn about violating constraints — enclosing mismatched squares, cutting
off a required dot. The agent becomes **more conservative**: it stops barrelling to the
exit with an invalid path. Those avoided wrong-solutions have to go somewhere, and they
land either in a correct solve or in a corner with no legal move.

**That single behavioural change explains both the success and the failure.** The trade
is visible already at trial 2, and the two models resolve it oppositely:

- **gpt-oss @T2**: −23 wrong-solutions, **+22 solved**, dead-ends unchanged (+1). Nearly
  every avoided wrong solution became a *correct* one.
- **Qwen3-32B @T2**: −28 wrong-solutions, **+2 solved**, **+26 dead-ends**. Nearly every
  avoided wrong solution became a *stall*.

So Reflexion induces the same caution in both models; only gpt-oss can execute on it.
The 32B understands "don't seal off that region" well enough to stop committing to a bad
path, but not well enough to find a good one — so it hesitates into a dead end instead.
This is a more specific account of the earlier "memory needs competence to act on"
claim: the competence required is not diagnosing the error but *replanning* around it.

The 0.6B is too weak for the trade to appear at all — its deltas are single digits in
both directions, and its dominant failure (dead-end, 61% at baseline) barely moves.

> **Correction to an earlier reading.** We initially suggested the reflections help
> because they target dead-ends, the dominant failure. The data says the opposite:
> dead-ends *increase* under memory on both large models. What memory suppresses is the
> wrong-solution failure, and whether that is a gain or a loss depends entirely on
> whether the model can convert the freed-up attempts into correct paths.

---

## 3. Cost and efficiency

Trial 1 is counted once per model, since both arms share it.

| Model | GPUs | Root | Memory arm | Control arm | **Total GPU-h** |
|---|---|---|---|---|---|
| gpt-oss-120b | 4 | 3.8 | 36.7 | 20.7 | **61.2** |
| Qwen3-32B | 4 | 12.8 | 114.4 | 100.7 | **227.9** |
| Qwen3-0.6B | 1 | 1.9 | 12.8 | 14.6 | **29.2** |
| | | | | | **318.3** |

### Token cost

Total across the whole arm:

| Model | Memory arm | of which reflection | Control arm | Reflection share |
|---|---|---|---|---|
| gpt-oss-120b | 298.5M | 9.2M | 197.8M | 3.1% |
| Qwen3-32B | 376.5M | 8.8M | 320.8M | 2.3% |
| Qwen3-0.6B | 421.2M | 9.1M | 374.6M | 2.2% |

**Reflection generation is negligible — 2–3% of tokens.** The cost of Reflexion is not
writing reflections; it is that every injected reflection lengthens *every actor prompt
for every step of every subsequent trial*. gpt-oss's memory arm spent 100.7M more tokens
than its control, of which only 9.2M were reflections. The other 91M is inflated actor
prompts.

Cumulative tokens at each checkpoint (actor + reflection):

| Model | Baseline (T1) | Control @T2 | Memory @T2 | Delta @T2 | Control @T10 | Memory @T10 | Delta @T10 |
|---|---|---|---|---|---|---|---|
| gpt-oss-120b | 27.4M | 51.3M | 65.8M | +14.5M (**+28%**) | 197.8M | 298.5M | +100.7M (**+51%**) |
| Qwen3-32B | 37.5M | 72.4M | 77.5M | +5.1M (+7%) | 320.8M | 376.5M | +55.7M (+17%) |
| Qwen3-0.6B | 41.1M | 78.5M | 80.9M | +2.4M (+3%) | 374.6M | 421.2M | +46.6M (+12%) |

The memory arm's overhead **compounds**: +28% at trial 2 grows to +51% by trial 10 on
gpt-oss. Each trial's reflection is added to a memory that is already being replayed
through every step, so the prompt tax grows until the window caps at 3.

### Efficiency

Tokens per solved puzzle — lower is better:

| Model | Baseline (T1) | Control @T2 | Memory @T2 | Delta @T2 | Control @T10 | Memory @T10 | Delta @T10 |
|---|---|---|---|---|---|---|---|
| gpt-oss-120b | 0.39M | 0.52M | 0.55M | +0.02M | **1.24M** | 1.67M | +0.42M |
| Qwen3-32B | 0.68M | 0.94M | 0.98M | +0.04M | **2.41M** | 2.94M | +0.53M |
| Qwen3-0.6B | 2.94M | 3.02M | **2.79M** | −0.23M | 7.49M | **7.26M** | −0.23M |

Two things stand out. **Efficiency collapses with trial count regardless of arm** — every
model costs 3–4× more per solve at trial 10 than at trial 2, because later trials retry
an increasingly hard residue. And **at trial 2 the arms are near-parity** (+0.02M, +0.04M,
−0.23M), whereas by trial 10 memory is clearly worse on both large models.

Marginal cost of the puzzles memory wins *over* the control:

| Model | Extra solved @T2 | Extra tokens @T2 | Cost / extra solve @T2 | Extra solved @T10 | Extra tokens @T10 | Cost / extra solve @T10 |
|---|---|---|---|---|---|---|
| gpt-oss-120b | +22 | +14.5M | **0.7M** | +20 | +100.7M | **5.0M** |
| Qwen3-32B | +2 | +5.1M | 2.5M | −5 | +55.7M | n/a (no gain) |
| Qwen3-0.6B | +3 | +2.4M | 0.8M | +8 | +46.6M | 5.8M |

This is the strongest quantitative argument in the report. On gpt-oss the first
reflection buys 22 extra puzzles at **0.7M tokens each** — roughly in line with what that
arm was paying per solve anyway at trial 2 (0.55M). Running on to trial 10 buys *fewer*
extra puzzles (20) at **5.0M tokens each: a 7× worse marginal rate, and 3× the memory
arm's own average cost per solve** (1.67M). Every trial after the second buys back the
same 20-puzzle margin at steadily worse prices.

### Token cost per difficulty

Actor tokens only, at trial 10. (Reflection tokens are aggregated per trial, not per
puzzle, so they cannot be attributed to a difficulty band — at 2–3% of the total this
does not affect the picture.)

**gpt-oss-120b**

| Difficulty | n | Control | Memory | Delta | Ctl solved | Mem solved | Ctl tok/solve | Mem tok/solve |
|---|---|---|---|---|---|---|---|---|
| 1 | 86 | 14.6M | 18.7M | +4.0M | 65 | 71 | 0.23M | 0.26M |
| 2 | 118 | 38.3M | 53.2M | +14.9M | 53 | 57 | 0.72M | 0.93M |
| 3 | 121 | 49.1M | 71.5M | +22.4M | 30 | 40 | 1.64M | 1.79M |
| 4 | 86 | 43.7M | 66.7M | +23.0M | 9 | 9 | 4.85M | **7.41M** |
| 5 | 89 | 52.1M | 79.2M | +27.1M | 2 | 2 | 26.05M | **39.61M** |

**Qwen3-32B**

| Difficulty | n | Control | Memory | Delta | Ctl solved | Mem solved | Ctl tok/solve | Mem tok/solve |
|---|---|---|---|---|---|---|---|---|
| 1 | 86 | 31.1M | 34.9M | +3.8M | 58 | 52 | 0.54M | 0.67M |
| 2 | 118 | 62.5M | 70.7M | +8.2M | 42 | 44 | 1.49M | 1.61M |
| 3 | 121 | 79.5M | 90.4M | +10.9M | 25 | 28 | 3.18M | 3.23M |
| 4 | 86 | 65.6M | 79.1M | +13.5M | 6 | 4 | 10.93M | **19.77M** |
| 5 | 89 | 82.1M | 92.6M | +10.5M | 2 | 0 | 41.04M | **— (0 solved)** |

**Qwen3-0.6B**

| Difficulty | n | Control | Memory | Delta | Ctl solved | Mem solved | Ctl tok/solve | Mem tok/solve |
|---|---|---|---|---|---|---|---|---|
| 1 | 86 | 51.0M | 57.7M | +6.7M | 21 | 26 | 2.43M | 2.22M |
| 2 | 118 | 81.0M | 87.3M | +6.3M | 15 | 17 | 5.40M | 5.13M |
| 3 | 121 | 88.5M | 97.9M | +9.4M | 12 | 11 | 7.38M | 8.90M |
| 4 | 86 | 72.5M | 79.0M | +6.5M | 2 | 3 | 36.27M | 26.34M |
| 5 | 89 | 81.6M | 90.2M | +8.6M | 0 | 1 | — | 90.21M |

**The spend is inverted against the payoff.** Difficulty scales token cost (puzzles get
longer and are retried more often) while collapsing the solve rate, so cost per solve
rises by two orders of magnitude across the bands — 0.26M at difficulty 1 to 39.6M at
difficulty 5 on gpt-oss.

Breaking down gpt-oss's +91.4M extra *actor* tokens by where they went:

| Difficulty | Extra tokens | Extra solved |
|---|---|---|
| 1 | +4.0M | +6 |
| 2 | +14.9M | +4 |
| 3 | +22.4M | +10 |
| 4 | +23.0M | **0** |
| 5 | +27.1M | **0** |

**55% of the memory arm's extra tokens (50.2M) were spent on difficulties 4–5 and bought
exactly zero additional puzzles.** The reflections are generated, stored and replayed
through every step of every retry for 175 puzzles that neither arm ever solves. This is
the single largest source of waste in the experiment, and it is structural: Reflexion
keeps re-attempting whatever is unsolved, with no notion of a puzzle being out of reach.

### The practical implication

Since the entire effect is at trial 2, **T=2 captures it at a fifth of the cost.**
Trials 3–10 on gpt-oss consumed roughly 8 GPU-hours per arm to reproduce the control
almost exactly. A T=2 design across all three models would have cost on the order of
60–70 GPU-hours instead of 318, and supported the same conclusions.

---

## 4. Conclusions

1. **Reflexion gives a small, single-shot benefit on the strongest model only.**
   +4.0pp on gpt-oss-120b, nothing on Qwen3-32B, nothing on Qwen3-0.6B.
2. **Most of the apparent gain is resampling.** Every model's own curve rises steeply;
   ~80% of gpt-oss's rise is reproduced by a control with no memory at all. Any Reflexion
   result reported without a trial-matched control is uninterpretable.
3. **The iterative premise does not hold here.** The paper's mechanism is refinement
   accumulating over trials. On SPaRC the benefit is entirely in the first reflection,
   and trials 3–10 match the control on all three models.
4. **The limiting factor is reflection diversity, not quality.** Reflections are
   well-formed, on-topic and freshly worded every trial, but re-state one insight. The
   model can identify *what* went wrong and cannot generate a materially different *plan*.
5. **Memory needs competence to act on.** The gain is confined to difficulties 1–3 on
   gpt-oss and is exactly zero at 4–5. A verbal hint does not substitute for capability.
6. **It is not free.** 2–3% of tokens for reflections, but 30–50% more total tokens from
   the longer prompts they cause, and worse tokens-per-solve than the control on two of
   three models.

### Threats to validity

- **Single seed per trial index.** One run per arm; no variance estimate across repeats.
- **`solved` is ground truth** from the environment, so the evaluator is stronger than
  the paper's self-evaluated setup — results are not directly comparable to its AlfWorld
  numbers.
- **Difficulty 4–5 is near-floor** for all three models, so those rows cannot show an
  effect in either direction.

---

## Reproducing

```bash
# 1. trial 1, shared by both arms
python scripts/run_reflexion.py --phase root --split test --limit 0
# 2. the two arms
python scripts/run_reflexion.py --phase arm --arm memory  --root <root.json> --trials 10
python scripts/run_reflexion.py --phase arm --arm control --root <root.json> --trials 10
# 3. per-model and cross-model reports
python scripts/run_reflexion.py --phase compare --memory <m.json> --control <c.json>
python scripts/compare_all_arms.py
```

Arms checkpoint after every trial; resubmit with `--resume <arm.json>` to continue.
See [architecture.md](architecture.md) for the design and its method-agnostic seams.

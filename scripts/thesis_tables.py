#!/usr/bin/env python
"""Emit the thesis's LaTeX result tables from thesis_report.py's registry. No GPU, stdlib only.

    python scripts/thesis_tables.py T1 > results/thesis/t1_spatialgym.tex
    python scripts/thesis_tables.py T2 > results/thesis/t2_stepgame.tex
    python scripts/thesis_tables.py T5|T6|T7|M5   # one of the other tables
    python scripts/thesis_tables.py all           # every table (the default)
"""
import argparse, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import thesis_report as R

# (display name, RUNS label) per group; labels must match thesis_report's RUNS.
SPARC_GROUPS = [
    ("MemP", [("written procedure", "MemP — script v2, retrieved"),
              ("written procedure, random", "MemP — script v2, random"),
              ("step by step", "MemP — trajectory, retrieved"),
              ("step by step, random", "MemP — trajectory, random"),
              ("both together", "MemP — proceduralization v2, retrieved"),
              ("both together, random", "MemP — proceduralization v2, random"),
              ("own-successes bank", "MemP — agent-only bank v2, retrieved"),
              ("own-successes bank, random", "MemP — agent-only bank v2, random")]),
    ("MemoryBank", [("rules given", "MemoryBank — rp, retrieved"),
                    ("rules given, random", "MemoryBank — rp, random"),
                    ("rules withheld", "MemoryBank — norp, retrieved"),
                    ("rules withheld, random", "MemoryBank — norp, random")]),
    ("Contrastive", [("retrieved", "Contrastive — retrieved"),
                     ("random", "Contrastive — random")]),
]
STEP_GROUPS = [
    ("MemP", [("written procedure", "MemP — script, retrieved"),
              ("written procedure, random", "MemP — script, random"),
              ("step by step", "MemP — trajectory, retrieved"),
              ("step by step, random", "MemP — trajectory, random"),
              ("both together", "MemP — proceduralization, retrieved"),
              ("both together, random", "MemP — proceduralization, random"),
              ("own-successes bank", "MemP — agent-only bank, retrieved"),
              ("own-successes bank, random", "MemP — agent-only bank, random")]),
    ("MemoryBank", [("rules given", "MemoryBank — rp, retrieved"),
                    ("rules given, random", "MemoryBank — rp, random"),
                    ("rules withheld", "MemoryBank — norp, retrieved"),
                    ("rules withheld, random", "MemoryBank — norp, random")]),
    ("Contrastive", [("retrieved", "Contrastive — retrieved"),
                     ("random", "Contrastive — random")]),
]


def cell(run, base, n):
    """accuracy (change in points), or an em dash for an experiment not finished."""
    if run is None or not run.ok:
        return r"\textemdash"
    ids = [k for k in base.rec if k in run.rec]
    a = sum(1 for k in ids if run.rec[k].get("success"))
    b = sum(1 for k in ids if base.rec[k].get("success"))
    acc, d = 100 * a / len(ids), 100 * (a - b) / len(ids)
    return f"{acc:.1f} ({d:+.1f})"


def table(bench):
    actors = R.SPARC if bench == "sparc" else R.STEP
    groups = SPARC_GROUPS if bench == "sparc" else STEP_GROUPS
    runs = dict(R.SPARC_RUNS if bench == "sparc" else R.STEP_RUNS)
    bases = {t: R.Run(R.BASE[bench].format(t=t)) for t, _ in actors}
    n = 500 if bench == "sparc" else 1000
    unit = "puzzles" if bench == "sparc" else "items"
    lab = "Spatial-Gym" if bench == "sparc" else "StepGame"

    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\begin{tabular}{l" + "r" * len(actors) + "}", r"\toprule",
         "Experiment & " + " & ".join(n for _, n in actors) + r" \\",
         r"\midrule",
         "memoryless baseline & " +
         " & ".join(f"{100*sum(1 for v in bases[t].rec.values() if v.get('success'))/n:.1f}"
                    for t, _ in actors) + r" \\"]
    for gname, rows in groups:
        L.append(r"\midrule")
        L.append(r"\multicolumn{%d}{l}{\textit{%s}} \\" % (1 + len(actors), gname))
        for disp, key in rows:
            tmpl = runs.get(key)
            cells = [cell(R.Run(tmpl.format(t=t)) if tmpl else None, bases[t], n)
                     for t, _ in actors]
            L.append(f"\\quad {disp} & " + " & ".join(cells) + r" \\")
    # Reflexion is compared with its own control, not the baseline.
    L.append(r"\midrule")
    L.append(r"\multicolumn{%d}{l}{\textit{Reflexion} (against its trial-matched control)} \\"
             % (1 + len(actors)))
    mem, ctl = R.REFL[bench]
    for cp, lab_cp in ((2, "after one reflection"), (10, "after ten trials")):
        cells = []
        for t, _ in actors:
            m, c = R.Run(mem.format(t=t)), R.Run(ctl.format(t=t))
            if not (m.ok and c.ok):
                cells.append(r"\textemdash"); continue
            ids = [k for k in m.rec if k in c.rec]
            sol = lambda r, k: (r.rec[k].get("solved_at_trial") or 99) <= cp
            a = sum(1 for k in ids if sol(m, k)); b = sum(1 for k in ids if sol(c, k))
            cells.append(f"{100*a/len(ids):.1f} ({100*(a-b)/len(ids):+.1f})")
        L.append(f"\\quad {lab_cp} & " + " & ".join(cells) + r" \\")
    take = (r"Every experiment improves gpt-oss-120b, none improves Qwen3-32B, and none "
            r"moves Qwen3-0.6B." if bench == "sparc" else
            r"Every cross-task experiment improves both actors; Reflexion falls behind its "
            r"own control.")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{\textbf{Every experiment on %s.} Accuracy over %d test %s, with the change"
          % (lab, n, unit),
          r"  from each actor's own baseline in points; dashes are experiments still running. "
          + take + "}",
          r"\label{tab:%s}" % ("spatialgym" if bench == "sparc" else "stepgame"),
          r"\end{table}"]
    return "\n".join(L)


DISPLAY = {  # registry key -> short display name, for the failure-mode table
    "MemP — script v2, retrieved": "MemP written procedure",
    "MemP — script v2, random": "MemP written procedure, random",
    "MemP — trajectory, retrieved": "MemP step by step",
    "MemP — trajectory, random": "MemP step by step, random",
    "MemP — proceduralization v2, retrieved": "MemP both together",
    "MemP — proceduralization v2, random": "MemP both together, random",
    "MemP — agent-only bank v2, retrieved": "MemP own-successes bank",
    "MemP — agent-only bank v2, random": "MemP own-successes bank, random",
    "MemoryBank — rp, retrieved": "MemoryBank rules given",
    "MemoryBank — rp, random": "MemoryBank rules given, random",
    "MemoryBank — norp, retrieved": "MemoryBank rules withheld",
    "MemoryBank — norp, random": "MemoryBank rules withheld, random",
    "Contrastive — retrieved": "Contrastive",
    "Contrastive — random": "Contrastive, random",
}


def failure_table(actor="gptoss120b", actor_name="gpt-oss-120b"):
    """T5 -- solved / wrong-solution / dead-end shares per experiment (Spatial-Gym)."""
    base = R.Run(R.BASE["sparc"].format(t=actor))
    def split(run):
        s = sum(1 for v in run.rec.values() if v.get("success"))
        w = sum(1 for v in run.rec.values()
                if not v.get("success") and v.get("terminal_reason") == "wrong-solution")
        d = sum(1 for v in run.rec.values()
                if not v.get("success") and v.get("terminal_reason") == "dead-end")
        n = len(run.rec)
        return 100*s/n, 100*w/n, 100*d/n
    bs, bw, bd = split(base)

    body = []
    for key, tmpl in R.SPARC_RUNS:
        run = R.Run(tmpl.format(t=actor))
        if not run.ok:
            continue
        s, w, d = split(run)
        body.append((w, DISPLAY.get(key, key), s, w, d))
    body.sort()                                    # by wrong solutions, ascending

    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\begin{tabular}{lrrr}", r"\toprule",
         r"Experiment & solved & wrong solutions & dead ends \\", r"\midrule",
         f"memoryless baseline & {bs:.1f} & {bw:.1f} & {bd:.1f} \\\\",
         r"\midrule"]
    for _, name, s, w, d in body:
        L.append(f"{name} & {s:.1f} ({s-bs:+.1f}) & {w:.1f} ({w-bw:+.1f}) & "
                 f"{d:.1f} ({d-bd:+.1f}) \\\\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{\textbf{Memory swaps one kind of failure for another.} Share of the 500",
          r"  Spatial-Gym episodes for %s ending solved, with a broken rule, or with no legal" % actor_name,
          r"  move, and the change from the memoryless baseline in percentage points; rows are",
          r"  ordered by wrong solutions. Reading down the last two columns, they move in",
          r"  opposite directions, and MemP appears at both ends.}",
          r"\label{tab:failures}", r"\end{table}"]
    return "\n".join(L)


def cost_table(sparc_actor="gptoss120b", sparc_name="gpt-oss-120b",
               step_actor="qwen32b", step_name="Qwen3-32B"):
    """T6 -- accuracy against tokens per solved task, one actor per benchmark."""
    sp_runs, st_runs = dict(R.SPARC_RUNS), dict(R.STEP_RUNS)
    sp_base = R.Run(R.BASE["sparc"].format(t=sparc_actor))
    st_base = R.Run(R.BASE["stepgame"].format(t=step_actor))

    def cells(run, scale, dec):
        """accuracy, cost per solved task and prompt share; dashes if not finished."""
        if run is None or not run.ok:
            return [r"\textemdash"] * 3
        acc = 100 * run.solved() / len(run.rec)
        pr = sum(v.get("prompt_tokens") or 0 for v in run.rec.values())
        return [f"{acc:.1f}", f"{run.toks()/max(run.solved(),1)/scale:.{dec}f}",
                f"{100*pr/max(run.toks(),1):.0f}"]

    def row(disp, sp_key, st_key):
        c = (cells(R.Run(sp_runs[sp_key].format(t=sparc_actor)) if sp_key else None, 1e6, 2)
             + cells(R.Run(st_runs[st_key].format(t=step_actor)) if st_key else None, 1e3, 1))
        return f"\\quad {disp} & " + " & ".join(c) + r" \\"

    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"& \multicolumn{3}{c}{Spatial-Gym --- %s} & \multicolumn{3}{c}{StepGame --- %s} \\"
         % (sparc_name, step_name),
         r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
         r"Experiment & accuracy & Mtok & prompt \% & accuracy & ktok & prompt \% \\",
         r"\midrule",
         "memoryless baseline & " + " & ".join(cells(sp_base, 1e6, 2)
                                               + cells(st_base, 1e3, 1)) + r" \\"]
    # The cross-task methods, paired by display name across the two benchmarks.
    for gname, sp_rows in SPARC_GROUPS:
        st_rows = dict((d, k) for g, rs in STEP_GROUPS if g == gname for d, k in rs)
        L.append(r"\midrule")
        L.append(r"\multicolumn{7}{l}{\textit{%s}} \\" % gname)
        for disp, sp_key in sp_rows:
            L.append(row(disp, sp_key, st_rows.get(disp)))
    # Reflexion pays for every attempt, so its cost is cumulative over trials 1..T.
    L.append(r"\midrule")
    L.append(r"\multicolumn{7}{l}{\textit{Reflexion} (cumulative over trials)} \\")
    for upto, lab in ((2, "after one reflection"), (None, "after ten trials")):
        c = []
        for tmpl, actor, scale, dec in ((R.REFL["sparc"][0], sparc_actor, 1e6, 2),
                                        (R.REFL["stepgame"][0], step_actor, 1e3, 1)):
            r = R.Run(tmpl.format(t=actor))
            if not r.ok:
                c += [r"\textemdash", r"\textemdash"]; continue
            n, tk = r.solved_upto(upto), r.toks_upto(upto)
            pr = sum(v.get("prompt_tokens") or 0 for v in r.rec.values())
            c += [f"{100*n/len(r.rec):.1f}", f"{tk/max(n,1)/scale:.{dec}f}",
                  f"{100*pr/max(r.toks(),1):.0f}" if upto is None else r"\textemdash"]
        L.append(f"\\quad {lab} & " + " & ".join(c) + r" \\")

    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{\textbf{The methods differ in cost far more than in accuracy.} Tokens"
          r" spent per solved task (millions on Spatial-Gym, thousands on StepGame), beside"
          r" the accuracy they buy and the share of those tokens"
          r" that is prompt rather than generated. On Spatial-Gym the memory is re-sent at"
          r" every step and costs vary widely, but the cheapest methods score as high as the"
          r" most expensive ones. On StepGame it is sent once per item and they all cost about"
          r" the same. The prompt share barely moves on Spatial-Gym, where memory makes the"
          r" actor generate more as well, and roughly triples on StepGame, where it is added"
          r" to a short prompt and the answer stays the same length. Reflexion costs the most"
          r" on both, because its later trials retry puzzles it almost never solves.}",
          r"\label{tab:cost}", r"\end{table}"]
    return "\n".join(L)


def steps_table(actor="gptoss120b", actor_name="gpt-oss-120b"):
    """T7 -- how many steps an episode runs, overall and by how it ended (Spatial-Gym)."""
    def row(name, run):
        if not run.ok:
            return None
        v = list(run.rec.values())
        def mean(sel):
            xs = [x["steps"] for x in v if sel(x)]
            return f"{sum(xs)/len(xs):.1f}" if xs else r"\textemdash"
        end = lambda r: (lambda x: x.get("terminal_reason") == r)
        return (f"{name} & {sum(x['steps'] for x in v)/len(v):.1f} & "
                + " & ".join(mean(end(r)) for r in
                             ("solved", "wrong-solution", "dead-end")) + r" \\")

    base = R.Run(R.BASE["sparc"].format(t=actor))
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\begin{tabular}{lrrrr}", r"\toprule",
         r"& & \multicolumn{3}{c}{mean steps by how the episode ended} \\",
         r"\cmidrule(lr){3-5}",
         r"Experiment & mean steps & solved & wrong solution & dead end \\",
         r"\midrule", row("memoryless baseline", base), r"\midrule"]
    for key, tmpl in R.SPARC_RUNS:
        r = row(DISPLAY.get(key, key), R.Run(tmpl.format(t=actor)))
        if r:
            L.append(r)
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{\textbf{Memory makes episodes longer, not shorter.} Steps taken per"
          r" Spatial-Gym episode for %s, over all 500 episodes and split by how the episode"
          r" ended. Every experiment runs longer than the baseline, and solved episodes are"
          r" the shortest everywhere. The methods that leave the fewest dead ends are the ones"
          r" whose dead ends run longest, so what memory removes are the short ones." % actor_name
          + r"}",
          r"\label{tab:steps}", r"\end{table}"]
    return "\n".join(L)


# One bank per benchmark and actor; all cross-task methods distil the same episodes.
BANKS = [("Spatial-Gym", [("Qwen3-0.6B", "memp_bank_qwen06b_v2.json"),
                          ("Qwen3-32B", "memp_bank_qwen32b_v2.json"),
                          ("gpt-oss-120b", "memp_bank_gptoss120b_v2.json")]),
         ("StepGame", [("Qwen3-0.6B", "memp_bank_stepgame_qwen06b_upstream-header.json"),
                       ("Qwen3-32B", "memp_bank_stepgame_qwen32b_upstream-header.json")])]


def bank_table():
    """M5 -- what the banks are made of."""
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\begin{tabular}{llrrr}", r"\toprule",
         r"Benchmark & Bank built by & entries & from its own runs & from the dataset \\",
         r"\midrule"]
    for bench, rows in BANKS:
        for i, (actor, fname) in enumerate(rows):
            st = R.load_json(fname)["stats"]
            a, g = st["n_agent"], st["n_gold"]
            n = a + g
            L.append(f"{bench if i == 0 else ''} & {actor} & {n} & "
                     f"{a} ({100*a/n:.1f}\\%) & {g} ({100*g/n:.1f}\\%) \\\\")
        if bench != BANKS[-1][0]:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{\textbf{What the banks are made of.} Each bank holds one entry per"
          r" training item. An entry is distilled from the actor's own run where it solved the"
          r" item and from the dataset's solution where it did not, so the split is the"
          r" actor's accuracy on the training split. All three cross-task methods are built"
          r" from the same episodes and therefore share these counts; in the contrastive bank"
          r" the same split separates strategies from warnings. Spatial-Gym banks are almost"
          r" entirely dataset-derived, and only Qwen3-32B on StepGame builds a bank mostly"
          r" from its own runs.}",
          r"\label{tab:banks}", r"\end{table}"]
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="?", default="all", choices=["T1", "T2", "T5", "T6", "T7", "M5", "all"])
    a = ap.parse_args()
    if a.which in ("T1", "all"): print(table("sparc"))
    if a.which == "all": print()
    if a.which in ("T2", "all"): print(table("stepgame"))
    if a.which == "all": print()
    if a.which in ("T5", "all"): print(failure_table())
    if a.which == "all": print()
    if a.which in ("T6", "all"): print(cost_table())
    if a.which == "all": print()
    if a.which in ("T7", "all"): print(steps_table())
    if a.which == "all": print()
    if a.which in ("M5", "all"): print(bank_table())

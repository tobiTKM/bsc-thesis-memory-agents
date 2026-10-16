#!/usr/bin/env python
"""Emit the methodology's example memory blocks as LaTeX. No GPU, stdlib only.

Blocks are rendered from the banks and runs on disk via the eval's prompts.format_*.

    python scripts/thesis_blocks.py M6 > results/thesis/m6_memp_forms.tex
    python scripts/thesis_blocks.py M7 > results/thesis/m7_method_blocks.tex
    python scripts/thesis_blocks.py M7 --nofloat   # non-floating, for an appendix
"""
import argparse, pathlib, sys, textwrap

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from memory_agents import prompts as P            # noqa: E402
from thesis_report import load_json               # noqa: E402

WIDTH = 86                                        # fits \footnotesize verbatim
FOLD = {"\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "---", "\u2018": "'",
        "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u00d7": "x", "\u00a0": " ",
        "\u2026": "..."}


def ascii_(s):
    for k, v in FOLD.items():
        s = s.replace(k, v)
    s = "".join(" " if ch.isspace() and ch not in "\n\t" else ch for ch in s)
    return s.encode("ascii", "replace").decode()


def lines(text):
    """ASCII, wrapped to the page, with blank runs collapsed."""
    out = []
    for ln in ascii_(text).splitlines():
        ln = ln.rstrip()
        if not ln.strip():
            if out and out[-1] != "":
                out.append("")
            continue
        out += textwrap.wrap(ln, WIDTH, subsequent_indent="  ") or [""]
    return out


def cut(ls, n):
    return ls[:n] + (["    [...]"] if len(ls) > n else [])


def at(ls, marker):
    return next(i for i, l in enumerate(ls) if marker in l)


def splice(ls, *spans):
    """Selected stretches of a block, separated by the mark that says we cut."""
    out = []
    for i, (a, b) in enumerate(spans):
        out += ls[a:b]
        if b < len(ls):
            out.append("    [...]")
    return out


def panel(label, body):
    return (f"\\medskip\\noindent\\textbf{{{label}}}\n"
            "\\begin{verbatim}\n" + "\n".join(body) + "\n\\end{verbatim}")


def bank(name):
    return load_json(name)["entries"]


def memp_entry():
    """One entry, in both the script bank and the trajectory bank."""
    e = bank("memp_bank_gptoss120b_v2.json")[0]
    t = {x["source_pid"]: x for x in bank("memp_bank_gptoss120b_v2_traj.json")}
    return e, t[e["source_pid"]]


def M6():
    e, t = memp_entry()
    head = lines(P.format_procedures([e], "script"))
    head = [l for l in head if l][:4]             # the shared header paragraph
    body = lambda txt: [l for l in lines(txt) if l]

    scr = body(P.format_procedures([e], "script"))[4:]
    tra = body(P.format_procedures([t], "trajectory"))[4:]
    pro = body(P.format_procedures([t], "proceduralization"))[4:]

    # the panels show where the forms differ, not the first N lines of each
    tra_p = splice(tra, (0, 5), (at(tra, "move by move"), at(tra, "move by move") + 3),
                   (at(tra, "final grid"), at(tra, "final grid") + 1))
    pro_p = splice(pro, (0, 3), (at(pro, "starting grid"), at(pro, "starting grid") + 2))

    L = [r"\begin{figure}[t]", r"\footnotesize",
         panel("Every form opens with the same header", cut(head, 3)),
         panel("script --- the written procedure, and a one-line example",
               splice(scr, (0, 4), (at(scr, "Example ("), len(scr)))),
         panel("trajectory --- the solved run instead, step by step", tra_p),
         panel("proceduralization --- the procedure, then that same run", pro_p),
         r"\caption{\textbf{MemP's three forms are three renderings of one bank entry.}"
         r" The same entry as the actor is shown it, cut where marked, all on Spatial-Gym."
         r" The written"
         r" procedure generalises over a class of puzzles and names no coordinates; the"
         r" step-by-step form drops it for the solved run itself, grids and all; the third"
         r" concatenates the two. The bank never changes between them, only what is"
         r" rendered from it.}",
         r"\label{fig:mempforms}", r"\end{figure}"]
    return "\n".join(L)


def M7():
    e, t = memp_entry()
    ct = bank("contrastive_bank_gptoss120b_ex.json")
    mb = bank("memorybank_bank_gptoss120b_rp.json")
    refl = load_json("reflexion_memory_gptoss120b_test.json")["memory"]

    do = next(x for x in ct if x["provenance"] == "agent")
    dont = next(x for x in ct if x["provenance"] == "failed")
    one = next(v[0] for v in refl.values() if v)

    con = [l for l in lines(P.format_contrastive([do, dont], "avoid",
                                                 exemplars=False)) if l]
    con_p = splice(con, (0, at(con, "Item 1:") + 4),
                   (at(con, "Item 2:"), at(con, "Item 2:") + 4))

    L = [r"\begin{figure}[t]", r"\footnotesize",
         panel("Reflexion, on a retry of the same puzzle",
               cut([l for l in lines(P.format_reflections([one])) if l], 7)),
         panel("MemP",
               cut([l for l in lines(P.format_procedures([e], "script")) if l], 8)),
         panel("MemoryBank",
               cut([l for l in lines(P.format_cases([mb[0]], "case")) if l], 8)),
         panel("Contrastive memory, one entry of each sign", con_p),
         r"\caption{\textbf{What each method puts in front of the actor.} One real block"
         r" per method, cut where marked, all for gpt-oss-120b on Spatial-Gym. Reflexion's"
         r" is about the very puzzle being retried and names no solution; the three"
         r" cross-task methods describe other puzzles, and differ in whether that is a"
         r" procedure to adapt, a case to reason from, or a pair of a strategy and a"
         r" warning.}",
         r"\label{fig:blocks}", r"\end{figure}"]
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["M6", "M7"])
    ap.add_argument("--nofloat", action="store_true",
                    help="emit as ordinary text that may break across pages, with a "
                         "numbered caption via \\captionof (needs \\usepackage{caption}); "
                         "use this in an appendix, where a too-tall float would be "
                         "deferred or dropped")
    a = ap.parse_args()
    out = {"M6": M6, "M7": M7}[a.which]()
    if a.nofloat:
        kind = "figure"
        out = (out.replace(r"\begin{figure}[t]", "")
                  .replace(r"\end{figure}", "")
                  .replace(r"\caption{", "\\captionof{%s}{" % kind)).strip()
    print(out)

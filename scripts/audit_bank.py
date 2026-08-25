#!/usr/bin/env python
"""Does a Memp bank state SPaRC's rules correctly? A regex proxy, run on any bank.

docs/memp.md section 3.5 established that the banks are frequently *wrong* about the
rules -- and section 3.6 showed that is why the LLM-written script contributes nothing
on top of the mechanically-extracted trajectory. That audit was ad-hoc; this script
makes it reproducible, so the same measurement can be pointed at a bank built before
and after a change to the distillation prompt.

What it looks for, per entry (an entry can trip several at once -- these are
independent checks, not a partition):

  region_ok       the script frames squares/stars the way SPaRC actually defines them:
                  a constraint on the REGIONS the path encloses (same-colour squares
                  share a region; a star pairs with exactly one same-colour element).
  visit_wrong     the script tells the agent to visit / enter / step on / collect a
                  square, star or polyshape, or calls one a "waypoint" or "required
                  cell". The path goes *around* those cells; only dots are visited.
                  This is the triangle rule (edge touching) misapplied, or a plain
                  Hamiltonian-path assumption.
  count_wrong     the script reads the key text's census line ("rules: square x3" --
                  how many square CELLS the grid holds) as a number of required
                  visits, entries or crossings. No rule of that form exists.
  example_copy    the script is a near-verbatim copy of the single few-shot procedure
                  in the distillation prompt, i.e. the entry carries no puzzle-specific
                  information at all (110/500 of Qwen3-0.6B's bank, per section 3.5).

Only entries whose source puzzle actually HAS a square or star rule are scored for
region_ok / visit_wrong, since the others have nothing to get right or wrong there.

Caveat, inherited from section 3.5 and worth repeating in anything that cites this: a
regex is a proxy for a hand-labelled audit. It is reliable for *comparing two banks
under one definition*, which is what it is for; treat an absolute percentage as
indicative, not exact.

    python scripts/audit_bank.py results/json/memp_bank_*.json
    python scripts/audit_bank.py --verbose results/json/memp_bank_gptoss120b.json
"""

import argparse
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from memory_agents.prompts import _PROCEDURE_EXAMPLE

# Squares/stars framed as what they are: a property of the enclosed region.
REGION_OK = re.compile(
    r"region|enclos|separat|same colou?r|colou?r\w*\s+(?:group|zone|side|region)"
    r"|wall between|partition|divide", re.I)

# Squares/stars/polys framed as cells the path travels through. Two shapes: a verb of
# entry aimed at one of those symbols, or the symbol relabelled as a thing to be
# collected in passing ("waypoint", "required cell", "must-visit").
_ENTER = (r"visit|pass(?:es|ing)?\s+through|step(?:s|ping)?\s+on|enter|collect"
          r"|travel\s+through|go\s+through|touch|thread|traverse")
_SYMBOL = r"square|star|poly(?:shape|omino)?"
VISIT_WRONG = re.compile(
    rf"(?:{_ENTER})\w*\s+(?:the\s+|each\s+|every\s+|all\s+|a\s+)?(?:\w+[\s-]+){{0,3}}(?:{_SYMBOL})s?"
    rf"|(?:{_SYMBOL})s?[^.\n]{{0,40}}?(?:way[\s-]?point|must[\s-]be[\s-]visited"
    rf"|mandatory\s+cell|required\s+cell|must\s+(?:be\s+)?(?:{_ENTER}))"
    rf"|(?:way[\s-]?point|mandatory|required)\s+(?:cells?\s+)?\(?[^.\n]{{0,30}}?(?:{_SYMBOL})",
    re.I)

# The census line read as a multiplicity: "square x3" -> "each square needs 3 entries".
COUNT_WRONG = re.compile(
    r"(?:required|need\w*|exact\w*|specified)\s+(?:number\s+of\s+)?"
    r"(?:visit|entr|cross|pass|touch)\w*"
    r"|visit\s+count|entry\s+count|(?:visit|enter|cross)\w*\s+(?:it\s+)?"
    r"(?:twice|three|four|\d+)\s*(?:times)?"
    r"|(?:{sym})\w*\s*\(\s*\d+\s*\)|(?:{sym})\s*(?:×|x)\s*\d+\s*(?:visit|entr|time)"
    .format(sym=_SYMBOL), re.I)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split().__str__()


def _shingles(text: str, n: int = 8) -> set:
    words = re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split()
    return {tuple(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


def example_overlap(script: str, example_shingles: set) -> float:
    """Fraction of the script's 8-word shingles that also appear in the few-shot
    example. ~0 for an original procedure, high for a paraphrase-free copy."""
    sh = _shingles(script)
    if not sh:
        return 0.0
    return len(sh & example_shingles) / len(sh)


def entry_text(entry: dict) -> str:
    """The LLM-written prose of one bank entry, whichever method wrote it.

    Memp banks call it `script` (a procedure), MemoryBank banks `case_note` (a
    recollection). The checks below are about whether the prose states SPaRC's
    rules correctly, which is a question worth asking of either -- a case note
    that says "I stepped on the blue square" misinforms the actor exactly as a
    procedure that says "visit each square" does.
    """
    for field in ("script", "case_note"):
        if field in entry:
            return entry[field]
    raise KeyError(f"entry {entry.get('id')} has neither 'script' nor 'case_note'")


def audit(bank_path, verbose=False, copy_threshold=0.5) -> dict:
    data = json.loads(pathlib.Path(bank_path).read_text())
    entries = data.get("entries", [])
    example_sh = _shingles(_PROCEDURE_EXAMPLE)
    text = {e["id"]: entry_text(e) for e in entries}

    # Scored only where the puzzle actually carries the rule in question.
    scored = [e for e in entries
              if re.search(r"square|star", e["key_text"].split("rules:")[-1])]
    region = [e for e in scored if REGION_OK.search(text[e["id"]])]
    visit = [e for e in scored if VISIT_WRONG.search(text[e["id"]])]
    both = [e for e in scored
            if REGION_OK.search(text[e["id"]]) and VISIT_WRONG.search(text[e["id"]])]
    count = [e for e in entries if COUNT_WRONG.search(text[e["id"]])]
    # example_copy only means anything for a bank whose prompt HAS a few-shot
    # example. MemoryBank's case-note prompts deliberately have none, so the
    # measure is reported as 0 there rather than compared against Memp's.
    copies = [e for e in entries
              if example_overlap(text[e["id"]], example_sh) >= copy_threshold]

    n, ns = len(entries), max(len(scored), 1)
    result = {
        "bank": str(bank_path),
        "schema": data.get("schema", "?"),
        "model_id": data.get("config", {}).get("model_id", "?"),
        # None for a Memp bank, which has no such knob; True/False for MemoryBank.
        "rules_preamble": data.get("config", {}).get("rules_preamble"),
        "entries": n,
        "scored_for_square_star": len(scored),
        "region_ok_pct": 100 * len(region) / ns,
        "visit_wrong_pct": 100 * len(visit) / ns,
        "both_pct": 100 * len(both) / ns,
        "count_wrong_pct": 100 * len(count) / max(n, 1),
        "example_copies": len(copies),
    }
    if verbose:
        for label, sample in (("visit_wrong", visit), ("count_wrong", count)):
            print(f"\n----- sample {label} -----")
            for e in sample[:4]:
                hit = (VISIT_WRONG if label == "visit_wrong" else COUNT_WRONG
                       ).search(text[e["id"]])
                print(f"  {e['id']} ({e['provenance']}): ...{hit.group(0).strip()}...")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("banks", nargs="+",
                    help="memp_bank_*.json / memorybank_bank_*.json files")
    ap.add_argument("--verbose", action="store_true",
                    help="print sample matches for the two error checks")
    args = ap.parse_args()

    rows = [audit(b, verbose=args.verbose) for b in args.banks]
    head = (f"{'bank':<34} {'n':>4} {'region ok':>10} {'visit wrong':>12} "
            f"{'both':>7} {'count wrong':>12} {'ex-copies':>10}")
    print("\n" + head)
    print("-" * len(head))
    for r in rows:
        print(f"{pathlib.Path(r['bank']).stem:<34} {r['entries']:>4} "
              f"{r['region_ok_pct']:>9.1f}% {r['visit_wrong_pct']:>11.1f}% "
              f"{r['both_pct']:>6.1f}% {r['count_wrong_pct']:>11.1f}% "
              f"{r['example_copies']:>10}")
    print("\nregion ok = squares/stars framed as a region constraint (higher is better)")
    print("visit wrong / count wrong / ex-copies: lower is better")
    print("Scored over entries whose puzzle has a square or star rule; regex proxy, "
          "see the module docstring.")


if __name__ == "__main__":
    main()

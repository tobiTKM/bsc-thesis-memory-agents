#!/usr/bin/env python3
"""Mirror the *rectified* StepGame corpus into data/stepgame/raw/.

Why the rectified corpus and not the original: Li, Hogg & Cohn (AAAI-24) found 14
of StepGame's 214 crowdsourced templates express a different relation than the
label they were generated under. The corruption grows with k -- 7.6% of clean
k=1 instances up to 54.3% of clean k=10, and 74.2% of noisy k=10 (their Table 2)
-- so on the original data a model is graded against wrong answers precisely
where the benchmark is supposed to be hardest. Their `correct.py` drops every
instance built from an unfixable template; those filtered files are what we
mirror. `verify_stepgame_solver.py` re-derives Table 2's removal percentages from
what lands on disk, so a silent switch back to the original data would be caught.

Why fetch instead of vendoring the files into the repo: the upstream repository
publishes no LICENSE (checked: raw.githubusercontent.com/.../LICENSE -> 404), so
redistributing its data or its template table inside this repo would be a
licensing claim we are not in a position to make. Fetching at build time from a
pinned commit keeps provenance exact without redistributing anything, and the
manifest records what we actually got.

`sentence_to_relation.py` is fetched alongside the data: it is the template ->
relation table that `stepgame_solver.py` reads, and it must come from the same
commit as the corpus it decodes.

    python scripts/fetch_stepgame.py                      # clean+noise, train+test
    python scripts/fetch_stepgame.py --splits train valid test
    python scripts/fetch_stepgame.py --variants clean --ks 1 2 3
"""

import argparse
import hashlib
import json
import pathlib
import sys
import urllib.error
import urllib.request

# Pinned so the corpus cannot move under a half-finished campaign. Bumping this
# invalidates every embedding and every bank built from the old text, exactly
# like changing a key-fn -- so bump the manifest's `commit` deliberately, and
# rebuild the splits, rather than tracking main.
COMMIT = "1114cb58382e1cea8c66c8bf98b5749fb32d7575"
REPO = "Fangjun-Li/SpatialLM-StepGame"
RAW = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}"

# Upstream directory name per variant. "correct_" is the rectified prefix; the
# unrectified `clean`/`noise` directories in the same repo are deliberately not
# reachable from here, so a typo cannot quietly select the corrupted corpus.
VARIANT_DIR = {"clean": "correct_clean", "noise": "correct_noise"}

TEMPLATE_FILE = "sentence_to_relation.py"

DEFAULT_KS = list(range(1, 11))
DEFAULT_SPLITS = ["train", "test"]      # `valid` exists upstream; no arm uses it
DEFAULT_VARIANTS = ["clean", "noise"]


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: pathlib.Path, force: bool) -> tuple[bool, int]:
    """Fetch `url` to `dest`. Returns (downloaded?, size). Existing files are kept
    unless --force: a full mirror is ~200 MB and re-running the script to add one
    variant should not re-pull the other."""
    if dest.exists() and not force:
        return False, dest.stat().st_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as r, tmp.open("wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    except urllib.error.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"[fetch] {url} -> HTTP {exc.code}. "
                         f"Is COMMIT={COMMIT[:8]} still valid?") from exc
    # Rename only after the body is complete, so an interrupted download can
    # never be mistaken for a cached file on the next run.
    tmp.replace(dest)
    return True, dest.stat().st_size


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS,
                    choices=["clean", "noise"])
    ap.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS,
                    choices=["train", "valid", "test"])
    ap.add_argument("--ks", nargs="+", type=int, default=DEFAULT_KS,
                    help="hop counts to mirror (1-10)")
    ap.add_argument("--out", default=None,
                    help="target dir (default: <repo>/data/stepgame/raw)")
    ap.add_argument("--force", action="store_true",
                    help="re-download files that already exist")
    args = ap.parse_args()

    out = pathlib.Path(args.out) if args.out else repo_root() / "data/stepgame/raw"
    out.mkdir(parents=True, exist_ok=True)

    files: dict[str, dict] = {}
    fetched = cached = 0
    total_bytes = 0

    # The template table first: without it the corpus cannot be decoded, so
    # failing here should abort before pulling 200 MB of stories.
    dest = out / TEMPLATE_FILE
    got, size = download(f"{RAW}/{TEMPLATE_FILE}", dest, args.force)
    fetched, cached = (fetched + 1, cached) if got else (fetched, cached + 1)
    total_bytes += size
    files[TEMPLATE_FILE] = {"sha256": sha256(dest), "bytes": size}
    print(f"[fetch] {'GET ' if got else 'have'} {TEMPLATE_FILE} ({size:,} B)")

    for variant in args.variants:
        for split in args.splits:
            for k in args.ks:
                name = f"qa{k}_{split}.txt"
                rel = f"{variant}/{name}"
                url = f"{RAW}/data/{VARIANT_DIR[variant]}/{name}"
                dest = out / variant / name
                got, size = download(url, dest, args.force)
                fetched, cached = ((fetched + 1, cached) if got
                                   else (fetched, cached + 1))
                total_bytes += size
                files[rel] = {"sha256": sha256(dest), "bytes": size}
                print(f"[fetch] {'GET ' if got else 'have'} {rel} ({size:,} B)")

    manifest = out / "MANIFEST.json"
    # Merged rather than overwritten: a later run that adds `valid` must not
    # erase the checksums of the files an earlier run already verified.
    previous = (json.loads(manifest.read_text()).get("files", {})
                if manifest.exists() else {})
    if previous and json.loads(manifest.read_text()).get("commit") != COMMIT:
        print("[fetch] WARNING: manifest was written at a different commit; "
              "stale entries dropped", file=sys.stderr)
        previous = {}
    manifest.write_text(json.dumps({
        "source": f"https://github.com/{REPO}",
        "commit": COMMIT,
        "note": "rectified StepGame (Li, Hogg & Cohn, AAAI-24) over the original "
                "corpus of Shi, Zhang & Lipani (AAAI-22). Not redistributed: "
                "fetched at build time, upstream publishes no LICENSE.",
        "files": {**previous, **files},
    }, indent=2, sort_keys=True) + "\n")

    print(f"\n[fetch] {fetched} downloaded, {cached} cached, "
          f"{total_bytes / 1e6:.1f} MB total -> {out}")
    print(f"[fetch] manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build WhoDunit's "train_old" split and push it to OpenReward.

train_old is the subset of the train cases that use no post-2000 concepts, language
("lingo") or knowledge: a murder mystery a reader on 1999-12-31 could follow without
meeting anything that did not exist yet. Each case is judged by deepseek-v4.1-flash.

How a verdict is reached, so that it can be trusted:

* Every case is judged TWICE, independently. It is dropped if either verdict flags it.
  The split is meant to be clean, so a disagreement resolves toward dropping.
* A flag only counts with evidence. A "post-2000" verdict must quote the offending text,
  and a quote only counts if it appears verbatim in the case. A flag the model cannot
  back with a real quote from the case -- a hallucinated one -- is discarded.
* A keyword scan runs first, and its hits are passed to the model as HINTS, never as
  verdicts: words like "tablet", "stream", "cloud" and "zoom" all have ordinary
  pre-2000 meanings, so only the model, reading them in context, decides.
* The four shared exhibits (shown in every case by view_exhibits) are judged too. They
  cannot be filtered per case, so a flagged exhibit is reported loudly instead.

Verdicts are cached, so a --dry-run followed by a real run pays for classification once.

Then, unless --dry-run:
  1. writes split_train_old.json (task ids + problem names, and every dropped case
     with its quoted evidence);
  2. uploads it to the environment's OpenReward file store, where the deployed server
     reads its data (/orwd_data);
  3. commits it with whodunnit.py (which serves the new split) and this script, and
     pushes to GitHub -- the linked OpenReward environment rebuilds from main. If main
     rejects the push, the commit goes to a branch instead.

The upload happens before the push on purpose: the server lists train_old only when the
file is present, so there is no moment at which the new code serves a missing split.

Usage:
    export KIMI_API_KEY=...          # infer.gr.inc
    python3 scripts/make_train_old.py --dry-run     # classify and report only
    python3 scripts/make_train_old.py               # classify, write, upload, push
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent
ENV_REF = "GeneralReasoning/WhoDunit"
GATEWAY_URL = "https://infer.gr.inc/v1/chat/completions"
MODEL = "deepseek-v4.1-flash"
VOTES = 2
SPLIT_FILE = REPO / "split_train_old.json"
# The identity the GitHub token belongs to. No co-author trailers.
GIT_NAME = "louisrch"
GIT_EMAIL = "louis.roche@icloud.com"
CACHE = Path.home() / ".cache" / "whodunnit_train_old" / "verdicts.jsonl"

# Hints only (see module docstring). Deliberately broad: a false hint costs nothing.
HINT_TERMS = [
    "smartphone", "iphone", "android", "app", "apps", "selfie", "influencer", "podcast",
    "stream", "streaming", "livestream", "binge", "vlog", "blog", "hashtag", "meme",
    "emoji", "text message", "texted", "texting", "google", "googled", "wikipedia",
    "facebook", "twitter", "tweet", "instagram", "tiktok", "youtube", "netflix",
    "uber", "lyft", "airbnb", "crypto", "bitcoin", "nft", "blockchain", "wifi", "wi-fi",
    "bluetooth", "tablet", "ipad", "cloud", "zoom", "drone", "gps", "usb", "dvd",
    "online", "website", "internet", "email", "e-mail", "startup", "tech bro", "ai",
    "algorithm", "ghosted", "ghosting", "unfriend", "fomo", "woke", "vape", "vaping",
    "e-scooter", "hoverboard", "3d print", "smartwatch", "fitbit", "wearable",
]
_HINT_RE = re.compile(r"\b(" + "|".join(re.escape(t) for t in HINT_TERMS) + r")\b", re.I)

RUBRIC = """\
You are auditing a murder-mystery puzzle for a dataset restricted to what existed on
or before 1999-12-31. Decide whether the text below relies on anything from after 1999.

FLAG the case (post_2000 = true) if it contains ANY of:
- CONCEPTS: technology, products, services, platforms, institutions or social
  phenomena that did not exist, or were not in ordinary use, before 2000 --
  smartphones and apps, social media, texting as an everyday habit, streaming,
  podcasts, ride-hailing, influencers, crypto, consumer drones, Wi-Fi, smartwatches,
  video calls as routine, and the like.
- LANGUAGE / LINGO: words, slang or senses of words that date from after 2000, or took
  their modern meaning after 2000 -- e.g. "selfie", "influencer", "ghosting",
  "binge-watch", "vlog", "hashtag", "meme" in the internet sense, "google" as a verb,
  "unfriend", "emoji", "livestream", "FOMO", "woke", "tech bro".
- KNOWLEDGE: references to events, people, works, discoveries or facts from after 1999.

DO NOT FLAG:
- Settings in any past era, or timeless ones.
- Pre-2000 technology: landlines, fax, pagers, VCRs, early computers, email and the
  early web of the 1990s, mobile phones used as phones, CD-ROMs.
- Invented or fantastical elements (a spell, a time machine, a robot butler) that are
  fiction rather than a post-2000 real-world concept.
- Ordinary words that merely resemble modern terms: a stone TABLET, a mountain STREAM,
  a storm CLOUD, a camera ZOOM lens, a honey-bee DRONE.

Judge the CONTENT, not the date of writing: a puzzle written recently is fine if
nothing in it is post-1999.

Some words were flagged by a keyword scan and are listed as hints. A hint is only a
lead: decide from context whether it is a post-2000 use.

Reply with ONLY a JSON object:
{"post_2000": true or false,
 "evidence": [{"quote": "an exact, short quotation from the text", "why": "what is post-2000 about it"}]}
Every flagged item MUST be backed by a quote copied character-for-character from the
text. If post_2000 is false, evidence is [].
"""


def load_tasks() -> list[dict]:
    """Same files, same order as whodunnit.py, so indices are task_ids."""
    tasks: list[dict] = []
    for name in ("tasks_elementary.json", "tasks_impossible.json"):
        path = REPO / name
        if path.exists():
            tasks.extend(json.loads(path.read_text()))
    return tasks


def render(obj) -> str:
    """Every piece of text a player could be shown, one item per line."""
    lines: list[str] = []

    def walk(x, key=""):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, k)
        elif isinstance(x, list):
            for v in x:
                walk(v, key)
        elif isinstance(x, str):
            lines.append(f"{key}: {x}" if key else x)

    walk(obj)
    return "\n".join(lines)


def _norm(s: str) -> str:
    return " ".join(s.lower().replace("’", "'").replace("“", '"').replace("”", '"').split())


def call_model(text: str, hints: list[str]) -> dict:
    key = os.environ["KIMI_API_KEY"]
    hint_line = ", ".join(sorted(set(h.lower() for h in hints))) or "(none)"
    user = f"{RUBRIC}\nKEYWORD HINTS: {hint_line}\n\nTEXT:\n{text}"
    last = None
    for attempt in range(4):
        try:
            r = requests.post(
                GATEWAY_URL,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": MODEL, "temperature": 0.2, "max_tokens": 1500,
                      "messages": [{"role": "user", "content": user}]},
                timeout=180,
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"] or ""
            m = re.search(r"\{.*\}", content, re.S)
            if not m:
                raise ValueError(f"no JSON in reply: {content[:200]!r}")
            return json.loads(m.group(0))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"model call failed after retries: {last}")


def judge(kind: str, index: int, text: str, cache: dict, cache_lock_path: Path) -> dict:
    """VOTES independent verdicts, each keeping only evidence quoted verbatim."""
    digest = hashlib.sha1(text.encode()).hexdigest()[:16]
    hints = _HINT_RE.findall(text)
    norm_text = _norm(text)
    votes = []
    for v in range(VOTES):
        ck = f"{kind}:{index}:{digest}:{v}"
        raw = cache.get(ck)
        if raw is None:
            raw = call_model(text, hints)
            with open(cache_lock_path, "a") as f:
                f.write(json.dumps({"key": ck, "verdict": raw}) + "\n")
            cache[ck] = raw
        evidence = [e for e in (raw.get("evidence") or [])
                    if isinstance(e, dict) and e.get("quote")
                    and _norm(str(e["quote"])) in norm_text]
        claimed = bool(raw.get("post_2000"))
        votes.append({
            "claimed_post_2000": claimed,
            "flagged": claimed and bool(evidence),   # a flag needs a real quote
            "evidence": evidence,
            "unverified_claim": claimed and not evidence,
        })
    return {"kind": kind, "index": index, "hints": sorted(set(h.lower() for h in hints)),
            "flagged": any(v["flagged"] for v in votes), "votes": votes}


def run(cmd: list[str], dry: bool, check: bool = True) -> subprocess.CompletedProcess:
    print("+ " + " ".join(cmd))
    if dry:
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return subprocess.run(cmd, cwd=REPO, check=check, text=True, capture_output=not check)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="classify and report, but do not write, upload or push")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if not os.environ.get("KIMI_API_KEY"):
        print("error: KIMI_API_KEY is not set", file=sys.stderr)
        return 1

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    cache: dict = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                cache[rec["key"]] = rec["verdict"]

    tasks = load_tasks()
    exhibits = json.loads((REPO / "exhibits.json").read_text())
    print(f"judging {len(tasks)} cases and {len(exhibits)} shared exhibits "
          f"with {MODEL}, {VOTES} votes each ({len(cache)} cached verdicts)")

    jobs = [("task", i, render(t)) for i, t in enumerate(tasks)] + \
           [("exhibit", i, render(e)) for i, e in enumerate(exhibits)]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(lambda j: judge(j[0], j[1], j[2], cache, CACHE), jobs))

    task_results = [r for r in results if r["kind"] == "task"]
    exhibit_results = [r for r in results if r["kind"] == "exhibit"]
    kept = [{"task_id": r["index"], "problem_name": tasks[r["index"]]["problem_name"]}
            for r in task_results if not r["flagged"]]
    dropped = [{"task_id": r["index"], "problem_name": tasks[r["index"]]["problem_name"],
                "evidence": [e for v in r["votes"] for e in v["evidence"]]}
               for r in task_results if r["flagged"]]
    unverified = sum(v["unverified_claim"] for r in results for v in r["votes"])
    split_votes = sum(1 for r in task_results
                      if len({v["flagged"] for v in r["votes"]}) > 1)
    flagged_exhibits = [{"exhibit_id": exhibits[r["index"]].get("exhibit_id"),
                         "title": exhibits[r["index"]].get("title"),
                         "evidence": [e for v in r["votes"] for e in v["evidence"]]}
                        for r in exhibit_results if r["flagged"]]

    print(f"\nkept {len(kept)} / {len(tasks)} cases; dropped {len(dropped)}")
    print(f"  cases where the two votes disagreed (resolved to drop): {split_votes}")
    print(f"  post-2000 claims discarded for lack of a verbatim quote: {unverified}")
    for d in dropped:
        ev = "; ".join(f'"{e["quote"]}" ({e["why"]})' for e in d["evidence"][:3])
        print(f"  - DROP #{d['task_id']:>3} {d['problem_name']}: {ev}")
    if flagged_exhibits:
        print("\nWARNING: shared exhibits flagged -- these are shown in EVERY case, "
              "including train_old ones, so filtering cases cannot remove them:")
        for e in flagged_exhibits:
            print(f"  - exhibit {e['exhibit_id']} {e['title']}: " +
                  "; ".join(f'"{x["quote"]}"' for x in e["evidence"][:3]))
    else:
        print("  shared exhibits: none flagged")

    if not kept:
        print("error: no case survived; refusing to write an empty split", file=sys.stderr)
        return 1

    split = {
        "split": "train_old",
        "description": "Train cases using no post-2000 concepts, language or knowledge.",
        "model": MODEL,
        "votes_per_case": VOTES,
        "rule": "dropped if any vote flags it with a quote found verbatim in the case",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tasks": kept,
        "dropped": dropped,
        "flagged_exhibits": flagged_exhibits,
    }

    if args.dry_run:
        print("\n(dry run: nothing written, uploaded or pushed; verdicts are cached)")
        return 0

    SPLIT_FILE.write_text(json.dumps(split, indent=2) + "\n")
    print(f"\nwrote {SPLIT_FILE.name} ({len(kept)} cases)")

    # Sanity: the server, loading this file, must serve exactly the kept cases.
    check = subprocess.run(
        [sys.executable, "-c",
         # Load from this checkout even if the host has an /orwd_data of its own.
         "import os, json; _e=os.path.exists; os.path.exists=lambda p: p!='/orwd_data' and _e(p);"
         "import whodunnit as w; os.path.exists=_e; ids=[t['task_id'] for t in w.Whodunnit.list_tasks('train_old')];"
         "print(json.dumps({'splits': w.Whodunnit.list_splits(), 'ids': ids}))"],
        cwd=REPO, capture_output=True, text=True)
    served = json.loads(check.stdout.strip().splitlines()[-1])
    if served["ids"] != [t["task_id"] for t in kept]:
        print(f"error: server would serve {served}, not the kept cases", file=sys.stderr)
        return 1
    print(f"server check: splits={served['splits']}, train_old serves {len(served['ids'])} cases")

    print("\n=== upload to OpenReward file store ===")
    run(["orwd", "upload", ENV_REF, SPLIT_FILE.name], dry=False)

    print("\n=== commit and push ===")
    run(["git", "add", "whodunnit.py", SPLIT_FILE.name, "scripts/make_train_old.py"], dry=False)
    msg = (f"Add train_old split: {len(kept)} of {len(tasks)} cases with no post-2000 content\n\n"
           f"Each case is judged twice by {MODEL} for post-2000 concepts, language or\n"
           f"knowledge, and dropped if either verdict flags it with a quote found verbatim\n"
           f"in the case. split_train_old.json lists the kept cases and every dropped\n"
           f"case with its evidence. The server lists train_old only when that file is\n"
           f"present, and ignores any entry whose problem_name no longer matches its index.")
    run(["git", "-c", f"user.name={GIT_NAME}", "-c", f"user.email={GIT_EMAIL}",
         "commit", "-q", "-m", msg], dry=False)
    push = subprocess.run(["git", "push", "origin", "HEAD:main"], cwd=REPO, text=True)
    if push.returncode != 0:
        print("push to main was rejected; pushing a branch instead")
        run(["git", "push", "origin", "HEAD:refs/heads/train-old-split"], dry=False)
        print("open a PR from train-old-split; the environment rebuilds once it is merged")
        return 0
    print(f"\ndone: {ENV_REF} rebuilds from main; train_old will list {len(kept)} cases")
    print(f"      follow it with: orwd deployments {ENV_REF}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

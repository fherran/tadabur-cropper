"""Build mutashabihat.json: Quran-wide pairs of ayahs sharing >=80% of their
words (al-mutashabihat al-lafdhiyya), with the differing word ranges the
cropper's verification duel needs. Text-only; run once."""
import difflib
import json
import ssl
import urllib.request
from collections import Counter

import certifi

API = "https://quranapi.pages.dev/api/{surah}.json"
THRESHOLD = 0.8


def fetch(surah):
    ctx = ssl.create_default_context(cafile=certifi.where())
    req = urllib.request.Request(API.format(surah=surah), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
        return json.loads(r.read().decode())["arabic2"]


def differ_range(wa, wb):
    sm = difflib.SequenceMatcher(a=wa, b=wb, autojunk=False)
    da, db = set(range(len(wa))), set(range(len(wb)))
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            da -= set(range(i1, i2))
            db -= set(range(j1, j2))
    return ([min(da), max(da)] if da else None), ([min(db), max(db)] if db else None)


def main():
    words = {}
    for s in range(1, 115):
        for a, t in enumerate(fetch(s), start=1):
            words[f"{s}:{a}"] = t.split()
    print(f"fetched {len(words)} ayahs")

    keys = list(words)
    counters = {k: Counter(v) for k, v in words.items()}
    table = {}
    n_pairs = 0
    for i, ka in enumerate(keys):
        wa, ca = words[ka], counters[ka]
        for kb in keys[i + 1:]:
            wb, cb = words[kb], counters[kb]
            shared = sum((ca & cb).values())
            if shared < THRESHOLD * min(len(wa), len(wb)):
                continue
            ra, rb = differ_range(wa, wb)
            table.setdefault(ka, {})[kb] = {"mine": ra, "theirs": rb, "n_mine": len(wa), "n_theirs": len(wb)}
            table.setdefault(kb, {})[ka] = {"mine": rb, "theirs": ra, "n_mine": len(wb), "n_theirs": len(wa)}
            n_pairs += 1
    import os
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mutashabihat.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False)
    print(f"pairs: {n_pairs}   ayahs with rivals: {len(table)}")
    print("26:161 rivals:", sorted(table.get("26:161", {})))


if __name__ == "__main__":
    main()

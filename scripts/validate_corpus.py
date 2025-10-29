import argparse
import json
import os


def main():
    """Validate text files by length and emit a JSON summary."""
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    args = ap.parse_args()
    rep = {"files": 0, "ok": 0, "too_short": 0, "empty": 0, "samples": [], "read_errors": 0}
    for name in sorted(os.listdir(args.dir)):
        if not name.endswith(".txt"):
            continue
        path = os.path.join(args.dir, name)
        rep["files"] += 1
        try:
            with open(path, "r", encoding="utf-8") as source:
                txt = source.read()
        except OSError:
            rep["read_errors"] += 1
            continue
        wc = len(txt.split())
        if wc == 0:
            rep["empty"] += 1
        elif wc < 80:
            rep["too_short"] += 1
        else:
            rep["ok"] += 1
            if len(rep["samples"]) < 5:
                snippet = txt[:200].replace("\n", " ")
                rep["samples"].append({"file": name, "words": wc, "preview": snippet})
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

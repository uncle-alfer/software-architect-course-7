import subprocess
import sys

CASES = [
    ("Кто такой Ноктиль-Мор?", "Ноктиль-Мор"),
    ("Где находится Силентор?", "Силентор"),
    ("Что такое Нордеверо?", "Нордеверо"),
    ("Кто такой Альваро Дорн?", "Альваро"),
]


def parse_top3(stdout: str):
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip().startswith("[")]
    return lines[:3]


def run_case(q, expect_substr):
    print(f"\n=== {q} ===")
    p = subprocess.run(
        ["python", "scripts/index/search_index.py", "--query", q, "--top-k", "5", "--cand-mult", "10"],
        capture_output=True,
        text=True,
    )
    out = p.stdout + p.stderr
    print(out)
    top3 = "\n".join(parse_top3(out))
    ok = expect_substr.lower() in top3.lower()
    return ok


def main():
    bad = 0
    for q, exp in CASES:
        if not run_case(q, exp):
            bad += 1
    if bad:
        print(f"\nSanity checks failed: {bad} of {len(CASES)} not found in TOP-3.")
        sys.exit(1)
    print("\nAll sanity checks passed (TOP-3).")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Verify the main results of the FEGAR manuscript from the released per-subject scores.

Usage:  python3 verify_main_results.py

Reads data/main_scores.csv, data/backbone_scores.csv, and data/d4j_scores.csv, recomputes
every value below, and prints it next to the value reported in the paper:
  * Table 3: success rate and mutation score of the seven approaches on HumanEval-Java and
    LeetCode-Java, and the paired comparison of FEGAR with each baseline;
  * the seven-backbone comparison of FEGAR with vanilla prompting;
  * the Defects4J comparison of FEGAR with EvoSuite and vanilla prompting.
A subject without a valid test scores zero. One LeetCode-Java subject has no MUTGEN run, since
its released pipeline cannot process that subject; as in the paper, it scores zero in the mean
and is left out of the paired comparison. Comparisons are paired at the subject level: the
two-sided Wilcoxon signed-rank test (normal approximation with tie correction) for mutation
scores, the exact McNemar test for success, and the Vargha-Delaney A12 effect size. The
confirmatory family has 32 tests under Bonferroni correction (threshold 0.05/32).
Only the Python standard library is required. Exit status 0 means every value matches.
"""
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent / "data"
ALPHA = 0.05 / 32

# Values as reported in the paper.
PAPER = {
    "main": {
        "HumanEval-Java|EvoSuite": {
            "success": "100.0",
            "ms": "64.4",
            "a12": "0.87",
            "p": "<0.001",
            "bonferroni": "yes"
        },
        "HumanEval-Java|EvoSuite_mut": {
            "success": "98.1",
            "ms": "62.8",
            "a12": "0.88",
            "p": "<0.001",
            "bonferroni": "yes"
        },
        "HumanEval-Java|ChatUniTest": {
            "success": "91.3",
            "ms": "76.5",
            "a12": "0.65",
            "p": "<0.001",
            "bonferroni": "yes"
        },
        "HumanEval-Java|HITS": {
            "success": "100.0",
            "ms": "91.2",
            "a12": "0.55",
            "p": "0.018",
            "bonferroni": "no"
        },
        "HumanEval-Java|MUTGEN": {
            "success": "99.0",
            "ms": "85.8",
            "a12": "0.62",
            "p": "<0.001",
            "bonferroni": "yes"
        },
        "HumanEval-Java|Vanilla": {
            "success": "91.3",
            "ms": "83.4",
            "a12": "0.58",
            "p": "<0.001",
            "bonferroni": "yes"
        },
        "HumanEval-Java|FEGAR": {
            "success": "100.0",
            "ms": "93.4"
        },
        "LeetCode-Java|EvoSuite": {
            "success": "92.0",
            "ms": "66.1",
            "a12": "0.74",
            "p": "<0.001",
            "bonferroni": "yes"
        },
        "LeetCode-Java|EvoSuite_mut": {
            "success": "95.0",
            "ms": "70.6",
            "a12": "0.72",
            "p": "<0.001",
            "bonferroni": "yes"
        },
        "LeetCode-Java|ChatUniTest": {
            "success": "61.0",
            "ms": "45.3",
            "a12": "0.82",
            "p": "<0.001",
            "bonferroni": "yes"
        },
        "LeetCode-Java|HITS": {
            "success": "88.0",
            "ms": "75.0",
            "a12": "0.59",
            "p": "<0.001",
            "bonferroni": "yes"
        },
        "LeetCode-Java|MUTGEN": {
            "success": "96.0",
            "ms": "74.3",
            "a12": "0.64",
            "p": "<0.001",
            "bonferroni": "yes"
        },
        "LeetCode-Java|Vanilla": {
            "success": "88.0",
            "ms": "68.1",
            "a12": "0.72",
            "p": "<0.001",
            "bonferroni": "yes"
        },
        "LeetCode-Java|FEGAR": {
            "success": "100.0",
            "ms": "90.0"
        }
    },
    "backbone": {
        "Qwen2.5-Coder-32B|HumanEval-Java": {
            "success_vanilla": "91.3",
            "success_fegar": "100.0",
            "ms_vanilla": "83.4",
            "ms_fegar": "93.4",
            "a12": "0.58",
            "p": "<0.001",
            "bonferroni": "yes",
            "mcnemar_p": "0.004"
        },
        "Qwen2.5-Coder-32B|LeetCode-Java": {
            "success_vanilla": "88.0",
            "success_fegar": "100.0",
            "ms_vanilla": "68.1",
            "ms_fegar": "90.0",
            "a12": "0.72",
            "p": "<0.001",
            "bonferroni": "yes",
            "mcnemar_p": "<0.001"
        },
        "DeepSeek-Coder-33B|HumanEval-Java": {
            "success_vanilla": "76.9",
            "success_fegar": "94.2",
            "ms_vanilla": "65.8",
            "ms_fegar": "83.1",
            "a12": "0.61",
            "p": "<0.001",
            "bonferroni": "yes",
            "mcnemar_p": "<0.001"
        },
        "DeepSeek-Coder-33B|LeetCode-Java": {
            "success_vanilla": "38.0",
            "success_fegar": "93.0",
            "ms_vanilla": "27.3",
            "ms_fegar": "72.0",
            "a12": "0.80",
            "p": "<0.001",
            "bonferroni": "yes",
            "mcnemar_p": "<0.001"
        },
        "CodeLlama-34B|HumanEval-Java": {
            "success_vanilla": "35.6",
            "success_fegar": "97.1",
            "ms_vanilla": "31.3",
            "ms_fegar": "86.5",
            "a12": "0.81",
            "p": "<0.001",
            "bonferroni": "yes",
            "mcnemar_p": "<0.001"
        },
        "CodeLlama-34B|LeetCode-Java": {
            "success_vanilla": "27.0",
            "success_fegar": "60.0",
            "ms_vanilla": "13.2",
            "ms_fegar": "43.3",
            "a12": "0.70",
            "p": "<0.001",
            "bonferroni": "yes",
            "mcnemar_p": "<0.001"
        },
        "Llama-3.3-70B|HumanEval-Java": {
            "success_vanilla": "83.7",
            "success_fegar": "100.0",
            "ms_vanilla": "75.4",
            "ms_fegar": "93.0",
            "a12": "0.63",
            "p": "<0.001",
            "bonferroni": "yes",
            "mcnemar_p": "<0.001"
        },
        "Llama-3.3-70B|LeetCode-Java": {
            "success_vanilla": "76.0",
            "success_fegar": "99.0",
            "ms_vanilla": "57.9",
            "ms_fegar": "87.5",
            "a12": "0.72",
            "p": "<0.001",
            "bonferroni": "yes",
            "mcnemar_p": "<0.001"
        },
        "Claude Sonnet 5|HumanEval-Java": {
            "success_vanilla": "84.6",
            "success_fegar": "100.0",
            "ms_vanilla": "79.0",
            "ms_fegar": "95.1",
            "a12": "0.61",
            "p": "<0.001",
            "bonferroni": "yes",
            "mcnemar_p": "<0.001"
        },
        "Claude Sonnet 5|LeetCode-Java": {
            "success_vanilla": "50.0",
            "success_fegar": "100.0",
            "ms_vanilla": "44.8",
            "ms_fegar": "92.6",
            "a12": "0.80",
            "p": "<0.001",
            "bonferroni": "yes",
            "mcnemar_p": "<0.001"
        },
        "GPT-5.6 Sol|HumanEval-Java": {
            "success_vanilla": "98.1",
            "success_fegar": "100.0",
            "ms_vanilla": "92.6",
            "ms_fegar": "95.5",
            "a12": "0.57",
            "p": "0.003",
            "bonferroni": "no",
            "mcnemar_p": "0.500"
        },
        "GPT-5.6 Sol|LeetCode-Java": {
            "success_vanilla": "90.0",
            "success_fegar": "100.0",
            "ms_vanilla": "83.2",
            "ms_fegar": "95.8",
            "a12": "0.67",
            "p": "<0.001",
            "bonferroni": "yes",
            "mcnemar_p": "0.002"
        },
        "Gemini 3.1 Pro|HumanEval-Java": {
            "success_vanilla": "66.3",
            "success_fegar": "100.0",
            "ms_vanilla": "62.8",
            "ms_fegar": "93.0",
            "a12": "0.65",
            "p": "<0.001",
            "bonferroni": "yes",
            "mcnemar_p": "<0.001"
        },
        "Gemini 3.1 Pro|LeetCode-Java": {
            "success_vanilla": "42.0",
            "success_fegar": "100.0",
            "ms_vanilla": "38.2",
            "ms_fegar": "86.1",
            "a12": "0.77",
            "p": "<0.001",
            "bonferroni": "yes",
            "mcnemar_p": "<0.001"
        }
    },
    "d4j": {
        "EvoSuite": {
            "success": "98.8",
            "ms": "52.5",
            "a12": "0.61",
            "p": "0.001"
        },
        "Vanilla": {
            "success": "39.8",
            "ms": "22.3",
            "a12": "0.81",
            "p": "<0.001"
        },
        "FEGAR": {
            "success": "98.8",
            "ms": "65.5"
        }
    }
}


def a12(x, y):
    gt = sum(1 for a in x for b in y if a > b)
    eq = sum(1 for a in x for b in y if a == b)
    return (gt + 0.5 * eq) / (len(x) * len(y))


def wilcoxon(x, y):
    d = [a - b for a, b in zip(x, y) if a != b]
    n = len(d)
    if n < 6:
        return float("nan")
    ranked = sorted((abs(v), i) for i, v in enumerate(d))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ranked[j + 1][0] == ranked[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[ranked[k][1]] = (i + j) / 2 + 1
        i = j + 1
    w_plus = sum(r for r, v in zip(ranks, d) if v > 0)
    mu = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24 - sum(t ** 3 - t for t in Counter(v for v, _ in ranked).values()) / 48
    z = (w_plus - mu) / math.sqrt(var)
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def mcnemar(x, y):
    b = sum(1 for u, v in zip(x, y) if u and not v)
    c = sum(1 for u, v in zip(x, y) if v and not u)
    n = b + c
    if n == 0:
        return float("nan")
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def fmt_p(p):
    return "<0.001" if p < 1e-3 else f"{p:.3f}"


def load(name, keys):
    table = defaultdict(dict)
    with open(HERE / name, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            table[tuple(row[k] for k in keys)][row["subject"]] = (int(row["success"]), float(row["ms"]))
    return table


def series(table, key, subjects):
    """Scores and success flags over `subjects`; a subject without a run scores zero."""
    got = [table[key].get(s, (0, 0.0)) for s in subjects]
    return [v[1] for v in got], [v[0] for v in got]


checks = []


def check(where, what, got, paper):
    checks.append(got == paper)
    print(f"  [{'PASS' if got == paper else 'FAIL'}] {where:<38} {what:<16} computed {got:>8}   paper {paper:>8}")


def main():
    main_t = load("main_scores.csv", ("benchmark", "approach"))
    print("Table 3 (Qwen2.5-Coder-32B; MUTGEN on its own backbone)")
    for bench in ("HumanEval-Java", "LeetCode-Java"):
        subjects = sorted(main_t[(bench, "FEGAR")])
        for approach in ("EvoSuite", "EvoSuite_mut", "ChatUniTest", "HITS", "MUTGEN", "Vanilla", "FEGAR"):
            ms, ok = series(main_t, (bench, approach), subjects)
            exp = PAPER["main"][f"{bench}|{approach}"]
            where = f"{bench} {approach}"
            check(where, "success (%)", f"{100 * sum(ok) / len(ok):.1f}", exp["success"])
            check(where, "MS (%)", f"{100 * sum(ms) / len(ms):.1f}", exp["ms"])
            if approach != "FEGAR":
                common = [s for s in subjects if s in main_t[(bench, approach)]]
                f_c, _ = series(main_t, (bench, "FEGAR"), common)
                b_c, _ = series(main_t, (bench, approach), common)
                p = wilcoxon(f_c, b_c)
                check(where, "A12", f"{a12(f_c, b_c):.2f}", exp["a12"])
                check(where, "p", fmt_p(p), exp["p"])
                check(where, "Bonferroni", "yes" if p < ALPHA else "no", exp["bonferroni"])

    bb = load("backbone_scores.csv", ("backbone", "benchmark", "approach"))
    print("\nFEGAR against vanilla prompting on seven backbones")
    for key, exp in PAPER["backbone"].items():
        backbone, bench = key.split("|")
        subjects = sorted(bb[(backbone, bench, "FEGAR")])
        f_ms, f_ok = series(bb, (backbone, bench, "FEGAR"), subjects)
        v_ms, v_ok = series(bb, (backbone, bench, "Vanilla"), subjects)
        where = f"{backbone} {bench}"
        p = wilcoxon(f_ms, v_ms)
        check(where, "success vanilla", f"{100 * sum(v_ok) / len(v_ok):.1f}", exp["success_vanilla"])
        check(where, "success FEGAR", f"{100 * sum(f_ok) / len(f_ok):.1f}", exp["success_fegar"])
        check(where, "MS vanilla", f"{100 * sum(v_ms) / len(v_ms):.1f}", exp["ms_vanilla"])
        check(where, "MS FEGAR", f"{100 * sum(f_ms) / len(f_ms):.1f}", exp["ms_fegar"])
        check(where, "A12", f"{a12(f_ms, v_ms):.2f}", exp["a12"])
        check(where, "p", fmt_p(p), exp["p"])
        check(where, "Bonferroni", "yes" if p < ALPHA else "no", exp["bonferroni"])
        check(where, "McNemar p", fmt_p(mcnemar(f_ok, v_ok)), exp["mcnemar_p"])

    d4j = defaultdict(dict)
    with open(HERE / "d4j_scores.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            d4j[row["approach"]][row["subject"]] = (int(row["success"]), float(row["ms"]))
    print("\nDefects4J (83 real-world classes)")
    subjects = sorted(d4j["FEGAR"])
    f_ms = [d4j["FEGAR"][s][1] for s in subjects]
    for approach in ("EvoSuite", "Vanilla", "FEGAR"):
        ms = [d4j[approach][s][1] for s in subjects]
        ok = [d4j[approach][s][0] for s in subjects]
        exp = PAPER["d4j"][approach]
        check(f"Defects4J {approach}", "success (%)", f"{100 * sum(ok) / len(ok):.1f}", exp["success"])
        check(f"Defects4J {approach}", "MS (%)", f"{100 * sum(ms) / len(ms):.1f}", exp["ms"])
        if approach != "FEGAR":
            check(f"Defects4J {approach}", "A12", f"{a12(f_ms, ms):.2f}", exp["a12"])
            check(f"Defects4J {approach}", "p", fmt_p(wilcoxon(f_ms, ms)), exp["p"])

    passed = sum(checks)
    print(f"\n{passed} of {len(checks)} values match the paper.")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())

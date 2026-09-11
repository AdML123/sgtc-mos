"""Lint paper/main.tex against docs/WRITING-RULES.md (Q1/Q4/Q6/Q7 automatable part).

Run:  python -m sgtc.lint_paper
"""
import re
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent / "paper" / "main.tex"
text = PAPER.read_text(encoding="utf-8")
issues = []

# strip comments and math for prose checks
prose = re.sub(r"(?s)%.*", "", text)
prose = re.sub(r"(?s)\\begin\{equation\}.*?\\end\{equation\}", " EQ ", prose)
prose = re.sub(r"\$[^$]*\$", " MATH ", prose)

# Q1 checks
banned = ["delve", "shed light", "advance our understanding", "pave the way",
          "it is worth noting", "plays a pivotal role", "plays a critical role",
          "not only", "in contrast", "taken together", "overall,", "in summary",
          "rather than", "robust", "dynamic", "significant improvement"]
for w in banned:
    n = len(re.findall(re.escape(w), prose, flags=re.I))
    if n:
        issues.append(f"Q1 banned phrase '{w}': {n}x")
n_em = prose.count("---") + len(re.findall(r"(?<![-\w])--(?![-\w])", prose))
if n_em > 2:
    issues.append(f"Q1 em-dash-like '--': {n_em}x (max 2)")
n_seq = len(re.findall(r"(?i)\bfirst[,.].*\bsecond[,.]", prose))
if n_seq:
    issues.append(f"Q1 explicit First/Second sequencing: {n_seq}x")
n_colon = len(re.findall(r"[a-z]\s*:\s+[a-z]", prose))
if n_colon > 3:
    issues.append(f"Q1 colon-explanation: {n_colon}x (max 3)")
n_semi = prose.count("; ")
if n_semi > 4:
    issues.append(f"Q1 semicolon joins: {n_semi}x (max 4)")

# Q4: citation order strictly ascending on first appearance
firsts = []
for m in re.finditer(r"\\cite\{([^}]*)\}", text):
    for k in m.group(1).split(","):
        k = k.strip()
        if k and k not in firsts:
            firsts.append(k)
print("citation first-appearance order:", firsts)

# Q6: abbreviations used before expansion (rough check for our list)
abbr_checks = [("LCC", "linear correlation coefficient"),
               ("SRCC", "Spearman rank correlation coefficient"),
               ("RMSE", "root mean square error"),
               ("MOS", "mean opinion score"),
               ("FFN", None)]
for ab, full in abbr_checks:
    i_ab = prose.find(ab)
    i_full = prose.lower().find((full or "").lower()) if full else -1
    if full and 0 <= i_ab < i_full:
        issues.append(f"Q6 abbreviation {ab} appears before its expansion")

print(f"\n{len(issues)} issue(s):")
for i in issues:
    print(" -", i)
if not issues:
    print("ALL AUTOMATED CHECKS PASS")

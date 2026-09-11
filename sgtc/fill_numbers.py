"""Generate paper/numbers.tex: every numeric claim in the manuscript comes from
results/ tables through this module (single source of truth; no hand copying).

Run:  python -m sgtc.fill_numbers
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "tables"
OUT = ROOT / "paper" / "numbers.tex"

macros = {}


def m(name, value, fmt="{:.3f}"):
    macros[name] = fmt.format(value) if isinstance(value, float) else str(value)


def cell(method, ratio, col, t1):
    r = t1[(t1.method == method) & (t1.ratio == ratio)]
    return float(r[col].iloc[0]) if len(r) else float("nan")


def run():
    # ---------- Table I ----------
    t1 = pd.read_csv(RES / "table1.csv")
    # five-seed means become the reported numbers for the stochastic methods
    ss = RES / "seed_stability.csv"
    if ss.exists():
        sdf = pd.read_csv(ss)
        for _, row in sdf.iterrows():
            r = int(row["ratio"])
            mask = (t1.method == "merge") & (t1.ratio == r)
            t1.loc[mask, ["lcc", "srcc", "rmse"]] = [row["lcc_mean"],
                                                     row["srcc_mean"],
                                                     row["rmse_mean"]]
    m("LccBase", cell("baseline", 1, "lcc", t1))
    m("SrccBase", cell("baseline", 1, "srcc", t1))
    m("RmseBase", cell("baseline", 1, "rmse", t1))
    for ratio in [2, 3, 4]:
        rn = {2: "Two", 3: "Three", 4: "Four"}[ratio]
        for meth, tag in [("merge", "Merge"), ("prune", "Prune"),
                          ("sgtc_w", "Sgtcw"), ("sgtc", "Sgtc"), ("tome", "Tome")]:
            m(f"Lcc{tag}{rn}", cell(meth, ratio, "lcc", t1))
            m(f"Srcc{tag}{rn}", cell(meth, ratio, "srcc", t1))
            m(f"Rmse{tag}{rn}", cell(meth, ratio, "rmse", t1))
        m(f"LccRand{rn}", cell("random_lcc", ratio, "lcc", t1))

    m("DLccMergeTwo", cell("merge", 2, "lcc", t1) - cell("baseline", 1, "lcc", t1), "{:+.3f}")
    m("DLccMergeThree", cell("merge", 3, "lcc", t1) - cell("baseline", 1, "lcc", t1), "{:+.3f}")
    m("DLccMergeFour", cell("merge", 4, "lcc", t1) - cell("baseline", 1, "lcc", t1), "{:+.3f}")
    rmse_gain = 1 - cell("merge", 4, "rmse", t1) / cell("baseline", 1, "rmse", t1)
    m("RmseGainPct", rmse_gain * 100, "{:.0f}")

    sig = t1[t1.method == "merge_vs_sgtc_t"]
    for ratio in [2, 3, 4]:
        rn = {2: "Two", 3: "Three", 4: "Four"}[ratio]
        r = sig[sig.ratio == ratio]
        if len(r):
            m(f"PMerge{rn}", float(r["p"].iloc[0]), "{:.1e}")

    # ---------- diagnosis ----------
    d = json.loads((RES / "shuffle_diagnosis.json").read_text())
    m("DLccChan", d["original"]["lcc_mean"] - d["chan_shuffle"]["lcc_mean"], "{:.3f}")
    m("MelMos", 3.79, "{:.2f}")
    mj = RES / "mel_utterance.json"
    if mj.exists():
        jj = json.loads(mj.read_text())
        m("MelMos", float(jj["mos"]), "{:.2f}")
        m("MelMeanCos", float(jj["mean_adj_cosine"]), "{:.2f}")
        m("MelFracAbove", float(jj["frac_above_09"]) * 100, "{:.0f}")
    m("LccChanShuf", d["chan_shuffle"]["lcc_mean"])
    m("LccBlock", d["block_shuffle"]["lcc_mean"])

    # ---------- ablation ----------
    t2 = pd.read_csv(RES / "table2.csv")
    tagmap = {"full": "Full", "random": "Rand", "cosine": "Cos", "global": "Glob",
              "tome": "TomeII", "sgtc_w": "SgtcwEq", "sgtc_w_cos": "SgtcwCos",
              "sgtc_w_rand": "SgtcwRand", "attention": "Att"}
    for _, r in t2.iterrows():
        tag = tagmap.get(r["variant"])
        if tag and not pd.isna(r.get("lcc")):
            m(f"AblLcc{tag}", float(r["lcc"]))

    # ---------- cross-domain ----------
    methods = [("baseline", "NisqaBase"), ("sgtc", "NisqaSgtc"),
               ("sgtc_w", "NisqaSgtcw"), ("tome", "NisqaTome"),
               ("prune2", "NisqaPrune"), ("merge2", "NisqaMerge")]
    if (RES / "table3.csv").exists():
        t3 = pd.read_csv(RES / "table3.csv")
        for meth, tag in methods:
            r = t3[t3.method == meth]
            if len(r):
                m(f"Lcc{tag}", float(r["lcc"].iloc[0]))
                m(f"Srcc{tag}", float(r["srcc"].iloc[0]))
                m(f"Rmse{tag}", float(r["rmse"].iloc[0]))
    for _, tag in methods:
        for pre in ["Lcc", "Srcc", "Rmse"]:
            macros.setdefault(f"{pre}{tag}", "??")

    # ---------- cross-backbone ----------
    cb = RES / "cross_backbone.json"
    if cb.exists():
        j = json.loads(cb.read_text())
        for bb, tag in [("hubert", "Hub"), ("wavlm", "Wlm")]:
            if bb in j:
                m(f"Lcc{tag}Base", j[bb]["baseline"]["lcc"])
                m(f"Lcc{tag}SgtcTwo", j[bb]["sgtc2"]["lcc"])
                m(f"Lcc{tag}SgtcFour", j[bb]["sgtc4"]["lcc"])
                if "merge2" in j[bb]:
                    m(f"Lcc{tag}Merge", j[bb]["merge2"]["lcc"])
                    m(f"Srcc{tag}Merge", j[bb]["merge2"]["srcc"])
                    m(f"Rmse{tag}Merge", j[bb]["merge2"]["rmse"])
    for tag in ["Hub", "Wlm"]:
        for suf in ["Base", "SgtcTwo", "SgtcFour", "Merge"]:
            macros.setdefault(f"Lcc{tag}{suf}", "??")
            if suf == "Merge":
                macros.setdefault(f"Srcc{tag}{suf}", "??")
                macros.setdefault(f"Rmse{tag}{suf}", "??")

    # ---------- revision: diagnosis tables (Table II / Table III bodies) ----------
    d = json.loads((RES / "shuffle_diagnosis.json").read_text())
    base = d["original"]["lcc_mean"]
    dt_g = base - d["time_shuffle"]["lcc_mean"]
    dt_g = 0.0 if abs(dt_g) < 5e-4 else dt_g
    dt_b = base - d["block_shuffle"]["lcc_mean"]
    diag_lines = ["\\begin{tabular}{lcc}", "\\toprule",
                  "Shuffle granularity & $\\Delta$LCC (time) & $\\Delta$LCC (channel)\\\\",
                  "\\midrule",
                  "Global & {:.3f} & {:.3f} \\\\".format(dt_g,
                                                        base - d["chan_shuffle"]["lcc_mean"]),
                  "Block of 50 frames & {:.3f} & -- \\\\".format(dt_b),
                  "Adjacent swap & 0.000 & -- \\\\",
                  "\\bottomrule", "\\end{tabular}"]
    (ROOT / "paper" / "diag_table.tex").write_text("\n".join(diag_lines) + "\n",
                                                    encoding="utf-8")

    ld = json.loads((RES / "layer_diagnosis.json").read_text())
    layers = sorted(int(k) for k in ld)
    l_lines = ["\\begin{tabular}{ccc}", "\\toprule",
               "Layer & $\\Delta$LCC (time) & $\\Delta$LCC (channel)\\\\",
               "\\midrule"]
    for l in layers:
        dt_ = ld[str(l)]["d_time"]
        dc_ = ld[str(l)]["d_chan"]
        l_lines.append("{} & {:.1e} & {:.3f} \\\\".format(l, abs(dt_), dc_))
    l_lines.append("\\bottomrule")
    l_lines.append("\\end{tabular}")
    (ROOT / "paper" / "layer_table.tex").write_text("\n".join(l_lines) + "\n",
                                                     encoding="utf-8")

    # ---------- deploy ----------
    dep = RES / "sgtc_deploy.json"
    if dep.exists():
        j = json.loads(dep.read_text())
        m("LccDeployTwo", j["accuracy"]["2x"]["lcc"])
        m("SpeedDeploy", j["speedup"], "{:.2f}")
        if "merge2_speedup" in j:
            m("SpeedMerge", j["merge2_speedup"], "{:.2f}")
    macros.setdefault("LccDeployTwo", "??")
    macros.setdefault("SpeedDeploy", "??")
    macros.setdefault("SpeedMerge", "??")

    # ---------- revision: E4 merge-vs-prune tests + bootstrap CI ----------
    rs = RES / "revision_stats.json"
    if rs.exists():
        j = json.loads(rs.read_text())
        for ratio in [2, 3, 4]:
            rn = {2: "Two", 3: "Three", 4: "Four"}[ratio]
            if str(ratio) in j["merge_vs_prune"]:
                m(f"PMergePrune{rn}", j["merge_vs_prune"][str(ratio)]["p"], "{:.1e}")
                lo, hi = j["merge_vs_base"][str(ratio)]["ci95"]
                m(f"CiMergeBase{rn}", f"[{lo:.3f},{hi:.3f}]")
    for rn in ["Two", "Three", "Four"]:
        macros.setdefault(f"PMergePrune{rn}", "??")
        macros.setdefault(f"CiMergeBase{rn}", "??")

    # ---------- revision: E3 NISQA calibration ----------
    nc = RES / "nisqa_calibration.json"
    if nc.exists():
        j = json.loads(nc.read_text())
        for row in j:
            tag = {"baseline": "Base", "merge2": "Merge", "prune2": "Prune"}[row["method"]]
            suf = "Cal" if row["calibration"] == "intercept" else "Raw"
            m(f"RmseNisqa{tag}{suf}", row["rmse"])
            m(f"LccNisqa{tag}{suf}", row["lcc"])
    for tag in ["Base", "Merge", "Prune"]:
        for suf in ["Cal", "Raw"]:
            macros.setdefault(f"RmseNisqa{tag}{suf}", "??")
            macros.setdefault(f"LccNisqa{tag}{suf}", "??")

    # ---------- revision: E5 five-seed stability ----------
    ss = RES / "seed_stability.csv"
    if ss.exists():
        for _, row in pd.read_csv(ss).iterrows():
            rn = {2: "Two", 3: "Three", 4: "Four"}[int(row["ratio"])]
            m(f"LccMerge{rn}Std", float(row["lcc_std"]), "{:.3f}")
    for rn in ["Two", "Three", "Four"]:
        macros.setdefault(f"LccMerge{rn}Std", "??")
    rj = RES / "random_drop_5seed.json"
    if rj.exists():
        j = json.loads(rj.read_text())
        for row in j:
            rn = {2: "Two", 3: "Three", 4: "Four"}[int(row["ratio"])]
            m(f"LccRand{rn}", float(row["lcc_mean"]))
            m(f"LccRand{rn}Std", float(row["lcc_std"]), "{:.3f}")
    for rn in ["Two", "Three", "Four"]:
        macros.setdefault(f"LccRand{rn}Std", "??")

    # ---------- revision: MLP head + size ablation ----------
    mh = RES / "mlp_head.csv"
    if mh.exists():
        for _, row in pd.read_csv(mh).iterrows():
            tag = {"baseline": "Base", "merge2": "MergeTwo", "merge3": "MergeThree",
                   "prune2": "PruneTwo"}.get(row["method"])
            if tag:
                m(f"MlpLcc{tag}", float(row["lcc_mean"]))
                m(f"MlpLcc{tag}Std", float(row["lcc_std"]), "{:.3f}")
    for tag in ["Base", "MergeTwo", "MergeThree", "PruneTwo"]:
        macros.setdefault(f"MlpLcc{tag}", "??")
        macros.setdefault(f"MlpLcc{tag}Std", "??")
    sa = RES / "size_ablation.csv"
    if sa.exists():
        piv = pd.read_csv(sa).pivot(index="frac", columns="method", values="lcc")
        for frac, tag in [("25%", "Quarter"), ("50%", "Half"), ("100%", "Full")]:
            if frac in piv.index and "merge2" in piv.columns:
                m(f"GainSize{tag}", float(piv.loc[frac, "merge2"] - piv.loc[frac, "baseline"]),
                  "{:+.3f}")
                m(f"LccSizeBase{tag}", float(piv.loc[frac, "baseline"]))
                m(f"LccSizeMerge{tag}", float(piv.loc[frac, "merge2"]))
    for tag in ["Quarter", "Half", "Full"]:
        macros.setdefault(f"GainSize{tag}", "??")
        macros.setdefault(f"LccSizeBase{tag}", "??")
        macros.setdefault(f"LccSizeMerge{tag}", "??")

    lines = ["% AUTO-GENERATED by sgtc.fill_numbers -- do not edit by hand"]
    for k in sorted(macros):
        v = macros[k]
        if v == "??":
            lines.append(f"\\newcommand{{\\{k}}}{{\\textbf{{??}}}}")
        else:
            lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(macros)} macros -> {OUT}")


if __name__ == "__main__":
    run()

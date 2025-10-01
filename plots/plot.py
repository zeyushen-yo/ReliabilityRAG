import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

plt.rcParams.update({'font.size': 22})

def find_repo_root(start: Path) -> Path:
    cand = start
    while cand != cand.parent:
        if (cand / "output").exists() or (cand / ".git").exists() or (cand / "README.md").exists():
            return cand
        cand = cand.parent
    return start

REPO_ROOT = find_repo_root(Path(__file__).resolve().parent)
OUTPUT_DIR = REPO_ROOT / "output"
FIG_DIR = REPO_ROOT / "figs" / "accuracy_new"
VALUES_DIR = FIG_DIR / "values"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(VALUES_DIR, exist_ok=True)

plots = {
    "results_with_baselines": {
        "datasets": ["realtimeqa_allrel_perturb",
                     "open_nq_allrel_perturb",
                     "triviaqa_allrel_perturb"],
        "models": ["mistral7b"],
        "attacks": ["PIA"],
        "defenses": [
            {"name": "MIS"},
            {"name": "keyword", "params": {"gamma": [1.0]}},
        ],
    },
}

dataset_labels = {
    "realtimeqa_allrel_perturb": "RQA",
    "open_nq_allrel_perturb"   : "NQ",
    "triviaqa_allrel_perturb"  : "TQA",
}

map_label = {
    "MIS" : "Sampling + MIS",
    "keyword"        : "RobustRAG (Keyword)",
}

style_map = {
    "keyword"        : ("tab:green" , "^"),
    "MIS" : ("tab:purple", "D"),
}

# ---------------------------------------------------------------------------
corruption_sizes = [0, 5, 10, 15, 20]
reps            = ["rep2", "rep3", "rep5"]          # 5 trials total

BIGGER_SIZE = 22
plt.rc('font', size=BIGGER_SIZE)
plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=BIGGER_SIZE)
plt.rc('xtick', labelsize=BIGGER_SIZE)
plt.rc('ytick', labelsize=BIGGER_SIZE)
plt.rc('legend', fontsize=BIGGER_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

for plot_name, cfg in plots.items():
    datasets = cfg["datasets"]
    models   = cfg["models"]
    attacks  = cfg["attacks"]
    defenses = cfg["defenses"]

    print("\nPlotting for:", plot_name)
    for dataset in datasets:
        for model in models:

            txt_path = str(VALUES_DIR / f"{model}_{dataset}_accuracy_corruptionsz.txt")
            header_written = False

            fig, axes = plt.subplots(
                nrows=1, ncols=len(attacks),
                figsize=(7 * len(attacks), 5),
                constrained_layout=True
            )
            if len(attacks) == 1:
                axes = [axes]

            for i, attack in enumerate(attacks):
                ax = axes[i]
                all_data = {}

                for defense in defenses:
                    key = defense["name"]

                    for z in corruption_sizes:
                        acc_trials = []

                        # ---------- gather trials from every CSV ----------
                        for rep in reps:
                            if key == "none":
                                fpath = str(OUTPUT_DIR / (
                                  f"{dataset}-{model}-none-{rep}"
                                  f"-top50-attack{attack}-attackpos0"
                                  f"-corruptionsz{z}.csv"))
                            elif key in ["astuterag", "instructrag_icl"]:
                                fpath = str(OUTPUT_DIR / (
                                  f"{dataset}-{model}-{key}-{rep}"
                                  f"-top50-attack{attack}-attackpos0"
                                  f"-corruptionsz{z}.csv"))
                            elif key == "keyword":
                                for gamma in defense["params"]["gamma"]:
                                    fpath = str(OUTPUT_DIR / (
                                      f"{dataset}-{model}-keyword-0.3-3.0"
                                      f"-gamma{gamma}-{rep}-top50-attack{attack}"
                                      f"-attackpos0-corruptionsz{z}.csv"))
                                    if os.path.exists(fpath):
                                        df = pd.read_csv(fpath)
                                        acc_trials.extend(df["acc"].tolist())
                                continue  # finished keyword; go to next z
                            elif key == "MIS":
                                fpath = str(OUTPUT_DIR / (
                                  f"{dataset}-{model}--{rep}"
                                  f"-top50-attack{attack}-attackpos0"
                                  f"-corruptionsz{z}.csv"))
                            else:
                                continue

                            if os.path.exists(fpath):
                                df = pd.read_csv(fpath)
                                acc_trials.extend(df["acc"].tolist())

                        # -------------------- aggregate --------------------
                        if acc_trials:             # at least one datapoint
                            acc_trials = acc_trials[:5]
                            mean_acc = np.mean(acc_trials)
                            std_acc  = (np.std(acc_trials, ddof=1)
                                        if len(acc_trials) > 1 else 100)
                            all_data.setdefault(key, []).append(
                                (z, mean_acc, std_acc))

                # ------------ save raw numbers (one .txt per fig) ------------
                if all_data:
                    mode = "w" if not header_written else "a"
                    with open(txt_path, mode, encoding="utf-8") as f:
                        if not header_written:
                            f.write("defense\tattack\tcorruptionsz\taccuracy\n")
                            header_written = True
                        for dkey, vals in all_data.items():
                            for z, acc, _ in vals:
                                f.write(f"{map_label[dkey]}\t{attack}\t{z}\t{acc:.4f}\n")

                # ----------------------- plotting ----------------------------
                for dkey, vals in all_data.items():
                    xs   = [z   for z, _, _ in vals]
                    ys   = [acc for _, acc, _ in vals]
                    yerr = [std for _, _, std in vals]
                    color, marker = style_map[dkey]

                    ax.plot(xs, ys, marker=marker, linestyle='--',
                            color=color, markersize=8, linewidth=2,
                            label=map_label[dkey])

                    # ───── shaded 1-σ band ─────
                    ys_arr   = np.asarray(ys)
                    err_arr  = np.asarray(yerr)
                    ax.fill_between(
                        xs,
                        ys_arr - err_arr, ys_arr + err_arr,
                        color=color, alpha=0.18, linewidth=0, zorder=-1
                    )

                ax.set_title(f"{model}; {attack}; {dataset_labels[dataset]}")
                ax.set_xlabel("Corruption size (z)")
                ax.set_ylabel("Accuracy")
                ax.set_ylim(0, 1)
                ax.grid(True)
                ax.set_xticks(corruption_sizes)

            # ------------- legend & save figure -----------------------------
            handles, labels = axes[0].get_legend_handles_labels()
            if "MIS" in labels:           # keep it last
                idx = labels.index("Sampling + MIS")
                handles.append(handles.pop(idx))
                labels.append(labels.pop(idx))

            fig.legend(handles, labels, loc="upper center",
                       bbox_to_anchor=(0.5, 1.14),
                       ncol=3, frameon=False, fontsize=22)

            fname = f"{model}_{dataset}_{plot_name}_accuracy_corruptionsz_50.png"
            plt.savefig(str(FIG_DIR / fname), bbox_inches='tight')
            print(f"Saved figure: {FIG_DIR / fname}")
            print(f"   → values written to: {txt_path}")
            plt.close()
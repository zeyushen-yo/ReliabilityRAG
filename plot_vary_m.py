import pandas as pd
import os
import matplotlib.pyplot as plt

def get_metrics(df):
    row = df.iloc[0]
    return row["acc"], row["asr"]

SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 22 # Increased from 20 for title
AXES_LABEL_SIZE = 22 # Slightly smaller for axes labels if BIGGER_SIZE is too large for them
TICK_LABEL_SIZE = 22 # Slightly smaller for tick labels

plt.rc('font', size=AXES_LABEL_SIZE) # Default font size
plt.rc('axes', titlesize=BIGGER_SIZE) # Title of the axes
plt.rc('axes', labelsize=AXES_LABEL_SIZE) # x and y labels
plt.rc('xtick', labelsize=TICK_LABEL_SIZE)
plt.rc('ytick', labelsize=TICK_LABEL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE) # Legend font size
plt.rc('figure', titlesize=BIGGER_SIZE) # Figure-level title

plots = {
    "sampleMIS_vary_m_all_pos": {
        "datasets": ["realtimeqa_sorted"],
        "models": ["mistral7b"],
        "attack_types": ["PIA"],
        "defenses": [
            {"name": "sampleMIS", "params": {"gamma": [0.9], "T": [20], "m": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}},
        ],
        "plot_attack_positions_file": [0, 24, 49],
        "plot_attack_positions_display": [1, 25, 50],
    },
}

markers = ["o", "s", "^", "D", "P", "X", "*", "v", "<", ">"] # Refined marker list

key_marker_map = {} # Reset for each script run if needed, or manage globally

label_map = {
    "sampleMIS": "Sampling + MIS",
}
dataset_map = {
    "realtimeqa_sorted": "RQA",
}

# --- CHOOSE LEGEND STYLE ---
LEGEND_STYLE = "inside_upper_right" # Options: "top_horizontal", "inside_best", "inside_upper_right"

for plot_name, plot_config in plots.items():
    datasets = plot_config["datasets"]
    models = plot_config["models"]
    attack_types = plot_config["attack_types"]
    defenses = plot_config["defenses"]
    plot_attack_positions_file = plot_config["plot_attack_positions_file"]
    plot_attack_positions_display = plot_config["plot_attack_positions_display"]

    print("\nPlotting for:", plot_name)
    for dataset in datasets:
        for model in models:
            fig, ax_acc = plt.subplots(figsize=(8, 6)) # Standard figsize

            if len(attack_types) == 1:
                plot_title = f"Accuracy vs. m for {model}-{dataset_map.get(dataset)}"
            else:
                plot_title = f"Accuracy vs. m for {model}-{dataset_map.get(dataset)}"
            ax_acc.set_title(plot_title)

            marker_idx = 0
            # Store legend handles and labels for reordering or custom placement
            legend_handles = []
            legend_labels = []

            for i, attack_pos_file in enumerate(plot_attack_positions_file):
                attack_pos_disp = plot_attack_positions_display[i]
                for attack_type in attack_types:
                    # This inner loop for attack_type might be redundant if only one attack type
                    # but kept for generality based on original structure.

                    for defense in defenses: # This loop structure implies different defense lines
                        defense_name = defense["name"]
                        params = defense.get("params", {})
                        gammas = params.get("gamma", [1.0])
                        Ts = params.get("T", [10])
                        ms_values = params.get("m", []) # Renamed to avoid conflict

                        for gamma in gammas: # Loop for gamma
                            for T_val in Ts: # Loop for T (renamed to T_val)
                                base_defense_label = label_map.get(defense_name, defense_name)
                                # Key for the line in the legend
                                current_label = f"{base_defense_label} (T={T_val}, $\\gamma$={gamma}), @Pos {attack_pos_disp}"

                                data_points = []
                                for m_val in ms_values:
                                    file_path = f"./output/{dataset}-{model}-sampleMIS-T{T_val}-m{m_val}-gamma{gamma}-rep1-top50-attack{attack_type}-attackpos{attack_pos_file}.csv"
                                    if os.path.exists(file_path):
                                        df = pd.read_csv(file_path)
                                        data_points.append((m_val, *get_metrics(df)))
                                    else:
                                        print(f"File not found: {file_path}")

                                if data_points:
                                    if current_label not in key_marker_map:
                                        key_marker_map[current_label] = markers[marker_idx % len(markers)]
                                        marker_idx +=1
                                    marker = key_marker_map[current_label]

                                    sorted_values = sorted(data_points, key=lambda item: item[0])
                                    x = [item[0] for item in sorted_values]
                                    y_acc = [item[1] for item in sorted_values]

                                    line, = ax_acc.plot(x, y_acc, label=current_label, marker=marker, linestyle='-')
                                    if current_label not in legend_labels: # Avoid duplicate legend entries if structure implies it
                                        legend_handles.append(line)
                                        legend_labels.append(current_label)


            ax_acc.set_xlabel("m (Size of Context)")
            ax_acc.set_ylabel("Accuracy")
            ax_acc.set_ylim(0, 1.02)
            ax_acc.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax_acc.grid(True)

            if LEGEND_STYLE == "top_horizontal":
                # Place legend horizontally above the plot
                # May need to adjust ncol and bbox_to_anchor
                fig.legend(legend_handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=min(3, len(legend_handles)), fancybox=True, shadow=False)
                plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust rect to make space for legend at top
            elif LEGEND_STYLE == "inside_best":
                ax_acc.legend(handles=legend_handles, labels=legend_labels, loc='best')
                plt.tight_layout(pad=0.5)
            elif LEGEND_STYLE == "inside_upper_right":
                ax_acc.legend(handles=legend_handles, labels=legend_labels, loc='upper right')
                plt.tight_layout(pad=0.5)
            else: # Default to inside best or a reasonable default
                ax_acc.legend(handles=legend_handles, labels=legend_labels, loc='best')
                plt.tight_layout(pad=0.5)


            os.makedirs("./figs_vary_m/", exist_ok=True)
            save_path = f"./figs_vary_m/{model}_{dataset}_{plot_name}_combined_vary_m.png"
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Saved figure: {save_path}")
            plt.close(fig)
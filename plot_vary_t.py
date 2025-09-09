import pandas as pd
import os
import matplotlib.pyplot as plt

def get_metrics(df):
    row = df.iloc[0]
    acc = row.get("acc", float('nan'))
    asr = row.get("asr", float('nan')) # ASR is still read but not used for plotting
    return acc, asr

# Consistent font sizes - adjust the base BIGGER_SIZE if needed
# The plt.rc calls will use these.
SMALL_SIZE = 16
MEDIUM_SIZE = 18 # Good for legend if BIGGER_SIZE is too large
BIGGER_SIZE = 22

plt.rc('font', size=BIGGER_SIZE) # Default for text elements not otherwise specified
plt.rc('axes', titlesize=BIGGER_SIZE)    # Title of the axes (the plot's title)
plt.rc('axes', labelsize=BIGGER_SIZE)   # X and Y axis labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)  # X-axis tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)  # Y-axis tick labels
plt.rc('legend', fontsize=SMALL_SIZE) # Legend font size from your first script
plt.rc('figure', titlesize=BIGGER_SIZE) # Overall figure title (if using fig.suptitle)

plots = {
    "sampleMIS_vary_T_all_pos_AccuracyOnly": {
        "datasets": ["realtimeqa_sorted"],
        "models": ["mistral7b"],
        "attack_types": ["PIA"],
        "defenses_configs": [
            {"name": "sampleMIS", "params": {"gamma": [0.9], "T_values": [5, 10, 15, 20, 25], "m": [2]}},
        ],
        "plot_attack_positions_file": [0, 24, 49],
        "plot_attack_positions_display": [1, 25, 50],
    },
}

markers = ["o", "s", "^", "D", "P", "X", "*"]
linestyles = ['-'] # Different linestyle per attack_pos

# Map (base_label_for_m) to a marker
# Map (line_label) to a linestyle (though we'll cycle linestyle by attack_pos index)
key_marker_map = {}


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
    defenses_configs = plot_config["defenses_configs"]
    plot_attack_positions_file = plot_config["plot_attack_positions_file"]
    plot_attack_positions_display = plot_config["plot_attack_positions_display"]

    print("\nPlotting for:", plot_name)
    for dataset in datasets:
        for model in models:
            for i_attack_type, attack_type_val in enumerate(attack_types):
                fig, ax_acc = plt.subplots(
                    nrows=1, ncols=1,
                    figsize=(8, 6), # Start with a reasonable size
                )
                plot_title = f"Accuracy vs. T for {model}-{dataset_map.get(dataset)}"
                ax_acc.set_title(plot_title)

                # For collecting handles and labels if needed for manual legend order
                # However, ax.legend() usually handles this automatically if labels are unique.
                # legend_handles = []
                # legend_labels = []

                for i_pos, attack_pos_file in enumerate(plot_attack_positions_file):
                    attack_pos_disp = plot_attack_positions_display[i_pos]
                    current_linestyle = linestyles[i_pos % len(linestyles)]

                    for defense_config in defenses_configs:
                        defense_name = defense_config["name"]
                        params = defense_config.get("params", {})
                        gammas = params.get("gamma", [1.0])
                        ms = params.get("m", [1])
                        T_values = params.get("T_values", [])

                        gamma_val = gammas[0]
                        m_val = ms[0]

                        base_label_for_m = f"{label_map.get(defense_name, defense_name)} (m={m_val}, $\\gamma$={gamma_val})"
                        line_label = f"{base_label_for_m}, @Pos {attack_pos_disp}"

                        data_points_for_line = []
                        for T_val_x in T_values:
                            file_path = f"./output/{dataset}-{model}-sampleMIS-T{T_val_x}-m{m_val}-gamma{gamma_val}-rep1-top50-attack{attack_type_val}-attackpos{attack_pos_file}.csv"
                            if os.path.exists(file_path):
                                df = pd.read_csv(file_path)
                                data_points_for_line.append((T_val_x, *get_metrics(df)))
                            else:
                                print(f"File not found: {file_path}")

                        if data_points_for_line:
                            if base_label_for_m not in key_marker_map:
                                key_marker_map[base_label_for_m] = markers[len(key_marker_map) % len(markers)]
                            marker = key_marker_map[base_label_for_m]

                            sorted_values = sorted(data_points_for_line, key=lambda item: item[0])
                            x_T = [item[0] for item in sorted_values]
                            y_acc_vals = [item[1] for item in sorted_values]

                            ax_acc.plot(x_T, y_acc_vals, label=line_label, marker=marker, linestyle=current_linestyle)
                            # No need to manually collect handles/labels if ax_acc.legend() is used directly
                            # and labels are unique.

                ax_acc.set_xlabel("T (Number of Sampling Rounds)")
                ax_acc.set_ylabel("Accuracy")
                ax_acc.set_ylim(0, 1.02)
                ax_acc.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                ax_acc.grid(True)

                if LEGEND_STYLE == "outside_right":
                    # This was the style attempted in your original code for the first script.
                    # It requires adjusting the plot area to make space.
                    ax_acc.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
                    # To make space for the legend placed outside, adjust subplot parameters
                    # The 'right' parameter controls how much space is on the right of the plot
                    # Value less than 1.0 (e.g., 0.75 or 0.8) makes the plot area narrower.
                    fig.subplots_adjust(right=0.70) # YOU WILL LIKELY NEED TO TUNE THIS VALUE (0.70)
                elif LEGEND_STYLE == "inside_upper_right":
                    ax_acc.legend(loc='upper right')
                elif LEGEND_STYLE == "best":
                    ax_acc.legend(loc='best')
                else: # Default if style not recognized
                    ax_acc.legend(loc='best')

                # If not using fig.subplots_adjust, plt.tight_layout() can sometimes help,
                # but it can conflict with bbox_to_anchor when legend is outside.
                # If LEGEND_STYLE is not "outside_right", tight_layout is safer.
                if LEGEND_STYLE != "outside_right":
                    plt.tight_layout(pad=0.5)


                os.makedirs("./figs_vary_t/", exist_ok=True)
                save_filename = f"./figs_vary_t/{model}_{dataset}_{plot_name}_varyT.png"
                # Use bbox_inches='tight' to try and crop whitespace, but subplots_adjust is more direct for legend space
                plt.savefig(save_filename, bbox_inches='tight')
                print(f"Saved figure: {save_filename}")
                plt.close(fig)
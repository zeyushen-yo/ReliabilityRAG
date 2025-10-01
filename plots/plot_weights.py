import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Consistent font sizes
SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 22

plt.rc('font', size=BIGGER_SIZE)
plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=BIGGER_SIZE)
plt.rc('xtick', labelsize=BIGGER_SIZE)
plt.rc('ytick', labelsize=BIGGER_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE) # Or MEDIUM_SIZE if SMALL_SIZE is too small
plt.rc('figure', titlesize=BIGGER_SIZE)

# Hardcoded data
# Accuracies for Attack @ Pos 1, 25, 50 respectively
data_decay_schemes = {
    "Exponential Decay ($\\gamma$=0.9)": [0.55, 0.67, 0.72],
    "Linear Decay": [0.59, 0.64, 0.70],
}

# X-axis labels (Attack Positions)
attack_positions_display = [1, 25, 50]
# Numeric positions for plotting if x-axis is treated as categorical
attack_positions_numeric = np.arange(len(attack_positions_display))

# Model and Dataset details for the title (assuming these are constant for this plot)
model_name = "Mistral7B"
dataset_name = "RQA"
# Assuming these parameters were constant for the experiments you're comparing
# If not, they might need to be part of the legend or title.
# For this plot, the varying part is the decay scheme.
# fixed_params_note = "(m=2, T=20)" # Example if m and T were fixed

# Markers and linestyles for different decay schemes
markers = ["o"] # Add more if you have more schemes
linestyles = ['-']

# Create the plot
fig, ax = plt.subplots(figsize=(8, 6)) # Single plot

plot_title = f"Accuracy by Decay Scheme for {model_name}-{dataset_name}"
# if fixed_params_note:
#     plot_title += f"\n{fixed_params_note}"
ax.set_title(plot_title) # Added padding for title

for i, (scheme_name, accuracies) in enumerate(data_decay_schemes.items()):
    ax.plot(attack_positions_numeric, accuracies, label=scheme_name,
            marker=markers[i % len(markers)],
            linestyle=linestyles[i % len(linestyles)],
            linewidth=2, markersize=8)

# Formatting the plot
ax.set_xlabel("Attack Position")
ax.set_ylabel("Accuracy")
ax.set_xticks(attack_positions_numeric)
ax.set_xticklabels(attack_positions_display)

ax.set_ylim(0, 1.02) # Start y-axis from 0 for accuracy, slight top padding
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.grid(True, linestyle='-', alpha=0.7) # Lighter grid

# Legend
# Common locations: 'best', 'upper left', 'upper right', 'lower left', 'lower right'
ax.legend(loc='upper right') # 'best' tries to find an optimal spot

plt.tight_layout() # Adjust layout

def find_repo_root(start: Path) -> Path:
    cand = start
    while cand != cand.parent:
        if (cand / "output").exists() or (cand / ".git").exists() or (cand / "README.md").exists():
            return cand
        cand = cand.parent
    return start

REPO_ROOT = find_repo_root(Path(__file__).resolve().parent)
FIG_DIR = REPO_ROOT / "figs" / "decay_scheme"
os.makedirs(FIG_DIR, exist_ok=True)
save_path = FIG_DIR / f"{model_name}_{dataset_name}_decay_schemes_comparison.png"
plt.savefig(str(save_path), bbox_inches='tight')
print(f"Saved figure: {save_path}")

plt.show() # Display the plot
plt.close(fig) # Close the figure object
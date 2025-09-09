import matplotlib.pyplot as plt
import numpy as np # For x-axis tick positions if needed
import os

# Consistent font sizes from your previous scripts
SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 22

plt.rc('font', size=BIGGER_SIZE)
plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=BIGGER_SIZE)
plt.rc('xtick', labelsize=BIGGER_SIZE) # Adjusted for clarity of x-axis values
plt.rc('ytick', labelsize=BIGGER_SIZE) # Adjusted for clarity of y-axis values
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

# Hardcoded data
# For m = 2, T = 20, Mistral-7B, RQA
# Accuracies for Attack @ Pos 1, 25, 50 respectively
data = {
    0.85: [0.48, 0.73, 0.74], # Gamma = 0.85
    0.90: [0.55, 0.67, 0.72], # Gamma = 0.90
    0.95: [0.57, 0.62, 0.68], # Gamma = 0.95
}

# X-axis labels (Attack Positions)
attack_positions_display = [1, 25, 50]
attack_positions_numeric = np.arange(len(attack_positions_display)) # For plotting

# Model and Dataset details for the title
model_name = "mistral7B"
dataset_name = "RQA"
defense_params = "m=2, T=20" # Fixed parameters

# Markers and linestyles for different gamma values
markers = ["o", "s", "^"]
linestyles = ['-']

# Create the plot
fig, ax = plt.subplots(figsize=(8, 6)) # Single plot

plot_title = f"{model_name}-{dataset_name}, varying $\\gamma$"
ax.set_title(plot_title) # Added padding for title

for i, (gamma, accuracies) in enumerate(data.items()):
    label = f"Sampling + MIS (m = 2, T = 20, $\\gamma$={gamma})"
    ax.plot(attack_positions_numeric, accuracies, label=label,
            marker=markers[i % len(markers)],
            linestyle=linestyles[i % len(linestyles)],
            linewidth=2, markersize=8)

# Formatting the plot
ax.set_xlabel("Attack Position")
ax.set_ylabel("Accuracy")
ax.set_xticks(attack_positions_numeric) # Set x-ticks to the numeric positions
ax.set_xticklabels(attack_positions_display) # Set the display labels for x-ticks

ax.set_ylim(0, 1.02)
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.grid(True, linestyle='-', alpha=0.7)

# Legend
# Place legend inside, trying 'best' location or a specific one like 'lower left' or 'upper right'
# if data allows.
ax.legend(loc='upper right') # Example: 'lower left', 'upper right', 'best'

plt.tight_layout() # Adjust layout to prevent labels from overlapping

# Save the figure
output_dir = "./figs_vary_gamma/"
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, f"{model_name}_{dataset_name}_m2_T20_varyGammaAttackPos.png")
plt.savefig(save_path, bbox_inches='tight')
print(f"Saved figure: {save_path}")
plt.show() # Display the plot
plt.close(fig)
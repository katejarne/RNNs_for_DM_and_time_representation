# plot_SI_task_compare.py
"""
MIT License - C. Jarne V. 1.0 - 2025
This code compares sequential processing patterns
in neural networks across different cognitive tasks
(Simple DM, Long-Short DM, Integral DM) by analyzing
Sequentiality Index (SI) scores. It visualizes performance
differences between orthogonally-initialized and standard
networks using comparative box plots with error bars,
employing task-specific color gradients to highlight
initialization effects while maintaining cross-task
comparability in temporal information processing efficiency.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

# Configuration
DIRECTORIES = {
    "Simple DM": "./plots/SI_results_simple_DM",
    "Long-Short DM": "./plots/SI_results_DM_long_short",
    "Integral DM": "./plots/SI_results_integral_DM_20"
}

COLOR_SCHEME = {
    "Simple DM": "#1f77b4",
    "Integral DM": "#ff7f0e",
    "Long-Short DM": "#2ca02c"
}

OUTPUT_PLOT = "plots/multi_task_si_comparison_final.png"


def generate_colors(base_color):
    """Generate color variations for initialization types"""
    base_rgb = np.array(to_rgb(base_color))
    return [
        np.clip(base_rgb * 0.7, 0, 1),  # Non-orthogonal (lighter)
        np.clip(base_rgb * 1.3, 0, 1)   # Orthogonal (darker)
    ]


def process_directory(dir_path):
    """Process all files in a directory"""
    data = {"ortho": {"means": [], "sems": []},
           "non_ortho": {"means": [], "sems": []}}

    for filename in os.listdir(dir_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(dir_path, filename)
            scores = []

            with open(filepath, 'r') as f:
                for line in f.readlines()[1:]:  # Skip header
                    entries = line.strip().split('\t')
                    if len(entries) >=3 and entries[2].lower() != 'nan' and float(entries[2])>0:
                        try:
                            scores.append(abs(float(entries[2])))
                        except (ValueError, IndexError):
                            continue

            if scores:
                mean = np.mean(scores)
                sem = np.std(scores, ddof=1)/np.sqrt(len(scores))

                if "orthogonal" in filename.lower():
                    data["ortho"]["means"].append(mean)
                    data["ortho"]["sems"].append(sem)
                else:
                    data["non_ortho"]["means"].append(mean)
                    data["non_ortho"]["sems"].append(sem)
    return data


# Process all directories
all_data = {}
for task_name, dir_path in DIRECTORIES.items():
    all_data[task_name] = process_directory(dir_path)

# Create visualization
plt.figure(figsize=(13, 7))
ax = plt.gca()

# Plot parameters
box_width = 0.6
group_spacing = 2.5
positions = []
current_pos = 1
legend_elements = []

# Plot each task group
for task_idx, (task_name, task_data) in enumerate(all_data.items()):
    colors = generate_colors(COLOR_SCHEME[task_name])

    # Plot non-orthogonal (lighter color)
    box = plt.boxplot(task_data["non_ortho"]["means"],
                    positions=[current_pos],
                    widths=box_width,
                    patch_artist=True,
                    showfliers=False,
                    medianprops={'color': 'black', 'linewidth': 1.5})
    plt.setp(box['boxes'], facecolor=colors[0], alpha=0.5)

    # Plot orthogonal (darker color)
    box = plt.boxplot(task_data["ortho"]["means"],
                    positions=[current_pos + 1],
                    widths=box_width,
                    patch_artist=True,
                    showfliers=False,
                    medianprops={'color': 'black', 'linewidth': 1.5})
    plt.setp(box['boxes'], facecolor=colors[1], alpha=0.6)

    # Add individual points with error bars
    plt.errorbar(x=current_pos + np.random.normal(0, 0.1, len(task_data["non_ortho"]["means"])),
                y=task_data["non_ortho"]["means"],
                yerr=task_data["non_ortho"]["sems"],
                fmt='o', color=colors[0], alpha=0.7, markersize=8, capsize=4)

    plt.errorbar(x=current_pos + 1 + np.random.normal(0, 0.1, len(task_data["ortho"]["means"])),
                y=task_data["ortho"]["means"],
                yerr=task_data["ortho"]["sems"],
                fmt='o', color=colors[1], alpha=0.7, markersize=8, capsize=4)

    # Legend elements
    legend_elements.append(Patch(facecolor=colors[0], label=f'{task_name} Normal'))
    legend_elements.append(Patch(facecolor=colors[1], label=f'{task_name} Orthogonal'))

    positions.extend([current_pos, current_pos + 1])
    current_pos += group_spacing

# X-axis setup
group_centers = [np.mean([positions[i], positions[i+1]]) for i in range(0, len(positions), 2)]
plt.xticks(group_centers, DIRECTORIES.keys(), fontsize=12)

# Create unified legend
legend = ax.legend(handles=legend_elements,
                  ncol=3,
                  loc='upper center',
                  bbox_to_anchor=(0.5, 1.18),  # Adjusted position
                  frameon=True,
                  fontsize=10,
                  title="Initialization Types",
                  title_fontsize=12)

# Styling
plt.title('Task-Specific Network Dynamics Comparison\nOrthogonal vs. Non-Orthogonal Initializations',
          fontsize=14, pad=25, y=1.15)
plt.ylabel('Sequentiality Index (SI Score)', fontsize=12)
plt.xlabel('Task Type', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.xlim(0, current_pos - group_spacing + 2)
plt.ylim(-0.5, 5)
# Save output
plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
plt.close()

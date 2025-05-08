# plot_eigen_task_violin.py
"""
MIT License - C. Jarne V. 1.0 - 2025
This code aggregates and visualizes the top eigenvalues
from multiple neural network configurations stored in text files.
It generates comparative violin plots sorted by median values,
showing eigenvalue magnitude distributions across different
experimental conditions to analyze spectral properties
of network weight matrices.
output in plot dir.
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Configure paths and parameters
BASE_DIR = "./plots/eigen_tasks"
DIR_PREFIX = "eigen_results_"
FILE_PATTERN = "*ortho*.txt"


def extract_category_name(dir_path):
    """Extracts category name from directory path"""
    print(dir_path)
    return os.path.basename(dir_path).split(DIR_PREFIX)[-1]


def process_directory(dir_path):
    """Process files and return eigenvalues"""
    eigenvalues = []
    for file_path in glob.glob(os.path.join(dir_path, FILE_PATTERN)):

        with open(file_path, 'r') as f:
            for line in f:
                if "Top5_Eigenvalues" in line:
                    next_line = next(f).strip().split('\t')
                    eigenvalues.extend(map(float, next_line))
                    break
    return eigenvalues


# Main processing
categories = []
all_data = []

# Collect data
for dir_path in sorted(glob.glob(os.path.join(BASE_DIR, f"{DIR_PREFIX}*"))):
    category = extract_category_name(dir_path)
    eigenvals = process_directory(dir_path)


    if eigenvals:
        categories.append(category)
        all_data.append(eigenvals)

# Sort by median value
paired = sorted(zip(categories, all_data), key=lambda x: np.median(x[1]))
sorted_categories = [pair[0] for pair in paired]
sorted_data = [pair[1] for pair in paired]

# Create violin plot
plt.figure(figsize=(14, 8))

# Create custom color palette
colors = plt.cm.tab20(np.linspace(0, 1, len(sorted_categories)))

# Create violin plots
violins = plt.violinplot(
    sorted_data,
    showmeans=True,
    showmedians=True,
    showextrema=False
)

# Set individual colors for each violin
for i, (pc, color) in enumerate(zip(violins['bodies'], colors)):
    pc.set_facecolor(color)
    pc.set_edgecolor('gray')
    pc.set_alpha(0.6)

# Configure median and mean lines
violins['cmedians'].set_color('black')
violins['cmedians'].set_linewidth(2)
violins['cmeans'].set_color('yellow')
violins['cmeans'].set_linewidth(2)

# Add individual data points with jitter
for i, data in enumerate(sorted_data):
    x = np.random.normal(i + 1, 0.15, len(data))
    plt.scatter(x, data, color='black', alpha=0.15, s=20, zorder=2)

# Configure plot aesthetics
plt.title("Top 5 Eigenvalues Distribution by Experimental Task", pad=20)
plt.ylabel("Eigenvalue Magnitude")
plt.xlabel("Task")
plt.xticks(range(1, len(sorted_categories)+1), sorted_categories, rotation=30, ha="right")
plt.grid(True, linestyle="--", alpha=0.4)

# Create custom legend

legend_elements = [
    Patch(facecolor=colors[i], edgecolor='gray', label=sorted_categories[i])
    for i in range(len(sorted_categories))
]

# Adjust layout and save
plt.tight_layout()
plt.savefig("./plots/violin_plot_eigenvalues.png", dpi=300, bbox_inches="tight")
plt.close()

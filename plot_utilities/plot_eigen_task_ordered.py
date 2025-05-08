# plot_eigen_task_ordered.py
"""
MIT License - C. Jarne V. 1.0 - 2025
This code aggregates and visualizes the top eigenvalues
from multiple neural network configurations stored in text files.
It generates comparative box plots sorted by median values,
showing eigenvalue magnitude distributions across different
experimental conditions to analyze spectral properties
of network weight matrices.
output in plot dir.
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# Configure paths and parameters
BASE_DIR = "../plots/eigen_tasks"
DIR_PREFIX = "eigen_results_"
FILE_PATTERN = "*ortho*.txt"


def extract_category_name(dir_path):
    """Extracts the category name from the directory"""
    return os.path.basename(dir_path).split(DIR_PREFIX)[-1]


def process_directory(dir_path):
    """Processes files and returns eigenvalues"""
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


for dir_path in sorted(glob.glob(os.path.join(BASE_DIR, f"{DIR_PREFIX}*"))):
    category = extract_category_name(dir_path)
    eigenvals = process_directory(dir_path)

    if eigenvals:
        categories.append(category)
        all_data.append(eigenvals)

# Order by mean/median

paired = sorted(zip(categories, all_data), key=lambda x: np.mean(x[1]))
sorted_categories = [pair[0] for pair in paired]
sorted_data = [pair[1] for pair in paired]

# Plot
plt.figure(figsize=(14, 8))
box = plt.boxplot(
    sorted_data,
    labels=sorted_categories,
    showmeans=True,
    meanprops={"marker": "D", "markerfacecolor": "orange", "markersize": 8},
    patch_artist=True,
    boxprops={"facecolor": "lightblue", "alpha": 0.6},
    whiskerprops={"color": "gray"},
    capprops={"color": "gray"}
)

# Individual dots
for i, data in enumerate(sorted_data):
    x = np.random.normal(i + 1, 0.15, len(data))
    plt.scatter(x, data, alpha=0.5, color="navy", s=20, zorder=2)

# Median line
medians = [np.median(d) for d in sorted_data]
# plt.plot(range(1, len(medians)+1), medians, '--', color='red', alpha=0.6, label='Median Trend')

plt.title("Top 5 Eigenvalues Sorted by Median Value", pad=20)
plt.ylabel("Eigenvalue Magnitude")
plt.xlabel("Experimental Configuration")
plt.xticks(rotation=45, ha="right")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()

# Save
plt.tight_layout()
plt.savefig("../plots/sorted_ortho_eigenvalues.png", dpi=300, bbox_inches="tight")
plt.close()

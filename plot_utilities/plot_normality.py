# plot_normality.py
"""
MIT License - C. Jarne V. 1.0 - 2025
This code analyzes and visualizes matrix normality deviations
across different neural network initialization methods using
Henrici's H metric. It aggregates H-values (quantifying
departure from matrix normality) from multiple experimental
configurations stored in "C"/"S"-prefixed directories,
comparing orthogonal vs normal initializations through
comparative box plots with jittered points.
The visualization highlights spectral structure differences
by plotting distributions against a normality baseline (H=0).
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

# Configuration
# root_dir = "plots/Tasks_results_H_number"
root_dir = "../plots/Task_condition_comparison"
target_file = "_H_number.txt"
output_name = "../plots/Henrici_plot_r.png"
plot_title = "Henrici's departure from normality \n(Random Orthogonal vs Random Normal)"
# plot_title = "Henrici's departure from normality \n(Initial condition: Random Orthogonal)"

# Data collection
data_dict = {}

# Traverse directory tree
for root, dirs, files in os.walk(root_dir):
    # Check for OK/ok directories at any level
    path_parts = root.split(os.sep)
    if not any(part.lower() == "ok" for part in path_parts):
        continue

    # Find the first parent directory with numbered name
    main_dir = None
    current_path = root
    print(root)
    while current_path != root_dir:
        current_path, dir_name = os.path.split(current_path)
        # if dir_name and dir_name[0].isdigit() and dir_name.startswith("0"):
        if dir_name and (dir_name.startswith("C") or dir_name.startswith("S")):
            print(dir_name)
            main_dir = dir_name
            break

    if not main_dir:
        continue  # Skip if no numbered parent found

    # Collect all H values from files
    h_values = []
    for file in files:
        if file == target_file:
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r') as f:
                    next(f)  # Skip header
                    for line in f:
                        if line.strip():
                            h_values.append(float(line.split()[1]))
            except Exception as e:
                print(f"Error reading {filepath}: {e}")

    if h_values:
        data_dict.setdefault(main_dir, []).extend(h_values)

# Exit if no data
if not data_dict:
    print("No data found!")
    exit()

# Prepare plot
plt.figure(figsize=(8, 6))
colors = cm.tab10(np.linspace(0, 1, len(data_dict)))

# Create boxplot with individual points
boxes = []
labels = sorted(data_dict.keys())  # Sort directories numerically
all_values = [data_dict[label] for label in labels]
plt.axhline(y=0, linestyle='--', alpha=0.5, color="red", label="Normal Matrices")
# plt.axhline(y=0.1, linestyle='--', alpha=0.5, color="gray")
# plt.axhline(y=0.2, linestyle='--', alpha=0.5, color="gray")
# plt.axhline(y=0.3, linestyle='--', alpha=0.5, color="gray")
# plt.axhline(y=0.4, linestyle='--', alpha=0.5, color="gray")
# plt.axhline(y=0.5, linestyle='--', alpha=0.5, color="gray")
# plt.axhline(y=0.6, linestyle='--', alpha=0.5, color="gray")
# plt.axhline(y=0.7, linestyle='--', alpha=0.5, color="gray")
# Create boxplot
bp = plt.boxplot(all_values,
                positions=range(len(labels)),
                widths=0.6,
                patch_artist=True,
                labels=labels)

# Set colors and jitter points
for i, (box, color) in enumerate(zip(bp['boxes'], colors)):
    box.set(facecolor=color, alpha=0.6)

    # Add jittered points
    y = data_dict[labels[i]]
    x = np.random.normal(i, 0.1, size=len(y))
    plt.scatter(x, y, color=color, alpha=0.6, edgecolor='k')

# Formatting
plt.title(plot_title, fontsize=14)#, fontweight='bold')
plt.xticks(rotation=30, ha='right')
plt.ylabel('H value')


plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(loc=4)
plt.tight_layout()

# Save and close
plt.savefig(output_name, dpi=300)
plt.close()

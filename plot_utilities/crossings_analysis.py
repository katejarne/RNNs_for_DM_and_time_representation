# crossings_analysis.py
# MIT License - C. Jarne V. 1.0 - 2025
import numpy as np
import matplotlib.pyplot as plt


def count_and_plot_crossings(X_pca_list, condition_labels, colors, plot_dir, epsilon=0.1, freeze_threshold=10):
    """
    Counts the crossings in the given PCA trajectories while ignoring crossings after the trajectory stabilizes.
    Generates a bar plot of crossings per condition.

    Args:
        X_pca_list (list): List of tuples (x, y, z) for each condition, where
                           x, y, and z are arrays representing the trajectory.
        condition_labels (list): Labels for each condition.
        plot_dir (str): Directory to save the plot.
        epsilon (float): Distance threshold to consider a crossing.
        freeze_threshold (int): Number of consecutive time steps at the same position before stopping the count.
    """
    counts = []

    for X_pca in X_pca_list:
        x, y, z = X_pca  # Extract x, y, z components of the trajectory
        trajectory = np.vstack((x, y, z)).T  # Shape (time_steps, 3)

        # Set to store visited points (with rounding to reduce floating point errors)
        visited_points = set()
        count = 0
        freeze_start = None  # Index where the trajectory becomes stationary

        for t in range(len(trajectory)):
            point = tuple(np.round(trajectory[t] / epsilon) * epsilon)

            # Check if the last `freeze_threshold` points are the same
            if t >= freeze_threshold and all(
                np.allclose(trajectory[t], trajectory[t - i], atol=epsilon) for i in range(1, freeze_threshold + 1)
            ):
                freeze_start = t - freeze_threshold  # Mark the start of the stationary phase
                break  # Stop counting from here

            if point in visited_points:
                count += 1
            else:
                visited_points.add(point)

        counts.append(count)

    # Generate bar plot
    plt.figure(figsize=(8, 4))
    bars = plt.bar(condition_labels, counts, color=colors)

    # Add labels with values
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height}', ha='center', va='bottom', fontsize=9)

    plt.axhline(y=9, color='gray', linewidth=1, linestyle='dashed', label="Fixed point limit")
    plt.ylabel('Number of Crossings', fontsize=10)
    plt.title('Crossings in PCA Trajectories per Condition', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.ylim([0, 20])
    plt.legend(loc=1)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/crossings_summary.png", dpi=300, bbox_inches='tight')
    plt.close()

    return counts

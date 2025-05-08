# Time interval comparison task (TICT) from H. Diaz et al. https://doi.org/10.1073/pnas.2420356122
"""
MIT License - C. Jarne V. 1.0 - 2025
This code analyzes an RNN performing temporal interval comparisons
between paired stimuli. It generates two experimental conditions
(Short-Long vs Long-Short interval sequences) with randomized temporal
jitter, examining how the network encodes relative timing through neural
dynamics and 3D PCA trajectories. The implementation reconstructs
trial-aligned temporal activity patterns showing input-target-output
relationships while projecting hidden states into low-dimensional space
to reveal condition-specific trajectory separation. Through explained
variance analysis and multi-view 3D projections, the study characterizes
how temporal comparison information becomes geometrically organized in
state space, demonstrating the RNN's capacity to maintain and compare
interval durations through distinct dynamical regimes.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utilities import cm2inch
import tensorflow as tf
from numpy.random import seed
from scipy import signal

# Configuration
MODEL_PATH = "./weights/10_interval_compare/weights_N_100_4/100_final.hdf5"
N_NEURONS = 100
PLOT_DIR = "./plots/PCA"
CONDITION_LABELS = ["Short->Long", "Long->Short"]
COLORS = ['blue', 'red']


def load_trained_model(model_path):
    # Load model
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(loss='mse', optimizer='Adam')
    print(model.summary())
    return model


def generate_comparison_trials(size, mem_gap=20, trial_type=None):
    seed(1)
    base_params = {
        'first_in': 50,
        'stim_dur': 10,
        'var_delay_length': 70,
        'interval_sep': 100
    }

    sequence_duration = 450
    x_train = np.zeros((size, sequence_duration, 1))
    y_train = np.zeros((size, sequence_duration, 1))
    trial_types = np.zeros(size, dtype=int)

    for sample in range(size):
        # Determine the type of test
        if trial_type is not None:
            current_trial = trial_type
        else:
            current_trial = sample % 2
        trial_types[sample] = current_trial

        # Assign intervals according to condition
        if current_trial == 0:  # Short-long
            delta1, delta2 = 25, 50
        else:  # Long-short
            delta1, delta2 = 50, 25

        # Calculate pulse positions
        pulse_starts = [
            base_params['first_in'] + np.random.randint(base_params['var_delay_length']),
            None, None, None]

        pulse_starts[1] = pulse_starts[0] + base_params['stim_dur'] + delta1
        pulse_starts[2] = pulse_starts[1] + base_params['stim_dur'] + base_params['interval_sep']
        pulse_starts[3] = pulse_starts[2] + base_params['stim_dur'] + delta2

        win = signal.windows.hann(10)
        for i, start in enumerate(pulse_starts):
            end = start + base_params['stim_dur']
            x_train[sample, start:end, 0] = signal.convolve(
                np.ones(base_params['stim_dur']), win, mode='same')/sum(win)

        # Define response
        response_start = pulse_starts[-1] + base_params['stim_dur'] + mem_gap
        y_train[sample, response_start:, 0] = 1.0 if current_trial == 0 else -1.0

    return x_train, y_train, sequence_duration, trial_types


def process_condition(model, condition_type, ax):

    x, y, seq_dur, _ = generate_comparison_trials(1, trial_type=condition_type)
    # Predictions
    y_pred = model.predict(x)
    print(y_pred.shape)
    layer_output = model.layers[0](x).numpy()[0]

    # Plotting
    ax.plot(x[0, :, 0], color='g', label='Input')
    ax.plot(y[0, :, 0], color='gray', linewidth=2, label='Target')
    ax.plot(y_pred[0, :, 0], color='r', linewidth=1.5, label='Output')
    ax.set_ylim([-2.5, 2.5])

    # Neural activity (only first 10 units)
    for n in range(10):
        ax.plot(layer_output[:, n], alpha=0.3, linewidth=0.7)
        if n==9:
            ax.plot(layer_output[:, n], color="black",linewidth=0.7)

    ax.set_title(CONDITION_LABELS[condition_type], fontsize=9)
    ax.axis('off')
    return layer_output


# PCA analysis
X_pca_list = []


def plot_pca_3d(ax, pca_data):
    markers = ['o', '^']
    for i, data in enumerate(pca_data):
        pca = PCA(n_components=3)
        X_pca_ = pca.fit(data.T)
        X_pca = pca.components_

        x = X_pca[0]
        y = X_pca[1]
        z = X_pca[2]
        X_pca_list.append((x, y, z))
        N = len(z)
        print(N)

        # Plot trajectory
        for ik in range(N-1):
            ax.plot(x[ik:ik+2], y[ik:ik+2], z[ik:ik+2], color=COLORS[i], alpha=0.3)

        if i == 0:
            ax.scatter(x[0], y[0], z[0], s=70, c='k', marker="^", label=' Start ')
        ax.scatter(x[-1], y[-1], z[-1], s=70, color=COLORS[i], marker="^", label=CONDITION_LABELS[i])

    ax.legend(fontsize=7)
    ax.view_init(elev=15, azim=80)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.legend(fontsize=6, loc='lower right')


def plot_explained_variance(pca_data, plot_dir):
    """
    Plots the cumulative explained variance as a function of the number of principal components.

    Parameters:
        pca_data (list): List of neural activity arrays for each condition.
        plot_dir (str): Directory where the graph will be saved.
    """

    # Calculate PCA for all components (up to N_NEURONS)
    plt.figure(figsize=(8, 6))
    for i, d in enumerate(pca_data):
        all_data = d
        print(all_data.shape)
        pca = PCA(n_components=N_NEURONS)
        pca.fit(all_data)

        # Obtain explained variance
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_explained_variance = np.cumsum(explained_variance_ratio)

        # Plot

        plt.plot(range(1, 101), cumulative_explained_variance * 100, marker='o', alpha=1, linestyle='-',
                 color=COLORS[i], label=f"{CONDITION_LABELS}")
    plt.xlabel('Components', fontsize=12)
    plt.ylabel('Explained Variance (%)', fontsize=12)
    plt.title('Explained Variance vs Components (Task Interval comparison)', fontsize=14)

    plt.axhline(y=85, linestyle='-.', color="gray", alpha=0.3)
    plt.axhline(y=90, linestyle='-.', color="gray", alpha=0.3)
    plt.axhline(y=95, linestyle='-.', color="gray", alpha=0.3)
    plt.axhline(y=100, linestyle='-.', color="gray", alpha=0.3)
    plt.axvline(x=3, linestyle='-.', color="gray", alpha=0.3)
    plt.axvline(x=4, linestyle='-.', color="gray", alpha=0.3)
    plt.axvline(x=5, linestyle='-.', color="gray", alpha=0.3)

    plt.xlim([0, 18])
    plt.xticks(np.arange(0, 20, 1))
    plt.yticks(np.arange(40, 110, 5))
    plt.legend(loc=4)
    plt.savefig(f"{plot_dir}/explained_variance_int_comp.png", dpi=300, bbox_inches='tight')
    plt.close()


def main():
    plt.figure(figsize=cm2inch(12, 6))
    gs = plt.GridSpec(1, 2)
    axes = [plt.subplot(gs[i]) for i in range(2)]

    model = load_trained_model(MODEL_PATH)
    all_activations = []

    for i, condition in enumerate([0, 1]):
        activations = process_condition(model, condition, axes[i])
        all_activations.append(activations)

    plt.suptitle("Interval comparison", y=1.1)
    plt.savefig(f"{PLOT_DIR}/activity_comparison.png", dpi=300)
    plt.close()

    # PCA
    fig = plt.figure(figsize=cm2inch(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    plot_pca_3d(ax, all_activations)
    plt.title("PCA of Neural Activity Across Conditions", y=0.95, fontsize=12)
    plt.savefig(f"{PLOT_DIR}/pca_comparison.png", dpi=300)
    plt.close()
    plot_explained_variance(all_activations, PLOT_DIR)


if __name__ == "__main__":
    main()

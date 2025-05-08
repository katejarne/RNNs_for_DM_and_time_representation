# Windowed Integration Decision-Making
"""
MIT License - C. Jarne V. 1.0 - 2025
This code investigates an RNN performing windowed integration
of noisy sensory signals to make binary decisions. It compares
positive vs negative integral outcomes through temporal neural
activity patterns and 3D PCA trajectory analysis. The
implementation generates temporally extended input signals with
additive noise, computes their time-integrated values as targets,
and analyzes how network dynamics evolve during evidence accumulation.
Dual visualization strategies reveal stimulus-response alignment
in time-domain plots while PCA projections characterize the geometric
separation of positive/negative decision states. By quantifying
explained variance across principal components, the analysis
identifies the minimal dimensionality required to capture integration
dynamics.
"""
import numpy as np
from numpy.random import seed
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from sklearn.decomposition import PCA
import tensorflow as tf
from utilities import cm2inch


# Configuration parameters
MODEL_PATH = "./weights/05_Perceptual_dm_delayed_response/orthogonal_rrn_no_bias_term/weights_N_100_1/100_final.hdf5"
PLOT_DIR = "./plots/PCA"
N_NEURONS = 100  # Should match your trained network


def generate_single_trial(mem_gap=20, first_in=50, stim_dur=100, ensure_negative_integral=False):
    seed(4)
    stim_noise = 1
    out_gap = 250 - 20
    rec_noise = 0

    xor_seed_A = np.array([[0], [1]])
    seq_dur = first_in + stim_dur + mem_gap + (out_gap - mem_gap)
    win = signal.hann(10)

    out_t = mem_gap + first_in + stim_dur
    x_train_ = np.zeros((seq_dur, 1))
    x_train = np.zeros((seq_dur, 1))
    y_train = 0.045 * np.ones((seq_dur, 1))

    trial_type = np.random.randint(2)
    if ensure_negative_integral:
        x_train_[first_in:first_in + stim_dur, 0] = -(xor_seed_A[trial_type, 0] + stim_noise * np.random.randn(stim_dur))
    else:
        x_train_[first_in:first_in + stim_dur, 0] = xor_seed_A[trial_type, 0] + stim_noise * np.random.randn(stim_dur)

    x_train[first_in:first_in + stim_dur, 0] = signal.convolve(x_train_[first_in:first_in + stim_dur, 0],
                                                               win, mode='same') / sum(win)

    y_train[out_t:, 0] = np.sign(scipy.integrate.simps(x_train[first_in:first_in + stim_dur, 0],
                                                       x=None, dx=1, axis=-1, even='avg'))

    return x_train, y_train, seq_dur, first_in + stim_dur, out_t


def load_trained_model(model_path):
    """Load and compile trained RNN model"""
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(loss='mse', optimizer='Adam')
    return model


def process_condition(model, condition, ax):
    """Process one condition and plot results"""
    # Generate input/target using integration task function
    if condition == "positive":
        x_train, y_train, seq_dur, _, _ = generate_single_trial(ensure_negative_integral=False)
    else:
        x_train, y_train, seq_dur, _, _ = generate_single_trial(ensure_negative_integral=True)

    # Get model predictions and activations
    y_pred = model.predict(x_train[np.newaxis, ...])
    layer_output = model.layers[0](x_train[np.newaxis, ...]).numpy()[0]

    # Plotting
    ax.plot(x_train[:, 0], color='g', label='Input')
    ax.plot(y_train[:, 0], color='gray', linewidth=2, label='Target')
    ax.plot(y_pred[0, :, 0], color='r', linewidth=1.5, label='Output')
    ax.set_ylim([-2.5, 2.5])

    # Plot neural activity (all neurons)
    colors = plt.cm.rainbow(np.linspace(0, 1, N_NEURONS))
    for n in range(N_NEURONS):
        ax.plot(layer_output[:, n], color=colors[n], alpha=0.1, linewidth=0.5)
        if n == N_NEURONS-1:  # Highlight last neuron
            ax.plot(layer_output[:, n], color="k", linewidth=0.7)

    ax.set_title(f"Integral: {condition.capitalize()}", fontsize=10)
    ax.axis('off')
    return layer_output


colors = ['blue', 'red']


def plot_pca_3d(ax, pca_data, conditions):
    """Plot 3D PCA of neural activity for both conditions"""

    markers = ['o', '^']

    for i, (data, cond) in enumerate(zip(pca_data, conditions)):
        # Fit PCA
        pca = PCA(n_components=3)
        pca_proj = pca.fit_transform(data.T)
        X_pca = pca.components_

        # Plot trajectory
        x, y, z = X_pca[0], X_pca[1], X_pca[2]
        N = len(z)

        for ik in range(N-1):
            ax.plot(x[ik:ik+2], y[ik:ik+2], z[ik:ik+2],
                    color=colors[i], alpha=0.4, linewidth=1.5)

        # Plot markers
        ax.scatter(x[0], y[0], z[0], s=100,
                   edgecolor='k', facecolor=colors[i],
                   marker='*', label=f'{cond.capitalize()} Start')
        ax.scatter(x[-1], y[-1], z[-1], s=100,
                   edgecolor='k', facecolor=colors[i],
                   marker='X', label=f'{cond.capitalize()} End')

    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    ax.set_zticks(())
    ax.view_init(elev=20, azim=60)
    ax.legend(fontsize=8, loc='lower right')
    ax.set_title("Neural State Trajectories in PCA Space", y=0.95)


def plot_explained_variance(pca_data, plot_dir):
    """
    Plots the cumulative explained variance as a function of the number of principal components.

    Parameters:
        pca_data (list): List of neural activity arrays for each condition.
        plot_dir (str): Directory where the graph will be saved.
    """

    # Calculate PCA for all components (up to N_NEURONS)
    plt.figure(figsize=(10, 6))
    for i,d in enumerate(pca_data):
        all_data=d
        print(all_data.shape)

        pca = PCA(n_components=N_NEURONS)
        pca.fit(all_data)

        # Obtain the cumulative explained variance
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_explained_variance = np.cumsum(explained_variance_ratio)

        # Plot

        plt.plot(range(1, N_NEURONS+1), cumulative_explained_variance * 100, marker='o', alpha=1, linestyle='-',
                 color=colors[i], label=f"Sample {i}")
    plt.xlabel('PCA Components', fontsize=12)
    plt.ylabel('Explained Variance (%)', fontsize=12)
    plt.title('Task Integral DM: Explained Variance vs PCA Components', fontsize=14)
    # plt.grid(True)
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
    plt.savefig(f"{plot_dir}/explained_variance_integral.png", dpi=300, bbox_inches='tight')
    plt.close()


def main():
    # Initialize plots
    plt.figure(figsize=cm2inch(12, 14))
    gs = plt.GridSpec(2, 1)
    axes = [plt.subplot(gs[i]) for i in range(2)]

    # Load trained model once
    model = load_trained_model(MODEL_PATH)

    # Process both conditions
    conditions = ["positive", "negative"]
    all_activations = []

    for i, cond in enumerate(conditions):
        activations = process_condition(model, cond, axes[i])
        all_activations.append(activations)

    # Add legend to first subplot
    axes[0].legend(fontsize=8, loc='lower right')

    # Save activity plot
    plt.suptitle("Integration Task: Neural Network Responses", y=0.96, fontsize=14)
    plt.savefig(f"{PLOT_DIR}/integration_activity.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Create 3D PCA plot
    fig = plt.figure(figsize=cm2inch(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    plot_pca_3d(ax, all_activations, conditions)
    plt.savefig(f"{PLOT_DIR}/integration_pca.png", dpi=300, bbox_inches='tight')
    plt.close()
    plot_explained_variance(all_activations, PLOT_DIR)


if __name__ == "__main__":
    main()

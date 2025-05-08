# Multi-interval Amplitude-based Decision-Making
"""
MIT License - C. Jarne V. 1.0 - 2025
Code analyzes the neural dynamics of a trained (RNN) in a multi-amplitude task.
It generates input stimuli with varying pulse amplitudes,
simulates the RNN’s responses using a pre-trained model, and
visualizes both temporal activity patterns and 3D PCA trajectories of hidden
states. The temporal plots juxtapose network inputs, outputs, and target signals
with neural activations, while the PCA projection illustrates how distinct amplitude
conditions evolve in a low-dimensional state space. By combining time-resolved
activity visualization and dimensionality reduction, the script aims to characterize
how the RNN encodes and transitions between different output amplitude
regimes during task execution.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
import tensorflow as tf
from sklearn.decomposition import PCA
from utilities import cm2inch

# Configuration
MODEL_PATH = "./weights/06_multi_amplitude_task/orthogonal_rrn_no_bias_term/weights_N_100_4/100_final.hdf5"
PLOT_DIR = "./plots/PCA/"

# Mapping pulse height to output amplitude
height_to_amplitude = {
    1: 0.25,
    2: 0.5,
    3: 0.75,
    4: 1.0,
    5: 1.25,
    6: 1.5
}


def generate_single_trial(mem_gap, stim_dur=20, stim_noise=0.05, var_delay_length=70):
    np.random.seed(1)
    first_in = 50
    out_gap = 250 - 20
    seq_dur = first_in + stim_dur + mem_gap + var_delay_length + (out_gap - mem_gap)
    win = scipy.signal.windows.hann(10)
    var_delay = 0  # Retardo fijo

    trials = []
    labels = []

    # Generate 6 amplitude conditions
    for pulse_height in range(1, 7):
        x_trial = np.zeros((seq_dur, 1))
        y_trial = np.zeros((seq_dur, 1))

        # Input signal
        pulse_signal = np.ones(stim_dur) * pulse_height
        convolved = signal.convolve(pulse_signal, win, mode='same')/sum(win)

        start_idx = first_in + var_delay
        x_trial[start_idx:start_idx+stim_dur, 0] = convolved

        # Output amplitude
        output_amp = height_to_amplitude[pulse_height]
        output_start = start_idx + stim_dur + 50  # Tiempo fijo de respuesta
        y_trial[output_start:, 0] = output_amp

        trials.append((x_trial, y_trial))
        labels.append(f"Amplitude {output_amp:.2f} (Pulse {pulse_height})")

    return trials, seq_dur, labels


def main():
    # Load model
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(loss='mse', optimizer='Adam')

    # 1. Time Activity plot
    fig_activity, axes_activity = plt.subplots(2, 3, figsize=(18, 12))
    fig_activity.suptitle("Neural Activity for Different Amplitude Conditions", fontsize=16)
    axes_activity = axes_activity.flatten()

    # Process conditions
    mem_gap = 0
    trials, seq_dur, labels = generate_single_trial(mem_gap)
    all_activations = []

    for i, (x_train, y_train) in enumerate(trials):
        x_expanded = np.expand_dims(x_train, axis=0)
        rnn_layer = model.layers[1] if isinstance(model.layers[0], tf.keras.layers.InputLayer) else model.layers[0]
        layer_output = rnn_layer(x_expanded).numpy()[0]
        all_activations.append(layer_output)
        y_pred = model.predict(x_expanded)[0]

        # Plot activity
        ax = axes_activity[i]
        colors = plt.cm.rainbow(np.linspace(0, 1, 100))
        for n in range(100):
            ax.plot(layer_output[:, n], color=colors[n], alpha=0.3, linewidth=0.5)
            if n == 99:
                ax.plot(layer_output[:, n], color="k", linewidth=0.7)

        ax.plot(x_train[:, 0], color='g', linewidth=1.5, label="Input")
        ax.plot(y_train, color='gray', linewidth=2, label="Target")
        ax.plot(y_pred[:, 0], color='r', linewidth=1.5, label='Output')
        ax.set_title(labels[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim([-6.5, 6.5])
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/amplitude_conditions_activity.png", dpi=200, bbox_inches='tight')
    plt.close()

    # 2. Plot PCA 3D
    fig_pca = plt.figure(figsize=cm2inch(14, 12))
    ax_pca = fig_pca.add_subplot(111, projection='3d')

    # Visual config
    # colors = plt.cm.viridis(np.linspace(0, 1, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 6))
    markers = ['o', '^', 's', 'D', 'p', '*']

    # Process each condition
    all_ends = []
    for i, (activations, label) in enumerate(zip(all_activations, labels)):
        pca = PCA(n_components=3)
        pca_proj = pca.fit_transform(activations)

        x, y, z = pca_proj[:,0], pca_proj[:,1], pca_proj[:,2]
        all_ends.append([x[-1], y[-1], z[-1]])

        # Path
        for ik in range(len(z)-1):
            ax_pca.plot(x[ik:ik+2], y[ik:ik+2], z[ik:ik+2], color=colors[i], alpha=0.4, linewidth=1.5)

        # Markers
        ax_pca.scatter(x[-1], y[-1], z[-1], s=120, edgecolor='k', facecolor=colors[i],
                       marker=markers[i], label=f'{label.split("(")[0]} End', zorder=10)

        ax_pca.scatter(x[0], y[0], z[0], s=80, edgecolor='k', facecolor=colors[i],
                       marker='*', label=f'{label.split("(")[0]} Start')

    all_ends = np.array(all_ends)
    padding = 0.5
    ax_pca.set_box_aspect([1, 1, 1])
    ax_pca.view_init(elev=15, azim=35)
    # ax_pca.legend(fontsize=7, loc='upper left', bbox_to_anchor=(0.01, 0.93))
    ax_pca.legend(fontsize=8, loc='lower right', bbox_to_anchor=(0.01, 0.01), borderaxespad=0.1)
    ax_pca.set_axis_off()
    ax_pca.set_title("Neural State Trajectories by Output Amplitude", y=0.95)
    plt.savefig(f"{PLOT_DIR}/amplitude_conditions_pca.png", dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()

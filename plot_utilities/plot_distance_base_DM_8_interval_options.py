# Multi-interval Distance-based Decision-Making
"""
MIT License - C. Jarne V. 1.0 - 2025
This code analyzes an RNN trained to measure temporal intervals
between paired input pulses. It generates eight stimulus conditions
with pulse separations ranging from 20-160 ms, tracking how neural
population activity encodes these intervals through temporal dynamics
and 3D PCA trajectories. The temporal plots align stimulus inputs
with network outputs and targets, while the PCA visualization reveals
interval-specific neural state trajectories in reduced dimensions.
By systematically rotating the 3D projection and analyzing trajectory
crossings, the script characterizes how temporal interval information
becomes progressively disentangled in state space, demonstrating
the RNN's implementation of a temporal interval-to-space transformation
during delayed response execution.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
import tensorflow as tf
from sklearn.decomposition import PCA
from utilities import cm2inch
from crossings_analysis import count_and_plot_crossings

# Configuration
MODEL_PATH = "./weights/08_DM_delayed_response_8_times_intervals/orthogonal_rrn_no_bias_term/weights_N_100_8/100_final.hdf5"
PLOT_DIR = "./plots/PCA/"


# 8 possible pulse spacings (T) in ms
T_options = [20, 40, 60, 80, 100, 120, 140, 160]


def generate_single_trial(mem_gap, stim_dur=20, stim_noise=0.05, var_delay_length=70):
    np.random.seed(1)
    first_in = 50
    win = scipy.signal.windows.hann(10)
    var_delay = 0  # Fixed delay

    trials = []
    labels = []

    for T in T_options:
        # Temporal sequence
        seq_dur = first_in + 2*stim_dur + 2*T + 100  # Asegurar suficiente duración

        x_trial = np.zeros((seq_dur, 1))
        y_trial = np.zeros((seq_dur, 1))

        # First Pulse
        start_1 = first_in + var_delay
        end_1 = start_1 + stim_dur

        # Second pulse (T ms after the first)
        start_2 = end_1 + T
        end_2 = start_2 + stim_dur

        # Generate pulses
        pulse = np.ones(stim_dur)
        convolved = signal.convolve(pulse, win, mode='same')/sum(win)

        # Assign both pulses
        x_trial[start_1:end_1, 0] = convolved
        x_trial[start_2:end_2, 0] = convolved

        # Response T ms after the second pulse
        response_start = end_2 + T
        y_trial[response_start:, 0] = 1

        trials.append((x_trial, y_trial))
        labels.append(f"T = {T} ms")

    return trials, seq_dur, labels


def main():
    # Load model
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(loss='mse', optimizer='Adam')

    # 1. Temporal activity plots

    fig_activity, axes_activity = plt.subplots(2, 4, figsize=(20, 10))
    fig_activity.suptitle("Neural Activity: Pulse Separation Determines Response Delay", fontsize=14)
    axes_activity = axes_activity.flatten()

    # Generate trials
    trials, seq_dur, labels = generate_single_trial(mem_gap=0)
    all_activations = []

    for i, (x_train, y_train) in enumerate(trials):
        x_expanded = np.expand_dims(x_train, axis=0)
        layer_output = model.predict(x_expanded)[0]  # Asume que el modelo tiene salida directa
        rnn_layer = model.layers[1] if isinstance(model.layers[0], tf.keras.layers.InputLayer) else model.layers[0]
        layer_output = rnn_layer(x_expanded).numpy()[0]
        all_activations.append(layer_output)
        y_pred = model.predict(x_train[np.newaxis, ...])

        # Plotting
        ax = axes_activity[i]
        colors = plt.cm.rainbow(np.linspace(0, 1, 100))
        for n in range(100):
            ax.plot(layer_output[:, n], color=colors[n], alpha=0.2, linewidth=0.5)
            if n == 99:  # Highlight last neuron
                ax.plot(layer_output[:, n], color="k", linewidth=0.7)

        ax.plot(x_train[:, 0], color='g', linewidth=1.5, label="Input")
        ax.plot(y_train[:, 0], color='gray', linewidth=2, label="Target")
        # ax.plot(layer_output[:, 0], color='r', linewidth=1.5, label='Output')
        ax.plot(y_pred[0, :, 0], color='r', linewidth=1.5, label='Output')

        ax.set_title(labels[i], fontsize=10)
        ax.set_ylim([-1.5, 1.5])
        ax.legend(fontsize=8, loc='upper right')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim([-2.5, 2.5])
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/pulse_separation_activity.png", dpi=200, bbox_inches='tight')
    plt.close()

    # 2. PCA Plot (similar al original pero con nuevas etiquetas)
    fig_pca = plt.figure(figsize=cm2inch(16, 14))
    ax_pca = fig_pca.add_subplot(111, projection='3d')

    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    markers = ['o'] * 8
    X_pca_list = []
    all_ends = []
    for i, (activations, label) in enumerate(zip(all_activations, labels)):
        pca = PCA(n_components=3)
        pca_proj = pca.fit_transform(activations.T)
        X_pca = pca.components_
        x = X_pca[0]
        y = X_pca[1]
        z = X_pca[2]

        # x, y, z = pca_proj[:,0], pca_proj[:,1], pca_proj[:,2]
        N = len(z)
        print("Trajectory length", N)
        X_pca_list.append((x, y, z))
        all_ends.append([x[-1], y[-1], z[-1]])

        # Trayectory
        ax_pca.plot(x, y, z, color=colors[i], alpha=0.4, linewidth=1.5)

        # Markers
        if i == 0:
            ax_pca.scatter(x[0], y[0], z[0], s=80, edgecolor='k', facecolor=colors[i], marker='*', label='Start')
        ax_pca.scatter(x[-1], y[-1], z[-1], s=120, edgecolor='k', facecolor=colors[i], marker='o', label=label)

    ax_pca.set_box_aspect([1, 1, 1])
    ax_pca.set_title("PCA: Neural Dynamics by Pulse Separation", y=1)
    # ax_pca.legend(fontsize=8, loc='upper left', bbox_to_anchor=(0.05, 0.95))
    ax_pca.legend(fontsize=8, loc='lower right', bbox_to_anchor=(0.01, 0.01),borderaxespad=0.1)
    ax_pca.set_title("Neural State Trajectories in PCA Space", y=0.95)
    all_ends = np.array(all_ends)
    ax_pca.set_axis_off()
    padding = 0.1
    ax_pca.set_xlim(all_ends[:, 0].min()-padding, all_ends[:, 0].max()+padding)
    ax_pca.set_ylim(all_ends[:, 1].min()-padding, all_ends[:, 1].max()+padding)
    ax_pca.set_zlim(all_ends[:, 2].min()-padding, all_ends[:, 2].max()+padding)

    for azi in [0, 10, 20, 30, 40, 50, 60, 70, 90, 80, 100, 120, 130, 140, 150, 160, 170]:
        ax_pca.view_init(elev=15, azim=azi)
        plt.savefig(f"{PLOT_DIR}/pulse_separation_pca_{azi}.png", dpi=300, bbox_inches='tight')
    plt.close()

    count_and_plot_crossings(X_pca_list, labels, colors, PLOT_DIR, epsilon=0.0000015)


if __name__ == "__main__":
    main()

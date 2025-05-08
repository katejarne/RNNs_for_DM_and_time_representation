# Multi-interval Amplitude-based Decision-Making
"""
MIT License - C. Jarne V. 1.0 - 2025
This code generates input pulses mapped to 8 distinct response delays,
simulates neural dynamics using a pre-trained RNN model,
and visualizes both temporal activation patterns and 3D PCA
trajectories. The temporal plots align network inputs,
outputs, and target signals with population activity, while the PCA
analysis captures state-space evolution across different response
timing conditions. By rendering multiple azimuthal views of the 3D
trajectories and highlighting initial/final states, the script reveals
how the RNN encodes temporal information and transitions between
delay-specific representations during task execution.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
import tensorflow as tf
from sklearn.decomposition import PCA
from utilities import cm2inch

# Configuration
MODEL_PATH = "./weights/04_DM_delayed_response_8_times/orthogonal_rrn_no_bias_term/weights_20_N_100_gap_4/100_final.hdf5"
PLOT_DIR = "./plots/PCA/"

# Mapping pulse height to response time
height_to_time = {
    1: 25, 2: 50, 3: 75, 4: 100,
    5: 125, 6: 150, 7: 175, 8: 200
}


def generate_single_trial(mem_gap, stim_dur=20, stim_noise=0.05, var_delay_length=70):
    np.random.seed(1)
    first_in = 50
    out_gap = 250 - 20
    seq_dur = first_in + stim_dur + mem_gap + var_delay_length + (out_gap - mem_gap)
    win = scipy.signal.windows.hann(10)
    var_delay = 0   # Fixed delay

    trials = []
    labels = []

    # Generate 8 conditions with 1 feature (only signal, no cue)
    for pulse_height in range(1, 9):
        x_trial = np.zeros((seq_dur, 1))  # Cambiar a 1 feature
        y_trial = np.zeros((seq_dur, 1))

        # Input signal
        pulse_signal = np.ones(stim_dur) * pulse_height
        convolved = signal.convolve(pulse_signal, win, mode='same')/sum(win)

        start_idx = first_in + var_delay
        x_trial[start_idx:start_idx+stim_dur, 0] = convolved  # Use only channel 0

        # Response time
        response_time = height_to_time[pulse_height]
        output_start = start_idx + stim_dur + response_time
        y_trial[output_start:, 0] = 1

        trials.append((x_trial, y_trial))
        labels.append(f"Pulse {pulse_height} ({response_time} ms)")

    return trials, seq_dur, labels


def main():
    # Load model
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(loss='mse', optimizer='Adam')
    print(model.summary())

    # 1. Temporal activity plots
    fig_activity, axes_activity = plt.subplots(2, 4, figsize=(20, 10))
    fig_activity.suptitle("Neural Activity for Different Pulse Stimuli", fontsize=16)
    axes_activity = axes_activity.flatten()

    # Process conditions
    mem_gap = 0
    trials, seq_dur, labels = generate_single_trial(mem_gap)
    all_activations = []

    for i, (x_train, y_train) in enumerate(trials):
        x_expanded = np.expand_dims(x_train, axis=0)
        layer_output = model.layers[0](x_expanded).numpy()[0]
        rnn_layer = model.layers[1] if isinstance(model.layers[0], tf.keras.layers.InputLayer) else model.layers[0]
        layer_output = rnn_layer(x_expanded).numpy()[0]
        all_activations.append(layer_output)
        y_pred = model.predict(x_train[np.newaxis, ...])

        # Plot activity
        ax = axes_activity[i]
        colors = plt.cm.rainbow(np.linspace(0, 1, 100))
        for n in range(100):
            ax.plot(layer_output[:, n], color=colors[n], alpha=0.2, linewidth=0.5)
            if n == 99:  # Highlight last neuron
                ax.plot(layer_output[:, n], color="k", linewidth=0.7)

        ax.plot(x_train[:, 0], color='g', linewidth=1.5, label="Input")
        ax.plot(y_train, color='gray', linewidth=2, label="Target")
        ax.plot(y_pred[0, :, 0], color='r', linewidth=1.5, label='Output')
        ax.set_title(labels[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim([-6.5, 6.5])
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/pulse_stimuli_activity.png", dpi=200, bbox_inches='tight')
    plt.close()

    # 2. 3D PCA plot
    fig_pca = plt.figure(figsize=cm2inch(16, 14))
    ax_pca = fig_pca.add_subplot(111, projection='3d')

    # visual config
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    markers = ['o', '^', 's', 'D', 'p', '*', 'X', 'P']

    # Process each condition
    all_ends = []
    for i, (activations, label) in enumerate(zip(all_activations, labels)):
        pca = PCA(n_components=3)
        pca_proj = pca.fit_transform(activations)

        x, y, z = pca_proj[:,0], pca_proj[:,1], pca_proj[:,2]
        all_ends.append([x[-1], y[-1], z[-1]])

        # Trajectory
        for ik in range(len(z)-1):
            ax_pca.plot(x[ik:ik+2], y[ik:ik+2], z[ik:ik+2], color=colors[i], alpha=0.4, linewidth=1.5)

        # Markers
        ax_pca.scatter(x[-1], y[-1], z[-1], s=120, edgecolor='k', facecolor=colors[i],
                       marker='o', label=f'{label} End', zorder=10)

        ax_pca.scatter(x[0], y[0], z[0], s=80, edgecolor='k', facecolor=colors[i],
                       marker='*', label=f'{label} Start')

    # Final adjustments
    all_ends = np.array(all_ends)
    padding = 0.5
    # ax_pca.set_xlim(all_ends[:, 0].min()-padding, all_ends[:,0].max()+padding)
    # ax_pca.set_ylim(all_ends[:, 1].min()-padding, all_ends[:,1].max()+padding)
    # ax_pca.set_zlim(all_ends[:, 2].min()-padding, all_ends[:,2].max()+padding)
    ax_pca.set_box_aspect([1, 1, 1])
    ax_pca.view_init(elev=10, azim=55) # weight 0
    # ax_pca.view_init(elev=10, azim=60) #weight 1/2
    # ax_pca.legend(fontsize=7, loc='upper left', bbox_to_anchor=(0.05, 1.1))
    # Opción 2: Mayor separación horizontal
    ax_pca.legend(fontsize=8, loc='lower right', bbox_to_anchor=(0.01, 0.01),borderaxespad=0.1)
    ax_pca.set_axis_off()
    # ax_pca.legend(fontsize=8, loc='lower right')
    ax_pca.set_title("Neural State Trajectories in PCA Space", y=0.95)
    for azi in [0, 10, 20, 30, 40, 50, 60, 70, 90, 80, 100, 120, 130, 140, 150, 160, 170]:
        ax_pca.view_init(elev=15, azim=azi)
        plt.savefig(f"{PLOT_DIR}/pulse_stimuli_pca_{azi}.png", dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()

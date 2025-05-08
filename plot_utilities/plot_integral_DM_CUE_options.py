# Cued Integration time Decision-Making
"""
MIT License - C. Jarne V. 1.0 - 2025
This code analyzes an RNN performing a perceptual decision-making
task with contextual cue modulation. It generates trials combining
noisy integral signals with binary cues that either enforce or
disable behavioral responses. The script compares four conditions
(positive/negative signals with/without cues), visualizing temporal
neural activity aligned with sensory inputs, contextual cues, and
target/predicted outputs. Through 3D PCA trajectory analysis and
multi-angle projections, it examines how cue availability modulates
population dynamics during evidence integration. The inclusion of
trajectory crossing analysis further quantifies condition-specific
neural state separability, revealing how contextual cues reshape
the RNN's decision-making pathways in low-dimensional space.
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
MODEL_PATH = "./weights/07_Perceptual_dm_delayed_response_signal_dont_end_cue/orthogonal_rrn_no_bias_term/weights_N_100_4/100_final.hdf5"
PLOT_DIR = "./plots/PCA"


def generate_single_trial(mem_gap, stim_dur=50, stim_noise=1, var_delay_length=150):
    np.random.seed(1)
    first_in = 50   # Initial time of the first stimulus
    seq_dur = first_in + stim_dur + mem_gap + var_delay_length + (250 - mem_gap - 20)
    win = scipy.signal.windows.hann(10)  # Hanning window
    var_delay = np.random.randint(var_delay_length) + 1 if var_delay_length != 0 else 0

    # Generate random base signal
    base_signal = stim_noise * np.random.randn(stim_dur + mem_gap + 100)
    processed_signal = signal.convolve(base_signal, win, mode='same') / sum(win)
    output_signal = np.sign(scipy.integrate.simps(processed_signal[:stim_dur]))

    trials = []
    labels = ["Negative + Cue", "Positive+ Cue", "Negative no Cue", "Positive no Cue"]

    for cue, invert in [(1, False), (1, True), (0, False), (0, True)]:
        x_trial = np.zeros((seq_dur, 2))
        y_trial = np.zeros((seq_dur, 1))

        signal_used = -processed_signal if invert else processed_signal

        x_trial[first_in + var_delay:first_in + stim_dur + var_delay + mem_gap + 100, 0] = signal_used
        x_trial[first_in + var_delay - 20:first_in + var_delay - 10, 1] = cue
        y_trial[first_in + stim_dur + var_delay + mem_gap + 10:, 0] = output_signal if not invert else -output_signal
        if cue == 0:
            y_trial[first_in + stim_dur + var_delay + mem_gap + 10:, 0] =0

        trials.append((x_trial, y_trial))

    return trials, seq_dur, labels


def process_condition(model, condition_idx, axes):
    trials, seq_dur, labels = generate_single_trial(condition_idx)
    activations_list = []

    for i, (x_train, y_train) in enumerate(trials):
        print(f"Procesando variante {i}: {labels[i]}")

        # Ensure the model receives the input with the correct dimension
        x_train = np.expand_dims(x_train, axis=0)  # (1, seq_dur, 2)
        y_pred = model.predict(x_train)[0]  # Tomamos la primera predicción
        layer_output = model.layers[0](x_train[np.newaxis, ...]).numpy()[0]
        activations_list.append(layer_output)

        # Plot input, expected output, and predicted output
        ax = axes[i]
        ax.plot(x_train[0, :, 0], color='g', label="Input")
        ax.plot(x_train[0, :, 1], color='pink', label="Cue")
        ax.plot(y_train, color='gray', linewidth=2, label="Expected Output")
        ax.plot(y_pred, color='blue', linestyle="dashed", label="Predicted Output")
        ax.set_ylim([-2.5, 2.5])
        ax.legend(fontsize=7, loc='lower right')
        ax.legend(fontsize=6, loc=3)
        ax.set_xticks(np.arange(0, seq_dur + 50, 100))
        ax.set_title(labels[i], fontsize=12)

    return activations_list


def main():
    # Load trained model
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(loss='mse', optimizer='Adam')

    # 1. Create time activity plots
    fig_activity, axes_activity = plt.subplots(1, 4, figsize=(18, 6))

    fig_activity.suptitle("Integral decision with or without CUE", fontsize=16)

    # Process conditions and obtain neural activations
    mem_gap = 50  # parameter
    trials, seq_dur, labels = generate_single_trial(mem_gap)
    all_activations = []  # Will store activations of the 4 conditions

    # Process each trial
    for i, (x_train, y_train) in enumerate(trials):
        # Obtener predicciones y activaciones
        x_expanded = np.expand_dims(x_train, axis=0)  # Add batch dimension
        y_pred = model.predict(x_expanded)[0]
        layer_output = model.layers[0](x_expanded).numpy()[0]
        all_activations.append(layer_output)  # Save for PCA

        # Plot in corresponding subplot
        ax = axes_activity[i]
        colors = plt.cm.rainbow(np.linspace(0, 1, 100))
        for n in range(100):
            ax.plot(layer_output[:, n], color=colors[n], alpha=0.2, linewidth=0.7)
            if n == 99:
                ax.plot(layer_output[:, n], color="k", linewidth=0.7)
        ax.plot(x_train[:, 0], color='g', label="Input")
        ax.plot(x_train[:, 1], color='hotpink', label="Cue", linewidth=2)
        ax.plot(y_train, color='gray', linewidth=2, label="Expected Output")
        ax.plot(y_pred, color='red', linewidth=2, label="Predicted Output")
        ax.set_ylim([-2.5, 2.5])
        ax.set_xlim([0, 350])
        ax.legend(fontsize=12, loc=3)
        ax.set_xticks(np.arange(0, seq_dur + 50, 100))
        ax.set_title(labels[i], fontsize=10)
        ax.axis('off')

    # Save activity plot
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/Perceptual_DM_cue_Predictions.png", dpi=200)
    plt.close()

    # 2. Crate PCA 3D
    fig_pca = plt.figure(figsize=cm2inch(14, 12))
    ax_pca = fig_pca.add_subplot(111, projection='3d')

    # Colors
    colors = ['blue', 'red', 'green', 'purple']
    markers = ['o', '^', 's', 'D']

    all_ends = []
    X_pca_list = []
    for i, (activations, label) in enumerate(zip(all_activations, labels)):

        # Components
        pca = PCA(n_components=3)
        pca_proj = pca.fit_transform(activations.T)
        # pca = PCA(n_components=3)
        X_pca_ = pca.fit(activations.T)
        X_pca = pca.components_

        x = X_pca[0]
        y = X_pca[1]
        z = X_pca[2]
        X_pca_list.append((x, y, z))
        N = len(z)
        print("len", N)
        all_ends.append([x[-1], y[-1], z[-1]])
        padding = 0.8  # padding

        # Trajectory

        if i == 0:
            ax_pca.scatter(x[0], y[0], z[0], s=80, edgecolor='k', facecolor=colors[i], marker='*', label='Start')

        ax_pca.scatter(x[-1], y[-1], z[-1], s=90, edgecolor='k', facecolor=colors[i],
                       marker=markers[i], label=f'{label} End')

        if i == 2:
            for ik in range(N-1):
                ax_pca.plot(x[ik:ik+2], y[ik:ik+2], z[ik:ik+2], color=colors[i], alpha=0.4, linewidth=2.5)

        else:
            for ik in range(N-1):
                ax_pca.plot(x[ik:ik+2], y[ik:ik+2], z[ik:ik+2],
                            color=colors[i], alpha=0.4, linewidth=1.5)
            # Markers start/end

    # visualization adjustments
    all_ends = np.array(all_ends)
    ax_pca.view_init(elev=20, azim=45)
    ax_pca.legend(fontsize=8, loc='upper left')
    ax_pca.set_title("PCA of Neural Activity Across Conditions", y=0.95)
    ax_pca.legend(fontsize=7, loc='lower right')
    ax_pca.axes.get_xaxis().set_ticks([])
    ax_pca.axes.get_yaxis().set_ticks([])
    # ax_pca.set_xlim(all_ends[:,0].min()-padding, all_ends[:,0].max()+padding)
    # ax_pca.set_ylim(all_ends[:,1].min()-padding, all_ends[:,1].max()+padding)
    # ax_pca.set_zlim(all_ends[:,2].min()-padding, all_ends[:,2].max()+padding)
    ax_pca.set_zticks(())
    ax_pca.view_init(elev=10, azim=120)
    ax_pca.set_box_aspect([1, 1, 1])
    for azi in [0, 10, 20, 30, 40, 50, 60, 70, 90, 80, 100, 120, 130, 140, 150, 160, 170]:
        ax_pca.view_init(elev=15, azim=azi)
        # save PCA plot
        plt.savefig(f"{PLOT_DIR}/integration_cue_pca_{azi}.png", dpi=300, bbox_inches='tight')
    plt.close()
    count_and_plot_crossings(X_pca_list, labels, colors, PLOT_DIR, epsilon=0.0000015)


if __name__ == "__main__":
    main()


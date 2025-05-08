# Context-dependent Binary Decision-Making
"""
MIT License - C. Jarne V. 1.0 - 2025
This code analyzes an RNN performing context-dependent
binary decisions based on pulse amplitude and polarity.
It generates four stimulus conditions combining short/long
durations with positive/negative polarities, examining how
these features shape neural dynamics through temporal
activity plots and 3D PCA trajectories.
The implementation reconstructs network architecture from
saved weights, aligns stimulus inputs with target/output signals
in time-domain visualizations, and quantifies state-space
separability using PCA. By calculating also cumulative explained variance across
PCA dimensions and plotting condition-specific neural trajectories,
the study reveals how temporal and categorical features of stimuli
become encoded in low-dimensional representations during decision formation.
"""
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy import signal
import numpy as np
from utilities import cm2inch
import h5py
from tensorflow.keras.layers import Input, SimpleRNN, Dense
from tensorflow.keras.models import Model

# Configuration parameters
# MODEL_PATH = "weights/02_DM_delayed_response_long-short/Random_normal_rnn_no_bias_term_sigma_1.05/try_15050/weights_20_N_100_gap_4/100_final.hdf5"
MODEL_PATH = "./weights/02_DM_delayed_response_long-short/orthogonal_rrn_no_bias_term/weights_20_N_100_gap_4/100_final.hdf5"
#MODEL_PATH = "weights/02_DM_delayed_response_long-short/orthogonal_rrn_no_bias_term/weights_20_N_100_gap_4/simple_DM_weights-20.keras"
N_NEURONS = 100  # Should match trained network
PLOT_DIR = "./plots/PCA"

"""
def load_trained_model(model_path):
    #Load and compile trained RNN model
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(loss='mse', optimizer='Adam')
    return model
"""


def generate_single_trial(pulse_height, pulse_sign, mem_gap=0, first_in=50, stim_dur=20):
    xor_seed_A = np.array([[0], [1]])
    seq_dur = first_in + stim_dur + mem_gap + (250 - 20 - mem_gap)
    win = signal.hann(10)

    out_t = mem_gap + first_in + stim_dur
    x_train_ = np.zeros((seq_dur, 1))
    x_train = np.zeros((seq_dur, 1))
    y_train = np.zeros((seq_dur, 1))

    trial_type = 1  # Use a fixed trial type for simplicity
    stimulus = xor_seed_A[trial_type, 0] * pulse_height * pulse_sign
    convolved_stimulus = signal.convolve(stimulus * np.ones(stim_dur), win, mode='same') / sum(win)
    x_train[first_in:first_in + stim_dur, 0] = convolved_stimulus

    if pulse_height == 1 and pulse_sign == 1:
        y_train[out_t + 50:, 0] = xor_seed_A[trial_type, 0]
    elif pulse_height == 1 and pulse_sign == -1:
        y_train[out_t + 50:, 0] = -1 * xor_seed_A[trial_type, 0]
    elif pulse_height == 2 and pulse_sign == 1:
        y_train[out_t + 100:, 0] = xor_seed_A[trial_type, 0]
    elif pulse_height == 2 and pulse_sign == -1:
        y_train[out_t + 100:, 0] = -1 * xor_seed_A[trial_type, 0]

    return x_train, y_train, seq_dur, first_in + stim_dur, out_t


def inspect_weights(model_path):
    """Inspecciona los nombres de las capas en el archivo de pesos."""
    with h5py.File(model_path, 'r') as f:
        print("Capas en el archivo de pesos:")
        for layer in f.keys():
            print(layer)


def build_model():
    """Defines the model architecture exactly as in training."""
    input_layer = Input(shape=(None, 1), name="input_layer")  # Secuencia de longitud arbitraria con 1 feature
    rnn_layer = SimpleRNN(100, activation="tanh", return_sequences=True, use_bias=False, name="optimizer_weights")(
        input_layer)
    output_layer = Dense(1, name="output_layer")(rnn_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model


def load_trained_model(model_path):
    """Loads weights into a newly built model with matching architecture"""
    model = build_model()

    try:
        model.load_weights(model_path)  # Carga solo los pesos
        print(" Weights loaded correctly.")
    except Exception as e:
        print(f"Error loading weights: {e}")

    model.compile(loss='mse', optimizer='adam')  # Compila el modelo
    return model


def process_condition(model, pulse_height, pulse_sign, ax):
    """Process one condition and plot results"""
    # Generate input/target using provided function

    x_train, y_train, seq_dur, end_of_pulse, change_point = generate_single_trial(pulse_height, pulse_sign)

    # Get model predictions and activations
    y_pred = model.predict(x_train[np.newaxis, ...])
    print(y_pred)
    layer_output = model.layers[1](x_train[np.newaxis, ...]).numpy()[0]
    print(layer_output.shape)

    # Plotting
    ax.plot(x_train[:, 0], color='g', label='Input')
    ax.plot(y_train[:, 0], color='gray', linewidth=2, label='Target')
    ax.plot(y_pred[0, :, 0], color='r', linewidth=1.5, label='Output')
    ax.set_ylim([-2.5, 2.5])

    # Plot neural activity (first 5 neurons)
    colors = plt.cm.rainbow(np.linspace(0, 1, N_NEURONS))
    for n in range(N_NEURONS):
        ax.plot(layer_output[:, n], color=colors[n], alpha=0.3, linewidth=0.7)
        if n == N_NEURONS-1:
            ax.plot(layer_output[:, n], color="k", linewidth=0.7)

    ax.set_title(f"Amp: {pulse_height}, Sign: {pulse_sign}", fontsize=9)
    ax.axis('off')
    return layer_output


# Call the function to see the structure of the weights file
inspect_weights(MODEL_PATH)
X_pca_list = []
colors = ['blue', 'green', 'red', 'purple']
conditions_label = ["Short Positive", "Short Negative", "Long Positive", "Long Negative"]


def plot_pca_3d(ax, pca_data, conditions):
    """Plot 3D PCA of neural activity for all conditions"""
    markers = ['o', '^', 's', 'D']
    conditions = [(1, 1), (1, -1), (2, 1), (2, -1)]
    for i, (data, cond) in enumerate(zip(pca_data, conditions)):
        # Fit PCA
        pca = PCA(n_components=3)
        X_pca_ = pca.fit(data.T)
        X_pca = pca.components_

        print("------------")

        x = X_pca[0]
        y = X_pca[1]
        z = X_pca[2]

        X_pca_list.append((x, y, z))
        N = len(z)
        print(N)
        # Plot trajectory
        for ik in range(N-1):
            ax.plot(x[ik:ik+2], y[ik:ik+2], z[ik:ik+2], color=colors[i], alpha=0.3)
        if conditions_label[i] == "Long Positive":
            ax.scatter(x[0], y[0], z[0], s=70,c='k', marker="^", label=' Start ')
        ax.scatter(x[-1], y[-1],z[-1], s=70,color=colors[i], marker="^", label=conditions_label[i])
        # ax.scatter(x[-1], y[-1],z[-1], s=70, c='b', marker="^", label=' Stop ')

        # Plot start/end markers
    # ax.set_xlabel('PC1', fontsize=8)
    # ax.set_ylabel('PC2', fontsize=8)
    # ax.set_zlabel('PC3', fontsize=8)

    ax.legend(fontsize=7, loc='lower right')
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    ax.set_zticks(())
    ax.view_init(elev=10, azim=70)


def plot_explained_variance(pca_data, plot_dir):
    """
    Plots the cumulative explained variance as a function of the number of principal components.

    Parameters:
        pca_data (list): List of neural activity arrays for each condition.
        plot_dir (str): Directory where the graph will be saved.
    """

    # Calculate PCA for all components (up to N_NEURONS)
    plt.figure(figsize=(10, 6))
    for i, d in enumerate(pca_data):
        all_data = d
        print(all_data.shape)

        pca = PCA(n_components=N_NEURONS)
        pca.fit(all_data)

        # Obtain the cumulative explained variance
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_explained_variance = np.cumsum(explained_variance_ratio)

        # Crear la gráfica

        plt.plot(range(1, 101), cumulative_explained_variance * 100, marker='o', alpha=1, linestyle='-',
                 color=colors[i], label=f"{conditions_label[i]}")
    plt.xlabel('PCA Components', fontsize=12)
    plt.ylabel('Explained Variance (%)', fontsize=12)
    plt.title('Task DM: Explained Variance vs PCA Components', fontsize=14)
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

    # Guardar la gráfica
    plt.savefig(f"{plot_dir}/explained_variance.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Initialize plots
    plt.figure(figsize=cm2inch(16, 10))
    gs = plt.GridSpec(2, 2)
    axes = [plt.subplot(gs[i//2, i % 2]) for i in range(4)]

    # Load trained model once
    input_shape = (370, 1)
    model = load_trained_model(MODEL_PATH)

    # Process all conditions
    conditions = [(1, 1), (1, -1), (2, 1), (2, -1)]
    all_activations = []

    for i, (amp, sign) in enumerate(conditions):
        # Process condition and store activations
        activations = process_condition(model, amp, sign, axes[i])
        all_activations.append(activations)

    # Add legend to first subplot
    axes[0].legend(fontsize=6, loc='lower right')

    # Save activity plot
    plt.suptitle("Neural Network Responses to Different Input Conditions", y=0.98, fontsize=12)
    plt.savefig(f"{PLOT_DIR}/combined_activity.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Create 3D PCA plot
    fig = plt.figure(figsize=cm2inch(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    plot_pca_3d(ax, all_activations, conditions)
    plt.title("PCA of Neural Activity Across Conditions", y=0.95, fontsize=12)
    plt.savefig(f"{PLOT_DIR}/combined_pca.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("activations_shape",len(all_activations))
    plot_explained_variance(all_activations, PLOT_DIR)
    conditions_label = ["Short Positive", "Short Negative", "Long Positive", "Long Negative"]
    # count_and_plot_crossings(X_pca_list, conditions_label, colors, PLOT_DIR, epsilon=0.0000015)


if __name__ == "__main__":
    main()

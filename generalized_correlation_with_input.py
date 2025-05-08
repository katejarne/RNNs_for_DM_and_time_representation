# generalized_correlation_with_input.py
"""
MIT License - C. Jarne V. 1.0 - 2025
This script analyzes the neural activity of a trained RNN
during a simple delayed-response task. It computes the generalized correlation between
output weights and neural activity across three task periods: the signal period,
intermediate period, and decision period. Input and output signals are classified as
positive, negative, or null, based on defined thresholds. The script generates plots
for individual samples showing input-output dynamics and calculates averages with error
bars for correlation results grouped by decision type. Outputs are saved as visual
summaries to provide insights into decision-making and task dynamics.
"""
import os
from numpy.linalg import norm
import matplotlib.pyplot as plt
# GPU configuration
os.environ['TF_DISABLE_GPU'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import Initializer
from utils.net_constraint_create import *


# Path to the trained model
current_directory = os.path.dirname(__file__)
r_dir = current_directory+"/weights/05_Perceptual_dm_delayed_response/orthogonal_rrn_with_bias_term/weights_N_100_0/"
# r_dir = current_directory+"/weights/01_DM_delayed_response/orthogonal_rrn_with_bias_term/weights_20_N_100_gap_0/"
# r_dir = current_directory+"/weights/05_Perceptual_dm_delayed_response/large_out_weights/weights_N_100_2/"
path = r_dir + "100_final.hdf5"

task = "Integral DM"  # "Simple DM"#
# task: "Simple DM", "Simple DM Long-short", "Simple DM 4 times","Simple DM 8 times", "Integral DM",
# "Integral DM signal keep", "Integral DM Cue"

# Define helper functions and custom layers

# Variables to capture positive/negative samples
positive_sample_activity = None
negative_sample_activity = None


def custom_simple_rnn(**config):
    if 'time_major' in config:
        del config['time_major']  # Remove unrecognized argument
    return tf.keras.layers.SimpleRNN(**config)


class IdentityInitializer(Initializer):
    def __call__(self, shape, dtype=None):
        if shape[0] != shape[1]:
            raise ValueError("Identity matrix initializer requires a square matrix shape.")
        return np.identity(shape[0], dtype=dtype)


def classify_decision(y_signal, threshold=0.1):
    """Classify the decision as positive, negative or null."""
    max_val = np.max(y_signal)
    min_val = np.min(y_signal)

    if max_val > threshold:
        return "positive"
    elif min_val < -threshold:
        return "negative"
    else:
        return "no_signal"

"""
    positive_indices = np.where(y_signal > threshold)[0]
    negative_indices = np.where(y_signal < -threshold)[0]
    return positive_indices, negative_indices
"""

# ========== Dictionary for storing positive, negative or null results ==========

"""
results = {
    "positive": {"signal": [], "inter": [], "decision": []},
    "negative": {"signal": [], "inter": [], "decision": []},
}
"""

results = {
    "positive": {"signal": [], "inter": [], "decision": []},
    "negative": {"signal": [], "inter": [], "decision": []},
    "no_signal": {"signal": [], "inter": [], "decision": []}
}


def generalized_correlation(W_in, array_red):
    """
    Calculates the generalized correlation between W_in and neural activity (array_red).

    W_in: (input_dim, neurons)
    array_red: (neurons, time)
    """
    # Remove unnecessary dimensions from array_red
    array_red = np.squeeze(array_red)  # Ensures shape is (neurons x time)
    W_in = W_in.T  # Transpose (neurons, input_dim) to aligne with array_red
    # Ensure alignment of shapes
    if W_in.shape[0] != array_red.shape[0]:
        raise ValueError(f"Incompatible shapes: W_in {W_in.shape}, array_red {array_red.shape}")

    # Project neural activity onto the output weights

    projection = np.dot(W_in.T, array_red)  # (input_dim, time)

    # Calculate norms
    numerator = norm(projection, ord='fro')
    denominator = np.linalg.norm(W_in) * np.linalg.norm(array_red, ord='fro')

    return numerator / denominator if denominator != 0 else 0


def generalized_correlation_per_output(W_in, array_red):

    # For each output, correlation between its weights and the averaged activity
    rho_per_output = []
    for j in range(W_in.shape[1]):    # Iterar sobre outputs
        w_j = W_in[:, j]              # Pesos del output j (neuronas x 1)
        a_j = array_red.mean(axis=1)  # Actividad promedio por neurona (neuronas x 1)
        rho_j = np.dot(w_j, a_j) / (norm(w_j) * norm(a_j))
        rho_per_output.append(rho_j)
    rho = np.mean(rho_per_output)

    return rho


def extract_weights_and_activity(model, x_sample):
    """
    Extracts output weights and neural activity given a model and an input sample.
    """
    W_in = model.layers[0].get_weights()[0]
    layer = 0  # Index of the recurrent layer
    layer_outputs = model.layers[layer](x_sample)
    tensor_np = layer_outputs.numpy()
    array_red = tensor_np.T  # Shape: (N_neurons x T_time)
    return W_in, array_red


def plot_input_output(sample_idx, x_sample, y_sample, x_start, decision_start, save_dir="plots"):
    """
    Plots input vs output with dynamic vertical lines based on detected events.

    Parameters:
    sample_idx (int): Sample index.
    x_sample (numpy.ndarray): Input data (shape: time x features).
    y_sample (numpy.ndarray): Output data (shape: time x features).
    x_start (int): Start index of the input activity period.
    decision_start (int): Start index of the decision period.
    save_dir (str): Directory to save the plot.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(10, 6))
    plt.plot(x_sample[:, 0], label="Input", color="blue")
    plt.plot(y_sample[:, 0], label="Output", color="red")

    # Plot vertical lines only if values are valid
    if x_start is not None:
        plt.axvline(x=x_start, color="green", linestyle="--", label="Input Activity Start")
        plt.axvline(x=x_end, color="green", linestyle="--", label="Input Activity End")

    if decision_start is not None:
        plt.axvline(x=decision_start, color="purple", linestyle="--", label="Decision Start")

    plt.xlabel("Time", fontsize=10)
    plt.ylabel("Amplitude", fontsize=10)
    plt.ylim([-2.5, 2.5])
    plt.xlim(0, len(y_sample[:, 0]))
    plt.xticks(np.arange(0, len(y_sample[:, 0])+1, 50))
    plt.title(f"Input and Output for Sample {sample_idx}")
    plt.legend(loc='upper right')
    # plt.grid(True)

    save_path = os.path.join(save_dir, f"sample_{sample_idx}_input_output.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

# Load model with custom objects


custom_objects = {'NonNegLast': NonNegLast, 'NonNegLast_input': NonNegLast_input,
                  'my_init_exi_ini': my_init_exi_ini, 'my_init_rec': my_init_rec,
                  'SimpleRNN': custom_simple_rnn, 'IdentityInitializer': IdentityInitializer}
model = load_model(path, custom_objects=custom_objects, compile=False)

# Generate input data
if task == "Integral DM":
    from data_set_generators.generate_perceptual_dm import generate_trials
    mem_gap = 200

if task in ["Simple DM", "Simple DM Long-short"]:
    # be aware of internal conf for data_set_generator to create one data set or the other
    from data_set_generators.generate_DM_delayed_response_sample import *
    mem_gap = 0

int_time = 150

x_train, y_train, seq_dur = generate_trials(20, mem_gap)


def detect_events(signal, threshold=0.2):  # 1e-3
    """Detects start/end of activity on a signal."""
    # Find indices where the signal exceeds the threshold
    active_indices = np.where(np.abs(signal) > threshold)[0]

    if len(active_indices) == 0:
        return None, None  # No activity

    # First and last active index
    start = active_indices[0]
    end = active_indices[-1]

    return start, end


for sample_idx, (x_sample, y_sample) in enumerate(zip(x_train[:20], y_train[:20])):
    x_sample = x_sample[np.newaxis, :, :]  # Add batch dimension
    W_in, array_red = extract_weights_and_activity(model, x_sample)

    # Obtaining relevant signals (removing batch dimension)
    x_signal = x_sample[0, :, 0]  # Format (time,)
    y_signal = y_sample[:, 0]    # Format (time,)

    # ----------------------------
    # 1. Detect RANDOM PERIOD
    # ----------------------------
    # Start: first point where x != 0
    x_start, x_end = detect_events(x_signal)

    # End: 40 steps after 10 consecutive zeros in x
    null_window = 10
    post_null_buffer = 40

    # Search for windows of zeros
    zeros = np.where(x_signal == 0)[0]
    zero_windows = np.split(zeros, np.where(np.diff(zeros) != 1)[0]+1)

    # Find the first window of at least 10 consecutive zeros
    valid_window = next((w for w in zero_windows if len(w) >= null_window), None)

    if valid_window is not None:
        period_end = valid_window[-1] + post_null_buffer
    else:
        period_end = x_end  # Fallback

    random_period = array_red[:, x_start:period_end, :] \
        if x_start is not None and x_end is not None else np.zeros((array_red.shape[0], 0))

    # ----------------------------
    # 2. Detect DECISION PERIOD
    # ----------------------------
    # Start: first transition from 0 to non-zero in y
    y_active = np.where(y_signal != 0)[0]
    decision_start = y_active[0] if len(y_active) > 0 else 0

    decision_period = array_red[:, decision_start:, :]
    inter_period = array_red[:, period_end:decision_start, :] \
        if x_end is not None else np.zeros((array_red.shape[0], 0))

    # Classify the decision
    decision_type = classify_decision(y_signal)
    # ----------------------------
    # Post-calculations
    # ----------------------------

    # Capture first sample of each tip
    if decision_type == "positive" and positive_sample_activity is None:
        positive_sample_activity = array_red  # Format (neurons, time)
        print(f"Capturada muestra POSITIVA en sample {sample_idx}")
    elif decision_type == "negative" and negative_sample_activity is None:
        negative_sample_activity = array_red
        print(f"Capturada muestra NEGATIVA en sample {sample_idx}")

    rho_random = generalized_correlation(W_in, random_period)
    rho_decision = generalized_correlation(W_in, decision_period)
    rho_inter = generalized_correlation(W_in, inter_period)

    # Store results by category
    results[decision_type]["signal"].append(rho_random)
    results[decision_type]["inter"].append(rho_inter)
    results[decision_type]["decision"].append(rho_decision)

    print(f"Sample {sample_idx + 1}:")
    print(f"  Random Period: {x_start}-{period_end}")
    print(f"  Decision Start: {decision_start}")
    print(f"  Correlations: Random={rho_random:.4f}, Decision={rho_decision:.4f}\n")

    plot_input_output(sample_idx, x_sample[0], y_sample, x_start, decision_start, save_dir="plots")


print("\nFinal Results for All Samples:")
for category, data in results.items():
    print(f"\nCategory: {category}")
    print(f"  Signal Period Correlations: {data['signal']}")
    print(f"  Intermediate Period Correlations: {data['inter']}")
    print(f"  Decision Period Correlations: {data['decision']}")

# ========== Function to graph results ==========


def plot_decision_correlations(results, save_dir="plots"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    categories = ["positive", "negative", "no_signal"]
    # categories = ["positive", "negative"]
    labels = ["Signal Period", "Intermediate", "Decision Period"]
    colors = ["#4e79a7", "#f28e2b", "#e15759"]

    # Calcular promedios
    avg_data = {
        cat: [
            np.mean(results[cat]["signal"]) if results[cat]["signal"] else 0,
            np.mean(results[cat]["inter"]) if results[cat]["inter"] else 0,
            np.mean(results[cat]["decision"]) if results[cat]["decision"] else 0
        ]
        for cat in categories
    }

    # Set bar position
    x = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    # Calculate averages and standard errors
    error_data = {
        cat: [
            np.std(results[cat]["signal"]) / np.sqrt(len(results[cat]["signal"]))
            if len(results[cat]["signal"]) > 1 else 0,
            np.std(results[cat]["inter"]) / np.sqrt(len(results[cat]["inter"]))
            if len(results[cat]["inter"]) > 1 else 0,
            np.std(results[cat]["decision"]) / np.sqrt(len(results[cat]["decision"]))
            if len(results[cat]["decision"]) > 1 else 0
        ]
        for cat in categories
    }

    # Create bars with error bars for each segment
    for i, (label, color) in enumerate(zip(labels, colors)):
        values = [avg_data[cat][i] for cat in categories]
        errors = [error_data[cat][i] for cat in categories]  # Error bars for each category
        ax.bar(x + i * width, values, width, label=label, color=color, yerr=errors, capsize=5)

    # Customize the plot
    ax.set(frame_on=False)
    ax.set_ylabel("Average Correlation")
    ax.set_title(f"Correlation W^in vs Out activity by Decision Type and Period (Task: {task})")
    ax.set_xticks(x + width)
    ax.set_xticklabels(["Positive Decision", "Negative Decision", "Zero Input"], fontsize=10)
    # ax.set_xticklabels(["Positive Decision", "Negative Decision"])
    # ax.axhline(y=0.2, color="green", linestyle="--", label="Input Activity Start")
    ax.legend(loc='upper right')
    ax.set_ylim([0, 0.5])
    # ax.grid(True, linestyle='--', alpha=0.6)

    # Save the figure
    plt.tight_layout()
    save_path = os.path.join(save_dir, "decision_correlation_win_summary.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# Plot the correlation in each decision
# Calcular métricas de información para las muestras capturadas


def calculate_information_metrics(activity, label):
    """Calculate information metrics using hoi."""
    if activity is None:
        return
    print(activity.shape)
    print(activity.reshape(100, -1).shape)

    # Prepare data (time x neurons) and normalize
    data_ = activity.reshape(100, -1)  # (tiempo, neuronas)
    # data_ = (data_ - np.mean(data_, axis=0)) / np.std(data_, axis=0)
    data = data_.astype(np.float64)
    n_neurons = data.shape[1]
    comb = np.arange(n_neurons).reshape(1, -1)  # Combining activity
    # pepe = calculate_hoi_metrics(data, label, N_rec) #Not implemented


# Run calculations
if positive_sample_activity is not None:
    calculate_information_metrics(positive_sample_activity, "POSITIVA")
if negative_sample_activity is not None:
    calculate_information_metrics(negative_sample_activity, "NEGATIVA")


plot_decision_correlations(results)

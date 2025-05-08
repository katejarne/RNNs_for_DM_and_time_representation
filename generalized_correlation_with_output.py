# generalized_correlation_with_output.py
"""
MIT License - C. Jarne V. 1.0 - 2025
This code analyzes the relationship between output weights and neural dynamics in RNNs
performing decision-making tasks. It computes generalized correlations between output
connection weights and population activity across different task phases
(stimulus processing, delay period, decision execution), comparing positive
vs negative decisions. The implementation features temporal event detection in input/output
signals, sequentiality index calculation to quantify information flow patterns,
and correlation matrix visualization of population code structure.
By analyzing multiple trials and decision types, it characterizes how output weights
align with specific dynamical regimes during task execution, while evaluating the
temporal organization of neural activity through entropy-based metrics.
The analysis reveals mechanistic insights into how network architecture supports
decision formation through phase-specific neural coordination.
"""
import os
from numpy.linalg import norm
from utils.SI_computing import compute_SI

# GPU configuration
os.environ['TF_DISABLE_GPU'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import Initializer
from utils.net_constraint_create import *
import matplotlib.pyplot as plt

# Path to the trained model
current_directory = os.path.dirname(__file__)

# r_dir = current_directory + "/weights/05_Perceptual_dm_delayed_response/orthogonal_rrn_with_bias_term/weights_N_100_0/"
# r_dir = current_directory+"/weights/01_DM_delayed_response/orthogonal_rrn_with_bias_term/weights_20_N_100_gap_4/"
# r_dir = current_directory+"/weights/01_DM_delayed_response/Random_normal_rnn_no_bias_term_sigma_1.1/weights_20_N_100_gap_2/"
# r_dir = current_directory+"/weights/05_Perceptual_dm_delayed_response/large_out_weights/weights_N_100_1/"
r_dir = current_directory+"/weights/01_DM_delayed_response/orthogonal_rrn_no_bias_term/weights_20_N_100_gap_1_/"
path = r_dir + "100_final.hdf5"

task = "Simple DM"  # "Integral DM"
# task: "Simple DM", "Simple DM Long-short", "Simple DM 4 times","Simple DM 8 times", "Integral DM",            #
# "Integral DM signal keep", "Integral DM Cue"

# Define helper functions and custom layers

# Variables for storing sample periods
positive_sample_activity = None
negative_sample_activity = None

# Variables for storing sample periods
positive_sample_periods = None
negative_sample_periods = None

# Variables to capture all samples
all_samples_activity = {
    "positive": [],
    "negative": [],
    "no_signal": []
}

all_samples_periods = {
    "positive": [],
    "negative": [],
    "no_signal": []
}


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


results = {
    "positive": {"signal": [], "inter": [], "decision": []},
    "negative": {"signal": [], "inter": [], "decision": []},
    "no_signal": {"signal": [], "inter": [], "decision": []}
}


def generalized_correlation(W_out, array_red):
    """
    Calculates the generalized correlation between W_out and neural activity (array_red).

    Parameters:
    W_out (numpy.ndarray): Output weight matrix (shape: neurons x outputs, e.g., 100 x 1).
    array_red (numpy.ndarray): Neural activity (shape: neurons x time x 1 or neurons x time).

    Returns:
    float: Generalized correlation value.
    """
    # Remove unnecessary dimensions from array_red
    array_red = np.squeeze(array_red)  # Ensures shape is (neurons x time)

    # Ensure alignment of shapes
    if W_out.shape[0] != array_red.shape[0]:
        raise ValueError(f"Incompatible shapes: W_out {W_out.shape}, array_red {array_red.shape}")

    # Project neural activity onto the output weights
    projection = np.dot(W_out.T, array_red)  # Results in shape (1 x time)

    # Calculate norms
    numerator = norm(projection, ord='fro')
    denominator = norm(W_out, ord='fro') * norm(array_red, ord='fro')

    return numerator / denominator if denominator != 0 else 0


def generalized_correlation_per_output(W_out, array_red):

    # Para cada output, correlación entre sus pesos y la actividad promediada
    rho_per_output = []
    for j in range(W_out.shape[1]):   # Iterate over outputs
        w_j = W_out[:, j]             # weights from output j (Neurons x 1)
        a_j = array_red.mean(axis=1)  # Average activity per neuron (Neurons x 1)
        rho_j = np.dot(w_j, a_j) / (norm(w_j) * norm(a_j))
        rho_per_output.append(rho_j)
    rho = np.mean(rho_per_output)

    return rho


def extract_weights_and_activity(model, x_sample):
    """
    Extracts output weights and neural activity given a model and an input sample.
    """
    W_out = model.layers[-1].get_weights()[0]  # Output weights
    layer = 0  # Index of the recurrent layer
    layer_outputs = model.layers[layer](x_sample)
    tensor_np = layer_outputs.numpy()
    array_red = tensor_np.T  # Shape: (N_neurons x T_time)
    return W_out, array_red


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

    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.ylim([-2.5, 2.5])
    plt.xlim(0, len(y_sample[:, 0]))
    plt.xticks(np.arange(0, len(y_sample[:, 0])+1, 50))
    plt.title(f"Input and Output for Sample {sample_idx}")
    plt.legend(loc='upper right')
    plt.grid(True)

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


si_results = []

for sample_idx, (x_sample, y_sample) in enumerate(zip(x_train[:20], y_train[:20])):
    x_sample = x_sample[np.newaxis, :, :]  # Add batch dimension
    W_out, array_red = extract_weights_and_activity(model, x_sample)

    # Obtaining relevant signals (removing batch dimension)
    x_signal = x_sample[0, :, 0]  # Formato (time,)
    y_signal = y_sample[:, 0]    # Formato (time,)

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
    # Almacenar TODAS las muestras (no solo la primera)
    all_samples_activity[decision_type].append(array_red)
    all_samples_periods[decision_type].append((x_start, period_end, decision_start))
    # ----------------------------
    # Post-calculations
    # ----------------------------

    # Capture first sample of each tip
    if decision_type == "positive" and positive_sample_activity is None:
        positive_sample_activity = array_red  # Format (neurons, time)
        positive_sample_periods = (x_start, period_end, decision_start)
        print(f"Capturada muestra POSITIVA en sample {sample_idx}")
    elif decision_type == "negative" and negative_sample_activity is None:
        negative_sample_activity = array_red
        negative_sample_periods = (x_start, period_end, decision_start)
        print(f"Capturada muestra NEGATIVA en sample {sample_idx}")

    rho_random = generalized_correlation(W_out, random_period)
    rho_decision = generalized_correlation(W_out, decision_period)
    rho_inter = generalized_correlation(W_out, inter_period)

    # Store results by category
    results[decision_type]["signal"].append(rho_random)
    results[decision_type]["inter"].append(rho_inter)
    results[decision_type]["decision"].append(rho_decision)

    print(f"Sample {sample_idx + 1}:")
    print(f"  Random Period: {x_start}-{period_end}")
    print(f"  Decision Start: {decision_start}")
    print(f"  Correlations: Random={rho_random:.4f}, Decision={rho_decision:.4f}\n")

    plot_input_output(sample_idx, x_sample[0], y_sample, x_start, decision_start, save_dir="plots")
    # Estimate SI using all activity
    activity = np.squeeze(array_red)  # (Neurons x timr)
    si = compute_SI(activity)
    # Save results
    si_results.append({
        'sample_id': sample_idx,
        'decision_type': decision_type,
        'SI': si
    })

print("\nFinal Results for All Samples:")
for category, data in results.items():
    print(f"\nCategory: {category}")
    print(f"  Signal Period Correlations: {data['signal']}")
    print(f"  Intermediate Period Correlations: {data['inter']}")
    print(f"  Decision Period Correlations: {data['decision']}")

# ========== Function to graph results ==========


# save
def save_si_results(results, filename="si_results.txt"):
    header = "Sample_ID\tDecision_Type\tSI_Score\n"
    with open(filename, 'w') as f:
        f.write(header)
        for res in results:
            line = f"{res['sample_id']}\t{res['decision_type']}\t{res['SI']:.4f}\n"
            f.write(line)

    print(f"Results SI saved in {filename}")


def plot_decision_correlations(results, save_dir="plots"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    categories = ["positive", "negative", "no_signal"]
    labels = ["Signal Period", "Intermediate", "Decision Period"]
    colors = ["#4e79a7", "#f28e2b", "#e15759"]

    # Estimate average
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
    ax.set_title(f"Correlation W^out vs output by Decision Type and Period (Task: {task})")
    ax.set_xticks(x + width)
    ax.set_xticklabels(["Positive Decision", "Negative Decision", "Zero Input"], fontsize=10)
    # ax.axhline(y=0.2, color="green", linestyle="--", label="Input Activity Start")
    ax.set_ylim([0, 0.5])
    ax.legend(loc='upper right')
    # ax.grid(True, linestyle='--', alpha=0.6)

    # Save the figure
    plt.tight_layout()
    save_path = os.path.join(save_dir, "decision_correlation_wout_summary.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# Plot the correlation in each decision


def calculate_information_metrics(activity, label):
    """Calculate information metrics using hoi."""
    if activity is None:
        return
    print(activity.shape)
    print(activity.reshape(100, -1).shape)

    # Prepare data (time x neurons)
    data_ = activity.reshape(100, -1)  # (tiempo, neuronas)
    # data_ = (data_ - np.mean(data_, axis=0)) / np.std(data_, axis=0)
    data = data_.astype(np.float64)
    n_neurons = data.shape[1]
    comb = np.arange(n_neurons).reshape(1, -1)  # All nuerons combined
    # pepe = calculate_hoi_metrics(data, label, N_rec) # Not implemented


def plot_correlation_matrices(activity,sample_idx, periods, period_labels, save_dir="plots"):
    """

    Calculates and graph correlation matrices for 3 time windows.

    Parameters:
    activity (np.array): Neuron activity (neurons x time)
    periods (list): List of tuples with (start, end) for each period
    period_labels (list): Period names for titles
    save_dir (str): Directory to save the figure

    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Correlation matrices
    corr_matrices = []

    for (start, end) in periods:

        window_data = activity[:, start:end].squeeze()  # Eliminates unit dimensions
        print("window", window_data.shape)
        # Verificar que sea 2D (neuronas x tiempo)
        if window_data.ndim != 2:
            raise ValueError(f"Incorrect form: {window_data.shape}. Expected (neurons, time)")

        print(window_data)
        print(window_data.shape)

        # Calculate mean and std for each neuron (time axis)
        neuron_means = np.mean(window_data, axis=1, keepdims=True)  # Form (Neurons, 1)
        neuron_stds = np.std(window_data, axis=1, keepdims=True)    # Form (Neurons, 1)

        # Avoid division by zero (si std=0 usar 1e-8)
        neuron_stds[neuron_stds == 0] = 1e-8

        # Normalization: (X - μ)/σ
        window_data_normalized = (window_data - neuron_means) / neuron_stds

        # 2. Squeeze
        # --------------------------------
        window_data_normalized = window_data_normalized.squeeze()

        corr = np.corrcoef(window_data_normalized)
        print(corr.shape)
        print(corr)
        corr_matrices.append(corr)

    # Fig config
    fig = plt.figure(figsize=(15, 5))
    plt.suptitle("Neuronal Correlation Patterns Across Task Periods", y=1.05)

    # Subplots
    axes = []
    for i, (corr, label) in enumerate(zip(corr_matrices, period_labels)):
        print(corr)
        print("IN")
        # print(corr_matrices[i])

        ax = fig.add_subplot(1, 3, i+1)
        axes.append(ax)

        # Plot correlation matrix
        im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_title(label, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    # Col bar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.savefig(os.path.join(save_dir, f"correlation_matrices_{sample_idx}.png"),
                dpi=300, bbox_inches='tight')
    plt.close()

# Run calculations


"""
if positive_sample_activity is not None:
    calculate_information_metrics(positive_sample_activity, "POSITIVA")
if negative_sample_activity is not None:
    calculate_information_metrics(negative_sample_activity, "NEGATIVA")
"""

plot_decision_correlations(results)
print("Plotting correlation matrices for positive sample:")
print(f"Activity shape: {positive_sample_activity.shape}")

# Generate matrices for all samples
for category in ["positive", "negative", "no_signal"]:
    for i, (activity, periods_info) in enumerate(zip(all_samples_activity[category], all_samples_periods[category])):
        print(i)
        if activity is None:
            continue

        # Extract periods for this sample
        x_start, period_end, decision_start = periods_info
        periods = [
            (x_start, period_end),
            (period_end, decision_start),
            (decision_start, activity.shape[1])
        ]

        # Category and sample specific directory
        # save_dir = os.path.join("plots", f"{category}_correlations", f"sample_{i}")
        save_dir = os.path.join("plots", f"{category}")

        plot_correlation_matrices(
            activity, i,
            periods,
            ["Signal", "Inter", "Decision"],
            save_dir=save_dir
        )

save_si_results(si_results)

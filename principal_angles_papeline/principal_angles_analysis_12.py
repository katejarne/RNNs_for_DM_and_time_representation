"""
principal_angles_analysis.py

Computes principal angles between leading PC subspaces of trained RNN populations.
Automatically detects stimulus and decision periods SEPARATELY for positive and negative trials.
Phases: pre‑stimulus, stimulus, intermediate, decision (full trial is computed but not plotted in bar/heatmap).
Plots: bar chart with individual replica points (with jitter), heatmap, and summary table.
Additionally, for the first replica, saves phase verification figures for first 3 and last 3 trials.
Now also saves a table of explained variance (EVR) per PC component.
"""

import os
import re
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import subspace_angles
import tensorflow as tf

os.environ['TF_DISABLE_GPU'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

warnings.filterwarnings('ignore')

# ============================================================
#  CONFIGURATION
# ============================================================

task = "Simple DM"
base_dir = "../weights/01_DM_delayed_response/Same_replicas_for_PCA_angle"
# base_dir = "./weights/01_DM_delayed_response/orthogonal_rrn_with_bias_term"
# base_dir = "./weights/01_DM_delayed_response/Random_normal_rnn_no_bias_term_sigma_1.05"

# using long-short as simple dm

# base_dir = "./weights/02_DM_delayed_response_long-short/orthogonal_rrn_no_bias_term"
# base_dir = "./weights/02_DM_delayed_response_long-short/random_normal"
from utils.net_constraint_create import *
from plot_utilities.plot_neural_activity import *
from tensorflow.keras.initializers import Initializer

class AsymmetricInitializer(tf.keras.initializers.Initializer):
    def __init__(self, base_initializer, asymmetry_factor=1.0):
        self.base_initializer = base_initializer
        self.asymmetry_factor = asymmetry_factor

    def __call__(self, shape, dtype=None):
        W0 = self.base_initializer(shape, dtype=dtype)
        W_sym = (W0 + tf.transpose(W0)) / 2.0
        W_anti = (W0 - tf.transpose(W0)) / 2.0
        W = W_sym + self.asymmetry_factor * W_anti
        return W

    def get_config(self):
        return {
            'base_initializer': tf.keras.initializers.serialize(self.base_initializer),
            'asymmetry_factor': self.asymmetry_factor
        }

    @classmethod
    def from_config(cls, config):
        base_initializer = tf.keras.initializers.deserialize(config['base_initializer'])
        return cls(base_initializer, asymmetry_factor=config['asymmetry_factor'])

def custom_simple_rnn(**config):
    if 'time_major' in config:
        del config['time_major']
    return tf.keras.layers.SimpleRNN(**config)

class IdentityInitializer(Initializer):
    def __call__(self, shape, dtype=None):
        if shape[0] != shape[1]:
            raise ValueError("Identity matrix initializer requires a square matrix shape.")
        return np.identity(shape[0], dtype=dtype)

custom_objects = {'NonNegLast': NonNegLast, 'NonNegLast_input': NonNegLast_input,
                  'my_init_exi_ini': my_init_exi_ini,
                  'my_init_rec': my_init_rec,
                  'SimpleRNN': custom_simple_rnn, 'IdentityInitializer': IdentityInitializer,
                  'AsymmetricInitializer': AsymmetricInitializer,
                  }

n_components = 3
n_trials_per_cond = 50
mem_gap = 0

# Detection thresholds (adjust based on your signal amplitudes)
stimulus_threshold = 0.5      # input magnitude above this is considered stimulus
decision_threshold = 0.5      # output magnitude above this is decision onset
decision_fixed_duration = 100   # number of time steps after decision onset to include

comparison_mode = "pos_vs_neg"
tasks_to_compare = ["Simple DM", "Integral DM"]

output_dir = "../principal_angles_output"
os.makedirs(output_dir, exist_ok=True)

visualize_phases = True   # generate verification plots for first replica (first 3 and last 3 trials)

# ============================================================
#  DATASET GENERATOR
# ============================================================

def import_generator(task_name):
    if task_name in ["Simple DM", "Simple DM Long-short"]:
        from data_set_generators.generate_DM_delayed_response_sample import generate_trials
    elif task_name == "Simple DM 4 times":
        from data_set_generators.generate_DM_delayed_response_sample_mult_times_4 import generate_trials
    elif task_name == "Simple DM 8 times":
        from data_set_generators.generate_DM_delayed_response_sample_mult_times_8 import generate_trials
    elif task_name == "Simple DM 8 time encoded":
        from data_set_generators.generate_DM_delayed_response_sample_mult_times_8_intervals import generate_trials
    elif task_name == "Integral DM":
        from data_set_generators.generate_perceptual_dm import generate_trials
    elif task_name == "Integral DM signal keep":
        from data_set_generators.generate_perceptual_dm_sig_not_end import generate_trials
    elif task_name == "Integral DM Cue":
        from data_set_generators.generate_perceptual_dm_sig_not_end_cue_mod import generate_trials
    elif task_name == "Multi Ampli":
        from data_set_generators.generate_DM_delayed_response_sample_mult_amplitude_8 import generate_trials
    elif task_name == "interval compare":
        from data_set_generators.generate_interval_comparison import generate_trials
    else:
        raise ValueError(f"Unknown task: {task_name}")
    return generate_trials

def get_mem_gap(task_name):
    gaps = {
        "Simple DM": 0, "Simple DM Long-short": 0,
        "Simple DM 4 times": 0, "Simple DM 8 times": 0,
        "Simple DM 8 time encoded": 0,
        "Integral DM": 200, "Integral DM signal keep": 50,
        "Integral DM Cue": 50, "Multi Ampli": 100,
        "interval compare": 20,
    }
    return gaps.get(task_name, 0)

# ============================================================
#  MODEL LOADING
# ============================================================

def load_network(model_path):
    model = tf.keras.models.load_model(
        model_path, custom_objects=custom_objects, compile=False
    )
    model.compile(loss='mse', optimizer='Adam')
    for layer in model.layers:
        if 'rnn' in layer.name.lower() or 'simple_rnn' in layer.name.lower():
            weights = layer.get_weights()
            if len(weights) == 3:
                print(f"    RNN layer '{layer.name}': bias=True")
                layer.set_weights([np.array(weights[0]), np.array(weights[1]), np.array(weights[2])])
            elif len(weights) == 2:
                print(f"    RNN layer '{layer.name}': bias=False")
                layer.set_weights([np.array(weights[0]), np.array(weights[1])])
            break
    return model

def get_hidden_states(model, x_input):
    rnn_layer = None
    for layer in model.layers:
        if 'rnn' in layer.name.lower() or 'simple_rnn' in layer.name.lower():
            rnn_layer = layer
            break
    if rnn_layer is None:
        raise RuntimeError("No RNN layer found.")
    intermediate = tf.keras.Model(inputs=model.inputs, outputs=rnn_layer.output)
    return intermediate.predict(x_input, verbose=0)

# ============================================================
#  PRECISE PHASE DETECTION (with 1D flattening)
# ============================================================

def detect_stimulus_interval(input_signal, threshold):
    """Return (start, end) indices where |input| > threshold.
    If no stimulus detected, returns (0, 0)."""
    input_signal = np.asarray(input_signal).flatten()
    input_abs = np.abs(input_signal)
    above = input_abs > threshold
    if not np.any(above):
        return 0, 0
    start = np.argmax(above)               # first True
    end = len(above) - 1 - np.argmax(above[::-1])  # last True
    return start, end

def detect_decision_onset(output_signal, threshold, start_search):
    """Return first index >= start_search where |output| > threshold.
    If none, returns len(output_signal)-1."""
    output_signal = np.asarray(output_signal).flatten()
    output_abs = np.abs(output_signal)
    indices = np.arange(start_search, len(output_abs))
    candidates = indices[output_abs[indices] > threshold]
    if len(candidates) == 0:
        return len(output_abs) - 1
    return candidates[0]

def get_phase_windows(input_signal, output_signal,
                      stim_thresh, dec_thresh, dec_duration):
    """
    Returns dictionary with keys: 'pre-stimulus', 'stimulus', 'intermediate', 'decision', 'full trial'.
    Each value is (start, end) tuple (slice convention start:end).
    For full trial, value is None.
    """
    input_signal = np.asarray(input_signal).flatten()
    output_signal = np.asarray(output_signal).flatten()
    T = len(input_signal)

    # Stimulus
    stim_start, stim_end = detect_stimulus_interval(input_signal, stim_thresh)
    if stim_start == stim_end == 0 and np.all(np.abs(input_signal) <= stim_thresh):
        # No stimulus detected: set a minimal window (first few steps)
        stim_start, stim_end = 0, min(10, T-1)
        print(f"    Warning: No stimulus detected. Using fallback stimulus window [0, {stim_end}]")

    # Pre-stimulus: from beginning to just before stimulus start
    pre_start = 0
    pre_end = max(0, stim_start - 1)
    if pre_end < pre_start:
        pre_end = pre_start  # empty window, set to single point

    # Decision onset (search after stimulus end)
    dec_start = detect_decision_onset(output_signal, dec_thresh, stim_end)
    dec_end = min(dec_start + dec_duration, T-1)

    # Intermediate: from stimulus_end+1 to dec_start-1
    inter_start = stim_end + 1
    inter_end = dec_start - 1
    if inter_start > inter_end:
        # No intermediate period: set a small window (just one step after stimulus)
        inter_start = stim_end
        inter_end = stim_end
        if inter_start >= T:
            inter_start, inter_end = T-1, T-1

    # Ensure all windows are within bounds and have at least 1 step
    stim_end = min(stim_end, T-1)
    dec_end = min(dec_end, T-1)
    inter_end = min(inter_end, T-1)
    pre_end = min(pre_end, T-1)
    if inter_start >= T:
        inter_start = T-1
        inter_end = T-1
    if pre_start >= T:
        pre_start, pre_end = T-1, T-1

    return {
        'pre-stimulus': (pre_start, pre_end),
        'stimulus': (stim_start, stim_end),
        'intermediate': (inter_start, inter_end),
        'decision': (dec_start, dec_end),
        'full trial': None
    }

# ============================================================
#  ANALYSIS FUNCTIONS
# ============================================================


def pca_subspace(activity_matrix, n_components):
    A = activity_matrix - activity_matrix.mean(axis=0, keepdims=True)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    total_var = np.sum(s**2)
    evr = (s[:n_components]**2) / total_var if total_var > 0 else np.zeros(n_components)
    V = Vt[:n_components].T
    return V, evr


def compute_principal_angles(V1, V2):
    angles_rad = subspace_angles(V1, V2)
    return np.degrees(angles_rad)


def slice_phase(hidden, window):
    if window is None:
        return hidden
    t0, t1 = window
    T = hidden.shape[1]
    t0 = max(0, min(t0, T))
    t1 = max(0, min(t1, T))
    if t0 >= t1:
        t1 = min(t0 + 1, T)
    return hidden[:, t0:t1, :]


def separate_conditions(x_trials, y_trials):
    final_target = y_trials[:, -1, 0]
    idx_pos = np.where(final_target > 0)[0]
    idx_neg = np.where(final_target < 0)[0]
    return idx_pos, idx_neg

# ============================================================
#  VISUALIZATION OF DETECTED PHASES (for a specific trial index)
# ============================================================


def visualize_trial_phases(task_name, model, x_pos, x_neg, y_pos, y_neg,
                           phase_params, replica_idx, trial_idx_pos, trial_idx_neg,
                           output_dir):
    """
    Plot input and output for a specific positive trial and negative trial,
    with automatically detected phase boundaries for each.
    trial_idx_pos: index of positive trial to plot (0-based)
    trial_idx_neg: index of negative trial to plot (0-based)
    """
    # Extract the specific trials
    x_ex_pos = x_pos[trial_idx_pos:trial_idx_pos+1]
    x_ex_neg = x_neg[trial_idx_neg:trial_idx_neg+1]
    y_pred_pos = model.predict(x_ex_pos, verbose=0)[0]
    y_pred_neg = model.predict(x_ex_neg, verbose=0)[0]
    input_pos = x_ex_pos[0, :, 0] if x_ex_pos.shape[-1] >= 1 else x_ex_pos[0, :, 0]
    input_pos = np.asarray(input_pos).flatten()
    y_pred_pos = np.asarray(y_pred_pos[:, 0]).flatten()
    input_neg = x_ex_neg[0, :, 0] if x_ex_neg.shape[-1] >= 1 else x_ex_neg[0, :, 0]
    input_neg = np.asarray(input_neg).flatten()
    y_pred_neg = np.asarray(y_pred_neg[:, 0]).flatten()

    # Detect phase windows for this specific positive and negative trial
    windows_pos = get_phase_windows(input_pos, y_pred_pos,
                                    phase_params['stim_thresh'],
                                    phase_params['dec_thresh'],
                                    phase_params['dec_duration'])
    windows_neg = get_phase_windows(input_neg, y_pred_neg,
                                    phase_params['stim_thresh'],
                                    phase_params['dec_thresh'],
                                    phase_params['dec_duration'])

    T = len(input_pos)
    time = np.arange(T)

    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    fig.suptitle(f"Task: {task_name} | Replica {replica_idx} | Trial pos {trial_idx_pos}, neg {trial_idx_neg}\nAutomatically detected phases", fontsize=12)

    def add_markers(ax, windows, T_max, color_map):
        for name, win in windows.items():
            if win is None:
                continue
            t0, t1 = win
            if t0 < T_max:
                ax.axvspan(t0, min(t1, T_max), alpha=0.1, color=color_map.get(name, 'gray'),
                           label=name if name not in ax.get_legend_handles_labels()[1] else "")
                ax.axvline(t0, color=color_map.get(name, 'gray'), linestyle='--', alpha=0.5)
                ax.axvline(t1, color=color_map.get(name, 'gray'), linestyle='--', alpha=0.5)

    colors = {'pre-stimulus': 'purple', 'stimulus': 'blue', 'intermediate': 'orange', 'decision': 'red'}

    # Positive trial input
    ax1 = axes[0, 0]
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.plot(time, input_pos, 'g-', label='Input')
    ax1.set_ylabel('Input amplitude', fontsize=12)
    ax1.set_title('Positive trial - Input')
    add_markers(ax1, windows_pos, T, colors)
    ax1.legend(fontsize=9)
    # Positive trial output
    ax2 = axes[0, 1]
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.plot(time, y_pred_pos, 'r-', label='Network output')
    ax2.set_title('Positive trial - Output')
    add_markers(ax2, windows_pos, T, colors)
    ax2.legend(fontsize=8)

    # Negative trial input
    ax3 = axes[1, 0]
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.plot(time, input_neg, 'g-', label='Input')
    ax3.set_xlabel('Time step')
    ax3.set_ylabel('Input amplitude', fontsize=12)
    ax3.set_title('Negative trial - Input')
    add_markers(ax3, windows_neg, T, colors)
    ax3.legend(fontsize=9)
    # Negative trial output
    ax4 = axes[1, 1]
    ax4.spines['right'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    ax4.plot(time, y_pred_neg, 'r-', label='Network output')
    ax4.set_xlabel('Time step', fontsize=12)
    ax4.set_title('Negative trial - Output')
    add_markers(ax4, windows_neg, T, colors)
    ax4.legend(fontsize=9)

    plt.tight_layout()
    # Save both PNG and SVG
    out_path_png = os.path.join(output_dir, f"phase_verification_replica_{replica_idx}_pos{trial_idx_pos}_neg{trial_idx_neg}.png")
    out_path_svg = os.path.join(output_dir, f"phase_verification_replica_{replica_idx}_pos{trial_idx_pos}_neg{trial_idx_neg}.svg")
    plt.savefig(out_path_png, dpi=150, bbox_inches='tight')
    plt.savefig(out_path_svg, dpi=150, bbox_inches='tight')
    print(f"  Phase verification figures saved: {out_path_png} and {out_path_svg}")
    plt.close(fig)

# ============================================================
#  FIND REPLICA DIRECTORIES
# ============================================================

def find_replica_dirs(base_dir):
    if not os.path.isdir(base_dir):
        return []
    replica_dirs = []
    for item in os.listdir(base_dir):
        full_path = os.path.join(base_dir, item)
        if os.path.isdir(full_path) and item.startswith("replica_"):
            match = re.search(r'replica_(\d+)', item)
            if match:
                num = int(match.group(1))
                replica_dirs.append((num, full_path))
    replica_dirs.sort(key=lambda x: x[0])
    return replica_dirs

# ============================================================
#  MAIN ANALYSIS (separate phase windows per condition)
# ============================================================


def run_analysis(task_name, base_dir, n_components, n_trials_per_cond,
                 stim_thresh, dec_thresh, dec_duration,
                 output_dir, visualize=False):

    replica_list = find_replica_dirs(base_dir)
    if not replica_list:
        raise RuntimeError(f"No replica directories found in {base_dir}")
    n_replicas = len(replica_list)
    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"Found {n_replicas} replicas: {[r[0] for r in replica_list]}")
    print(f"Components: {n_components}")
    print(f"Detection: stim_thresh={stim_thresh}, dec_thresh={dec_thresh}, dec_duration={dec_duration}")
    print(f"{'='*60}")

    generate_trials = import_generator(task_name)
    mem_gap_val = get_mem_gap(task_name)

    total_trials = n_trials_per_cond * 4
    x_all, y_all, _ = generate_trials(total_trials, mem_gap_val)
    idx_pos, idx_neg = separate_conditions(x_all, y_all)
    n_pos = min(len(idx_pos), n_trials_per_cond)
    n_neg = min(len(idx_neg), n_trials_per_cond)
    if n_pos == 0 or n_neg == 0:
        raise RuntimeError("Could not find both positive and negative trials.")
    idx_pos = idx_pos[:n_pos]
    idx_neg = idx_neg[:n_neg]
    x_pos = x_all[idx_pos]
    x_neg = x_all[idx_neg]
    y_pos = y_all[idx_pos]
    y_neg = y_all[idx_neg]

    # Store phase windows separately for each condition (used for analysis)
    phase_windows_pos = None
    phase_windows_neg = None
    # Define phases in order: pre-stimulus, stimulus, intermediate, decision, full trial
    phase_list = ['pre-stimulus', 'stimulus', 'intermediate', 'decision', 'full trial']
    results = {phase: [] for phase in phase_list}
    evr_pos_all = []
    evr_neg_all = []
    visualization_done = False

    for rep_num, rep_path in replica_list:
        candidates = [
            os.path.join(rep_path, "100_final.hdf5"),
            os.path.join(rep_path, "100_final.keras"),
            os.path.join(rep_path, task_name.replace(" ", "_"), "100_final.hdf5"),
        ]
        model_path = next((c for c in candidates if os.path.exists(c)), None)
        if model_path is None:
            print(f"  [WARNING] Replica {rep_num}: no weight file. Skipping.")
            for ph in results:
                results[ph].append(np.full(n_components, np.nan))
            evr_pos_all.append(np.full(n_components, np.nan))
            evr_neg_all.append(np.full(n_components, np.nan))
            continue

        print(f"  Processing replica {rep_num} ...")
        model = load_network(model_path)

        # Determine phase windows for positive and negative conditions (first replica only)
        if phase_windows_pos is None or phase_windows_neg is None:
            # Positive trial (first)
            x_first_pos = x_pos[0:1]
            y_pred_pos_first = model.predict(x_first_pos, verbose=0)[0]
            y_pred_pos_first = np.asarray(y_pred_pos_first[:, 0]).flatten()
            input_first_pos = x_first_pos[0, :, 0] if x_first_pos.shape[-1] >= 1 else x_first_pos[0, :, 0]
            input_first_pos = np.asarray(input_first_pos).flatten()
            phase_windows_pos = get_phase_windows(input_first_pos, y_pred_pos_first,
                                                  stim_thresh, dec_thresh, dec_duration)
            # Negative trial (first)
            x_first_neg = x_neg[0:1]
            y_pred_neg_first = model.predict(x_first_neg, verbose=0)[0]
            y_pred_neg_first = np.asarray(y_pred_neg_first[:, 0]).flatten()
            input_first_neg = x_first_neg[0, :, 0] if x_first_neg.shape[-1] >= 1 else x_first_neg[0, :, 0]
            input_first_neg = np.asarray(input_first_neg).flatten()
            phase_windows_neg = get_phase_windows(input_first_neg, y_pred_neg_first,
                                                  stim_thresh, dec_thresh, dec_duration)
            print(f"    Positive windows (used for analysis): pre-stimulus={phase_windows_pos['pre-stimulus']}, "
                  f"stimulus={phase_windows_pos['stimulus']}, "
                  f"intermediate={phase_windows_pos['intermediate']}, decision={phase_windows_pos['decision']}")
            print(f"    Negative windows (used for analysis): pre-stimulus={phase_windows_neg['pre-stimulus']}, "
                  f"stimulus={phase_windows_neg['stimulus']}, "
                  f"intermediate={phase_windows_neg['intermediate']}, decision={phase_windows_neg['decision']}")

            # --- Visualization for first replica: first 3 and last 3 trials ---
            if visualize and not visualization_done and rep_num == replica_list[0][0]:
                phase_params = {
                    'stim_thresh': stim_thresh,
                    'dec_thresh': dec_thresh,
                    'dec_duration': dec_duration
                }
                # Indices: first 3 (0,1,2) and last 3 (n-3, n-2, n-1)
                n_pos_trials = x_pos.shape[0]
                n_neg_trials = x_neg.shape[0]
                trial_indices_pos_first = list(range(min(3, n_pos_trials)))
                trial_indices_pos_last = list(range(max(0, n_pos_trials-3), n_pos_trials))
                trial_indices_neg_first = list(range(min(3, n_neg_trials)))
                trial_indices_neg_last = list(range(max(0, n_neg_trials-3), n_neg_trials))
                # Combine unique indices (avoid duplicates if n<6)
                pos_indices = sorted(set(trial_indices_pos_first + trial_indices_pos_last))
                neg_indices = sorted(set(trial_indices_neg_first + trial_indices_neg_last))
                for p_idx in pos_indices:
                    # Find corresponding negative index (use same order, but ensure exists)
                    # We'll pair the i-th positive with the i-th negative if both exist, otherwise use first negative
                    # For simplicity, use the same index for both (if within bounds)
                    n_idx = p_idx if p_idx < n_neg_trials else 0
                    visualize_trial_phases(task_name, model, x_pos, x_neg, y_pos, y_neg,
                                           phase_params, rep_num, p_idx, n_idx, output_dir)
                # Also ensure last negative trials are covered if not already
                for n_idx in neg_indices:
                    p_idx = n_idx if n_idx < n_pos_trials else 0
                    # Avoid double plotting if already plotted
                    if n_idx not in pos_indices or p_idx != n_idx:
                        visualize_trial_phases(task_name, model, x_pos, x_neg, y_pos, y_neg,
                                               phase_params, rep_num, p_idx, n_idx, output_dir)
                visualization_done = True

        # Extract hidden states
        h_pos = get_hidden_states(model, x_pos)
        h_neg = get_hidden_states(model, x_neg)
        h_pos_mean = h_pos.mean(axis=0)
        h_neg_mean = h_neg.mean(axis=0)

        # Compute PCA angles for each phase using the corresponding windows
        for phase_name in phase_list:
            window_pos = phase_windows_pos[phase_name]
            window_neg = phase_windows_neg[phase_name]
            h_pos_phase = slice_phase(h_pos_mean[np.newaxis], window_pos)[0]
            h_neg_phase = slice_phase(h_neg_mean[np.newaxis], window_neg)[0]
            if h_pos_phase.shape[0] < n_components or h_neg_phase.shape[0] < n_components:
                print(f"    [WARNING] Phase '{phase_name}'"
                      f" has {h_pos_phase.shape[0]} steps (pos) and {h_neg_phase.shape[0]}"
                      f" steps (neg); need at least {n_components}. Skipping.")
                results[phase_name].append(np.full(n_components, np.nan))
                continue
            V_pos, evr_pos = pca_subspace(h_pos_phase, n_components)
            V_neg, evr_neg = pca_subspace(h_neg_phase, n_components)
            angles = compute_principal_angles(V_pos, V_neg)
            results[phase_name].append(angles)
            if phase_name == "full trial":
                evr_pos_all.append(evr_pos)
                evr_neg_all.append(evr_neg)
        print(f"    Full-trial angles: {np.round(results['full trial'][-1], 2)} deg")

    # Convert to arrays
    for ph in results:
        results[ph] = np.array(results[ph])
    evr_pos_all = np.array(evr_pos_all)
    evr_neg_all = np.array(evr_neg_all)
    return results, evr_pos_all, evr_neg_all

# ============================================================
#  TABLE OF EXPLAINED VARIANCE (NEW)
# ============================================================


def save_evr_table(evr_pos, evr_neg, task_name, output_dir):
    """
    Print and save a table of explained variance ratio (EVR) per PC component,
    averaged across replicas, for positive and negative trials (full trial).
    """
    if evr_pos.size == 0 or np.all(np.isnan(evr_pos)):
        print("No valid explained variance data to save.")
        return
    n_comp = evr_pos.shape[1]
    print(f"\n{'='*60}\nExplained Variance Ratio (full trial): {task_name}\n{'='*60}")
    header = f"{'Condition':<22}" + "".join([f"  PC{i+1:>12}" for i in range(n_comp)])
    print(header)
    print("-" * len(header))
    lines = []
    for cond_name, evr in [("Positive", evr_pos), ("Negative", evr_neg)]:
        if evr.size == 0 or np.all(np.isnan(evr)):
            continue
        row = f"{cond_name:<22}"
        for j in range(n_comp):
            col = evr[:, j]
            col_valid = col[~np.isnan(col)]
            if len(col_valid) == 0:
                row += "       NaN   "
            else:
                row += f"  {np.mean(col_valid)*100:5.1f}±{np.std(col_valid)*100:4.1f}%"
        print(row)
        lines.append(row)
    print()
    txt_path = os.path.join(output_dir, f"evr_table_{task_name.replace(' ', '_')}.txt")
    with open(txt_path, 'w') as f:
        f.write(f"Explained Variance Ratio (full trial): {task_name}\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for line in lines:
            f.write(line + "\n")
        f.write("\n")
    print(f"EVR table saved to: {txt_path}")

# ============================================================
#  PLOTTING FUNCTIONS (with jitter, exclude full trial from bar/heatmap)
# ============================================================


def plot_principal_angles(results, evr_pos, evr_neg, task_name, n_components, output_dir):
    # Exclude 'full trial' from the bar plot
    phase_names = [p for p in results.keys() if p != 'full trial' and results[p].size > 0 and not np.all(np.isnan(results[p]))]
    if not phase_names:
        print("No valid data to plot (excluding full trial).")
        return
    n_phases = len(phase_names)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_phases))
    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    x = np.arange(1, n_components+1)
    width = 0.8 / n_phases

    # Store bar positions and data for scatter with jitter
    bar_positions = {}  # phase -> list of (x_center, y_vals_per_replica) for each PC

    for i, (phase, color) in enumerate(zip(phase_names, colors)):
        data = results[phase]  # (n_replicas, n_components)
        valid = ~np.isnan(data).any(axis=1)
        data_valid = data[valid]
        if data_valid.shape[0] == 0:
            continue
        mean_a = np.nanmean(data_valid, axis=0)
        std_a = np.nanstd(data_valid, axis=0)
        offsets = (i - (n_phases-1)/2) * width
        # Bar plot
        bars = ax1.bar(x + offsets, mean_a, width=width*0.9, yerr=std_a,
                       color=color, alpha=0.85, label=phase,
                       error_kw=dict(elinewidth=1, capsize=3))
        # Store centers
        bar_centers = [bar.get_x() + bar.get_width()/2 for bar in bars]
        bar_positions[phase] = (bar_centers, data_valid)

    # Add individual replica points with jitter (random horizontal offset)
    for phase, (centers, data_valid) in bar_positions.items():
        for pc_idx, center in enumerate(centers):
            y_vals = data_valid[:, pc_idx]
            # Generate small jitter: uniform between -0.1*width and +0.1*width
            jitter_width = width * 0.1  # jitter range
            jitter = np.random.uniform(-jitter_width, jitter_width, size=len(y_vals))
            x_points = center + jitter
            ax1.scatter(x_points, y_vals, s=20, color='gray',
                        alpha=0.6, zorder=3, edgecolors='none')

    ax1.axhline(90, color='gray', linestyle='--', label='90° (orthogonal)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"PC{i}" for i in x])
    ax1.set_ylabel("Principal angle (deg)",fontsize=12)
    ax1.set_xlabel("Principal component",fontsize=12)
    ax1.set_ylim(0, 95)
    ax1.set_title(f"Principal angles between +/- subspaces\n{task_name}  (mean ± std, n={results[phase_names[0]].shape[0]} replicas)")
    ax1.legend(fontsize=9, loc="upper right", framealpha=0.7)
    ax1.spines[['top', 'right']].set_visible(False)

    # Right panel: explained variance (full trial only)
    ax2 = fig.add_subplot(gs[1])
    x_evr = np.arange(1, n_components+1)
    if evr_pos.shape[0] > 0 and not np.isnan(evr_pos).all():
        evr_pos_mean = np.nanmean(evr_pos, axis=0)
        evr_pos_std = np.nanstd(evr_pos, axis=0)
        evr_neg_mean = np.nanmean(evr_neg, axis=0)
        evr_neg_std = np.nanstd(evr_neg, axis=0)

        # Bars for positive and negative
        bars_pos = ax2.bar(x_evr - 0.2, evr_pos_mean*100, width=0.35, yerr=evr_pos_std*100,
                           color='steelblue', label='Positive trials', alpha=0.85,
                           error_kw=dict(elinewidth=1, capsize=3))
        bars_neg = ax2.bar(x_evr + 0.2, evr_neg_mean*100, width=0.35, yerr=evr_neg_std*100,
                           color='coral', label='Negative trials', alpha=0.85,
                           error_kw=dict(elinewidth=1, capsize=3))

        # Add individual points for positive trials
        for pc_idx, bar in enumerate(bars_pos):
            center = bar.get_x() + bar.get_width()/2
            y_vals = evr_pos[:, pc_idx] * 100  # convert to percentage
            # Remove NaNs
            y_vals = y_vals[~np.isnan(y_vals)]
            if len(y_vals) == 0:
                continue
            jitter_width = 0.15  # fixed small jitter for variance plot
            jitter = np.random.uniform(-jitter_width, jitter_width, size=len(y_vals))
            ax2.scatter(center + jitter, y_vals, s=15, color='gray', alpha=0.5, zorder=3)

        # Add individual points for negative trials
        for pc_idx, bar in enumerate(bars_neg):
            center = bar.get_x() + bar.get_width()/2
            y_vals = evr_neg[:, pc_idx] * 100
            y_vals = y_vals[~np.isnan(y_vals)]
            if len(y_vals) == 0:
                continue
            jitter_width = 0.15
            jitter = np.random.uniform(-jitter_width, jitter_width, size=len(y_vals))
            ax2.scatter(center + jitter, y_vals, s=15, color='gray', alpha=0.5, zorder=3)

    ax2.set_xticks(x_evr)
    ax2.set_xticklabels([f"PC{i}" for i in x_evr])
    ax2.set_ylabel("Explained variance (%)",fontsize=12)
    ax2.set_xlabel("Principal component",fontsize=12)
    ax2.axhline(90, color='gray', linestyle='--', label='90 %')
    ax2.set_title("Variance explained (full trial)",fontsize=12)
    ax2.legend(fontsize=9)
    ax2.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    # Save both PNG and SVG
    figpath_png = os.path.join(output_dir, f"principal_angles_{task_name.replace(' ', '_')}.png")
    figpath_svg = os.path.join(output_dir, f"principal_angles_{task_name.replace(' ', '_')}.svg")
    plt.savefig(figpath_png, dpi=200, bbox_inches='tight')
    plt.savefig(figpath_svg, dpi=200, bbox_inches='tight')
    print(f"\nFigures saved: {figpath_png} and {figpath_svg}")
    # plt.show()
    return fig


def plot_angle_heatmap(results, task_name, output_dir):
    # Exclude 'full trial' from heatmap
    phase_names = [p for p in results.keys() if p != 'full trial' and results[p].size > 0 and not np.all(np.isnan(results[p]))]
    if not phase_names:
        return
    n_phases = len(phase_names)
    n_comp = results[phase_names[0]].shape[1]
    mean_mat = np.zeros((n_phases, n_comp))
    for i, ph in enumerate(phase_names):
        mean_mat[i] = np.nanmean(results[ph], axis=0)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    im = ax.imshow(mean_mat, vmin=0, vmax=90, cmap='RdYlGn_r', aspect='auto')
    plt.colorbar(im, ax=ax, label='Mean principal angle (deg)')
    ax.set_xticks(np.arange(n_comp))
    ax.set_xticklabels([f"PC{i+1}" for i in range(n_comp)])
    ax.set_yticks(np.arange(n_phases))
    ax.set_yticklabels(phase_names)
    ax.set_title(f"Principal angles: +/- subspaces\n{task_name}")
    for i in range(n_phases):
        for j in range(n_comp):
            ax.text(j, i, f"{mean_mat[i, j]:.1f}", ha='center', va='center', fontsize=9,
                    color='white' if mean_mat[i, j]>55 else 'black')
    plt.tight_layout()
    # Save both PNG and SVG
    figpath_png = os.path.join(output_dir, f"angle_heatmap_{task_name.replace(' ', '_')}.png")
    figpath_svg = os.path.join(output_dir, f"angle_heatmap_{task_name.replace(' ', '_')}.svg")
    plt.savefig(figpath_png, dpi=200, bbox_inches='tight')
    plt.savefig(figpath_svg, dpi=200, bbox_inches='tight')
    print(f"Heatmap saved: {figpath_png} and {figpath_svg}")
    # plt.show()


def print_summary_table(results, task_name, output_dir):
    # Include all phases (including full trial) in the table
    phase_names = [p for p in results.keys() if results[p].size > 0 and not np.all(np.isnan(results[p]))]
    if not phase_names:
        return
    n_comp = results[phase_names[0]].shape[1]
    # Print to console
    print(f"\n{'='*60}\nSummary - Principal angles (degrees): {task_name}\n{'='*60}")
    header = f"{'Phase':<22}" + "".join([f"  PC{i+1:>6}" for i in range(n_comp)])
    print(header)
    print("-" * len(header))
    lines = []
    for ph in phase_names:
        data = results[ph]
        row = f"{ph:<22}"
        for j in range(n_comp):
            col = data[:, j]
            col_valid = col[~np.isnan(col)]
            if len(col_valid) == 0:
                row += "       NaN"
            else:
                row += f"  {np.mean(col_valid):5.1f}+/-{np.std(col_valid):4.1f}"
        print(row)
        lines.append(row)
    print()

    # Save to text file
    txt_path = os.path.join(output_dir, f"summary_table_{task_name.replace(' ', '_')}.txt")
    with open(txt_path, 'w') as f:
        f.write(f"Summary - Principal angles (degrees): {task_name}\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for line in lines:
            f.write(line + "\n")
        f.write("\n")
    print(f"Summary table saved to: {txt_path}")

# ============================================================
#  CROSS-TASK (simplified)
# ============================================================


def cross_task_angle_comparison(tasks_to_compare, base_dir, n_components,
                                 n_trials_per_cond, stim_thresh, dec_thresh, dec_duration, output_dir):
    print("Cross-task comparison not fully implemented with automatic phase detection.")
    print("Please use pos_vs_neg mode for automatic detection.")

# ============================================================
#  ENTRY POINT
# ============================================================


if __name__ == "__main__":
    if comparison_mode == "pos_vs_neg":
        results, evr_pos, evr_neg = run_analysis(
            task_name=task,
            base_dir=base_dir,
            n_components=n_components,
            n_trials_per_cond=n_trials_per_cond,
            stim_thresh=stimulus_threshold,
            dec_thresh=decision_threshold,
            dec_duration=decision_fixed_duration,
            output_dir=output_dir,
            visualize=visualize_phases,
        )
        print_summary_table(results, task, output_dir)   # now also saves to .txt
        save_evr_table(evr_pos, evr_neg, task, output_dir)  # NEW: explained variance table
        plot_principal_angles(results, evr_pos, evr_neg, task, n_components, output_dir)
        plot_angle_heatmap(results, task, output_dir)
    elif comparison_mode == "task_vs_task":
        cross_task_angle_comparison(
            tasks_to_compare=tasks_to_compare,
            base_dir=base_dir,
            n_components=n_components,
            n_trials_per_cond=n_trials_per_cond,
            stim_thresh=stimulus_threshold,
            dec_thresh=decision_threshold,
            dec_duration=decision_fixed_duration,
            output_dir=output_dir,
        )
    else:
        raise ValueError(f"Unknown comparison_mode: {comparison_mode}")

"""
principal_angles_multi_interval.py - Multi‑interval amplitude task
Compares reference condition (height=1, sign=+) against 15 other conditions.
Generates bar plots and heatmaps for PC1, PC2, PC3.
Now includes separate variance explained plots for each PC (with 90% threshold line and individual points).
Aesthetic improvements: green input traces, transparent phase regions, PNG+SVG output.
Also saves an explained variance table (mean ± std) for all conditions.
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
task = "Simple DM 8 times"
#base_dir = "./weights/04_DM_delayed_response_8_times/Same_replicas_for_PCA_angle"
base_dir = "../weights/04_DM_delayed_response_8_times/orthogonal_rrn_no_bias_term"

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
        return {'base_initializer': tf.keras.initializers.serialize(self.base_initializer),
                'asymmetry_factor': self.asymmetry_factor}
    @classmethod
    def from_config(cls, config):
        base_initializer = tf.keras.initializers.deserialize(config['base_initializer'])
        return cls(base_initializer, asymmetry_factor=config['asymmetry_factor'])

def custom_simple_rnn(**config):
    if 'time_major' in config: del config['time_major']
    return tf.keras.layers.SimpleRNN(**config)

class IdentityInitializer(Initializer):
    def __call__(self, shape, dtype=None):
        if shape[0] != shape[1]: raise ValueError("Identity matrix initializer requires a square matrix shape.")
        return np.identity(shape[0], dtype=dtype)

custom_objects = {'NonNegLast': NonNegLast, 'NonNegLast_input': NonNegLast_input,
                  'my_init_exi_ini': my_init_exi_ini, 'my_init_rec': my_init_rec,
                  'SimpleRNN': custom_simple_rnn, 'IdentityInitializer': IdentityInitializer,
                  'AsymmetricInitializer': AsymmetricInitializer}

n_components = 3
n_trials_per_cond = 50
mem_gap = 0
stimulus_threshold = 0.5
decision_threshold = 0.5
decision_fixed_duration = 100
output_dir = "../principal_angles_output_multi"
os.makedirs(output_dir, exist_ok=True)
visualize_phases = True

amplitudes = list(range(1,9))
ref_amplitude = 1
ref_sign = 1

target_conditions = []
for amp in amplitudes:
    if amp == ref_amplitude: continue
    target_conditions.append((amp, 1))
for amp in amplitudes:
    target_conditions.append((amp, -1))

n_comparisons = len(target_conditions)  # = 15
cond_labels = []
for amp, sgn in target_conditions:
    sign_str = '+' if sgn == 1 else '-'
    cond_labels.append(f"H{amp}{sign_str}")

# ============================================================
#  DATASET GENERATOR & HELPERS (same as before)
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
    gaps = {"Simple DM":0, "Simple DM Long-short":0, "Simple DM 4 times":0, "Simple DM 8 times":0,
            "Simple DM 8 time encoded":0, "Integral DM":200, "Integral DM signal keep":50,
            "Integral DM Cue":50, "Multi Ampli":100, "interval compare":20}
    return gaps.get(task_name, 0)

def load_network(model_path):
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
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

def detect_stimulus_interval(input_signal, threshold):
    input_signal = np.asarray(input_signal).flatten()
    input_abs = np.abs(input_signal)
    above = input_abs > threshold
    if not np.any(above): return 0, 0
    start = np.argmax(above)
    end = len(above) - 1 - np.argmax(above[::-1])
    return start, end

def detect_decision_onset(output_signal, threshold, start_search):
    output_signal = np.asarray(output_signal).flatten()
    output_abs = np.abs(output_signal)
    indices = np.arange(start_search, len(output_abs))
    candidates = indices[output_abs[indices] > threshold]
    if len(candidates) == 0: return len(output_abs) - 1
    return candidates[0]

def get_phase_windows(input_signal, output_signal, stim_thresh, dec_thresh, dec_duration):
    input_signal = np.asarray(input_signal).flatten()
    output_signal = np.asarray(output_signal).flatten()
    T = len(input_signal)
    stim_start, stim_end = detect_stimulus_interval(input_signal, stim_thresh)
    if stim_start == stim_end == 0 and np.all(np.abs(input_signal) <= stim_thresh):
        stim_start, stim_end = 0, min(10, T-1)
        print(f"    Warning: No stimulus detected. Using fallback [0,{stim_end}]")
    pre_start = 0
    pre_end = max(0, stim_start - 1)
    if pre_end < pre_start: pre_end = pre_start
    dec_start = detect_decision_onset(output_signal, dec_thresh, stim_end)
    dec_end = min(dec_start + dec_duration, T-1)
    inter_start = stim_end + 1
    inter_end = dec_start - 1
    if inter_start > inter_end:
        inter_start = stim_end
        inter_end = stim_end
        if inter_start >= T: inter_start, inter_end = T-1, T-1
    stim_end = min(stim_end, T-1)
    dec_end = min(dec_end, T-1)
    inter_end = min(inter_end, T-1)
    pre_end = min(pre_end, T-1)
    if inter_start >= T: inter_start, inter_end = T-1, T-1
    if pre_start >= T: pre_start, pre_end = T-1, T-1
    return {'pre-stimulus': (pre_start, pre_end), 'stimulus': (stim_start, stim_end),
            'intermediate': (inter_start, inter_end), 'decision': (dec_start, dec_end),
            'full trial': None}

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
    if window is None: return hidden
    t0, t1 = window
    T = hidden.shape[1]
    t0 = max(0, min(t0, T))
    t1 = max(0, min(t1, T))
    if t0 >= t1: t1 = min(t0 + 1, T)
    return hidden[:, t0:t1, :]

def separate_by_condition(x_all, y_all):
    cond_indices = {(amp, sgn): [] for amp in amplitudes for sgn in [-1, 1]}
    for idx in range(x_all.shape[0]):
        input_signal = x_all[idx, :, 0]
        stim_start, stim_end = detect_stimulus_interval(input_signal, stimulus_threshold)
        if stim_start == stim_end == 0: continue
        window = slice(max(0, stim_start-5), min(x_all.shape[1], stim_end+5))
        amp_est = np.max(np.abs(input_signal[window]))
        amp = int(round(amp_est))
        if amp < 1: amp = 1
        if amp > 8: amp = 8
        final_out = y_all[idx, -1, 0]
        sign = 1 if final_out > 0 else -1
        cond_indices[(amp, sign)].append(idx)
    return cond_indices

def visualize_trial_phases(task_name, model, x_trial, y_trial, cond_label,
                           phase_params, replica_idx, trial_idx, output_dir):
    x_ex = x_trial[np.newaxis, ...]
    y_pred = model.predict(x_ex, verbose=0)[0]
    input_sig = x_trial[:, 0].flatten()
    output_sig = y_pred[:, 0].flatten()
    T = len(input_sig)
    time = np.arange(T)
    windows = get_phase_windows(input_sig, output_sig,
                                phase_params['stim_thresh'],
                                phase_params['dec_thresh'],
                                phase_params['dec_duration'])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    fig.suptitle(f"{task_name} | Replica {replica_idx} | {cond_label} trial {trial_idx}", fontsize=10)
    colors = {'pre-stimulus': 'purple', 'stimulus': 'green', 'intermediate': 'orange', 'decision': 'red'}
    def add_markers(ax, windows, T_max):
        for name, win in windows.items():
            if win is None: continue
            t0, t1 = win
            if t0 < T_max:
                ax.axvspan(t0, min(t1, T_max), alpha=0.1, color=colors.get(name, 'gray'),
                           label=name if name not in ax.get_legend_handles_labels()[1] else "")
                ax.axvline(t0, color=colors.get(name, 'gray'), linestyle='--', alpha=0.5)
                ax.axvline(t1, color=colors.get(name, 'gray'), linestyle='--', alpha=0.5)
    ax1 = axes[0]
    ax1.plot(time, input_sig, 'g-', label='Input')
    ax1.set_ylabel('Input amplitude')
    ax1.set_title('Input')
    add_markers(ax1, windows, T)
    ax1.legend(fontsize=6)
    ax2 = axes[1]
    ax2.plot(time, output_sig, 'r-', label='Network output')
    ax2.set_title('Output')
    add_markers(ax2, windows, T)
    ax2.legend(fontsize=6)
    plt.tight_layout()
    out_png = os.path.join(output_dir, f"phase_verification_repl{replica_idx}_{cond_label}_trial{trial_idx}.png")
    out_svg = os.path.join(output_dir, f"phase_verification_repl{replica_idx}_{cond_label}_trial{trial_idx}.svg")
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.savefig(out_svg, dpi=150, bbox_inches='tight')
    print(f"  Phase verification figures saved: {out_png} and {out_svg}")
    plt.close(fig)

def find_replica_dirs(base_dir):
    if not os.path.isdir(base_dir): return []
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
#  MAIN ANALYSIS
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
    print(f"Reference condition: H{ref_amplitude}{'+' if ref_sign==1 else '-'}")
    print(f"Target conditions: {cond_labels}")
    print(f"{'='*60}")

    generate_trials = import_generator(task_name)
    mem_gap_val = get_mem_gap(task_name)

    total_trials_needed = n_trials_per_cond * len(amplitudes) * 2 * 5
    x_all, y_all, _ = generate_trials(total_trials_needed, mem_gap_val)
    cond_indices = separate_by_condition(x_all, y_all)

    cond_data = {}
    for (amp, sgn), idx_list in cond_indices.items():
        if len(idx_list) < n_trials_per_cond:
            print(f"  Warning: condition H{amp}{'+' if sgn==1 else '-'} has only {len(idx_list)} trials (<{n_trials_per_cond})")
            selected = idx_list[:min(len(idx_list), n_trials_per_cond)]
        else:
            selected = idx_list[:n_trials_per_cond]
        cond_data[(amp, sgn)] = (x_all[selected], y_all[selected])

    ref_key = (ref_amplitude, ref_sign)
    if ref_key not in cond_data:
        raise RuntimeError(f"Reference condition {ref_key} not found.")
    x_ref, y_ref = cond_data[ref_key]

    target_keys = target_conditions
    n_targets = len(target_keys)

    phase_list = ['pre-stimulus', 'stimulus', 'intermediate', 'decision', 'full trial']
    results = {phase: [[] for _ in range(n_targets)] for phase in phase_list}
    evr_ref_list = []
    evr_target_lists = [[] for _ in range(n_targets)]

    visualization_done = False
    phase_params = {'stim_thresh': stim_thresh, 'dec_thresh': dec_thresh, 'dec_duration': dec_duration}

    for rep_num, rep_path in replica_list:
        candidates = [os.path.join(rep_path, "100_final.hdf5"), os.path.join(rep_path, "100_final.keras"),
                      os.path.join(rep_path, task_name.replace(" ", "_"), "100_final.hdf5")]
        model_path = next((c for c in candidates if os.path.exists(c)), None)
        if model_path is None:
            print(f"  [WARNING] Replica {rep_num}: no weight file. Skipping.")
            for phase in phase_list:
                for j in range(n_targets):
                    results[phase][j].append(np.full(n_components, np.nan))
            evr_ref_list.append(np.full(n_components, np.nan))
            for j in range(n_targets):
                evr_target_lists[j].append(np.full(n_components, np.nan))
            continue

        print(f"  Processing replica {rep_num} ...")
        model = load_network(model_path)

        h_ref = get_hidden_states(model, x_ref)
        h_ref_mean = h_ref.mean(axis=0)
        _, evr_ref = pca_subspace(h_ref_mean, n_components)
        evr_ref_list.append(evr_ref)

        h_target_means = []
        for idx, (amp, sgn) in enumerate(target_keys):
            x_tar, _ = cond_data[(amp, sgn)]
            h_tar = get_hidden_states(model, x_tar)
            h_tar_mean = h_tar.mean(axis=0)
            h_target_means.append(h_tar_mean)
            _, evr_tar = pca_subspace(h_tar_mean, n_components)
            evr_target_lists[idx].append(evr_tar)

        # Cache windows for reference and targets
        if not hasattr(run_analysis, 'windows_cache'):
            run_analysis.windows_cache = {}
        cache_key_ref = (rep_num, 'ref')
        if cache_key_ref not in run_analysis.windows_cache:
            x_first_ref = x_ref[0:1]
            y_pred_ref = model.predict(x_first_ref, verbose=0)[0]
            input_ref = x_first_ref[0, :, 0].flatten()
            output_ref = y_pred_ref[:, 0].flatten()
            win_ref = get_phase_windows(input_ref, output_ref, stim_thresh, dec_thresh, dec_duration)
            run_analysis.windows_cache[cache_key_ref] = win_ref

        for idx, (amp, sgn) in enumerate(target_keys):
            cache_key_tar = (rep_num, idx)
            if cache_key_tar not in run_analysis.windows_cache:
                x_tar_first = cond_data[(amp, sgn)][0][0:1]
                y_pred_tar = model.predict(x_tar_first, verbose=0)[0]
                input_tar = x_tar_first[0, :, 0].flatten()
                output_tar = y_pred_tar[:, 0].flatten()
                win_tar = get_phase_windows(input_tar, output_tar, stim_thresh, dec_thresh, dec_duration)
                run_analysis.windows_cache[cache_key_tar] = win_tar

        for phase_name in phase_list:
            window_ref = run_analysis.windows_cache[cache_key_ref][phase_name]
            h_ref_phase = slice_phase(h_ref_mean[np.newaxis], window_ref)[0]
            for idx, h_tar_mean in enumerate(h_target_means):
                window_tar = run_analysis.windows_cache[(rep_num, idx)][phase_name]
                h_tar_phase = slice_phase(h_tar_mean[np.newaxis], window_tar)[0]
                if h_ref_phase.shape[0] < n_components or h_tar_phase.shape[0] < n_components:
                    print(f"    [WARNING] Phase '{phase_name}' target {cond_labels[idx]} insufficient steps.")
                    results[phase_name][idx].append(np.full(n_components, np.nan))
                    continue
                V_ref, _ = pca_subspace(h_ref_phase, n_components)
                V_tar, _ = pca_subspace(h_tar_phase, n_components)
                angles = compute_principal_angles(V_ref, V_tar)
                results[phase_name][idx].append(angles)

        if visualize and not visualization_done and rep_num == replica_list[0][0]:
            cond_label_ref = f"H{ref_amplitude}{'+' if ref_sign==1 else '-'}"
            visualize_trial_phases(task_name, model, x_ref[0], y_ref[0], cond_label_ref,
                                   phase_params, rep_num, 0, output_dir)
            for idx, (amp, sgn) in enumerate(target_keys):
                x_tar, y_tar = cond_data[(amp, sgn)]
                visualize_trial_phases(task_name, model, x_tar[0], y_tar[0], cond_labels[idx],
                                       phase_params, rep_num, 0, output_dir)
            extreme_key = (8, 1)
            if extreme_key in cond_data:
                x_ext, y_ext = cond_data[extreme_key]
                for t in range(min(3, x_ref.shape[0])):
                    visualize_trial_phases(task_name, model, x_ref[t], y_ref[t], cond_label_ref,
                                           phase_params, rep_num, t, output_dir)
                n_ref = x_ref.shape[0]
                for t in range(max(0, n_ref-3), n_ref):
                    visualize_trial_phases(task_name, model, x_ref[t], y_ref[t], cond_label_ref,
                                           phase_params, rep_num, t, output_dir)
                cond_label_ext = f"H{extreme_key[0]}{'+' if extreme_key[1]==1 else '-'}"
                for t in range(min(3, x_ext.shape[0])):
                    visualize_trial_phases(task_name, model, x_ext[t], y_ext[t], cond_label_ext,
                                           phase_params, rep_num, t, output_dir)
                n_ext = x_ext.shape[0]
                for t in range(max(0, n_ext-3), n_ext):
                    visualize_trial_phases(task_name, model, x_ext[t], y_ext[t], cond_label_ext,
                                           phase_params, rep_num, t, output_dir)
            visualization_done = True

    for phase in phase_list:
        for j in range(n_targets):
            results[phase][j] = np.array(results[phase][j])
    evr_ref_arr = np.array(evr_ref_list)
    evr_target_arr = [np.array(lst) for lst in evr_target_lists]
    return results, evr_ref_arr, evr_target_arr, cond_labels

# ============================================================
#  EXPLAINED VARIANCE TABLE (improved)
# ============================================================
def save_evr_table_multi(evr_ref, evr_targets, cond_labels, task_name, output_dir):
    """Print and save a compact EVR table (mean ± SD) for all conditions."""
    ref_label = f"H{ref_amplitude}{'+' if ref_sign==1 else '-'}"
    all_labels = [ref_label] + cond_labels
    n_comp = evr_ref.shape[1]

    rows = []
    # Reference
    if evr_ref.size > 0 and not np.all(np.isnan(evr_ref)):
        evr_vals = evr_ref * 100
        row = [ref_label]
        for j in range(n_comp):
            col = evr_vals[:, j]
            valid = col[~np.isnan(col)]
            if len(valid) == 0:
                row.append("NaN")
            else:
                row.append(f"{np.mean(valid):.1f}±{np.std(valid):.1f}")
        rows.append(row)
    # Targets
    for i, label in enumerate(cond_labels):
        evr_tar = evr_targets[i]
        if evr_tar.size == 0 or np.all(np.isnan(evr_tar)):
            row = [label] + ["NaN"] * n_comp
        else:
            evr_vals = evr_tar * 100
            row = [label]
            for j in range(n_comp):
                col = evr_vals[:, j]
                valid = col[~np.isnan(col)]
                if len(valid) == 0:
                    row.append("NaN")
                else:
                    row.append(f"{np.mean(valid):.1f}±{np.std(valid):.1f}")
        rows.append(row)

    # Console output
    print(f"\n{'='*80}")
    print(f"Explained Variance Ratio (full trial) - {task_name}")
    print(f"{'='*80}")
    header = f"{'Condition':<8}" + "".join([f"  PC{i+1:>10}" for i in range(n_comp)])
    print(header)
    print("-" * len(header))
    for row in rows:
        line = f"{row[0]:<8}" + "".join([f"{str(c):>12}" for c in row[1:]])
        print(line)
    print()

    # Save to file
    txt_path = os.path.join(output_dir, f"evr_table_{task_name.replace(' ', '_')}.txt")
    with open(txt_path, 'w') as f:
        f.write(f"Explained Variance Ratio (full trial) - {task_name}\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for row in rows:
            line = f"{row[0]:<8}" + "".join([f"{str(c):>12}" for c in row[1:]])
            f.write(line + "\n")
    print(f"EVR table saved to: {txt_path}")

# ============================================================
#  PLOTTING FUNCTIONS (unchanged)
# ============================================================
def plot_principal_angles_for_pc(results, cond_labels, task_name, n_components, output_dir, pc_index):
    pc_name = f"PC{pc_index+1}"
    phase_names = [p for p in results.keys() if p != 'full trial']
    n_phases = len(phase_names)
    n_targets = len(cond_labels)
    colors_phase = plt.cm.viridis(np.linspace(0.15, 0.85, n_phases))

    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.5, 1], wspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    x = np.arange(n_targets)
    width = 0.8 / n_phases
    bar_centers_per_target = {j: [] for j in range(n_targets)}

    for i_phase, phase in enumerate(phase_names):
        data_by_target = results[phase]
        means = []
        stds = []
        for j in range(n_targets):
            data = data_by_target[j]
            if data.shape[0] == 0:
                means.append(np.nan)
                stds.append(np.nan)
            else:
                means.append(np.nanmean(data[:, pc_index]))
                stds.append(np.nanstd(data[:, pc_index]))
        offsets = (i_phase - (n_phases-1)/2) * width
        bars = ax1.bar(x + offsets, means, width=width*0.9, yerr=stds,
                       color=colors_phase[i_phase], alpha=0.85, label=phase,
                       error_kw=dict(elinewidth=1, capsize=3))
        for j, bar in enumerate(bars):
            bar_centers_per_target[j].append(bar.get_x() + bar.get_width()/2)

    for j in range(n_targets):
        for i_phase, phase in enumerate(phase_names):
            data = results[phase][j]
            if data.shape[0] == 0: continue
            y_vals = data[:, pc_index]
            center = bar_centers_per_target[j][i_phase]
            jitter_width = width * 0.3
            jitter = np.random.uniform(-jitter_width, jitter_width, size=len(y_vals))
            ax1.scatter(center + jitter, y_vals, s=15, color='gray', alpha=0.5, zorder=3)

    ax1.axhline(90, color='gray', linestyle='--', label='90° (orthogonal)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(cond_labels, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel(f"Principal angle (deg) - {pc_name}")
    ax1.set_xlabel("Comparison condition")
    ax1.set_ylim(0, 95)
    ax1.set_title(f"Principal angles ({pc_name}) between reference and each condition\n{task_name}")
    ax1.legend(fontsize=8, loc='upper left', framealpha=0.7)
    ax1.spines[['top','right']].set_visible(False)

    ax2 = fig.add_subplot(gs[1])
    ax2.text(0.5, 0.5, "Variance explained\n(see separate plots)", ha='center', va='center', transform=ax2.transAxes)
    ax2.set_axis_off()
    plt.tight_layout()
    out_png = os.path.join(output_dir, f"principal_angles_{task_name.replace(' ', '_')}_{pc_name}.png")
    out_svg = os.path.join(output_dir, f"principal_angles_{task_name.replace(' ', '_')}_{pc_name}.svg")
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.savefig(out_svg, dpi=200, bbox_inches='tight')
    print(f"\nFigures saved: {out_png} and {out_svg}")
    plt.close()

def plot_variance_explained(evr_ref, evr_targets, cond_labels, task_name, output_dir, pc_index):
    pc_name = f"PC{pc_index+1}"
    n_targets = len(cond_labels)
    evr_ref_pc = evr_ref[:, pc_index] * 100
    means_ref = np.full(n_targets, np.nanmean(evr_ref_pc))
    stds_ref = np.full(n_targets, np.nanstd(evr_ref_pc))

    means_tar = []
    stds_tar = []
    for evr_tar in evr_targets:
        if evr_tar.shape[0] > 0:
            vals = evr_tar[:, pc_index] * 100
            means_tar.append(np.nanmean(vals))
            stds_tar.append(np.nanstd(vals))
        else:
            means_tar.append(np.nan)
            stds_tar.append(np.nan)

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(n_targets)
    width = 0.35
    bars_ref = ax.bar(x - width/2, means_ref, width=width, yerr=stds_ref,
                      color='steelblue', alpha=0.85, label='Reference (H1+)',
                      error_kw=dict(elinewidth=1, capsize=3))
    bars_tar = ax.bar(x + width/2, means_tar, width=width, yerr=stds_tar,
                      color='coral', alpha=0.85, label='Target conditions',
                      error_kw=dict(elinewidth=1, capsize=3))

    for j in range(n_targets):
        y_vals = evr_ref_pc[~np.isnan(evr_ref_pc)]
        center = x[j] - width/2
        jitter_width = width * 0.3
        jitter = np.random.uniform(-jitter_width, jitter_width, size=len(y_vals))
        ax.scatter(center + jitter, y_vals, s=15, color='gray', alpha=0.5, zorder=3)

    for j in range(n_targets):
        evr_tar = evr_targets[j]
        if evr_tar.shape[0] == 0: continue
        y_vals = evr_tar[:, pc_index] * 100
        y_vals = y_vals[~np.isnan(y_vals)]
        if len(y_vals) == 0: continue
        center = x[j] + width/2
        jitter_width = width * 0.3
        jitter = np.random.uniform(-jitter_width, jitter_width, size=len(y_vals))
        ax.scatter(center + jitter, y_vals, s=15, color='gray', alpha=0.5, zorder=3)

    ax.axhline(90, color='gray', linestyle='--', label='90% variance threshold')
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel(f"Explained variance (%) - {pc_name}")
    ax.set_xlabel("Comparison condition")
    ax.set_title(f"Variance explained (full trial) - {pc_name}\n{task_name}")
    ax.legend(fontsize=8)
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    out_png = os.path.join(output_dir, f"variance_explained_{task_name.replace(' ', '_')}_{pc_name}.png")
    out_svg = os.path.join(output_dir, f"variance_explained_{task_name.replace(' ', '_')}_{pc_name}.svg")
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.savefig(out_svg, dpi=200, bbox_inches='tight')
    print(f"Variance explained figure saved: {out_png} and {out_svg}")
    plt.close()

def plot_heatmap_for_pc(results, cond_labels, task_name, output_dir, pc_index):
    pc_name = f"PC{pc_index+1}"
    phase_names = [p for p in results.keys() if p != 'full trial']
    n_phases = len(phase_names)
    n_targets = len(cond_labels)
    mean_mat = np.zeros((n_phases, n_targets))
    for i, phase in enumerate(phase_names):
        for j in range(n_targets):
            data = results[phase][j]
            if data.shape[0] > 0:
                mean_mat[i, j] = np.nanmean(data[:, pc_index])
            else:
                mean_mat[i, j] = np.nan
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(mean_mat, vmin=0, vmax=90, cmap='RdYlGn_r', aspect='auto')
    plt.colorbar(im, ax=ax, label=f'Mean principal angle (deg) - {pc_name}')
    ax.set_xticks(np.arange(n_targets))
    ax.set_xticklabels(cond_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(np.arange(n_phases))
    ax.set_yticklabels(phase_names)
    ax.set_title(f"Principal angles ({pc_name}) between reference and each condition\n{task_name}")
    for i in range(n_phases):
        for j in range(n_targets):
            val = mean_mat[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha='center', va='center', fontsize=7,
                        color='white' if val > 55 else 'black')
    plt.tight_layout()
    out_png = os.path.join(output_dir, f"heatmap_{task_name.replace(' ', '_')}_{pc_name}.png")
    out_svg = os.path.join(output_dir, f"heatmap_{task_name.replace(' ', '_')}_{pc_name}.svg")
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.savefig(out_svg, dpi=200, bbox_inches='tight')
    print(f"Heatmap saved: {out_png} and {out_svg}")
    plt.close()

def print_summary_table(results, cond_labels, task_name, output_dir):
    phase_names = [p for p in results.keys()]
    n_targets = len(cond_labels)
    print(f"\n{'='*80}")
    print(f"Summary - Principal angles (degrees) - {task_name}")
    print(f"{'='*80}")
    lines = []
    for pc_idx in range(3):
        pc_name = f"PC{pc_idx+1}"
        print(f"\n--- {pc_name} ---")
        header = f"{'Phase':<18}" + "".join([f"{lab:>8}" for lab in cond_labels])
        print(header)
        print("-" * len(header))
        lines.append(f"--- {pc_name} ---")
        lines.append(header)
        lines.append("-" * len(header))
        for phase in phase_names:
            row = f"{phase:<18}"
            for j in range(n_targets):
                data = results[phase][j]
                if data.shape[0] == 0:
                    row += "     NaN"
                else:
                    mean_val = np.nanmean(data[:, pc_idx])
                    std_val = np.nanstd(data[:, pc_idx])
                    row += f"{mean_val:6.1f}±{std_val:4.1f}"
            print(row)
            lines.append(row)
        print()
        lines.append("")
    txt_path = os.path.join(output_dir, f"summary_table_{task_name.replace(' ', '_')}.txt")
    with open(txt_path, 'w') as f:
        f.write(f"Summary - Principal angles (degrees) - {task_name}\n")
        for line in lines:
            f.write(line + "\n")
    print(f"Summary table saved to: {txt_path}")

# ============================================================
#  ENTRY POINT
# ============================================================
if __name__ == "__main__":
    results, evr_ref, evr_targets, cond_labels = run_analysis(
        task_name=task, base_dir=base_dir, n_components=n_components,
        n_trials_per_cond=n_trials_per_cond, stim_thresh=stimulus_threshold,
        dec_thresh=decision_threshold, dec_duration=decision_fixed_duration,
        output_dir=output_dir, visualize=visualize_phases
    )
    print_summary_table(results, cond_labels, task, output_dir)
    # Explicited EVR table (now with improved formatting)
    save_evr_table_multi(evr_ref, evr_targets, cond_labels, task, output_dir)
    for pc in range(3):
        plot_principal_angles_for_pc(results, cond_labels, task, n_components, output_dir, pc)
        plot_heatmap_for_pc(results, cond_labels, task, output_dir, pc)
        plot_variance_explained(evr_ref, evr_targets, cond_labels, task, output_dir, pc)

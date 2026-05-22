"""
principal_angles_analysis_integral_dm.py

Computes principal angles between positive and negative subspaces
for the Integral DM task (perceptual decision-making with sustained noise input).
Uses deterministic phase windows based on task timing.
Output: bar chart, heatmap, EVR panel, summary tables.
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
task = "Integral DM"
# Ajustá esta ruta a tus réplicas entrenadas con Integral DM
base_dir = "../weights/05_Perceptual_dm_delayed_response/orthogonal_rrn_no_bias_term"

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
mem_gap = 0   # Not used; we use get_mem_gap

# Decision window duration after onset (time steps)
decision_fixed_duration = 100

output_dir = "../principal_angles_output_integral_dm"
os.makedirs(output_dir, exist_ok=True)

visualize_phases = True   # verification plots for first replica

# ============================================================
#  DATASET GENERATOR (same import logic)
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
#  DETERMINISTIC PHASE WINDOWS (based on Integral DM structure)
# ============================================================
def get_phase_windows_integral(mem_gap_val, seq_dur, dec_duration):
    """
    Returns phase windows for the Integral DM task.
    - Stimulus: from t=50 to t=50+mem_gap_val-1 (mem_gap_val = 200 by default)
    - Decision onset: 10 + first_in+stim_dur + 10 = 270
    - Intermediate: end of stimulus to just before decision onset
    """
    first_in = 50
    stim_dur = mem_gap_val
    out_t = 10 + first_in + stim_dur          # = 260
    decision_onset = out_t + 10               # = 270

    pre_start = 0
    pre_end = max(0, first_in - 1)            # 0..49

    stim_start = first_in                     # 50
    stim_end = first_in + stim_dur - 1        # 249

    inter_start = stim_end + 1                # 250
    inter_end = decision_onset - 1            # 269
    if inter_start > inter_end:
        inter_start = inter_end = stim_end    # fallback

    dec_start = decision_onset                # 270
    dec_end = min(decision_onset + dec_duration, seq_dur - 1)

    # Asegurar límites
    pre_end = min(pre_end, seq_dur - 1)
    stim_end = min(stim_end, seq_dur - 1)
    inter_end = min(inter_end, seq_dur - 1)
    dec_end = min(dec_end, seq_dur - 1)

    return {
        'pre-stimulus': (pre_start, pre_end),
        'stimulus': (stim_start, stim_end),
        'intermediate': (inter_start, inter_end),
        'decision': (dec_start, dec_end),
        'full trial': None
    }

# ============================================================
#  PCA & ANGLES
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
#  VISUALIZATION OF DETERMINED WINDOWS
# ============================================================
def visualize_trial_phases_integral(task_name, model, x_trial, y_trial,
                                    windows, replica_idx, trial_idx, output_dir):
    x_ex = x_trial[np.newaxis, ...]
    y_pred = model.predict(x_ex, verbose=0)[0]
    input_sig = x_trial[:, 0].flatten()
    output_sig = y_pred[:, 0].flatten()
    T = len(input_sig)
    time = np.arange(T)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    fig.suptitle(f"{task_name} | Replica {replica_idx} | Trial {trial_idx}", fontsize=10)
    colors = {'pre-stimulus': 'purple', 'stimulus': 'blue', 'intermediate': 'orange', 'decision': 'red'}

    def add_markers(ax, windows, T_max):
        for name, win in windows.items():
            if win is None: continue
            t0, t1 = win
            if t0 < T_max:
                ax.axvspan(t0, min(t1, T_max), alpha=0.1, color=colors.get(name, 'gray'),
                           label=name if name not in ax.get_legend_handles_labels()[1] else "")
                ax.axvline(t0, color=colors.get(name, 'gray'), linestyle='--', alpha=0.5)
                ax.axvline(t1, color=colors.get(name, 'gray'), linestyle='--', alpha=0.5)

    ax1 = axes[0]; ax1.plot(time, input_sig, 'g-', label='Input'); ax1.set_ylabel('Input amplitude'); ax1.set_title('Input')
    add_markers(ax1, windows, T); ax1.legend(fontsize=9, loc=1)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.set_xlabel('Time step')
    ax2 = axes[1]; ax2.plot(time, output_sig, 'r-', label='Network output'); ax2.set_title('Output')
    add_markers(ax2, windows, T); ax2.legend(fontsize=9, loc=1)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_xlabel('Time step')
    plt.tight_layout()
    out_png = os.path.join(output_dir, f"phase_verification_replica_{replica_idx}_trial{trial_idx}.png")
    out_svg = os.path.join(output_dir, f"phase_verification_replica_{replica_idx}_trial{trial_idx}.svg")
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.savefig(out_svg, dpi=150, bbox_inches='tight')
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
#  MAIN ANALYSIS
# ============================================================
def run_analysis(task_name, base_dir, n_components, n_trials_per_cond,
                 dec_duration, output_dir, visualize=False):

    replica_list = find_replica_dirs(base_dir)
    if not replica_list:
        raise RuntimeError(f"No replica directories found in {base_dir}")
    n_replicas = len(replica_list)
    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"Found {n_replicas} replicas: {[r[0] for r in replica_list]}")
    print(f"Components: {n_components}")
    print(f"Decision window duration: {dec_duration} steps")
    print(f"{'='*60}")

    generate_trials = import_generator(task_name)
    mem_gap_val = get_mem_gap(task_name)   # 200 for Integral DM

    total_trials = n_trials_per_cond * 4
    x_all, y_all, seq_dur = generate_trials(total_trials, mem_gap_val)
    print(f"Sequence duration: {seq_dur} time steps")

    # Phase windows (same for all trials because timing is fixed)
    phase_windows = get_phase_windows_integral(mem_gap_val, seq_dur, dec_duration)
    print("Phase windows:")
    for k, v in phase_windows.items():
        print(f"  {k}: {v}")

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

        # Hidden states and EVR
        h_pos = get_hidden_states(model, x_pos)
        h_neg = get_hidden_states(model, x_neg)
        h_pos_mean = h_pos.mean(axis=0)
        h_neg_mean = h_neg.mean(axis=0)

        # Compute angles per phase using deterministic windows
        for phase_name in phase_list:
            win = phase_windows[phase_name]   # same for both conditions
            h_pos_phase = slice_phase(h_pos_mean[np.newaxis], win)[0]
            h_neg_phase = slice_phase(h_neg_mean[np.newaxis], win)[0]
            if h_pos_phase.shape[0] < n_components or h_neg_phase.shape[0] < n_components:
                print(f"    [WARNING] Phase '{phase_name}' insufficient steps. Skipping.")
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

        # Visualization (first replica: first 3 and last 3 trials each condition)
        if visualize and not visualization_done and rep_num == replica_list[0][0]:
            n_pos_trials = x_pos.shape[0]
            n_neg_trials = x_neg.shape[0]
            #for cond, (x_trials, y_trials, n_t) in [('pos', x_pos, y_pos, n_pos_trials), ('neg', x_neg, y_neg, n_neg_trials)]:
            for cond, x_trials, y_trials, n_t in [('pos', x_pos, y_pos, n_pos_trials), ('neg', x_neg, y_neg, n_neg_trials)]:
                for t in range(min(3, n_t)):
                    visualize_trial_phases_integral(task_name, model, x_trials[t], y_trials[t],
                                                    phase_windows, rep_num, t, output_dir)
                for t in range(max(0, n_t-3), n_t):
                    visualize_trial_phases_integral(task_name, model, x_trials[t], y_trials[t],
                                                    phase_windows, rep_num, t, output_dir)
            visualization_done = True

    # Convert to arrays
    for ph in results:
        results[ph] = np.array(results[ph])
    evr_pos_all = np.array(evr_pos_all)
    evr_neg_all = np.array(evr_neg_all)
    return results, evr_pos_all, evr_neg_all, seq_dur

# ============================================================
#  TABLES AND PLOTS (unchanged from principal_angles_analysis_12.py)
# ============================================================
def save_evr_table(evr_pos, evr_neg, task_name, output_dir):
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

def plot_principal_angles(results, evr_pos, evr_neg, task_name, n_components, output_dir):
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
    bar_positions = {}

    for i, (phase, color) in enumerate(zip(phase_names, colors)):
        data = results[phase]
        valid = ~np.isnan(data).any(axis=1)
        data_valid = data[valid]
        if data_valid.shape[0] == 0:
            continue
        mean_a = np.nanmean(data_valid, axis=0)
        std_a = np.nanstd(data_valid, axis=0)
        offsets = (i - (n_phases-1)/2) * width
        bars = ax1.bar(x + offsets, mean_a, width=width*0.9, yerr=std_a,
                       color=color, alpha=0.85, label=phase,
                       error_kw=dict(elinewidth=1, capsize=3))
        bar_centers = [bar.get_x() + bar.get_width()/2 for bar in bars]
        bar_positions[phase] = (bar_centers, data_valid)

    for phase, (centers, data_valid) in bar_positions.items():
        for pc_idx, center in enumerate(centers):
            y_vals = data_valid[:, pc_idx]
            jitter_width = width * 0.1
            jitter = np.random.uniform(-jitter_width, jitter_width, size=len(y_vals))
            ax1.scatter(center + jitter, y_vals, s=20, color='gray', alpha=0.6, zorder=3, edgecolors='none')

    ax1.axhline(90, color='gray', linestyle='--', label='90° (orthogonal)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"PC{i}" for i in x])
    ax1.set_ylabel("Principal angle (deg)", fontsize=12)
    ax1.set_xlabel("Principal component", fontsize=12)
    ax1.set_ylim(0, 95)
    ax1.set_title(f"Principal angles between +/- subspaces\n{task_name}  (mean ± std, n={results[phase_names[0]].shape[0]} replicas)")
    ax1.legend(fontsize=9, loc="upper right", framealpha=0.7)
    ax1.spines[['top', 'right']].set_visible(False)

    ax2 = fig.add_subplot(gs[1])
    x_evr = np.arange(1, n_components+1)
    if evr_pos.shape[0] > 0 and not np.isnan(evr_pos).all():
        evr_pos_mean = np.nanmean(evr_pos, axis=0)
        evr_pos_std = np.nanstd(evr_pos, axis=0)
        evr_neg_mean = np.nanmean(evr_neg, axis=0)
        evr_neg_std = np.nanstd(evr_neg, axis=0)

        bars_pos = ax2.bar(x_evr - 0.2, evr_pos_mean*100, width=0.35, yerr=evr_pos_std*100,
                           color='steelblue', label='Positive trials', alpha=0.85,
                           error_kw=dict(elinewidth=1, capsize=3))
        bars_neg = ax2.bar(x_evr + 0.2, evr_neg_mean*100, width=0.35, yerr=evr_neg_std*100,
                           color='coral', label='Negative trials', alpha=0.85,
                           error_kw=dict(elinewidth=1, capsize=3))

        for pc_idx, bar in enumerate(bars_pos):
            center = bar.get_x() + bar.get_width()/2
            y_vals = evr_pos[:, pc_idx] * 100
            y_vals = y_vals[~np.isnan(y_vals)]
            if len(y_vals) == 0: continue
            jitter = np.random.uniform(-0.15, 0.15, size=len(y_vals))
            ax2.scatter(center + jitter, y_vals, s=15, color='gray', alpha=0.5, zorder=3)

        for pc_idx, bar in enumerate(bars_neg):
            center = bar.get_x() + bar.get_width()/2
            y_vals = evr_neg[:, pc_idx] * 100
            y_vals = y_vals[~np.isnan(y_vals)]
            if len(y_vals) == 0: continue
            jitter = np.random.uniform(-0.15, 0.15, size=len(y_vals))
            ax2.scatter(center + jitter, y_vals, s=15, color='gray', alpha=0.5, zorder=3)

    ax2.set_xticks(x_evr)
    ax2.set_xticklabels([f"PC{i}" for i in x_evr])
    ax2.set_ylabel("Explained variance (%)", fontsize=12)
    ax2.set_xlabel("Principal component", fontsize=12)
    ax2.axhline(90, color='gray', linestyle='--', label='90 %')
    ax2.set_title("Variance explained (full trial)", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    figpath_png = os.path.join(output_dir, f"principal_angles_{task_name.replace(' ', '_')}.png")
    figpath_svg = os.path.join(output_dir, f"principal_angles_{task_name.replace(' ', '_')}.svg")
    plt.savefig(figpath_png, dpi=200, bbox_inches='tight')
    plt.savefig(figpath_svg, dpi=200, bbox_inches='tight')
    print(f"\nFigures saved: {figpath_png} and {figpath_svg}")
    plt.close(fig)

def plot_angle_heatmap(results, task_name, output_dir):
    phase_names = [p for p in results.keys() if p != 'full trial' and results[p].size > 0 and not np.all(np.isnan(results[p]))]
    if not phase_names: return
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
    figpath_png = os.path.join(output_dir, f"angle_heatmap_{task_name.replace(' ', '_')}.png")
    figpath_svg = os.path.join(output_dir, f"angle_heatmap_{task_name.replace(' ', '_')}.svg")
    plt.savefig(figpath_png, dpi=200, bbox_inches='tight')
    plt.savefig(figpath_svg, dpi=200, bbox_inches='tight')
    print(f"Heatmap saved: {figpath_png} and {figpath_svg}")
    plt.close()

def print_summary_table(results, task_name, output_dir):
    phase_names = [p for p in results.keys() if results[p].size > 0 and not np.all(np.isnan(results[p]))]
    if not phase_names: return
    n_comp = results[phase_names[0]].shape[1]
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
#  ENTRY POINT
# ============================================================
if __name__ == "__main__":
    results, evr_pos, evr_neg, seq_dur = run_analysis(
        task_name=task,
        base_dir=base_dir,
        n_components=n_components,
        n_trials_per_cond=n_trials_per_cond,
        dec_duration=decision_fixed_duration,
        output_dir=output_dir,
        visualize=visualize_phases,
    )
    print_summary_table(results, task, output_dir)
    save_evr_table(evr_pos, evr_neg, task, output_dir)
    plot_principal_angles(results, evr_pos, evr_neg, task, n_components, output_dir)
    plot_angle_heatmap(results, task, output_dir)

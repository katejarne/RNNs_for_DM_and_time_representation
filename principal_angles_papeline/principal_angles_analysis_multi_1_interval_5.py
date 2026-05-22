"""
principal_angles_multi_interval_distance.py

Multi-interval distance-based task (8 intervals encoded by pulse separation).
Computes principal angles between leading PC subspaces of trained RNN populations.
Reference condition: T20 (interval 20 ms). Compared against T40, T60, ..., T160.
Generates bar plots and heatmaps for PC1, PC2, PC3.
Now also saves an explained variance table for all conditions.
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
task = "Simple DM 8 time encoded"
base_dir = "../weights/08_DM_delayed_response_8_times_intervals/orthogonal_rrn_no_bias_term"

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
decision_fixed_duration = 100   # duración fija de la ventana de decisión (pasos)

output_dir = "../principal_angles_output_multi_distance"
os.makedirs(output_dir, exist_ok=True)
visualize_phases = True   # genera figuras de verificación para la primera réplica

# Intervalos posibles (ms)
interval_options = [20, 40, 60, 80, 100, 120, 140, 160]
ref_interval = 20
target_intervals = [iv for iv in interval_options if iv != ref_interval]
cond_labels = [f"T{iv}" for iv in target_intervals]  # temporal

# ============================================================
#  DATASET GENERATOR & HELPERS
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
    if x_input.shape[0] == 0:
        return np.empty((0, 0, 0))
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
#  FUNCIONES PARA VENTANAS DETERMINISTAS (basadas en T y seq_dur)
# ============================================================
def get_phase_windows_from_T(T, seq_dur, dec_duration):
    """
    Calcula las ventanas de fase de forma determinista usando la estructura
    temporal de la tarea (pulsos en t=50 y t=70+T, respuesta en t=70+2T).

    Parámetros:
    - T: intervalo entre pulsos (ms)
    - seq_dur: duración total de la secuencia (pasos de tiempo)
    - dec_duration: duración fija de la ventana de decisión (pasos)

    Retorna:
    - diccionario con las ventanas: 'pre-stimulus', 'stimulus', 'intermediate', 'decision', 'full trial'
    """
    first_in = 50          # inicio del primer pulso (fijo)
    stim_dur = 20          # duración de cada pulso (fijo)

    start1 = first_in
    end1 = start1 + stim_dur
    start2 = end1 + T
    end2 = start2 + stim_dur
    response_start = end2 + T

    # Pre-estímulo: desde 0 hasta justo antes del primer pulso
    pre_start = 0
    pre_end = max(0, start1 - 1)

    # Estímulo: desde el inicio del primer pulso hasta el final del segundo pulso
    stim_start = start1
    stim_end = end2

    # Intermedio: desde el final del estímulo hasta justo antes de la respuesta
    inter_start = end2 + 1
    inter_end = response_start - 1
    if inter_start > inter_end:
        # Si no hay intermedio (T muy pequeño), se toma un solo paso
        inter_start = end2
        inter_end = end2

    # Decisión: desde el inicio de la respuesta hasta dec_duration pasos después
    dec_start = response_start
    dec_end = min(response_start + dec_duration, seq_dur - 1)

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

def visualize_trial_phases_deterministic(task_name, model, x_trial, y_trial, cond_label,
                                         T, seq_dur, dec_duration,
                                         replica_idx, trial_idx, output_dir):
    """
    Visualiza un trial con las ventanas deterministas, usando el intervalo T conocido.
    """
    x_ex = x_trial[np.newaxis, ...]
    y_pred = model.predict(x_ex, verbose=0)[0]
    input_sig = x_trial[:, 0].flatten()
    output_sig = y_pred[:, 0].flatten()
    time = np.arange(len(input_sig))

    windows = get_phase_windows_from_T(T, seq_dur, dec_duration)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    fig.suptitle(f"{task_name} | Replica {replica_idx} | {cond_label} trial {trial_idx} (T={T}ms)", fontsize=10)
    colors = {'pre-stimulus': 'purple', 'stimulus': 'green', 'intermediate': 'orange', 'decision': 'red'}

    def add_markers(ax, windows, T_max):
        for name, win in windows.items():
            if win is None: continue
            t0, t1 = win
            if t0 < T_max:
                ax.axvspan(t0, min(t1, T_max), alpha=0.2, color=colors.get(name, 'gray'),
                           label=name if name not in ax.get_legend_handles_labels()[1] else "")
                ax.axvline(t0, color=colors.get(name, 'gray'), linestyle='--', alpha=0.7)
                ax.axvline(t1, color=colors.get(name, 'gray'), linestyle='--', alpha=0.7)

    ax1 = axes[0]
    ax1.plot(time, input_sig, 'b-', label='Input')
    ax1.set_ylabel('Input amplitude')
    ax1.set_title('Input')
    add_markers(ax1, windows, len(input_sig))
    ax1.legend(fontsize=6)

    ax2 = axes[1]
    ax2.plot(time, output_sig, 'r-', label='Network output')
    ax2.set_title('Output')
    add_markers(ax2, windows, len(input_sig))
    ax2.legend(fontsize=6)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"phase_verification_repl{replica_idx}_{cond_label}_trial{trial_idx}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

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
#  MAIN ANALYSIS (con ventanas deterministas)
# ============================================================
def run_analysis(task_name, base_dir, n_components, n_trials_per_cond,
                 dec_duration, output_dir, visualize=False):
    global cond_labels

    replica_list = find_replica_dirs(base_dir)
    if not replica_list:
        raise RuntimeError(f"No replica directories found in {base_dir}")
    n_replicas = len(replica_list)
    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"Found {n_replicas} replicas: {[r[0] for r in replica_list]}")
    print(f"Components: {n_components}")
    print(f"Decision window duration: {dec_duration} steps")
    print(f"Reference condition: T{ref_interval}")
    print(f"Target conditions: {cond_labels}")
    print(f"{'='*60}")

    generate_trials = import_generator(task_name)
    mem_gap_val = get_mem_gap(task_name)

    # Generar suficientes trials para todos los intervalos
    total_trials_needed = n_trials_per_cond * len(interval_options) * 5
    x_all, y_all, seq_dur, Ts_all = generate_trials(total_trials_needed, mem_gap_val)
    print(f"Sequence duration: {seq_dur} time steps")

    # Separar trials por intervalo usando Ts_all
    cond_indices = {iv: [] for iv in interval_options}
    for idx, T in enumerate(Ts_all):
        # Ts_all contiene los intervalos exactos (20,40,...,160)
        if T in cond_indices:
            cond_indices[T].append(idx)
        else:
            # Por si acaso, redondear al más cercano
            closest = min(interval_options, key=lambda x: abs(x - T))
            cond_indices[closest].append(idx)

    cond_data = {}
    valid_intervals = []
    for iv in interval_options:
        idx_list = cond_indices[iv]
        if len(idx_list) < n_trials_per_cond:
            print(f"  Warning: condition T{iv} has only {len(idx_list)} trials (<{n_trials_per_cond})")
            if len(idx_list) == 0:
                continue
            selected = idx_list[:min(len(idx_list), n_trials_per_cond)]
        else:
            selected = idx_list[:n_trials_per_cond]
        cond_data[iv] = (x_all[selected], y_all[selected])
        valid_intervals.append(iv)

    if ref_interval not in cond_data:
        raise RuntimeError(f"Reference interval {ref_interval} not found.")
    x_ref, y_ref = cond_data[ref_interval]

    target_intervals = [iv for iv in valid_intervals if iv != ref_interval]
    if len(target_intervals) == 0:
        raise RuntimeError("No target intervals available.")
    cond_labels = [f"T{iv}" for iv in target_intervals]
    n_targets = len(target_intervals)

    # Definir las ventanas de fase para cada intervalo (son deterministas y dependen de T)
    # Precalcular las ventanas para todos los intervalos presentes
    phase_windows_by_T = {}
    for iv in valid_intervals:
        phase_windows_by_T[iv] = get_phase_windows_from_T(iv, seq_dur, dec_duration)

    phase_list = ['pre-stimulus', 'stimulus', 'intermediate', 'decision', 'full trial']
    results = {phase: [[] for _ in range(n_targets)] for phase in phase_list}
    evr_ref_list = []
    evr_target_lists = [[] for _ in range(n_targets)]

    visualization_done = False

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

        # Referencia
        h_ref = get_hidden_states(model, x_ref)
        if h_ref.size == 0:
            print(f"    Reference hidden states empty. Skipping replica.")
            continue
        h_ref_mean = h_ref.mean(axis=0)
        _, evr_ref = pca_subspace(h_ref_mean, n_components)
        evr_ref_list.append(evr_ref)

        # Targets
        h_target_means = []
        for idx, iv in enumerate(target_intervals):
            x_tar, _ = cond_data[iv]
            h_tar = get_hidden_states(model, x_tar)
            if h_tar.size == 0:
                h_target_means.append(np.empty((0,0)))
                evr_target_lists[idx].append(np.full(n_components, np.nan))
                continue
            h_tar_mean = h_tar.mean(axis=0)
            h_target_means.append(h_tar_mean)
            _, evr_tar = pca_subspace(h_tar_mean, n_components)
            evr_target_lists[idx].append(evr_tar)

        # Para cada fase, usar las ventanas precalculadas según el intervalo
        for phase_name in phase_list:
            # Ventana para la referencia (intervalo ref_interval)
            window_ref = phase_windows_by_T[ref_interval][phase_name]
            h_ref_phase = slice_phase(h_ref_mean[np.newaxis], window_ref)[0]
            for idx, iv in enumerate(target_intervals):
                h_tar_mean = h_target_means[idx]
                if h_tar_mean.size == 0:
                    results[phase_name][idx].append(np.full(n_components, np.nan))
                    continue
                window_tar = phase_windows_by_T[iv][phase_name]
                h_tar_phase = slice_phase(h_tar_mean[np.newaxis], window_tar)[0]
                if h_ref_phase.shape[0] < n_components or h_tar_phase.shape[0] < n_components:
                    print(f"    [WARNING] Phase '{phase_name}' target {cond_labels[idx]} has only {h_tar_phase.shape[0]} steps (<{n_components}). Skipping.")
                    results[phase_name][idx].append(np.full(n_components, np.nan))
                    continue
                V_ref, _ = pca_subspace(h_ref_phase, n_components)
                V_tar, _ = pca_subspace(h_tar_phase, n_components)
                angles = compute_principal_angles(V_ref, V_tar)
                results[phase_name][idx].append(angles)

        # Visualización (solo primera réplica, primeros trials)
        if visualize and not visualization_done and rep_num == replica_list[0][0]:
            # Referencia
            visualize_trial_phases_deterministic(task_name, model, x_ref[0], y_ref[0],
                                                 f"T{ref_interval}", ref_interval, seq_dur, dec_duration,
                                                 rep_num, 0, output_dir)
            # Cada target
            for idx, iv in enumerate(target_intervals):
                x_tar, y_tar = cond_data[iv]
                visualize_trial_phases_deterministic(task_name, model, x_tar[0], y_tar[0],
                                                     cond_labels[idx], iv, seq_dur, dec_duration,
                                                     rep_num, 0, output_dir)
            # Primeros y últimos 3 trials de referencia y del intervalo extremo
            extreme_iv = max(target_intervals)
            x_ext, y_ext = cond_data[extreme_iv]
            for t in range(min(3, x_ref.shape[0])):
                visualize_trial_phases_deterministic(task_name, model, x_ref[t], y_ref[t],
                                                     f"T{ref_interval}", ref_interval, seq_dur, dec_duration,
                                                     rep_num, t, output_dir)
            n_ref = x_ref.shape[0]
            for t in range(max(0, n_ref-3), n_ref):
                visualize_trial_phases_deterministic(task_name, model, x_ref[t], y_ref[t],
                                                     f"T{ref_interval}", ref_interval, seq_dur, dec_duration,
                                                     rep_num, t, output_dir)
            for t in range(min(3, x_ext.shape[0])):
                visualize_trial_phases_deterministic(task_name, model, x_ext[t], y_ext[t],
                                                     f"T{extreme_iv}", extreme_iv, seq_dur, dec_duration,
                                                     rep_num, t, output_dir)
            n_ext = x_ext.shape[0]
            for t in range(max(0, n_ext-3), n_ext):
                visualize_trial_phases_deterministic(task_name, model, x_ext[t], y_ext[t],
                                                     f"T{extreme_iv}", extreme_iv, seq_dur, dec_duration,
                                                     rep_num, t, output_dir)
            visualization_done = True

    for phase in phase_list:
        for j in range(n_targets):
            results[phase][j] = np.array(results[phase][j])
    evr_ref_arr = np.array(evr_ref_list)
    evr_target_arr = [np.array(lst) for lst in evr_target_lists]
    return results, evr_ref_arr, evr_target_arr, cond_labels

# ============================================================
#  EXPLAINED VARIANCE TABLE (NEW)
# ============================================================
def save_evr_table_multi_distance(evr_ref, evr_targets, cond_labels, task_name, output_dir, ref_interval):
    """
    Print and save a table of explained variance ratio (EVR) per PC component,
    for reference (T{ref_interval}) and all target intervals, averaged across replicas.
    """
    ref_label = f"T{ref_interval}"
    all_labels = [ref_label] + cond_labels
    n_comp = evr_ref.shape[1]
    print(f"\n{'='*80}")
    print(f"Explained Variance Ratio (full trial) - {task_name}")
    print(f"{'='*80}")
    header = f"{'Condition':<8}" + "".join([f"  PC{i+1:>16}" for i in range(n_comp)])
    print(header)
    print("-" * len(header))
    lines = []

    # Reference row
    evr_vals = evr_ref * 100  # percentage, shape (n_replicas, n_comp)
    row = f"{ref_label:<8}"
    for j in range(n_comp):
        col = evr_vals[:, j]
        col_valid = col[~np.isnan(col)]
        if len(col_valid) == 0:
            row += "       NaN       "
        else:
            row += f"  {np.mean(col_valid):5.1f}±{np.std(col_valid):4.1f}%"
    print(row)
    lines.append(row)

    # Target rows
    for i, label in enumerate(cond_labels):
        evr_tar = evr_targets[i] * 100
        if evr_tar.size == 0 or np.all(np.isnan(evr_tar)):
            row = f"{label:<8}" + "       NaN       " * n_comp
            print(row)
            lines.append(row)
            continue
        row = f"{label:<8}"
        for j in range(n_comp):
            col = evr_tar[:, j]
            col_valid = col[~np.isnan(col)]
            if len(col_valid) == 0:
                row += "       NaN       "
            else:
                row += f"  {np.mean(col_valid):5.1f}±{np.std(col_valid):4.1f}%"
        print(row)
        lines.append(row)
    print()

    # Save to file
    txt_path = os.path.join(output_dir, f"evr_table_{task_name.replace(' ', '_')}.txt")
    with open(txt_path, 'w') as f:
        f.write(f"Explained Variance Ratio (full trial) - {task_name}\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for line in lines:
            f.write(line + "\n")
        f.write("\n")
    print(f"EVR table saved to: {txt_path}")

# ============================================================
#  PLOTTING FUNCTIONS (sin cambios)
# ============================================================
def plot_principal_angles_for_pc(results, cond_labels, task_name, n_components, output_dir, pc_index):
    pc_name = f"PC{pc_index+1}"
    phase_names = [p for p in results.keys() if p != 'full trial']
    n_phases = len(phase_names)
    n_targets = len(cond_labels)
    colors_phase = plt.cm.viridis(np.linspace(0.15, 0.85, n_phases))

    # Determinar qué condiciones tienen datos válidos
    valid_targets = []
    valid_indices = []
    for j in range(n_targets):
        has_valid = False
        for phase in phase_names:
            data = results[phase][j]
            if data.size > 0 and np.any(np.isfinite(data[:, pc_index])):
                has_valid = True
                break
        if has_valid:
            valid_targets.append(cond_labels[j])
            valid_indices.append(j)
        else:
            print(f"  Skipping target {cond_labels[j]} (no valid data for PC{pc_index+1})")

    if len(valid_targets) == 0:
        print(f"No valid targets for {pc_name}, skipping plot.")
        return

    cond_labels_valid = valid_targets
    n_targets_valid = len(valid_targets)
    x = np.arange(n_targets_valid)
    width = 0.8 / n_phases

    # Fases con datos válidos
    valid_phases = []
    valid_phase_indices = []
    for i_phase, phase in enumerate(phase_names):
        has_val = False
        for j_orig in valid_indices:
            data = results[phase][j_orig]
            if data.size > 0 and np.any(np.isfinite(data[:, pc_index])):
                has_val = True
                break
        if has_val:
            valid_phases.append(phase)
            valid_phase_indices.append(i_phase)

    if len(valid_phases) == 0:
        print(f"No valid phases for {pc_name}, skipping plot.")
        return

    n_phases_valid = len(valid_phases)
    colors_phase_valid = [colors_phase[i] for i in valid_phase_indices]

    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.5, 1], wspace=0.35)
    ax1 = fig.add_subplot(gs[0])

    bar_centers_per_target = {j: [] for j in range(n_targets_valid)}

    for i_phase, phase in enumerate(valid_phases):
        means = []
        stds = []
        for j_idx, j_orig in enumerate(valid_indices):
            data = results[phase][j_orig]
            if data.size == 0:
                means.append(np.nan)
                stds.append(np.nan)
            else:
                vals = data[:, pc_index]
                if np.all(np.isnan(vals)):
                    means.append(np.nan)
                    stds.append(np.nan)
                else:
                    means.append(np.nanmean(vals))
                    stds.append(np.nanstd(vals))
        if np.all(np.isnan(means)):
            continue
        means_clean = [0 if np.isnan(m) else m for m in means]
        stds_clean = [0 if np.isnan(s) else s for s in stds]
        offsets = (i_phase - (n_phases_valid - 1) / 2) * width
        bars = ax1.bar(x + offsets, means_clean, width=width * 0.9, yerr=stds_clean,
                       color=colors_phase_valid[i_phase], alpha=0.85, label=phase,
                       error_kw=dict(elinewidth=1, capsize=3))
        for j_idx, bar in enumerate(bars):
            bar_centers_per_target[j_idx].append(bar.get_x() + bar.get_width() / 2)

    # Puntos individuales
    for j_idx, j_orig in enumerate(valid_indices):
        for i_phase, phase in enumerate(valid_phases):
            data = results[phase][j_orig]
            if data.size == 0:
                continue
            y_vals = data[:, pc_index]
            if np.all(np.isnan(y_vals)):
                continue
            if i_phase >= len(bar_centers_per_target[j_idx]):
                continue
            center = bar_centers_per_target[j_idx][i_phase]
            jitter_width = width * 0.3
            jitter = np.random.uniform(-jitter_width, jitter_width, size=len(y_vals))
            ax1.scatter(center + jitter, y_vals, s=15, color='gray', alpha=0.5, zorder=3)

    ax1.axhline(90, color='gray', linestyle='--', label='90° (orthogonal)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(cond_labels_valid, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel(f"Principal angle (deg) - {pc_name}")
    ax1.set_xlabel("Comparison condition")
    ax1.set_ylim(0, 95)
    ax1.set_title(f"Principal angles ({pc_name}) between reference and each condition\n{task_name}")
    ax1.legend(fontsize=8, loc='upper left', framealpha=0.7)
    ax1.spines[['top', 'right']].set_visible(False)

    ax2 = fig.add_subplot(gs[1])
    ax2.text(0.5, 0.5, "Variance explained\n(see table)", ha='center', va='center', transform=ax2.transAxes)
    ax2.set_axis_off()

    plt.tight_layout()
    figpath = os.path.join(output_dir, f"principal_angles_{task_name.replace(' ', '_')}_{pc_name}.png")
    plt.savefig(figpath, dpi=200, bbox_inches='tight')
    print(f"\nFigure saved: {figpath}")
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
            if data.size > 0:
                mean_mat[i, j] = np.nanmean(data[:, pc_index])
            else:
                mean_mat[i, j] = np.nan
    valid_cols = [j for j in range(n_targets) if not np.all(np.isnan(mean_mat[:, j]))]
    if len(valid_cols) == 0:
        print(f"No valid columns for heatmap {pc_name}, skipping.")
        return
    mean_mat = mean_mat[:, valid_cols]
    cond_labels_valid = [cond_labels[j] for j in valid_cols]
    fig, ax = plt.subplots(figsize=(max(6, len(cond_labels_valid) * 0.6), 4))
    im = ax.imshow(mean_mat, vmin=0, vmax=90, cmap='RdYlGn_r', aspect='auto')
    plt.colorbar(im, ax=ax, label=f'Mean principal angle (deg) - {pc_name}')
    ax.set_xticks(np.arange(len(cond_labels_valid)))
    ax.set_xticklabels(cond_labels_valid, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(np.arange(n_phases))
    ax.set_yticklabels(phase_names)
    ax.set_title(f"Principal angles ({pc_name}) between reference and each condition\n{task_name}")
    for i in range(n_phases):
        for j in range(len(cond_labels_valid)):
            val = mean_mat[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha='center', va='center', fontsize=7,
                        color='white' if val > 55 else 'black')
    plt.tight_layout()
    figpath = os.path.join(output_dir, f"heatmap_{task_name.replace(' ', '_')}_{pc_name}.png")
    plt.savefig(figpath, dpi=200, bbox_inches='tight')
    print(f"Heatmap saved: {figpath}")
    plt.close()

def print_summary_table(results, cond_labels, task_name):
    phase_names = [p for p in results.keys()]
    n_targets = len(cond_labels)
    print(f"\n{'='*80}")
    print(f"Summary - Principal angles (degrees) - {task_name}")
    print(f"{'='*80}")
    for pc_idx in range(3):
        pc_name = f"PC{pc_idx+1}"
        print(f"\n--- {pc_name} ---")
        header = f"{'Phase':<18}" + "".join([f"{lab:>8}" for lab in cond_labels])
        print(header)
        print("-" * len(header))
        for phase in phase_names:
            row = f"{phase:<18}"
            for j in range(n_targets):
                data = results[phase][j]
                if data.size == 0:
                    row += "     NaN"
                else:
                    vals = data[:, pc_idx]
                    if np.all(np.isnan(vals)):
                        row += "     NaN"
                    else:
                        mean_val = np.nanmean(vals)
                        std_val = np.nanstd(vals)
                        row += f"{mean_val:6.1f}±{std_val:4.1f}"
            print(row)
    print()

# ============================================================
#  ENTRY POINT
# ============================================================
if __name__ == "__main__":
    results, evr_ref, evr_targets, cond_labels = run_analysis(
        task_name=task,
        base_dir=base_dir,
        n_components=n_components,
        n_trials_per_cond=n_trials_per_cond,
        dec_duration=decision_fixed_duration,
        output_dir=output_dir,
        visualize=visualize_phases
    )
    print_summary_table(results, cond_labels, task)
    # Nueva tabla de varianza explicada para todas las condiciones
    save_evr_table_multi_distance(evr_ref, evr_targets, cond_labels, task, output_dir, ref_interval)
    for pc in range(3):
        plot_principal_angles_for_pc(results, cond_labels, task, n_components, output_dir, pc)
        plot_heatmap_for_pc(results, cond_labels, task, output_dir, pc)
